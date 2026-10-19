"""Every selected backend, through the chain a session depends on.

Four checks per backend, each on the configs that stand for it (``tests/backends.py``):

1. **playable** -- the adapter builds from the phase, resets on a seed, steps
   the phase's own keys, and what it hands the loop is what the loop takes:
   an RGB frame, JSON-able variables, a savestate or none, PCM shaped
   ``(samples, channels)`` or none, an outcome from the fixed vocabulary.
2. **recorded** -- a block played by the random policy (``agents/agent_play.py``,
   no window) writes a log of the documented shape: the lines in order, every
   frame's pixels in ``frames.h5``, every chunk of sound in ``audio.h5`` when
   the engine makes any, the savestates where the stride says.
3. **replays** -- that log, rebuilt in a second process under another hash
   seed, matches bit for bit (``tests/verify_replay.py``).
4. **runs** -- the real ``fmri-play`` loop plays the config headless, with a
   key script at the keyboard, and leaves a run folder a subject's would look like
   (``tests/headless.py``).
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import gymnasium
import numpy as np
import pytest

from fmri_gym.adapters import get_adapter
from fmri_gym.logging import (
    AUDIO_FILENAME, EVENTS_FILENAME, FRAMES_FILENAME, _json_default, read_audio, read_events,
    read_frames,
)
from tests import verify_replay
from tests.backends import REPO, game_phases
from tests.conftest import write_curriculum

OUTCOMES = {"playing", "terminated", "truncated", "won", "lost"}
STEPS = 12


def _check_frame(frame) -> None:
    frame = np.asarray(frame)
    assert frame.ndim == 3 and frame.shape[2] == 3, f"render() gave shape {frame.shape}"
    assert frame.dtype == np.uint8, f"render() gave dtype {frame.dtype}"
    assert frame.shape[0] > 0 and frame.shape[1] > 0


def test_playable(game_phase):
    backend, path, phase = game_phase
    try:
        adapter = get_adapter(backend, dict(phase))
    except (ModuleNotFoundError, gymnasium.error.DependencyNotInstalled) as e:
        if backend != "gym":
            raise
        # The gym backend plays any registered env; its box2d / mujoco families
        # are extras of their own, so a config needing one is a visible skip.
        pytest.skip(f"{phase['game']} needs an extra not synced here: {e}")
    rng = random.Random(0)
    seed = int(phase.get("seed", 1000))
    try:
        adapter.reset(seed)
        _check_frame(adapter.render())
        pool = [adapter.keymap.resolve(combo) for combo in adapter.keymap.combos]
        assert pool, f"{path}: the phase's keys resolve to no action"
        terminated = truncated = False
        for _ in range(STEPS):
            obs, reward, terminated, truncated, info = adapter.step(rng.choice(pool))
            float(reward)
            _check_frame(adapter.render())
            fs = adapter.capture(obs, info, want_blob=True)
            json.dumps(fs.variables, default=_json_default)
            assert fs.blob is None or isinstance(fs.blob, (bytes, bytearray))
            sound = adapter.sound()
            if sound is not None:
                pcm = np.asarray(sound.pcm)
                assert pcm.ndim == 2, f"sound() must be (samples, channels), got {pcm.shape}"
                assert sound.sample_rate > 0
            if terminated or truncated:
                seed += 1
                adapter.reset(seed)
        outcome, message = adapter.outcome(terminated, truncated)
        assert outcome in OUTCOMES and isinstance(message, str)
        extra = adapter.block_extra()
        assert extra is None or isinstance(extra, dict)
        native = adapter.native_fps()
        assert native is None or native > 0
    finally:
        adapter.close()


# --- a recorded block, shared by the log, audio and replay checks --------------

@dataclass
class Recording:
    """One smoke config's first game phase, played by the random policy."""

    backend: str
    config: Path
    phase: dict
    outdir: Path
    block: Path
    record_log: str


@pytest.fixture
def recorded(smoke_config, plan, tmp_path_factory, request) -> Recording:
    cache = request.config.fmri_recordings
    if smoke_config.path not in cache:
        outdir = tmp_path_factory.mktemp(f"rec-{Path(smoke_config.path).stem}")
        phase = game_phases(REPO / smoke_config.path)[0]
        config = write_curriculum([phase], outdir / "curriculum.json")
        try:
            verify_replay.record(str(config), str(outdir / "run"), plan.frames, 2)
        except subprocess.CalledProcessError as e:
            pytest.fail(f"recording {smoke_config.path} failed (exit {e.returncode})",
                        pytrace=False)
        blocks = sorted((outdir / "run").glob("block-*"))
        assert len(blocks) == 1, f"expected one block folder, found {blocks}"
        cache[smoke_config.path] = Recording(smoke_config.backend, config, phase, outdir,
                                             blocks[0], "")
    return cache[smoke_config.path]


def test_recorded_log(recorded):
    events = read_events(str(recorded.block))
    assert events[0]["type"] == "block_start" and events[-1]["type"] == "block_end"
    assert events[0]["backend"] == recorded.backend
    assert events[0]["phase"]["game"] == recorded.phase["game"]
    frames = [e for e in events if e["type"] == "frame"]
    assert frames, "no frame lines"
    assert [e["frame"] for e in frames] == list(range(len(frames)))
    starts = [e for e in events if e["type"] == "episode_start"]
    base = events[0]["base_seed"]
    assert [e["seed"] for e in starts] == [base + i for i in range(len(starts))]
    assert len([e for e in events if e["type"] == "episode_end"]) == len(starts)
    assert events[-1]["n_frames"] == len(frames)
    for e in frames:
        assert isinstance(e["variables"], dict)
        assert {"action", "reward", "terminated", "truncated", "ep_frame"} <= set(e)

    pixels, index = read_frames(str(recorded.block))
    assert pixels.dtype == np.uint8 and pixels.ndim == 4 and pixels.shape[3] == 3
    assert list(index) == [e["frame"] for e in frames], "frames.h5 does not cover every frame"

    # Savestates: every `state_stride` frames for an engine with a snapshot, once
    # at block end for a replay-backed one, none at all for an env that replays
    # from its seed and actions alone (the default adapter). The replay check is
    # what proves each of those reconstructs; this only checks the stride.
    stride = max(1, int(recorded.phase.get("state_stride", 1)))
    anchors = [e for e in frames if "state" in e]
    assert all(e["ep_frame"] % stride == 0 for e in anchors), "a savestate off the stride"
    if anchors:
        assert [e["frame"] for e in frames if e["ep_frame"] % stride == 0] == \
            [e["frame"] for e in anchors], "a frame on the stride without its savestate"

    manifest = json.loads((recorded.outdir / "run" / "manifest.json").read_text())
    entry = [p for p in manifest["phases"] if p.get("type") == "game"]
    assert entry and entry[0]["n_frames"] == len(frames)


def test_recorded_audio(recorded):
    events = read_events(str(recorded.block))
    frames = [e for e in events if e["type"] == "frame"]
    numbered = [e["audio_chunk"] for e in frames if e.get("audio_chunk", -1) >= 0]
    audio_file = recorded.block / AUDIO_FILENAME
    if not numbered:
        assert not audio_file.exists(), "audio.h5 exists but no frame names a chunk"
        pytest.skip(f"{recorded.backend}: the engine made no sound in this block")
    assert audio_file.exists(), "frames name sound chunks but audio.h5 is missing"
    assert numbered == list(range(len(numbered))), "chunk numbers are not 0, 1, 2, ..."
    samples, rate, offsets = read_audio(str(recorded.block))
    assert rate > 0 and samples.ndim == 2 and samples.shape[1] >= 1
    assert len(offsets) == len(numbered), "one chunk per frame that names one"
    assert offsets[-1] < len(samples)


def test_replays_bit_exact(recorded):
    done = verify_replay.check_in_second_process(str(recorded.outdir / "run"))
    assert done.returncode == 0, (
        f"replay of {recorded.config} does not match its log:\n{done.stdout}\n{done.stderr[-2000:]}")
    assert "PASS" in done.stdout


# --- the real loop, headless -------------------------------------------------

def _audio_output_here() -> bool:
    try:
        import sounddevice
        sounddevice.query_devices(kind="output")
        return True
    except Exception:
        return False


def test_headless_run(smoke_config, plan, tmp_path):
    phase = game_phases(REPO / smoke_config.path)[0]
    config = write_curriculum([{"type": "fixation", "duration": 0.5, "include_on_resume": True},
                               phase, {"type": "fixation", "duration": 0.5}],
                              tmp_path / f"{Path(smoke_config.path).stem}.json")
    no_audio = plan.no_audio or not _audio_output_here()
    cmd = [sys.executable, "-m", "tests.headless", "--curriculum", str(config),
           "--data-root", str(tmp_path / "data"), "--duration", str(plan.run_seconds)]
    cmd += ["--no-audio"] * no_audio + ["--window"] * plan.has_window
    env = {**os.environ}
    if not plan.has_window:
        env["SDL_VIDEODRIVER"] = "dummy"
    done = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                          timeout=plan.run_seconds + 180, env=env)
    assert done.returncode == 0, f"fmri-play exited {done.returncode}:\n{done.stderr[-3000:]}"

    manifests = list((tmp_path / "data").glob("sub-test/ses-001/beh/*/manifest.json"))
    assert len(manifests) == 1, f"expected one run folder, found {manifests}"
    manifest = json.loads(manifests[0].read_text())
    games = [p for p in manifest["phases"] if p.get("type") == "game"]
    assert len(games) == 1, f"the game phase did not finish logging: {manifest['phases']}"
    assert games[0]["n_frames"] > 0, "the game block logged no frame: were the keys read?"
    block = manifests[0].parent / games[0]["data_dir"].split("/")[-1]
    assert (block / EVENTS_FILENAME).exists() and (block / FRAMES_FILENAME).exists()
    events = read_events(str(block))
    frames = [e for e in events if e["type"] == "frame"]
    assert len(frames) == games[0]["n_frames"]
    assert any(e["type"] == "input_event" for e in events), "no key press was logged"
    if not no_audio and any(e.get("audio_chunk", -1) >= 0 for e in frames):
        assert (block / AUDIO_FILENAME).exists(), "sound was played but audio.h5 is missing"
