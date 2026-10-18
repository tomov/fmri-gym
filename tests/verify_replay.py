"""Prove a block replays: record it with a policy, replay it in another process, compare.

The product of a run is a log that recreates the game frame for frame, and the
only way to know a backend delivers that is to try: play a block, then rebuild
every one of its frames from ``episode_start`` seeds and ``frame`` actions in a
*second* process and compare pixels, logged variables, reward, termination and
-- where the block recorded sound -- each frame's PCM bit for bit. A second
process because the usual culprit is hash order, fixed for the life of one
interpreter (AGENTS.md: crafter passed every in-process check and diverged
across ``PYTHONHASHSEED``). The same pass then restores every savestate
anchor the block holds and plays on from it to the next, so an anchor is shown
to be a branch point and not just bytes; a block's end state (its
``block_end`` record's ``extra.state``) is restored and compared with its last frame.

Record and check::

    uv run python -m tests.verify_replay --curriculum configs/dbp_games/minihack__room5x5.json \\
        --max-frames 200 --n-episodes 2

Check a run folder that already exists (a subject's, say)::

    PYTHONHASHSEED=7 uv run python -m tests.verify_replay --check data/sub-01/ses-001/beh/<run>

Recording uses ``agents/agent_play.py`` with the random policy, so it needs
no window; the check builds each block's adapter from its ``block_start``
line, exactly as :mod:`fmri_gym.replay` does. Exit status is 0 only when every
frame and every anchor of every block matched. The test suite runs the same
two steps on every backend it selects (``tests/test_backends.py``).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
AGENT_PLAY = REPO / "agents" / "agent_play.py"


def record(curriculum: str, outdir: str, max_frames: int, n_episodes: int) -> None:
    """Play ``curriculum``'s game blocks with the random policy into ``outdir``.

    :raises subprocess.CalledProcessError: when the recorder fails.
    """
    env = {**os.environ, "PYTHONHASHSEED": "1"}
    subprocess.run([sys.executable, str(AGENT_PLAY), "--curriculum", curriculum,
                    "--outdir", outdir, "--max-frames", str(max_frames),
                    "--n-episodes", str(n_episodes)], check=True, env=env, cwd=REPO)


def check_run(outdir: str) -> bool:
    """Replay and compare every block under ``outdir``; print one line per block."""
    from fmri_gym.logging import EVENTS_FILENAME

    blocks = sorted(p.parent for p in Path(outdir).glob(f"block-*/{EVENTS_FILENAME}"))
    if not blocks:
        raise FileNotFoundError(f"no block folders under {outdir}")
    ok = True
    for block in blocks:
        result = check_block(str(block))
        ok &= result["pass"]
        print(_report(block.name, result))
    return ok


def check_block(block: str) -> dict[str, Any]:
    """Replay one block's episodes and anchors against its log.

    :return: counts of frames and anchors compared and of mismatches, the first
        mismatch's whereabouts, and ``pass``.
    """
    from fmri_gym.adapters import get_adapter
    from fmri_gym.logging import read_events, read_frames
    from fmri_gym.replay import reconstruct_episode

    events = read_events(block)
    start = events[0]
    lines = [e for e in events if e["type"] == "frame"]
    # A block that ended cleanly may carry the state it ended in (EnvAdapter.block_extra).
    end = events[-1] if events[-1]["type"] == "block_end" else {}
    end_state = (end.get("extra") or {}).get("state")
    frames, index = read_frames(block)
    pixels = {int(i): frames[k] for k, i in enumerate(index)}
    audio = _read_audio(block)
    tally = {"frames": 0, "frame_mismatches": 0, "anchors": 0, "anchor_mismatches": 0,
             "audio_chunks": 0, "first_mismatch": None}
    adapter = get_adapter(start["backend"], start["phase"])
    try:
        for episode_id in sorted({e["episode_id"] for e in lines}):
            episode = [e for e in lines if e["episode_id"] == episode_id]
            reconstruct_episode(block, episode_id, adapter=adapter, n_actions=0)
            tally["frames"] += len(episode)
            tally["frame_mismatches"] += _step_and_compare(adapter, episode, pixels, audio, tally)
            _check_anchors(adapter, episode, pixels, tally)

        if end_state and lines:
            _check_end_state(adapter, end_state, lines[-1], pixels, tally)
    finally:
        adapter.close()
    tally["pass"] = tally["frame_mismatches"] == 0 and tally["anchor_mismatches"] == 0
    return tally


def _read_audio(block: str) -> dict[int, np.ndarray] | None:
    """The block's logged PCM by chunk number, or ``None`` when it logged no sound."""
    from fmri_gym.logging import AUDIO_FILENAME, read_audio

    if not os.path.exists(os.path.join(block, AUDIO_FILENAME)):
        return None
    samples, _rate, offsets = read_audio(block)
    ends = np.append(offsets[1:], len(samples))
    return {k: samples[a:b] for k, (a, b) in enumerate(zip(offsets, ends))}


def _check_anchors(adapter: Any, episode: list[dict], pixels: dict, tally: dict) -> None:
    """Restore each anchor of ``episode`` and play on through the next one, comparing.

    What a restore is judged by is the play that follows it, not the picture
    at the instant of the restore: an emulator's snapshot restores the
    machine, and its screen buffer shows the last frame it emulated until the
    next step (the ALE). The frames stepped after the anchor, up to and
    including the next anchor's, are compared, so an anchor every frame still
    checks one step from each.
    """
    anchors = [k for k, e in enumerate(episode) if "state" in e]
    for n, k in enumerate(anchors):
        adapter.restore(zlib.decompress(base64.b64decode(episode[k]["state"])))
        tally["anchors"] += 1
        until = anchors[n + 1] + 1 if n + 1 < len(anchors) else len(episode)
        tally["anchor_mismatches"] += _step_and_compare(adapter, episode[k + 1:until], pixels,
                                                        None, tally)


def _check_end_state(adapter: Any, state: str, last: dict, pixels: dict, tally: dict) -> None:
    """Restore the block's end state and compare it with the block's last frame."""
    adapter.restore(zlib.decompress(base64.b64decode(state)))
    tally["anchors"] += 1
    if not np.array_equal(np.asarray(adapter.render()), pixels[last["frame"]]):
        tally["anchor_mismatches"] += 1
        _note(tally, last, "pixels differ after restoring the block's end state")


def _step_and_compare(adapter: Any, lines: list[dict], pixels: dict,
                      audio: dict[int, np.ndarray] | None, tally: dict) -> int:
    """Step ``lines``' actions and count the frames whose pixels, variables or sound differ.

    :param audio: the block's logged chunks by number, to compare each frame's
        ``sound()`` against its ``audio_chunk``; ``None`` skips the sound.
    """
    mismatches = 0
    for line in lines:
        obs, reward, terminated, truncated, info = adapter.step(_action(line["action"]))
        frame_state = adapter.capture(obs, info, want_blob=False)
        problems = []
        if not np.array_equal(np.asarray(adapter.render()), pixels[line["frame"]]):
            problems.append("pixels")
        if _jsonable(frame_state.variables) != line["variables"]:
            problems.append("variables " + ", ".join(
                _differing(frame_state.variables, line["variables"])))
        if (float(reward), bool(terminated), bool(truncated)) != (
                float(line["reward"]), line["terminated"], line["truncated"]):
            problems.append("reward/termination")
        if audio is not None and "audio_chunk" in line:
            problems += _compare_sound(adapter.sound(), line["audio_chunk"], audio, tally)
        if problems:
            mismatches += 1
            _note(tally, line, "; ".join(problems))
    return mismatches


def _compare_sound(sound: Any, chunk: int, audio: dict[int, np.ndarray], tally: dict) -> list[str]:
    """This frame's PCM against the chunk the log says it queued (-1: none)."""
    if chunk < 0:
        return ["sound where the log has none"] if sound is not None and len(sound.pcm) else []
    tally["audio_chunks"] += 1
    if sound is None or not np.array_equal(np.asarray(sound.pcm), audio[chunk]):
        return ["audio"]
    return []


def _action(logged: Any) -> Any:
    """The logged action as the env takes it: a JSON list is an array again."""
    return np.asarray(logged) if isinstance(logged, list) else logged


def _jsonable(variables: dict) -> Any:
    """``variables`` as they read back from the log (numpy to lists, the rest to str)."""
    from fmri_gym.logging import _json_default

    return json.loads(json.dumps(variables, default=_json_default))


def _differing(variables: dict, logged: dict) -> list[str]:
    """The names whose value differs between the replayed and the logged variables."""
    fresh = _jsonable(variables)
    return sorted(k for k in set(fresh) | set(logged) if fresh.get(k) != logged.get(k))


def _note(tally: dict, line: dict, what: str) -> None:
    """Remember the first mismatch, where a reader looks first."""
    if tally["first_mismatch"] is None:
        tally["first_mismatch"] = (f"episode {line['episode_id']} ep_frame {line['ep_frame']} "
                                   f"(frame {line['frame']}): {what}")


def _report(name: str, result: dict) -> str:
    verdict = "PASS" if result["pass"] else "FAIL"
    line = (f"{verdict} {name}: {result['frames']} frames, {result['frame_mismatches']} "
            f"mismatched; {result['anchors']} anchors, {result['anchor_mismatches']} mismatched; "
            f"{result['audio_chunks']} sound chunks compared")
    if result["first_mismatch"]:
        line += f"\n     first: {result['first_mismatch']}"
    return line


def check_in_second_process(outdir: str) -> subprocess.CompletedProcess:
    """Run ``--check`` on ``outdir`` in a fresh interpreter under another hash seed.

    :return: the finished process (its ``stdout`` holds the PASS/FAIL lines).
    """
    # A different hash seed from the recording's, so hash-order dependence shows.
    return subprocess.run([sys.executable, __file__, "--check", outdir], check=False,
                          env={**os.environ, "PYTHONHASHSEED": "2"}, cwd=REPO,
                          capture_output=True, text=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--curriculum", help="record this curriculum's game blocks, then check them")
    p.add_argument("--check", metavar="RUN_DIR", help="check an existing run folder instead")
    p.add_argument("--outdir", help="where to record (default: a temporary folder)")
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--n-episodes", type=int, default=2)
    args = p.parse_args()
    if bool(args.curriculum) == bool(args.check):
        p.error("give --curriculum or --check, not both")

    if args.check:
        sys.exit(0 if check_run(args.check) else 1)
    outdir = args.outdir or tempfile.mkdtemp(prefix="verify_replay_")
    record(args.curriculum, outdir, args.max_frames, args.n_episodes)
    print(f"recorded {outdir}; checking in a second process")
    done = check_in_second_process(outdir)
    sys.stdout.write(done.stdout)
    sys.stderr.write(done.stderr)
    sys.exit(done.returncode)


if __name__ == "__main__":
    main()
