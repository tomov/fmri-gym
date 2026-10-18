"""The engine-agnostic core, with no game engine: the log's round trip, the keymaps, the change map.

These run on every machine and on every PR. What they cannot say is whether
a game plays; ``test_backends`` does that, per backend.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces

from fmri_gym.adapters.base import Sound
from fmri_gym.audio import AudioBlockLog
from fmri_gym.adapters.keymap import make_keymap
from fmri_gym.curriculum import validate_curriculum
from fmri_gym.logging import (
    AUDIO_FILENAME, FRAMES_FILENAME, Logger, read_audio, read_events, read_frames, read_savestate,
)
from fmri_gym.replay import frame_arrays, reconstruction_plan
from fmri_gym.run import Clock
from tests.changed import select

PHASE = {"type": "game", "backend": "gym", "game": "CartPole-v1", "fps": 30,
         "keys": {"": 0, "LEFT": 0, "RIGHT": 1}, "mode": "episode", "n_episodes": 1}


def test_logger_round_trip(tmp_path):
    """What the logger writes is what the readers return: lines, pixels, states, sound."""
    clock = Clock()
    clock.trigger()
    logger = Logger(str(tmp_path), "sub-test", [PHASE], clock)
    logger.set_trigger_time()
    name = logger.open_block(0, "gym", "CartPole-v1", PHASE, base_seed=7)
    block = str(tmp_path / name)       # open_block names the folder; the manifest's data_dir
    logger.log(type="episode_start", episode_id=0, seed=7)
    frames = [np.full((4, 6, 3), i, dtype=np.uint8) for i in range(3)]
    states = [b"state-%d" % i for i in range(3)]
    chunks = [np.arange(10, dtype=np.int16).reshape(5, 2), np.ones((3, 2), dtype=np.int16)]
    for i in range(3):
        logger.log_frame({"episode_id": 0, "ep_frame": i, "action": i % 2, "reward": 0.5,
                          "terminated": False, "truncated": i == 2, "run_time": float(i),
                          "flip_time": float(i), "wall_time": 0.0,
                          "variables": {"obs": np.array([i, i])},
                          "audio_chunk": i if i < 2 else -1},
                         frame_rgb=frames[i], savestate=states[i])
        logger.log_audio(Sound(chunks[i], 44100.0) if i < 2 else None)
    logger.log(type="episode_end", episode_id=0, outcome="truncated", score=1.5)
    logger.close_block(n_episodes=1, n_frames=3, extra=None,
                       audio=AudioBlockLog(onsets=[], delay_ms=0.0, resyncs=0, trimmed_samples=0))
    logger.log_phase({"index": 0, "type": "game", "data_dir": name})
    manifest = json.loads(Path(logger.save_manifest()).read_text())
    logger.close()

    events = read_events(block)
    assert [e["type"] for e in events] == (
        ["block_start", "episode_start"] + ["frame"] * 3 + ["episode_end", "block_end"])
    assert events[0]["backend"] == "gym" and events[0]["phase"] == PHASE
    lines = [e for e in events if e["type"] == "frame"]
    assert [e["frame"] for e in lines] == [0, 1, 2]
    assert [read_savestate(e["state"]) for e in lines] == states
    assert lines[1]["variables"] == {"obs": [1, 1]}

    pixels, index = read_frames(block)
    assert pixels.shape == (3, 4, 6, 3) and pixels.dtype == np.uint8
    assert list(index) == [0, 1, 2]
    assert all(np.array_equal(pixels[i], frames[i]) for i in range(3))

    samples, rate, offsets = read_audio(block)
    assert rate == 44100.0 and list(offsets) == [0, 5]
    assert np.array_equal(samples, np.concatenate(chunks))

    arrays = frame_arrays(block)
    assert list(arrays["action"]) == [0, 1, 0] and arrays["obs"].shape == (3, 2)
    plan = reconstruction_plan(block)
    assert plan.seed == 7 and plan.actions == [0, 1, 0]
    assert manifest["phases"][0]["data_dir"] == name
    assert (tmp_path / "manifest.json").exists()


def test_logger_writes_no_audio_file_for_a_silent_block(tmp_path):
    clock = Clock()
    clock.trigger()
    logger = Logger(str(tmp_path), "sub-test", [PHASE], clock)
    name = logger.open_block(0, "gym", "CartPole-v1", PHASE, base_seed=1)
    logger.log_frame({"episode_id": 0}, frame_rgb=np.zeros((2, 2, 3), np.uint8))
    logger.log_audio(None)
    logger.close_block(n_frames=1)
    logger.close()
    assert (tmp_path / name / FRAMES_FILENAME).exists()
    assert not (tmp_path / name / AUDIO_FILENAME).exists()


def test_discrete_keymap_longest_combo_wins():
    km = make_keymap({"keys": {"": 0, "LEFT": 1, "LEFT+SPACE": 2}}, spaces.Discrete(3))
    assert km.resolve(frozenset()) == 0
    assert km.resolve(frozenset({"LEFT"})) == 1
    assert km.resolve(frozenset({"LEFT", "SPACE"})) == 2
    assert km.resolve(frozenset({"X"})) == 0
    assert km.turn_actions() == {"LEFT": 1}          # the no-key entry is not a press


def test_multibinary_keymap_combines_held_keys():
    km = make_keymap({"keys": {"LEFT": 0, "RIGHT": 1, "SPACE": 2}}, spaces.MultiBinary(3))
    assert list(km.resolve(frozenset({"LEFT", "SPACE"}))) == [1, 0, 1]
    assert list(km.resolve(frozenset())) == [0, 0, 0]


def test_box_keymap_takes_lists():
    km = make_keymap({"keys": {"": [0.0], "RIGHT": [1.0]}}, spaces.Box(-1.0, 1.0, (1,)))
    assert np.array_equal(km.resolve(frozenset({"RIGHT"})), np.array([1.0], dtype=np.float32))


def test_keymap_refuses_an_action_outside_the_space():
    with pytest.raises(ValueError):
        make_keymap({"keys": {"": 0, "LEFT": 5}}, spaces.Discrete(3))


@pytest.mark.parametrize("phase, problem", [
    ({**PHASE, "keys": {"": 0, "NOSUCHKEY": 1}}, "NOSUCHKEY"),
    ({k: v for k, v in PHASE.items() if k != "keys"}, "keys"),
    ({**PHASE, "fps": 0}, "fps"),
])
def test_validate_config_refuses(phase, problem):
    problems = validate_curriculum({"curriculum": [phase]})
    assert problems and any(problem in p for p in problems), problems


def test_validate_config_accepts_a_plain_game_phase():
    assert validate_curriculum({"curriculum": [PHASE]}) == []


@pytest.mark.parametrize("paths, direct, core", [
    (["fmri_gym/adapters/crafter.py"], {"crafter"}, False),
    (["fmri_gym/adapters/default.py"], {"gym"}, False),
    (["fmri_gym/adapters/base.py"], set(), True),
    (["fmri_gym/run.py"], set(), True),
    (["agents/agent_play.py", "README.md"], set(), True),
    (["gym/replay_state/replay_state_gym/wrappers.py"],
     {"minihack", "nethack", "vizdoom", "coom", "baba_auto", "rushhour", "aigamestore", "stk"},
     False),
    (["gym/crafter/pyproject.toml"], {"crafter"}, False),
    (["external/coom"], {"coom"}, False),
    (["configs/dbp_games/vizdoom__basic.json"], {"vizdoom"}, False),
    (["configs/demo_classic.json"], {"gym"}, False),
    (["README.md", "docs/SAVESTATES.md", "tests/README.md"], set(), False),
    (["tests/test_core.py"], set(), True),
])
def test_changed_files_select_backends(paths, direct, core):
    selection = select(paths)
    assert selection.direct == direct
    assert selection.core is core
