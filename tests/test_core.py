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
from fmri_gym.bids import task_label
from fmri_gym.adapters.keymap import make_keymap
from fmri_gym.config.curriculum import (
    check_shape, game_changes, is_common_file, load_curriculum, save_curriculum,
    shared_overrides, validate_curriculum,
)
from fmri_gym.logging import (
    AUDIO_FILENAME, FRAMES_FILENAME, Logger, read_audio, read_events, read_frames, read_state,
)
from fmri_gym.replay import frame_arrays, reconstruction_plan
from fmri_gym.run import Clock
from tests.changed import select

PHASE = {"type": "game", "backend": "gym", "game": "CartPole-v1", "fps": 30,
         "controls": {"": 0, "LEFT": 0, "RIGHT": 1}, "mode": "episode", "n_episodes": 1}


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
                         frame=frames[i], state=states[i])
        logger.log_audio(Sound(chunks[i], 44100.0) if i < 2 else None)
    logger.log(type="episode_end", episode_id=0, outcome="truncated", score=1.5)
    logger.close_block(n_episodes=1, n_frames=3, extra=None,
                       audio=AudioBlockLog(onsets=[], delay_ms=0.0, resyncs=0, trimmed_samples=0,
                                         stretched=[]))
    logger.log_phase({"index": 0, "type": "game", "data_dir": name})
    manifest = json.loads(Path(logger.save_manifest()).read_text())
    logger.close()

    events = read_events(block)
    assert [e["type"] for e in events] == (
        ["block_start", "episode_start"] + ["frame"] * 3 + ["episode_end", "block_end"])
    assert events[0]["backend"] == "gym" and events[0]["phase"] == PHASE
    lines = [e for e in events if e["type"] == "frame"]
    assert [e["frame"] for e in lines] == [0, 1, 2]
    assert [read_state(e["state"]) for e in lines] == states
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
    logger.log_frame({"episode_id": 0}, frame=np.zeros((2, 2, 3), np.uint8))
    logger.log_audio(None)
    logger.close_block(n_frames=1)
    logger.close()
    assert (tmp_path / name / FRAMES_FILENAME).exists()
    assert not (tmp_path / name / AUDIO_FILENAME).exists()


def test_discrete_keymap_longest_combo_wins():
    km = make_keymap({"controls": {"": 0, "LEFT": 1, "LEFT+SPACE": 2}}, spaces.Discrete(3))
    assert km.resolve(frozenset()) == 0
    assert km.resolve(frozenset({"LEFT"})) == 1
    assert km.resolve(frozenset({"LEFT", "SPACE"})) == 2
    assert km.resolve(frozenset({"X"})) == 0
    assert km.turn_actions() == {"LEFT": 1}          # the no-key entry is not a press


def test_multibinary_keymap_combines_held_keys():
    km = make_keymap({"controls": {"LEFT": 0, "RIGHT": 1, "SPACE": 2}}, spaces.MultiBinary(3))
    assert list(km.resolve(frozenset({"LEFT", "SPACE"}))) == [1, 0, 1]
    assert list(km.resolve(frozenset())) == [0, 0, 0]


def test_box_keymap_takes_lists():
    km = make_keymap({"controls": {"": [0.0], "RIGHT": [1.0]}}, spaces.Box(-1.0, 1.0, (1,)))
    assert np.array_equal(km.resolve(frozenset({"RIGHT"})), np.array([1.0], dtype=np.float32))


def test_keymap_refuses_an_action_outside_the_space():
    with pytest.raises(ValueError):
        make_keymap({"controls": {"": 0, "LEFT": 5}}, spaces.Discrete(3))


@pytest.mark.parametrize("phase, problem", [
    ({**PHASE, "controls": {"": 0, "NOSUCHKEY": 1}}, "NOSUCHKEY"),
    ({k: v for k, v in PHASE.items() if k != "controls"}, "controls"),
    ({**PHASE, "fps": 0}, "fps"),
    ({**PHASE, "instructions": {"title": "T", "controls_to_dismiss": ["SPACE"],
                                "text": ["a", "![pad](images/no-such.png)"]}}, "no-such.png"),
    ({"type": "message", "text": "![pad](README.md)"}, "README.md"),
])
def test_validate_config_refuses(phase, problem):
    problems = validate_curriculum({"curriculum": [phase]})
    assert problems and any(problem in p for p in problems), problems


def test_validate_config_accepts_a_plain_game_phase():
    assert validate_curriculum({"curriculum": [PHASE]}) == []


def test_an_image_line_is_drawn_as_its_image_between_the_text(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from fmri_gym.display import Display
    import pygame
    display = Display(size=(400, 300), vsync=False)
    try:
        display.draw_text("above\n![pad](images/controllers/xbox-controller-labelled.png)\nbelow")
        assert pygame.transform.average_color(display.canvas)[0] > 50     # the white diagram
        display.draw_text("![pad] (images/controllers/xbox-controller-labelled.png)")
        assert pygame.transform.average_color(display.canvas)[0] < 50     # a line of text
    finally:
        display.close()


@pytest.mark.parametrize("path, label", [
    ("configs/curricula/minihack/levels/room5x5.json", "minihackLevelsRoom5x5"),
    ("/elsewhere/curricula/baba_auto/levels/lock.json", "babaAutoLevelsLock"),
    ("configs/pong_v2.json", "pongV2"),
    ("configs/curricula/top_level.json", "topLevel"),
])
def test_task_label_is_the_path_under_curricula(path, label):
    assert task_label(path) == label


def _write(path: Path, content: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content))
    return path


def test_include_lays_the_phase_over_its_shared_one(tmp_path):
    shared = {**PHASE, "instructions": {"title": "CartPole", "text": ["a", "b"],
                                 "controls_to_dismiss": ["SPACE"]},
              "env_kwargs": {"x": 1, "y": 2}}
    _write(tmp_path / "common" / "cartpole.json", shared)
    path = _write(tmp_path / "level.json", {"curriculum": [
        {"type": "fixation", "duration": 1.0},
        {"__include": "common/cartpole.json", "fps": 15,
         "instructions": {"title": "Level 2"}, "env_kwargs": {"y": 3}},
    ]})
    game = load_curriculum(str(path))["curriculum"][1]
    assert "__include" not in game
    assert game["fps"] == 15 and game["controls"] == PHASE["controls"]
    assert game["instructions"] == {"title": "Level 2", "text": ["a", "b"],
                                    "controls_to_dismiss": ["SPACE"]}
    assert game["env_kwargs"] == {"x": 1, "y": 3}
    assert validate_curriculum({"curriculum": [game]}) == []
    assert is_common_file(str(tmp_path / "common" / "cartpole.json"))
    assert not is_common_file(str(path))


def test_include_nests_relative_to_each_file(tmp_path):
    _write(tmp_path / "common" / "base.json", PHASE)
    _write(tmp_path / "common" / "slow.json", {"__include": "base.json", "fps": 10})
    path = _write(tmp_path / "run.json", {"curriculum": [{"__include": "common/slow.json"}]})
    assert load_curriculum(str(path))["curriculum"][0] == {**PHASE, "fps": 10}


def test_comments_start_with_two_hashes():
    check_shape({"## why": "a note", "curriculum": [{**PHASE, "## keys": "LEFT, RIGHT"}]}, "c")
    with pytest.raises(ValueError, match="comment"):
        check_shape({"_note": "old", "curriculum": [PHASE]}, "c")
    with pytest.raises(ValueError, match="comment"):
        check_shape({"curriculum": [{**PHASE, "_keys_note": "old"}]}, "c")


def _level(tmp_path, monkeypatch) -> Path:
    """A curriculum in tmp_path/levels/ whose game phase includes ../common/game.json."""
    monkeypatch.chdir(tmp_path)
    shared = {k: v for k, v in PHASE.items() if k != "game"}
    _write(tmp_path / "common" / "game.json",
           {**shared, "instructions": {"text": ["a"], "controls_to_dismiss": ["SPACE"]}})
    return _write(tmp_path / "levels" / "one.json", {"curriculum": [
        {"type": "fixation", "duration": 1.0},
        {"__include": "../common/game.json", "game": "CartPole-v0",
         "instructions": {"title": "Level 1"}},
    ]})


def test_an_unchanged_curriculum_saves_as_it_was_written(tmp_path, monkeypatch):
    path = _level(tmp_path, monkeypatch)
    loaded = load_curriculum(str(path), keep_includes=True)
    assert loaded["curriculum"][1]["__include"] == "common/game.json"   # from the working dir
    assert shared_overrides(loaded) == []
    written = json.loads(path.read_text())
    save_curriculum(loaded, str(path))
    assert json.loads(path.read_text()) == written


def test_saving_elsewhere_keeps_the_include_pointing_at_the_shared_phase(tmp_path, monkeypatch):
    path = _level(tmp_path, monkeypatch)
    loaded = load_curriculum(str(path), keep_includes=True)
    save_curriculum(loaded, str(tmp_path / "copy.json"))
    assert json.loads((tmp_path / "copy.json").read_text())["curriculum"][1]["__include"] == \
        "common/game.json"
    assert load_curriculum("copy.json") == load_curriculum(str(path))


def test_a_change_to_the_shared_part_is_said_and_saved_as_the_phase_own(tmp_path, monkeypatch):
    path = _level(tmp_path, monkeypatch)
    loaded = load_curriculum(str(path), keep_includes=True)
    loaded["curriculum"][1]["fps"] = 60
    assert any("fps" in line and "own" in line for line in shared_overrides(loaded))
    save_curriculum(loaded, str(path))
    game = json.loads(path.read_text())["curriculum"][1]
    assert game == {"__include": "../common/game.json", "game": "CartPole-v0",
                    "instructions": {"title": "Level 1"}, "fps": 60}


def test_a_removal_from_the_shared_part_saves_the_phase_in_full(tmp_path, monkeypatch):
    path = _level(tmp_path, monkeypatch)
    loaded = load_curriculum(str(path), keep_includes=True)
    del loaded["curriculum"][1]["instructions"]["text"]
    assert any("instructions.text" in line and "in full" in line
               for line in shared_overrides(loaded))
    save_curriculum(loaded, str(path))
    game = json.loads(path.read_text())["curriculum"][1]
    assert "__include" not in game and "text" not in game["instructions"]


def test_game_changes_ignore_comments_and_resumable_fields():
    then = {**PHASE, "duration": 30.0, "## why": "a"}
    assert game_changes(then, {**PHASE, "duration": 60.0, "## why": "b"}) == []
    assert game_changes(then, {**PHASE, "game": "Acrobot-v1"}) == ["game"]
    assert game_changes({**PHASE, "_note": "a"}, PHASE) == ["_note"]


@pytest.mark.parametrize("include, shared, problem", [
    ("common/missing.json", None, "must name a file"),
    (3, None, "must name a file"),
    ("common/loop.json", {"__include": "loop.json"}, "includes itself"),
    ("common/list.json", [PHASE], "one JSON object"),
])
def test_include_refuses(tmp_path, include, shared, problem):
    if shared is not None:
        _write(tmp_path / str(include), shared)
    path = _write(tmp_path / "run.json", {"curriculum": [{"__include": include}]})
    with pytest.raises(ValueError, match=problem):
        load_curriculum(str(path))


@pytest.mark.parametrize("paths, direct, core", [
    (["fmri_gym/adapters/crafter.py"], {"crafter"}, False),
    (["fmri_gym/adapters/default.py"], {"gym"}, False),
    (["fmri_gym/adapters/base.py"], set(), True),
    (["fmri_gym/run.py"], set(), True),
    (["agents/agent_play.py", "README.md"], set(), True),
    (["gym/replay_state/replay_state_gym/wrappers.py"],
     {"minihack", "nethack", "vizdoom", "coom", "baba_auto", "rushhour", "aigamestore"},
     False),
    (["gym/crafter/pyproject.toml"], {"crafter"}, False),
    (["external/coom"], {"coom"}, False),
    (["configs/curricula/vizdoom/levels/basic.json"], {"vizdoom"}, False),
    (["configs/curricula/gym/classic_control.json"], {"gym"}, False),
    (["README.md", "docs/SAVESTATES.md", "tests/README.md"], set(), False),
    (["tests/test_core.py"], set(), True),
])
def test_changed_files_select_backends(paths, direct, core):
    selection = select(paths)
    assert selection.direct == direct
    assert selection.core is core


def test_a_curriculum_loads_once_into_typed_phases(tmp_path):
    from fmri_gym.config.curriculum import Curriculum
    from fmri_gym.config.phases import Fixation, GamePhase, Message
    path = _write(tmp_path / "run.json", {"curriculum": [
        {"type": "message", "text": ["a", "b"], "control": "any"},
        {"type": "fixation", "duration": 1.5},
        {**PHASE, "seed": 7}, PHASE]})
    loaded = Curriculum.load(str(path))
    assert [type(p) for p in loaded.phases] == [Message, Fixation, GamePhase, GamePhase]
    assert loaded.phases[0].text == "a\nb" and loaded.phases[1].duration == 1.5
    seeded, how = loaded.with_seeds(lambda i: 100 + i)
    assert [p.seed for p in seeded.phases[2:]] == [7, 103] and how == {2: "pinned", 3: "derived"}
    assert seeded.record()[3]["seed"] == 103 and loaded.record()[3] == PHASE
    with pytest.raises(ValueError, match="fps"):
        Curriculum.from_dict({"curriculum": [{**PHASE, "fps": 0}]}, "bad.json")


def test_every_phase_type_is_a_phase():
    from fmri_gym.config.phases import PHASE_CLASSES, Phase
    assert all(issubclass(cls, Phase) and cls.type == name for name, cls in PHASE_CLASSES.items())
    with pytest.raises(TypeError):
        Phase(spec={}, include_on_resume=False)  # abstract: only a phase type is read


def test_a_measurement_that_broke_part_way_is_reported_not_crashed_on():
    from fmri_gym.rig_measure import _html_frames, _html_section
    half = {"summaries": {"frames": {"blocks": []}}}      # it broke before refresh_rate
    note = _html_section(_html_frames, half, ".", {"status": "fail", "why": "KeyError: 'keys'"})
    assert "stopped before it finished" in note and "KeyError" in note


def test_each_rig_measurement_files_its_latencies_beside_the_rig_under_a_new_number(tmp_path):
    from fmri_gym.rig_measure import file_measured
    offset = {"median_ms": 12.0, "sd_ms": 0.3, "n_matched": 10, "n": 10}
    m = {"start_time": "2026-10-20T10:00:00", "task": "rigMeasurement", "status": "fail",
         "failed_tests": "audio", "run": {"label": "sub-rig_ses-001_task-rigMeasurement_run-001"},
         "tests": {"photodiode": {"status": "pass"}, "audio": {"status": "fail"}},
         "summaries": {"photodiode": {"offset": offset, "acoustic_offset": offset}},
         "rig": {"path": "laptop.json"}, "machine": {}, "versions": {}}
    rig_path = str(tmp_path / "laptop.json")
    first = file_measured(m, "run1", rig_path)
    second = file_measured(m, "run2", rig_path)
    assert first == str(tmp_path / "measured" / "laptop-1.json")
    assert second == str(tmp_path / "measured" / "laptop-2.json")
    latency = json.loads(Path(second).read_text())["latency"]
    assert latency["flip_to_photon"]["median_ms"] == 12.0
    assert latency["dac_to_ear"] is None          # its test failed
    m["tests"]["photodiode"]["status"] = "fail"   # no latency left: nothing filed
    assert file_measured(m, "run3", rig_path) is None
    assert len(list((tmp_path / "measured").iterdir())) == 2


def test_a_quick_rig_measurement_runs_on_its_own_and_files_its_report(tmp_path, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from dataclasses import replace

    import pygame

    from fmri_gym import rig_measure
    from fmri_gym.config.rig import RigConfig
    rig_path = tmp_path / "laptop.json"
    rig_path.write_text(json.dumps({"data_root": str(tmp_path / "data"), "has_audio": False,
                                    "triggers": {"sync": {"mode": "none"}}}))
    wait_for_control = rig_measure._wait_for_control

    def press_space_then_wait(inputs, display, control):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE, mod=0,
                                             unicode=" ", scancode=0))
        wait_for_control(inputs, display, control)

    monkeypatch.setattr(rig_measure, "_wait_for_control", press_space_then_wait)
    monkeypatch.setattr(rig_measure, "lsl_probe", lambda: "not probed in the test")
    monkeypatch.setattr(rig_measure, "machine", lambda: {})
    monkeypatch.setattr(rig_measure, "_VERDICT_S", 0.0)
    monkeypatch.setattr(rig_measure, "_SUMMARY_S", 0.0)
    settings = replace(rig_measure.QUICK, display_flips=10, frame_seconds=0.2)
    rig_measure.measure_rig(str(rig_path), settings)
    folder, = (tmp_path / "data" / "sub-rig" / "ses-001" / "beh").iterdir()
    m = json.loads((folder / "manifest.json").read_text())
    assert m["task"] == "rigMeasurement" and set(m["tests"]) == set(rig_measure.TESTS)
    assert list(m["summaries"]) == list(rig_measure.MEASUREMENTS)
    assert all(m["tests"][t]["status"] == "not run" for t in ("triggers", "controls",
                                                              "photodiode", "audio"))
    assert len(m["summaries"]["frames"]["blocks"]) == 2
    for name in ("report.html", "report.md", "rigmeasurement.tsv", "display.npz", "frames.npz"):
        assert (folder / name).exists(), name
    assert (tmp_path / "data" / "rigmeasurements.tsv").exists()
    assert RigConfig.open(str(rig_path)).measured_latency_warning   # no latency measured


def test_the_machine_is_described_and_pygame_left_as_it_was():
    import pygame

    from fmri_gym.rig_measure import machine
    pygame.quit()
    m = machine()
    assert {"cpu", "memory_gb", "displays", "gpus", "audio_devices", "input_devices",
            "usb_devices", "system", "hostname"} <= set(m)
    assert not pygame.display.get_init()


def test_a_rig_with_no_measured_latencies_says_so_until_one_is_filed(tmp_path):
    from fmri_gym.config.rig import RigConfig, next_measured_latency_file
    rig_path = tmp_path / "laptop.json"
    rig_path.write_text(json.dumps({"data_root": str(tmp_path / "data")}))
    rig = RigConfig.open(str(rig_path))
    assert "no measured latencies" in rig.measured_latency_warning
    assert rig.measured_latency_files == []
    first = next_measured_latency_file(str(rig_path))
    assert first == str(tmp_path / "measured" / "laptop-1.json")
    (tmp_path / "measured").mkdir()
    Path(first).write_text("{}")
    rig = RigConfig.open(str(rig_path))
    assert rig.measured_latency_warning is None and rig.measured_latency_files == [first]


def test_claude_md_imports_agents_md():
    """Claude Code reads AGENTS.md on its own only while no CLAUDE.md exists: the import keeps it."""
    lines = (Path(__file__).resolve().parents[1] / "CLAUDE.md").read_text().splitlines()
    assert lines[0] == "@AGENTS.md", "CLAUDE.md must start with the line @AGENTS.md"
