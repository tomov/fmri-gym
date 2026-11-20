"""The engine-agnostic core, with no game engine: the log's round trip, the keymaps, the change map.

These run on every machine and on every PR. What they cannot say is whether
a game plays; ``test_backends`` does that, per backend.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from gymnasium import spaces

from humans.outputs.sound import Sound
from humans.save.bids import task_label
from humans.adapters.base import EnvAdapter, HudSettings
from humans.adapters.keymap import make_keymap
from humans.config.curriculum import (
    check_shape, game_changes, is_common_file, load_curriculum, save_curriculum,
    shared_overrides, validate_curriculum,
)
from humans.save.logging import (
    AUDIO_FILENAME, EVENTS_FILENAME, FORMAT, FRAMES_FILENAME, GAME_EVENTS_FILENAME,
    RUN_EVENTS_FILENAME, VERSION,
    AudioBlockLog, Logger, read_audio, read_events, read_frames, read_lines, read_state,
)
from humans.save.replay import frame_arrays, reconstruction_plan
from tests.changed import select

#: A screen geometry for the rig files written here.
GEOMETRY = {"width_cm": 28.6, "height_cm": 17.9, "distance_cm": 50.0}
PHASE = {"type": "game", "backend": "gym", "game": "CartPole-v1", "fps": 30,
         "controls": {"": 0, "LEFT": 0, "RIGHT": 1}, "mode": "episode", "n_episodes": 1}


def test_logger_round_trip(tmp_path):
    """What the logger writes is what the readers return: lines, pixels, states, sound."""
    logger = Logger(str(tmp_path), "sub-test", [PHASE])
    logger.log_trigger_time(perf_time=100.0, wall_time=1.7e9)
    name = logger.open_block(0, "gym", "CartPole-v1", PHASE, base_seed=7, run_time=0.5)
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
    logger.close_block(n_episodes=1, n_frames=3, extra=None, run_time=3.5,
                       audio=AudioBlockLog(onsets=[(0, 1.5)], delay_ms=0.0,
                                           resyncs=0, trimmed_samples=0, stretched=[]))
    logger.log_phase({"index": 0, "type": "game", "data_dir": name})
    manifest = json.loads(Path(logger.save_manifest()).read_text())
    logger.close()

    events = read_events(block)
    assert [e["type"] for e in events] == (
        ["block_start", "episode_start"] + ["frame"] * 3 + ["episode_end", "block_end"])
    assert events[0]["backend"] == "gym" and events[0]["phase"] == PHASE
    assert (events[0]["run_time"], events[-1]["run_time"]) == (0.5, 3.5)
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
    with h5py.File(Path(block) / AUDIO_FILENAME, "r") as f:
        assert list(f["onset_chunk"][:]) == [0] and list(f["onset_time"][:]) == [1.5]

    arrays = frame_arrays(block)
    assert list(arrays["action"]) == [0, 1, 0] and arrays["obs"].shape == (3, 2)
    plan = reconstruction_plan(block)
    assert plan.seed == 7 and plan.actions == [0, 1, 0]
    assert manifest["phases"][0]["data_dir"] == name
    assert (manifest["trigger_perf"], manifest["start_epoch"]) == (100.0, 1.7e9)
    assert not (tmp_path / RUN_EVENTS_FILENAME).exists(), \
        "a run that logs no run line (an agent's) has no run's log"


def test_logger_writes_no_audio_file_for_a_silent_block(tmp_path):
    logger = Logger(str(tmp_path), "sub-test", [PHASE])
    name = logger.open_block(0, "gym", "CartPole-v1", PHASE, base_seed=1)
    logger.log_frame({"episode_id": 0}, frame=np.zeros((2, 2, 3), np.uint8))
    logger.log_audio(None)
    logger.close_block(n_frames=1)
    logger.close()
    assert (tmp_path / name / FRAMES_FILENAME).exists()
    assert not (tmp_path / name / AUDIO_FILENAME).exists()


def test_each_line_goes_once_to_the_log_of_what_produced_it(tmp_path):
    logger = Logger(str(tmp_path), "sub-test", [PHASE])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["log_format"] == {"format": FORMAT, "version": VERSION}, \
        "the manifest says how to read the logs before any is written"
    logger.log_run_event("input_event", 1.0, name="5", down=True)       # between blocks
    with pytest.raises(RuntimeError, match="no block open"):
        logger.log(type="episode_start")
    name = logger.open_block(0, "gym", "CartPole-v1", PHASE, base_seed=1)
    logger.log_run_event("input_event", 2.0, name="5", down=False)      # during the block
    logger.log_frame({"episode_id": 0}, frame=np.zeros((2, 2, 3), np.uint8))
    logger.close_block(n_frames=1)
    logger.close()
    assert read_lines(str(tmp_path / RUN_EVENTS_FILENAME)) == [
        {"type": "input_event", "perf_time": 1.0, "block": None, "name": "5", "down": True},
        {"type": "input_event", "perf_time": 2.0, "block": name, "name": "5", "down": False}]
    assert [line["type"] for line in read_events(str(tmp_path / name))] == [
        "block_start", "frame", "block_end"]
    assert len(read_frames(str(tmp_path / name))[0]) == 1


def test_the_console_is_logged_line_by_line_until_the_log_closes(tmp_path):
    import sys
    logger = Logger(str(tmp_path), "sub-test", [PHASE])
    logger.copy_console()
    try:
        print("one\ntwo")
        print("half", end="", file=sys.stderr)
    finally:
        logger.close()
    print("after")
    lines = read_lines(str(tmp_path / RUN_EVENTS_FILENAME))
    assert [(line["type"], line["stream"], line["text"]) for line in lines] == [
        ("console", "stdout", "one"), ("console", "stdout", "two"), ("console", "stderr", "half")]
    assert all(isinstance(line["perf_time"], float) for line in lines)


def test_each_flip_is_logged_with_what_it_showed(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from types import SimpleNamespace

    from humans.outputs.display import Display, PhotodiodePatch
    logged = []
    logger = SimpleNamespace(log_run_event=lambda line_type, perf_time, **fields:
                             logged.append((line_type, perf_time, fields)))
    display = Display(size=(200, 150), vsync=False, photodiode_patch=PhotodiodePatch("br", 10),
                      logger=logger)
    logged.clear()                                      # the window's opening
    try:
        text_flip = display.draw_text("Press SPACE")
        fixation_flip = display.draw_fixation()
        repeat_flip = display.redraw()
        display.alternate_photodiode_patch()
        frame_flip = display.draw_frame(np.zeros((4, 4, 3), np.uint8))
    finally:
        display.close()
    assert logged == [
        ("flip", text_flip, {"kind": "text", "text": "Press SPACE",
                             "is_photodiode_patch_lit": False}),
        ("flip", fixation_flip, {"kind": "fixation", "is_photodiode_patch_lit": False}),
        ("flip", repeat_flip, {"kind": "repeat", "is_photodiode_patch_lit": False}),
        ("flip", frame_flip, {"kind": "game_frame", "is_photodiode_patch_lit": True})]


def test_each_code_a_line_sends_is_logged_and_none_from_a_line_that_sends_nothing():
    from types import SimpleNamespace

    from humans.sync.triggers import Triggers
    logged = []
    logger = SimpleNamespace(log_run_event=lambda line_type, perf_time, **fields:
                             logged.append((line_type, fields)))
    triggers = Triggers.from_dict(None, logger)             # the null line sends nothing
    triggers.lifecycle("task_start")
    triggers.frame()
    triggers.block_end()
    assert logged == []
    triggers.enabled = True
    value = triggers.lifecycle("task_stop")
    frame_value = triggers.frame()
    triggers.block_end()
    assert logged == [("trigger_sent", {"name": "task_stop", "value": value}),
                      ("trigger_sent", {"name": "frame", "value": frame_value}),
                      ("trigger_sent", {"name": "clear", "value": 0})]


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
    ({**PHASE, "hud": {"show_score": "no"}}, "hud.show_score"),
    ({**PHASE, "hud": {"show_clock": False}}, "show_clock"),
    ({**PHASE, "hud": False}, "hud"),
    ({**PHASE, "menu": {"control": "X", "text": ["![pad](images/no-such.png)"]}}, "no-such.png"),
    ({**PHASE, "menu": {"control": "X", "text": [1]}}, "menu.text"),
    ({**PHASE, "hint": {"control": "X"}}, "hint.text"),
    ({**PHASE, "hint": {"text": "hi"}}, "hint.control"),
    ({**PHASE, "hint": {"control": "X", "text": "hi", "hold": -1}}, "hint.hold"),
    ({**PHASE, "hint": {"control": "X", "text": "hi", "colour": "red"}}, "colour"),
])
def test_validate_config_refuses(phase, problem):
    problems = validate_curriculum({"curriculum": [phase]})
    assert problems and any(problem in p for p in problems), problems


def test_validate_config_accepts_a_plain_game_phase():
    assert validate_curriculum({"curriculum": [PHASE]}) == []
    hud = {"show_score": False, "show_time_remaining": True}
    assert validate_curriculum({"curriculum": [{**PHASE, "hud": hud}]}) == []
    menu = {"control": "X", "text": ["![pad](images/controllers/xbox-controller.png)", "", "hi"]}
    assert validate_curriculum({"curriculum": [{**PHASE, "menu": menu}]}) == []
    hint = {"control": "X", "hold": 3.0, "text": ["![pad](images/controllers/xbox-controller.png)"]}
    assert validate_curriculum({"curriculum": [{**PHASE, "hint": hint}]}) == []


def test_a_hint_shows_once_held_long_enough_and_until_let_go(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from humans.popup import Hint
    from humans.outputs.display import Display
    clock = iter(range(100_000))
    monkeypatch.setattr("humans.popup.time.perf_counter", lambda: float(next(clock)))
    held = [frozenset({"X"})] * 6 + [frozenset()]
    held_now = lambda: held.pop(0) if len(held) > 1 else held[0]     # noqa: E731
    hint = Hint({"control": "X", "hold": 2.0, "text": "hi"}, 0.0, held_now)
    assert [hint.armed() for _ in range(3)] == [False, False, True] and hint.pending

    display = Display(size=(200, 150), vsync=False)
    try:
        assert hint.run(display, lambda: 1.0, lambda: []) == "resume"
    finally:
        display.close()
    assert not hint.pending and held == [frozenset()]
    assert hint.describe()["events"] == [{"opened": 1.0, "closed": 1.0}]


@pytest.mark.parametrize("hud, lines", [
    ({}, ["12 s", "Score: 3"]),
    ({"show_score": False}, ["12 s", ""]),
    ({"show_time_remaining": False}, ["", "Score: 3"]),
    ({"show_score": False, "show_time_remaining": False}, None),
])
def test_the_hud_keeps_time_left_and_score_right_whichever_is_hidden(hud, lines):
    adapter = EnvAdapter.__new__(EnvAdapter)
    adapter.hud_settings = HudSettings.from_dict(hud)
    assert adapter.hud(3.0, 12.4) == lines


def test_an_image_line_is_drawn_as_its_image_between_the_text(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from humans.outputs.display import Display
    import pygame
    display = Display(size=(400, 300), vsync=False)
    try:
        display.draw_text("above\n![pad](images/controllers/xbox-controller-labelled.png)\nbelow")
        assert pygame.transform.average_color(display.canvas)[0] > 50     # the white diagram
        display.draw_text("![pad] (images/controllers/xbox-controller-labelled.png)")
        assert pygame.transform.average_color(display.canvas)[0] < 50     # a line of text
    finally:
        display.close()


def test_the_photodiode_patch_alternates_with_each_game_frame_and_is_black_on_text(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import numpy as np
    import pygame

    from humans.outputs.display import Display, PhotodiodePatch
    patch = PhotodiodePatch("tl", 10)
    assert PhotodiodePatch("br", 10).rect((200, 150)) == pygame.Rect(190, 140, 10, 10)
    display = Display(size=(200, 150), vsync=False, photodiode_patch=patch)

    def patch_colour() -> tuple:
        return tuple(display.screen.get_at((5, 5)))[:3]
    try:
        white_game = np.full((15, 20, 3), 255, dtype=np.uint8)   # the patch is not the game's
        display.draw_frame(white_game)
        assert patch_colour() == (0, 0, 0)
        display.alternate_photodiode_patch()
        display.draw_frame(white_game)
        assert patch_colour() == (255, 255, 255)
        display.redraw()                                          # the same frame: no edge
        assert patch_colour() == (255, 255, 255)
        display.alternate_photodiode_patch()
        display.draw_frame(white_game)
        assert patch_colour() == (0, 0, 0)
        display.alternate_photodiode_patch()
        display.draw_fixation()
        assert patch_colour() == (0, 0, 0)
    finally:
        display.close()


def test_a_window_without_the_keyboard_focus_says_so(monkeypatch, capsys):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from humans.outputs.display import Display
    import pygame
    display = Display(size=(200, 150), vsync=False)
    try:
        assert display.take_focus()
        monkeypatch.setattr("humans.outputs.display.FOCUS_WAIT_S", 0.05)
        monkeypatch.setattr(pygame.key, "get_focused", lambda: False)
        assert not display.take_focus()
    finally:
        display.close()
    assert "does NOT have the keyboard focus" in capsys.readouterr().err


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
    (["humans/adapters/crafter.py"], {"crafter"}, False),
    (["humans/adapters/default.py"], {"gym"}, False),
    (["humans/adapters/base.py"], set(), True),
    (["humans/run.py"], set(), True),
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
    from humans.config.curriculum import Curriculum
    from humans.config.phases import Fixation, GamePhase, Message
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
    from humans.config.phases import PHASE_CLASSES, Phase
    assert all(issubclass(cls, Phase) and cls.type == name for name, cls in PHASE_CLASSES.items())
    with pytest.raises(TypeError):
        Phase(spec={}, include_on_resume=False)  # abstract: only a phase type is read


def test_a_measurement_that_broke_part_way_is_reported_not_crashed_on():
    from humans.standalone.rig_setup.report import _html_frames, _html_section
    half = {"summaries": {"frames": {"blocks": []}}}      # it broke before refresh_rate
    note = _html_section(_html_frames, half, ".", {"status": "fail", "why": "KeyError: 'keys'"})
    assert "stopped before it finished" in note and "KeyError" in note


def test_each_rig_measurement_files_its_latencies_beside_the_rig_under_a_new_number(tmp_path):
    from humans.standalone.rig_setup.report import file_measured
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


def _measure_laptop_rig(tmp_path, monkeypatch, measurement_names, **rig) -> dict:
    """A quick rig measurement of a laptop rig (``rig``: more of its file), headless, taking
    ``measurement_names``; its manifest, once its files are checked."""
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from dataclasses import replace

    import pygame

    from humans.config.rig import RigConfig
    from humans.standalone.rig_setup import cli, report
    from humans.standalone.rig_setup import measure as rig_measure
    rig_path = tmp_path / "laptop.json"
    rig_path.write_text(json.dumps({"data_root": str(tmp_path / "data"), "has_audio": False,
                                    "screen": {"geometry": GEOMETRY},
                                    "triggers": {"sync": {"mode": "none"}}, **rig}))
    wait_for_control = rig_measure._wait_for_control

    def press_space_then_wait(inputs, display, control, logger):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE, mod=0,
                                             unicode=" ", scancode=0))
        wait_for_control(inputs, display, control, logger)

    monkeypatch.setattr(rig_measure, "_wait_for_control", press_space_then_wait)
    monkeypatch.setattr(rig_measure, "lsl_probe", lambda: "not probed in the test")
    monkeypatch.setattr(report, "machine", lambda: {})
    monkeypatch.setattr(rig_measure, "_VERDICT_S", 0.0)
    monkeypatch.setattr(rig_measure, "_SUMMARY_S", 0.0)
    settings = replace(rig_measure.QUICK, display_flips=10, frame_seconds=0.2)
    cli.measure_rig(str(rig_path), settings, measurement_names)
    folder, = (tmp_path / "data" / "sub-rig" / "ses-001" / "beh").iterdir()
    m = json.loads((folder / "manifest.json").read_text())
    assert m["task"] == "rigMeasurement" and set(m["tests"]) == set(rig_measure.TESTS)
    assert list(m["summaries"]) == list(rig_measure.MEASUREMENTS)
    for name in ("report.html", "report.md", "rigmeasurement.tsv", "display.npz"):
        assert (folder / name).exists(), name
    assert (tmp_path / "data" / "rigmeasurements.tsv").exists()
    assert RigConfig.open(str(rig_path)).measured_latency_warning   # no latency measured
    return m


def test_a_quick_rig_measurement_runs_on_its_own_and_files_its_report(tmp_path, monkeypatch):
    from humans.standalone.rig_setup.measure import MEASUREMENTS

    m = _measure_laptop_rig(tmp_path, monkeypatch, tuple(MEASUREMENTS))
    assert all(m["tests"][t]["status"] == "not run" for t in ("triggers", "controls",
                                                              "photodiode", "audio"))
    assert len(m["summaries"]["frames"]["blocks"]) == 2


def test_a_measurement_unticked_is_not_run_and_fails_nothing(tmp_path, monkeypatch):
    from humans.standalone.rig_setup.measure import UNTICKED

    # the rig's controller is not plugged in: with its test unticked, it is not looked for
    m = _measure_laptop_rig(tmp_path, monkeypatch, ("display",),
                            inputs=["keyboard", "xbox_controller"])
    assert m["tests"]["frames"] == {"status": "not run", "why": UNTICKED}
    assert m["tests"]["controls"] == {"status": "not run", "why": UNTICKED}
    assert m["summaries"]["frames"] == {} and m["tests"]["display"]["status"] != "not run"
    assert "frames" not in (m["failed_tests"] or "")


def test_the_machine_is_described_and_pygame_left_as_it_was():
    import pygame

    from humans.standalone.rig_setup.report import machine
    pygame.quit()
    m = machine()
    assert {"cpu", "memory_gb", "displays", "gpus", "audio_devices", "input_devices",
            "usb_devices", "system", "hostname"} <= set(m)
    assert not pygame.display.get_init()


def test_a_rig_with_no_measured_latencies_says_so_until_one_is_filed(tmp_path):
    from humans.config.rig import RigConfig, next_measured_latency_file
    rig_path = tmp_path / "laptop.json"
    rig_path.write_text(json.dumps({"data_root": str(tmp_path / "data"),
                                    "screen": {"geometry": GEOMETRY}}))
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


def test_the_controls_a_calibration_asked_for_are_not_asked_for_again():
    import types

    from humans.inputs import ControlNames
    from humans.standalone.rig_setup.measure import _controls_to_ask

    def pad(name, buttons, sticks=()):
        return types.SimpleNamespace(name=name, BUTTON_NAMES=buttons, axis_controls=lambda: sticks)

    button_a, up = ControlNames.CONTROLLER_BUTTON_A, ControlNames.CONTROLLER_AXIS_LEFTY_UP
    calibrated = pad("calibrated", (button_a,), (up,))
    stopped = pad("stopped", (button_a,), (up,))
    no_stick = pad("no_stick", (ControlNames.CONTROLLER_BUTTON_B,))
    assert _controls_to_ask([(calibrated, {"measured": True})]) == {}
    assert _controls_to_ask([(stopped, {"measured": False}), (no_stick, {"measured": False})]) == {
        button_a: "on the stopped", up: "on the stopped",
        ControlNames.CONTROLLER_BUTTON_B: "on the no_stick"}


# -- the logs read together: the timeline, the events table, the dataset --------------------


LABEL = "sub-01_ses-001_task-x_run-001"
BLOCK = "block-00_gym_x"


def _run_folder(data_root: Path, folder: str = LABEL, has_started: bool = True,
                log_format: int | None = VERSION) -> Path:
    """A run's folder as a run leaves it: its manifest, its log and a block's.

    The trigger is at perf_counter 100; the display refreshes at 100 Hz; LEFT is pressed in the
    block's episode, a pop-up's B is given to no step, the scanner's = gives one pulse.
    """
    run = data_root / "sub-01" / "ses-001" / "beh" / folder
    (run / BLOCK).mkdir(parents=True)
    manifest = {
        "run": {"label": LABEL, "attempt": 1}, "start_epoch": 1.7e9 if has_started else None,
        "trigger_perf": 100.0 if has_started else None, "dummy_trigger": False,
        "rig": {"path": "configs/lab/rigs/desk.json",
                "triggers": {"sync": {"mode": "wait", "key": "EQUALS"}}},
        "display": {"size": [640, 480], "vsync": True, "refresh_rate": 100.0, "driver": "x11"},
        "versions": {"fmri_gym": "0.1", "git": "abc1234", "python": "3.12", "pygame": "2.5",
                     "sdl": "2.32"},
        "platform": {"system": "Linux", "hostname": "rig",
                     "perf_counter": {"implementation": "clock_gettime(CLOCK_MONOTONIC)"}}}
    if log_format is not None:
        manifest["log_format"] = {"format": "fmri-gym-log", "version": log_format}
    (run / "manifest.json").write_text(json.dumps(manifest))

    def line(line_type, perf_time, block=None, **fields):
        return {"type": line_type, "perf_time": perf_time, "block": block, **fields}
    left = {"type": 768, "key": 1073741904, "scancode": 80, "mod": 0, "unicode": ""}
    run_lines = [
        line("run_start", 98.0), line("flip", 98.5, kind="text", text="ready"),
        line("input_event", 99.0, name="SPACE", down=True),
        line("input_event", 99.1, name="SPACE", down=False),
        line("pygame_event", 100.0, event_name="KeyDown",
             event={"type": 768, "key": 61, "scancode": 46, "mod": 0, "unicode": "="}),
        line("input_event", 100.0, name="EQUALS", down=True), line("scanner_trigger", 100.0),
        line("flip", 100.01, kind="fixation"), line("flip", 100.02, kind="repeat"),
        line("flip", 101.0, kind="game_frame", block=BLOCK),        # 980 ms without a flip
        line("pygame_event", 101.62, BLOCK, event_name="KeyDown", event=left),
        line("input_event", 101.62, BLOCK, name="LEFT", down=True),
        line("flip", 101.7, BLOCK, kind="text", text="Paused\nResume"),
        line("input_event", 101.8, BLOCK, name="B", down=True),
        line("input_event", 101.9, BLOCK, name="LEFT", down=False),
        line("flip", 102.01, BLOCK, kind="game_frame"),
        line("trigger_sent", 103.0, name="task_stop", value=16),
        line("console", 103.5, stream="stdout", text="Saved run"),
        line("run_end", 104.0, how="finished")]
    block_lines = [
        {"type": "block_start", "run_time": 1.0, "backend": "gym", "game": "x"},
        {"type": "episode_start", "episode_id": 0, "flip_time": 1.5},
        {"type": "frame", "frame": 0, "controls_pressed": ["LEFT"], "run_time": 2.0,
         "flip_time": 2.01},
        {"type": "episode_end", "episode_id": 0, "outcome": "won", "score": 1.0, "run_time": 2.5},
        {"type": "block_end", "run_time": 3.0}]
    new = (log_format or 1) > 2
    for path, lines in ((run / (RUN_EVENTS_FILENAME if new else EVENTS_FILENAME), run_lines),
                        (run / BLOCK / (GAME_EVENTS_FILENAME if new else EVENTS_FILENAME),
                         block_lines)):
        path.write_text("".join(json.dumps(each) + "\n" for each in lines))
    return run


def test_the_timeline_joins_the_manifest_and_every_log(tmp_path):
    from humans.standalone.run_timeline import RunRecord, build_timeline
    timeline = build_timeline(RunRecord.read(str(_run_folder(tmp_path))))
    left, b = [press for press in timeline.presses if press.control in ("LEFT", "B")]
    assert left.onset == pytest.approx(1.62)
    assert (left.frame, left.episode, left.duration) == (0, 0, pytest.approx(0.28))
    assert (b.frame, b.episode) == (None, 0), "a pop-up's press is given to no step"
    assert [press.control for press in timeline.presses if press.is_scanner_pulse] == ["EQUALS"]
    assert [screen.kind for screen in timeline.screens] == ["text", "fixation", "text"]
    assert [screen.onset for screen in timeline.screens] == pytest.approx([-1.5, 0.01, 1.7])
    assert ("display", "no flip for 980.0 ms") in [(m.what, m.detail) for m in timeline.marks]
    quality = timeline.quality
    assert quality["pulses"]["count"] == 1 and quality["refreshes"]["missed"] == 4
    assert quality["presses"] == {"count": 3, "in_episodes": 2, "given_to_a_step": 1,
                                  "given_to_none": 1}


def test_the_events_table_has_a_row_per_screen_block_episode_and_press(tmp_path):
    from humans.standalone.events_table import event_rows, write_events_table
    from humans.standalone.run_timeline import RunRecord, build_timeline
    timeline = build_timeline(RunRecord.read(str(_run_folder(tmp_path))))
    rows = event_rows(timeline)
    assert [(row.trial_type, row.control) for row in rows] == [
        ("fixation", None), ("block", None), ("episode", None), ("press", "LEFT"),
        ("text", None), ("press", "B")], "nothing before the trigger, no scanner pulse"
    press = rows[3]
    assert (press.onset, press.frame) == (pytest.approx(1.62), 0)
    table = Path(write_events_table(timeline, str(tmp_path / "beh")))
    header, *cells = [row.split("\t") for row in table.read_text().splitlines()]
    assert header[:3] == ["onset", "duration", "trial_type"] and len(cells) == 6
    assert cells[4][-1] == "Paused\\nResume" and cells[5][1] == "n/a"
    sidecar = json.loads((tmp_path / "beh" / f"{LABEL}_events.json").read_text())
    assert set(sidecar) == set(header) and sidecar["onset"]["Units"] == "s"


def test_the_timeline_in_words_heads_with_the_record_and_its_checks(tmp_path):
    from humans.standalone.run_timeline import RunRecord, build_timeline
    from humans.standalone.timeline_text import timeline_text
    record = RunRecord.read(str(_run_folder(tmp_path)))
    text = timeline_text(record, build_timeline(record))
    assert text.startswith(f"run       {LABEL} (attempt 1)\ncode      fmri-gym 0.1, git abc1234")
    assert "scanner pulses  1\n" in text
    assert "LEFT, held 0.280 s, to frame 0 of episode 0" in text
    assert "episode  0 ends: won, score 1" in text and "code     task_stop = 16" in text


def _rig_of_team(tmp_path: Path, data_root: Path) -> str:
    """A rig file of team ``lab``, writing into ``data_root``."""
    from humans.config import rig
    team = tmp_path / "configs" / "lab"
    path = team / "rigs" / "desk.json"
    rig.save({**{k: "x" for k in rig.FIELDS}, "site": "lab", "rig": "desk", "pi": "MT",
              "modality": "fmri", "notes": "", "data_root": str(data_root),
              "screen": {"geometry": {"width_cm": 30.0, "height_cm": 20.0,
                                      "distance_cm": 60.0}}}, str(path))
    return str(path)


def test_bids_from_logs_writes_a_dataset_and_a_timeline_of_each_started_run(tmp_path, capsys):
    from humans.standalone.bids_from_logs import main
    data = tmp_path / "data"
    _run_folder(data)
    _run_folder(data, f"{LABEL}_02", has_started=False)          # stopped at start-up
    main(["--rig", _rig_of_team(tmp_path, data)])
    dataset = data / "bids"
    assert not (dataset / "dataset_description.json").exists(), "the user's to write"
    assert (dataset / "participants.tsv").read_text() == "participant_id\nsub-01\n"
    table = dataset / "sub-01" / "ses-001" / "beh" / f"{LABEL}_events.tsv"
    assert table.read_text().count("\tpress\t") == 2
    timelines = dataset / "derivatives" / "timelines"
    assert (timelines / "sub-01" / "ses-001" / "beh" / f"{LABEL}_timeline.txt").exists()
    assert "never reached its trigger" in capsys.readouterr().err


@pytest.mark.parametrize("log_format, problem", [
    (2, "2 attempts reached the trigger"),
    (None, "log format 1, recorded before runs kept a log of their own"),
])
def test_bids_from_logs_writes_nothing_when_it_would_have_to_choose(tmp_path, log_format,
                                                                      problem):
    from humans.standalone.bids_from_logs import main
    data = tmp_path / "data"
    _run_folder(data, log_format=log_format)
    if log_format is not None:
        _run_folder(data, f"{LABEL}_02")
    with pytest.raises(SystemExit, match=problem):
        main(["--rig", _rig_of_team(tmp_path, data)])
    assert not (data / "bids").exists()


def _device_recording(run: Path, cut_short: bool = False) -> Path:
    """The run's device recording (:func:`_run_folder`'s run), from perf_counter 95 to 110. Its
    device saw the scanner's = and LEFT before the run read them, and an A the run never read;
    ``cut_short``: it has no ``recording_end``."""
    event = {"type": "device_event", "device": 0, "event_type": 1}
    lines = [{"type": "recording_start", "reader": "linux evdev", "clock": "CLOCK_MONOTONIC",
              "devices": [{"name": "Box"}], "perf_time": 95.0, "wall_time": 1.7e9 - 5.0},
             {**event, "perf_time": 99.9995, "code": 13, "value": 1},     # KEY_EQUAL
             {**event, "perf_time": 101.615, "code": 105, "value": 1},    # KEY_LEFT
             {"type": "device_event", "perf_time": 101.615, "device": 0, "event_type": 0,
              "code": 0, "value": 0},
             {**event, "perf_time": 102.5, "code": 30, "value": 1}]       # KEY_A
    if not cut_short:
        lines.append({"type": "recording_end", "how": "stopped", "perf_time": 110.0,
                      "wall_time": 1.7e9 + 10.0})
    path = run / "device_events.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def test_a_runs_device_recording_gives_each_press_its_devices_time(tmp_path):
    from humans.standalone.events_table import event_rows
    from humans.standalone.run_timeline import RunRecord, build_timeline
    from humans.standalone.timeline_text import timeline_text
    run = _run_folder(tmp_path)
    _device_recording(run)
    record = RunRecord.read(str(run))
    timeline = build_timeline(record)
    left = next(press for press in timeline.presses if press.control == "LEFT")
    assert (left.onset, left.read_time, left.device_time) == pytest.approx((1.615, 1.62, 1.615))
    pulse = next(press for press in timeline.presses if press.is_scanner_pulse)
    assert pulse.onset == pytest.approx(-0.0005), "a pulse is timed by its device too"
    assert ("device", "device 0: A down") in [(m.what, m.detail) for m in timeline.marks]
    devices = timeline.quality["devices"]
    assert devices["presses_timed"] == 2 and devices["ended"] == "stopped"
    assert devices["read_delay_max_ms"] == pytest.approx(5.0)
    press_row = next(row for row in event_rows(timeline) if row.control == "LEFT")
    assert (press_row.onset, press_row.read_onset) == (pytest.approx(1.615), pytest.approx(1.62))
    text = timeline_text(record, timeline)
    assert "read 5.0 ms after its device sent it" in text
    assert "devices   1 device(s) (linux evdev)" in text


def test_a_device_recording_cut_short_says_so_and_still_times_the_presses(tmp_path):
    from humans.standalone.run_timeline import RunRecord, build_timeline
    run = _run_folder(tmp_path)
    _device_recording(run, cut_short=True)
    timeline = build_timeline(RunRecord.read(str(run)))
    assert timeline.quality["devices"]["ended"] == "without saying how"
    assert timeline.quality["devices"]["presses_timed"] == 2


def test_a_run_of_log_format_2_is_read_from_its_events_jsonl(tmp_path):
    from humans.standalone.run_timeline import RunRecord
    run = _run_folder(tmp_path, log_format=2)
    assert not (run / RUN_EVENTS_FILENAME).exists()
    assert not (run / BLOCK / GAME_EVENTS_FILENAME).exists()
    record = RunRecord.read(str(run))
    assert record.run_lines[0]["type"] == "run_start"
    assert record.block_lines[BLOCK][0]["type"] == "block_start"
    assert read_events(str(run / BLOCK))[0]["type"] == "block_start", \
        "resume reads a block of before format 3 from its events.jsonl"


