"""The rig config: reading a rig file and refusing a wrong one; and every rig file in the
repo is one a run would start on."""

from __future__ import annotations

import json

import pytest

from fmri_gym.config import rig
from fmri_gym.config.rig import RigConfig

FIELDS = {**{k: "x" for k in rig.FIELDS}, "site": "site1", "pi": "MT", "modality": "fmri",
          "notes": ""}


@pytest.mark.parametrize("path", [str(p) for p in sorted(rig.REPO_RIGS.glob("*/*.json"))])
def test_repo_rig_files_are_valid(path):
    assert RigConfig.open(path).path == path


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def write(name: str, **values) -> str:
    """The rig file ``site1/<name>.json``, written; its path."""
    path = f"site1/{name}.json"
    rig.save({**FIELDS, "rig": name, **values}, path)
    return path


def test_defaults_fill_what_the_file_leaves_out():
    desk = RigConfig.open(write("desk"))
    screen = desk.screen
    assert (screen.size, screen.is_fullscreen, screen.monitor, screen.has_vsync) == (
        (1024, 768), False, 0, True)
    assert (desk.has_pad, desk.has_audio, desk.data_root) == (True, True, "data")
    assert desk.record()["screen"] == rig.DEFAULTS["screen"]


def test_settings_are_read():
    scan = RigConfig.open(write("scan", screen={"size": "800x600", "is_fullscreen": True,
                                                "monitor": 1, "has_vsync": False},
                                has_pad=False, has_audio=False, data_root="/data"))
    screen = scan.screen
    assert (screen.size, screen.is_fullscreen, screen.monitor, screen.has_vsync) == (
        (800, 600), True, 1, False)
    assert (scan.has_pad, scan.has_audio, scan.data_root) == (False, False, "/data")


def test_a_missing_file_says_how_to_make_one():
    with pytest.raises(ValueError, match="no such file.*rig-check-long"):
        RigConfig.open("site1/none.json")


@pytest.mark.parametrize("change, problem", [
    ({"pi": "mt"}, "initials"),
    ({"screen": {"size": "big"}}, '"size"'),
    ({"screen": {"monitor": -1}}, '"monitor"'),
    ({"has_audio": "no"}, '"has_audio" must be true or false'),
    ({"audio": False}, "unknown key 'audio'"),
    ({"data_root": ""}, '"data_root"'),
    ({"bogus": 1}, "unknown key 'bogus'"),
    ({"rig": "other"}, "its folder is the site, its name the rig"),
])
def test_a_wrong_file_is_refused(change, problem):
    path = write("r")
    with open(path, "w") as f:
        json.dump({**FIELDS, "rig": "r", **change}, f)
    with pytest.raises(ValueError, match=problem):
        RigConfig.open(path)


def test_triggers_are_the_rigs():
    meg = RigConfig.open(write("meg", triggers={"sync": {"mode": "send"}, "backend": "lsl"}))
    assert meg.has_trigger_setup and meg.triggers.backend == "lsl"
    assert meg.without_trigger_line().triggers.sync.mode == "none"
    assert not RigConfig.open(write("plain")).has_trigger_setup
    with pytest.raises(ValueError, match="needs a port"):
        RigConfig.open(write("plain", triggers={"backend": "serial"}))


def test_a_game_key_the_scanner_types_is_refused_on_that_rig():
    from fmri_gym.config.curriculum import Curriculum
    from fmri_gym.config.run import RunConfig
    box = RigConfig.open(write("box", triggers={"sync": {"mode": "wait", "key": "5"}}))

    def run_of(keys: dict) -> RunConfig:
        game = {"type": "game", "game": "CartPole-v1", "fps": 30, "keys": keys}
        curriculum = Curriculum.from_dict({"curriculum": [game]}, "cartpole.json")
        return RunConfig(curriculum=curriculum, rig=box, subject="sub-01", ses=1, run=1,
                         duration=None, end_fixation=10.0, resume=False, loop=False,
                         dummy_trigger=True)

    assert run_of({"": 0, "5": 1}).problems() and not run_of({"": 0, "LEFT": 1}).problems()


def test_what_the_controller_cannot_press_is_refused_on_that_rig():
    from fmri_gym.config.curriculum import Curriculum
    from fmri_gym.config.run import RunConfig
    pad = RigConfig.open(write("pad", bindings={"A": "SPACE", "UP": "UP", "DOWN": "DOWN"}))
    keyboard = RigConfig.open(write("kb", has_pad=False, bindings={"A": "SPACE"}))

    def problems(phase: dict, on: RigConfig = pad) -> list[str]:
        curriculum = Curriculum.from_dict({"curriculum": [phase]}, "c.json")
        return RunConfig(curriculum=curriculum, rig=on, subject="sub-01", ses=1, run=1,
                         duration=None, end_fixation=10.0, resume=False, loop=False,
                         dummy_trigger=True).problems()

    def game(keys: dict, **more) -> dict:
        return {"type": "game", "game": "CartPole-v1", "fps": 30, "keys": keys, **more}

    assert problems(game({"": 0, "UP": 1, "W": 1, "SPACE": 2})) == []  # W only doubles UP
    assert "action 3 needs W" in problems(game({"": 0, "SPACE": 2, "W": 3}))[0]
    assert problems(game({"": 0, "W": 3}), on=keyboard) == []
    menu = {"key": "X", "move": ["UP", "DOWN"], "confirm": "SPACE"}
    assert "its menu's X" in problems(game({"": 0, "UP": 1}, menu=menu))[0]
    assert problems({"type": "message", "text": "hi", "keys_to_dismiss": ["Q", "SPACE"]}) == []
    assert "dismissing it needs Q" in problems({"type": "message", "text": "hi", "key": "q"})[0]
    assert problems({"type": "message", "text": "hi", "duration": 2.0, "key": "q"}) == []


def test_bindings_map_the_controllers_controls_to_keys():
    pad = RigConfig.open(write("pad", bindings={"A": "SPACE", "UP": "UP"}))
    assert pad.bindings == {"A": "SPACE", "UP": "UP"}
    with pytest.raises(ValueError, match="'Q' is not a control.*'NOPE' is not a key name"):
        RigConfig.open(write("pad", bindings={"Q": "SPACE", "B": "NOPE"}))


def test_every_key_is_optional_and_site_and_rig_are_the_files_place(tmp_path):
    (tmp_path / "site1").mkdir()
    (tmp_path / "site1" / "bare.json").write_text("{}")
    bare = RigConfig.open("site1/bare.json")
    assert bare.fields == {"site": "site1", "rig": "bare"} and bare.has_audio
    (tmp_path / "site1" / "bare.json").write_text(json.dumps({"site": "elsewhere"}))
    with pytest.raises(ValueError, match="its folder is the site"):
        RigConfig.open("site1/bare.json")
