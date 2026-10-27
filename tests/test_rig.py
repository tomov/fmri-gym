"""The rig config: reading a rig file and refusing a wrong one; and every rig file in the
repo is one a run would start on."""

from __future__ import annotations

import json
import sys

import pytest

from fmri_gym.config import rig
from fmri_gym.config.phases import phase_problems
from fmri_gym.config.rig import RigConfig

FIELDS = {**{k: "x" for k in rig.FIELDS}, "site": "site1", "pi": "MT", "modality": "fmri",
          "notes": ""}


@pytest.mark.parametrize("path", [str(p) for p in rig.repo_rig_files()])
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
    ({"rig": "other"}, "the site is its team.s folder, the rig its name"),
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
    with pytest.raises(ValueError, match="needs a port"):
        RigConfig.open(write("plain", triggers={"backend": "hid"}))
    with pytest.raises(ValueError, match="has no port 'C'"):
        RigConfig.open(write("plain", triggers={"backend": "hid", "port": "C"}))
    with pytest.raises(ValueError, match="unknown hid_device"):
        RigConfig.open(write("plain", triggers={"backend": "hid", "hid_device": "x", "port": "A"}))


class _FakeHid:
    """The ``hid`` module of hidapi, over one USB-1208FS whose reports are kept."""

    def __init__(self) -> None:
        self.reports: list[list[int]] = []
        reports = self.reports

        class device:  # noqa: N801 -- hidapi's name
            def open_path(self, path: bytes) -> None:
                assert path == b"if0"

            def write(self, report: list[int]) -> int:
                reports.append(list(report))
                return len(report)

            def close(self) -> None:
                reports.append(["closed"])

        self.device = device

    def enumerate(self, vid: int, pid: int) -> list[dict]:
        assert (vid, pid) == (0x09DB, 0x0082)
        return [{"interface_number": n, "path": f"if{n}".encode()} for n in (1, 0)]


def test_a_hid_box_is_set_to_output_zero_then_holds_each_code(monkeypatch):
    from fmri_gym.triggers import Triggers
    fake = _FakeHid()
    monkeypatch.setitem(sys.modules, "hid", fake)
    triggers = Triggers.from_dict({"sync": {"mode": "send"}, "backend": "hid", "port": "B"})
    assert triggers.active == "hid:mcc-1208fs:B"
    assert fake.reports == [[0x01, 1, 0], [0x04, 1, 0]]  # port B an output, at 0
    del fake.reports[:]
    triggers.frame()
    triggers.block_end()
    triggers.close()
    assert fake.reports == [[0x04, 1, 1], [0x04, 1, 0], [0x04, 1, 0], ["closed"]]


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
    with pytest.raises(ValueError, match="the site is its team's folder"):
        RigConfig.open("site1/bare.json")


def test_a_teams_rig_is_filed_under_the_team(tmp_path):
    (tmp_path / "MT" / "rigs").mkdir(parents=True)
    (tmp_path / "MT" / "rigs" / "mgh-prisma.json").write_text("{}")
    rig = RigConfig.open(str(tmp_path / "MT" / "rigs" / "mgh-prisma.json"))
    assert rig.fields == {"site": "MT", "rig": "mgh-prisma"}


@pytest.mark.parametrize("phase", [
    {"type": "check_photodiode", "readout": "soundcard"},
    {"type": "check_photodiode", "corner": "br"},
    {"type": "check_frames", "recording_hz": 1000},
    {"type": "check_controls", "keys": {"1": "index"}},
])
def test_a_check_curriculum_cannot_set_what_is_the_rigs(phase):
    assert any("rig-config" in p for p in phase_problems(phase))


@pytest.mark.parametrize("values, problem", [
    ({"response_keys": {"NOSUCHKEY": "index"}}, "not a key name"),
    ({"rig_check": {"photodiode": "eyes"}}, '"photodiode"'),
    ({"rig_check": {"has_mic": True}}, '"has_mic" needs'),
    ({"rig_check": {"frame_rates": [0]}}, '"frame_rates"'),
    ({"rig_check": {"what": 1}}, "unknown key"),
])
def test_a_wrong_rig_check_section_is_refused(values, problem):
    assert any(problem in p for p in rig.problems(values))
