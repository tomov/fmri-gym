"""The rig config: reading a rig file and refusing a wrong one; and every rig file in the
repo is one a run would start on."""

from __future__ import annotations

import json
import sys

import pytest

from humans.config import rig
from humans.config.phases import phase_problems
from humans.config.rig import RigConfig
from humans.inputs import ControlNames
from humans.outputs.display import PhotodiodePatch

FIELDS = {**{k: "x" for k in rig.FIELDS}, "site": "site1", "pi": "MT", "modality": "fmri",
          "notes": ""}
GEOMETRY = {"width_cm": 70.8, "height_cm": 39.8, "distance_cm": 120.0}


@pytest.mark.parametrize("path", [str(p) for p in rig.repo_rig_files()])
def test_repo_rig_files_are_valid(path):
    assert RigConfig.open(path).path == path


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def write(name: str, **values) -> str:
    """The rig file ``site1/<name>.json``, written; its path."""
    path = f"site1/{name}.json"
    rig.save({**FIELDS, "rig": name, "screen": {"geometry": GEOMETRY}, **values}, path)
    return path


def test_defaults_fill_what_the_file_leaves_out():
    desk = RigConfig.open(write("desk"))
    screen = desk.screen
    assert (screen.size, screen.is_fullscreen, screen.monitor, screen.has_vsync) == (
        (1024, 768), True, 0, True)
    assert (desk.inputs, desk.has_audio, desk.data_root) == (("keyboard",), True, "data")
    assert desk.record()["screen"] == {**rig.DEFAULTS["screen"], "geometry": GEOMETRY}


def test_settings_are_read():
    scan = RigConfig.open(write("scan", screen={"size": "800x600", "is_fullscreen": True,
                                                "monitor": 1, "has_vsync": False,
                                                "geometry": GEOMETRY},
                                inputs=["current_designs_gamepad3"], has_audio=False,
                                data_root="/data"))
    screen = scan.screen
    assert (screen.size, screen.is_fullscreen, screen.monitor, screen.has_vsync) == (
        (800, 600), True, 1, False)
    assert (scan.inputs, scan.has_audio, scan.data_root) == (("current_designs_gamepad3",),
                                                             False, "/data")


def test_a_missing_file_is_refused():
    with pytest.raises(ValueError, match="no such rig file"):
        RigConfig.open("site1/none.json")


@pytest.mark.parametrize("change, problem", [
    ({"pi": "mt"}, "initials"),
    ({"screen": {"size": "big"}}, '"size"'),
    ({"screen": {"monitor": -1}}, '"monitor"'),
    ({"has_audio": "no"}, '"has_audio" must be true or false'),
    ({"audio": False}, "unknown key 'audio'"),
    ({"data_root": ""}, '"data_root"'),
    ({"bogus": 1}, "unknown key 'bogus'"),
    ({"bindings": {"A": "SPACE"}}, "unknown key 'bindings'.*list the controller under"),
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
    from humans.sync.triggers import Triggers
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


class _FakeSerialPort:
    """A serial port that keeps what is written and answers each read with the next reply."""

    def __init__(self, url: str, replies: list[bytes]) -> None:
        self.url, self.replies, self.written, self.dtr = url, replies, bytearray(), False

    def reset_input_buffer(self) -> None:
        pass

    def write(self, data: bytes) -> None:
        self.written += data

    def read(self, n: int) -> bytes:
        return self.replies.pop(0) if self.replies else b""

    def close(self) -> None:
        self.written += b"<closed>"


def _serial_box(monkeypatch, replies: list[bytes]) -> list[_FakeSerialPort]:
    """The ``serial`` module, faked: each port opened answers ``replies``; the ports opened."""
    import types

    import humans.sync.triggers as triggers_module
    opened: list[_FakeSerialPort] = []

    def serial_for_url(url, **settings):
        assert settings["baudrate"] == 115200
        opened.append(_FakeSerialPort(url, list(replies)))
        return opened[-1]
    monkeypatch.setitem(sys.modules, "serial", types.SimpleNamespace(serial_for_url=serial_for_url))
    monkeypatch.setattr(triggers_module.time, "sleep", lambda s: None)
    return opened


def test_a_dlpio8_answers_its_ping_then_sets_each_line_with_a_byte_of_its_own(monkeypatch):
    from humans.sync.triggers import Triggers
    opened = _serial_box(monkeypatch, [b"Q"])
    triggers = Triggers.from_dict({"sync": {"mode": "send"}, "backend": "dlpio8",
                                   "port": "/dev/ttyUSB0"})
    port, = opened
    assert triggers.active == "dlpio8:/dev/ttyUSB0" and port.written == b"'QWERTYUI"
    del port.written[:]
    triggers.lifecycle("scanner_start")             # 64, pulsed then cleared
    triggers.close()
    assert bytes(port.written) == b"QWERTY7I" + b"QWERTYUI" * 2 + b"<closed>"


def test_a_meg_ttl_box_sets_its_lines_at_once_when_its_firmware_can(monkeypatch):
    from humans.sync.triggers import Triggers
    opened = _serial_box(monkeypatch, [b"MTB\x01\x01"])
    triggers = Triggers.from_dict({"sync": {"mode": "send"}, "backend": "megttlbox",
                                   "port": "/dev/ttyACM0"})
    port, = opened
    assert port.dtr and triggers.active == "megttlbox:/dev/ttyACM0"
    assert bytes(port.written) == bytes([1, 14, 0xFF])           # info, then all lines low
    del port.written[:]
    triggers.lifecycle("task_start")
    assert bytes(port.written) == bytes([17, 8, 14, 0xFF])


def test_a_meg_ttl_box_with_older_firmware_sets_its_lines_high_then_low(monkeypatch):
    from humans.sync.triggers import Triggers
    opened = _serial_box(monkeypatch, [])                          # no answer to get_info
    triggers = Triggers.from_dict({"sync": {"mode": "send"}, "backend": "megttlbox",
                                   "port": "/dev/ttyACM0"})
    assert "firmware sets the lines in two steps" in triggers.active
    del opened[0].written[:]
    triggers.lifecycle("task_start")
    assert bytes(opened[0].written) == bytes([13, 8, 14, 0xF7, 14, 0xFF])


@pytest.mark.parametrize("backend, replies, says", [
    ("dlpio8", [], "no DLP-IO8 answered its ping"),
    ("megttlbox", [b"OK\r\n!"], "not a MEG TTL box"),
])
def test_a_box_that_does_not_answer_as_itself_stops_the_run(monkeypatch, backend, replies, says):
    from humans.sync.triggers import TriggerError, Triggers
    opened = _serial_box(monkeypatch, replies)
    refused = f"cannot open {backend} triggers on /dev/ttyX: .*{says}"
    with pytest.raises(TriggerError, match=refused):
        Triggers.from_dict({"sync": {"mode": "send"}, "backend": backend, "port": "/dev/ttyX"})
    assert opened[0].written.endswith(b"<closed>")
    with pytest.raises(ValueError, match="needs a port"):
        RigConfig.open(write("plain", triggers={"backend": backend}))


def test_a_game_key_the_scanner_types_is_refused_on_that_rig():
    from humans.config.curriculum import Curriculum
    from humans.config.run import RunConfig
    box = RigConfig.open(write("box", triggers={"sync": {"mode": "wait", "key": "5"}}))

    def run_of(keys: dict) -> RunConfig:
        game = {"type": "game", "game": "CartPole-v1", "fps": 30, "controls": keys}
        curriculum = Curriculum.from_dict({"curriculum": [game]}, "cartpole.json")
        return RunConfig(curriculum=curriculum, rig=box, subject="sub-01", ses=1, run=1,
                         duration=None, resume=False, loop=False,
                         dummy_trigger=True, skip_load_games=False,
                         calibration_files=())

    assert run_of({"": 0, "5": 1}).problems() and not run_of({"": 0, "LEFT": 1}).problems()


def test_a_control_none_of_the_rigs_inputs_has_is_refused_on_that_rig():
    from humans.config.curriculum import Curriculum
    from humans.config.run import RunConfig
    pad = RigConfig.open(write("pad", inputs=["keyboard", "current_designs_gamepad3"]))
    keyboard = RigConfig.open(write("kb"))
    nothing = RigConfig.open(write("none", inputs=[]))

    def run(phase: dict, on: RigConfig) -> RunConfig:
        curriculum = Curriculum.from_dict({"curriculum": [phase]}, "c.json")
        return RunConfig(curriculum=curriculum, rig=on, subject="sub-01", ses=1, run=1,
                         duration=None, resume=False, loop=False, dummy_trigger=True,
                         skip_load_games=False,
                         calibration_files=())

    def game(keys: dict, **more) -> dict:
        return {"type": "game", "game": "CartPole-v1", "fps": 30, "controls": keys, **more}

    a, x, y = (ControlNames.CONTROLLER_BUTTON_A, ControlNames.CONTROLLER_BUTTON_X,
               ControlNames.CONTROLLER_BUTTON_Y)
    stick_up = ControlNames.CONTROLLER_AXIS_LEFTY_UP
    stick = game({"": 0, stick_up: 1, "UP": 1, a: 2})
    assert run(stick, pad).problems() == []
    assert f"needs {[stick_up, a]}, which none of" in run(stick, keyboard).problems()[0]
    assert run(game({"": 0, "UP": 1}), keyboard).problems() == []
    assert "needs ['UP'], which none of" in run(game({"": 0, "UP": 1}), nothing).problems()[0]
    assert run(stick, pad).required_controls() == {stick_up, "UP", a}
    menu = {"control": x, "move": ["UP", "DOWN"], "confirm": "SPACE"}
    assert f"needs {[x]}" in run(game({"": 0, "UP": 1}, menu=menu), keyboard).problems()[0]
    survey = {"type": "survey", "questions": ["ok?"], "controls": [y, a]}
    assert f"needs {[a, y]}" in run(survey, keyboard).problems()[0]
    xbox = RigConfig.open(write("xbox", inputs=["keyboard", "xbox_controller"]))
    assert run(stick, xbox).problems() == []       # the GMPD3's controls are the Xbox's too


def test_inputs_name_the_rigs_devices():
    both = ["current_designs_gamepad3", "xbox_controller"]
    assert RigConfig.open(write("pad", inputs=both)).inputs == tuple(both)
    for wrong in (["keyboard", "keyboard"], ["gmpd3"], ["joystick"], "keyboard"):
        with pytest.raises(ValueError, match=r'"inputs": expected a list of distinct names '
                                             r"from \['current_designs_gamepad3', 'keyboard', "
                                             r"'xbox_controller'\]"):
            RigConfig.open(write("pad", inputs=wrong))


def test_every_key_is_optional_and_site_and_rig_are_the_files_place(tmp_path):
    (tmp_path / "site1").mkdir()
    (tmp_path / "site1" / "bare.json").write_text("{}")
    bare = RigConfig.open("site1/bare.json")
    assert bare.fields == {"site": "site1", "rig": "bare"} and bare.has_audio
    assert bare.screen.geometry is None and "geometry" not in bare.record()["screen"]
    (tmp_path / "site1" / "bare.json").write_text(json.dumps({"site": "elsewhere"}))
    with pytest.raises(ValueError, match="the site is its team's folder"):
        RigConfig.open("site1/bare.json")


def test_a_teams_rig_is_filed_under_the_team(tmp_path):
    (tmp_path / "MT" / "rigs").mkdir(parents=True)
    (tmp_path / "MT" / "rigs" / "mgh-prisma.json").write_text(
        json.dumps({"screen": {"geometry": GEOMETRY}}))
    rig = RigConfig.open(str(tmp_path / "MT" / "rigs" / "mgh-prisma.json"))
    assert rig.fields == {"site": "MT", "rig": "mgh-prisma"}


@pytest.mark.parametrize("values, problem", [
    ({"response_keys": {"NOSUCHKEY": "index"}}, "not a keyboard key name"),
    ({"triggers": {"sync": {"mode": "wait", "key": "="}}}, "not a keyboard key name"),
    ({"rig_measurement": {"photodiode": "eyes"}}, '"photodiode"'),
    ({"rig_measurement": {"has_mic": True}}, '"has_mic" needs'),
    ({"rig_measurement": {"frame_rates": [0]}}, '"frame_rates"'),
    ({"rig_measurement": {"what": 1}}, "unknown key"),
    ({"rig_measurement": {"corner": "tl"}}, "unknown key 'corner': the photodiode's square is "
                                            'the rig\'s "photodiode_patch"'),
    ({"rig_measurement": {"photodiode": "soundcard"}}, 'so the rig needs a "photodiode_patch"'),
    ({"photodiode_patch": {"corner": "middle"}}, '"corner" must be one of'),
    ({"photodiode_patch": {"size_px": 0}}, '"size_px" must be a whole number of pixels'),
    ({"photodiode_patch": {"size_px": 120.5}}, '"size_px" must be a whole number of pixels'),
    ({"photodiode_patch": {"colour": "white"}}, "unknown key 'colour'"),
    ({"photodiode_patch": "tl"}, "expected {\"corner\", \"size_px\"} or null"),
])
def test_a_wrong_rig_measurement_section_is_refused(values, problem):
    assert any(problem in p for p in rig.problems(values))


def test_the_photodiode_patch_is_the_rigs_with_defaults_for_what_it_leaves_out():
    assert RigConfig.open(write("desk")).photodiode_patch is None
    meg = RigConfig.open(write("meg", photodiode_patch={"corner": "tl"},
                               rig_measurement={"photodiode": "recording"}))
    assert meg.photodiode_patch == PhotodiodePatch("tl", rig.PHOTODIODE_PATCH_DEFAULTS["size_px"])
    assert meg.record()["photodiode_patch"] == {"corner": "tl", "size_px": 120}


@pytest.mark.parametrize("geometry, problem", [
    ({**GEOMETRY, "distance_cm": 0}, "more than 0"),
    ({**GEOMETRY, "width_cm": "70"}, "more than 0"),
    ({"diagonal_in": 32, "distance_cm": 120}, '"geometry" must be'),
])
def test_a_wrong_screen_geometry_is_refused(geometry, problem):
    assert any(problem in p for p in rig.problems({"screen": {"geometry": geometry}}))


def test_the_geometry_says_what_a_degree_is():
    geometry = rig.ScreenGeometry(**GEOMETRY)
    assert geometry.cm_for_degrees(1) == pytest.approx(2.094, abs=1e-3)
    assert geometry.degrees_for_cm(geometry.cm_for_degrees(20)) == pytest.approx(20)
    assert geometry.px_per_cm((1920, 1080)) == pytest.approx((27.12, 27.14), abs=0.01)
    assert RigConfig.open(write("scan")).screen.geometry == geometry


def test_a_message_is_dismissed_by_a_control_name_not_a_character():
    message = {"type": "message", "text": "hi"}
    assert not phase_problems(message) and not phase_problems({**message, "control": "any"})
    assert not phase_problems({**message, "control": ControlNames.CONTROLLER_BUTTON_A})
    assert any("control name" in p for p in phase_problems({**message, "control": " "}))
