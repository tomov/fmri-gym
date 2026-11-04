"""The input devices with none plugged in: their events, their sticks, their calibration.

A scripted controller answers the calibration frame by frame, on a fake clock,
so a whole calibration plays in well under a second and writes into a
temporary folder. The controllers are fed the events SDL would send them.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import types
from datetime import datetime, timezone

import pygame
import pytest

import fmri_gym.inputs as inputs
from fmri_gym import pad_calibration as cal
from fmri_gym.inputs import (QUIT, CalibrationError, CurrentDesignsGamepad3, Edge, InputDevices,
                             InputEvent, Keyboard, XBoxController)
from fmri_gym.inputs import calibration as files
from fmri_gym.inputs.calibration import UNCALIBRATED, StickTuning
from fmri_gym.inputs.controller import (LEFT_STICK, RIGHT_STICK, STICK_DIRECTIONS, Controller,
                                        Stick)

MADE = "2026-01-21T14:15:00+00:00"
AXIS = {"rest": 0.0, "low": -1.0, "high": 1.0, "drift": 0.01}
FILE = {"device": "Scripted", "stick": LEFT_STICK, "tuned_from": None, "made": MADE,
        "x": AXIS, "y": AXIS, "on": 0.5, "off": 0.3, "diagonal": 1.0, "dwell_s": 0.05}
RIGHT_FILE = {**FILE, "stick": RIGHT_STICK, "on": 0.6}
#: The recording's columns for the scripted controller's sticks.
LX, LY, RX, RY = f"{LEFT_STICK}_x", f"{LEFT_STICK}_y", f"{RIGHT_STICK}_x", f"{RIGHT_STICK}_y"


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    """Calibrations written to and read from a folder of the test's own."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return files.calibrations_dir()


def _write(folder, name: str, blob: dict, mtime: int):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(json.dumps(blob))
    os.utime(path, (mtime, mtime))
    return path


def _set(blob: dict, path: str, value) -> dict:
    """``blob`` with the field at dotted ``path`` set to ``value`` (or removed, for ...)."""
    out = copy.deepcopy(blob)
    *parents, key = path.split(".")
    d = out
    for p in parents:
        d = d[p]
    if value is ...:
        del d[key]
    else:
        d[key] = value
    return out


def _calibration(blob: dict) -> tuple[None, files.StickCalibration]:
    """A calibration as a controller takes it: its (absent) file, and its contents."""
    return None, files.StickCalibration.from_dict(blob)


# -- the calibration file -----------------------------------------------------------------


def test_a_calibration_round_trips_through_its_file():
    loaded = files.StickCalibration.from_dict(copy.deepcopy(RIGHT_FILE))
    assert loaded.record() == RIGHT_FILE
    assert (loaded.stick, loaded.tuning.on) == (RIGHT_STICK, 0.6)
    assert loaded.made == datetime(2026, 1, 21, 14, 15, tzinfo=timezone.utc)


@pytest.mark.parametrize("field, value, says", [
    ("on", "0.5", "on: expected a number in (0, 1), got '0.5'"),
    ("off", 0.6, "off: expected a number in (0, on=0.5], got 0.6"),
    ("diagonal", 0, "diagonal: expected a number in (0, 1], got 0"),
    ("dwell_s", 0.5, "dwell_s: expected a number of seconds in [0, 0.3]"),
    ("x.drift", True, "x.drift: expected a number >= 0, got True"),
    ("x.low", 0.5, "x: expected low < rest < high"),
    ("x.noise", 0.001, "x.noise: unknown key"),
    ("made", "2026-10-21T14:15:00", "made: expected a time such as"),
    ("y", ..., "y: missing"),
    ("stick", "", "stick: expected a name, got ''"),
    ("device", ..., "device: missing"),
    ("tuned_from", 3, "tuned_from: expected null or a file name, got 3"),
    ("sectors", {}, "sectors: unknown key"),
])
def test_each_wrong_field_is_refused_by_its_file_and_name(tmp_path, field, value, says):
    path = tmp_path / "Scripted_LEFT_STICK_20261021-141500.json"
    path.write_text(json.dumps(_set(FILE, field, value)))
    with pytest.raises(CalibrationError, match=re.escape(f"{path}: {says}")):
        files.read(path)


def test_a_stick_is_read_through_its_newest_calibration(config_home):
    _write(config_home, "Scripted_a.json", _set(FILE, "on", 0.31), 1000)
    newest = _write(config_home, "Scripted_b.json", _set(FILE, "on", 0.33), 3000)
    _write(config_home, "Scripted_c.json", _set(FILE, "on", 0.32), 2000)
    _write(config_home, "Scripted_r.json", RIGHT_FILE, 4000)            # newer, the other stick
    _write(config_home, "Other_d.json", {**FILE, "device": "Other"}, 4000)   # newer, another pad
    _write(config_home, "Other_e.json", {"device": "Other", "axes": {}}, 5000)  # bad, not ours
    path, found = files.newest("Scripted", LEFT_STICK)
    assert (path, found.tuning.on) == (newest, 0.33)
    assert files.newest("Scripted", RIGHT_STICK)[1].tuning.on == 0.6
    assert files.newest("Nobody", LEFT_STICK) is None


def test_a_malformed_newest_calibration_stops_the_run_with_no_fallback(config_home):
    _write(config_home, "Scripted_a.json", FILE, 1000)
    bad = _write(config_home, "Scripted_b.json", _set(FILE, "on", "0.5"), 2000)
    with pytest.raises(CalibrationError, match=re.escape(f"{bad}: on")):
        files.newest("Scripted", LEFT_STICK)
    bad.write_text("{not json")                    # names no device: it may be this one's
    with pytest.raises(CalibrationError, match="not JSON"):
        files.newest("Scripted", LEFT_STICK)


# -- a controller with no SDL behind it -----------------------------------------------------


class _FakeHandle:
    """What a controller reads off SDL's handle: its name, the id its events carry, its axes
    at rest.

    A joystick's and a game controller's handles spell these differently; this one spells both.
    """

    def __init__(self, index: int = 0, name: str = "Scripted") -> None:
        self.name, self.id = name, 7
        self.open = True

    def get_name(self) -> str:
        return self.name

    def get_instance_id(self) -> int:
        return self.id

    def get_axis(self, axis: int) -> float:
        return 0.0

    def get_init(self) -> bool:
        return self.open

    def quit(self) -> None:
        self.open = False


def _bare(cls: type[Controller], name: str = "Scripted") -> Controller:
    """A controller of ``cls`` as if SDL had opened it, with events of id 7, uncalibrated."""
    return cls(_FakeHandle(name=name), calibrated=False)


def _calibrated(controller: Controller, *blobs: dict) -> Controller:
    """``controller`` reading each stick a blob names through that blob."""
    for blob in blobs:
        controller.set_stick_calibration(controller.stick(blob["stick"]), _calibration(blob))
    return controller


class _ScriptedController(Controller):
    """A two-stick, two-button controller whose readings are a script, one frame each; then
    ESC."""

    name = "scripted"
    STICK_NAMES = (LEFT_STICK, RIGHT_STICK)
    BUTTON_NAMES = ("S_A", "S_B")
    CONTROLS = frozenset({*BUTTON_NAMES, *STICK_DIRECTIONS[LEFT_STICK],
                          *STICK_DIRECTIONS[RIGHT_STICK]})
    HANDLE = _FakeHandle

    def __init__(self, handle=None, calibrated: bool = False, frames=()) -> None:
        super().__init__(handle or _FakeHandle(), calibrated)
        self.frames = iter(frames)

    @classmethod
    def accepts(cls, joystick, is_controller: bool) -> bool:
        return True

    def attach(self, handle) -> None:
        self.joystick, self.instance_id, self.device_name = handle, handle.id, handle.name

    def is_open(self) -> bool:
        return self.joystick.get_init()

    def handle(self, events, now: float) -> list[InputEvent]:
        """Take the next scripted frame as the controller's reading, ignoring the events."""
        frame = next(self.frames, None)
        if frame is None:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
            frame = (self.axes_at({}), set())
        axes, buttons = frame
        for stick in self.sticks:
            stick.x, stick.y = axes[f"{stick.name}_x"], axes[f"{stick.name}_y"]
        out = [e for b in self.BUTTON_NAMES for e in self._press(b, b in buttons)]
        return out + self._read_sticks(now)

    @staticmethod
    def axes_at(moved: dict[str, float]) -> dict[str, float]:
        return {c: moved.get(c, 0.0) for c in (LX, LY, RX, RY)}


def _stick_frames(d: _ScriptedController, stick: Stick, wrong: str | None = None) -> list:
    """Hands off, roll, let go, then each direction twice (``wrong`` pushed once first)."""
    x, y = cal.columns_of(stick)
    still = [(d.axes_at({x: 0.02}), set())] * 70
    circle = [(d.axes_at({x: math.cos(2 * math.pi * k / 48), y: math.sin(2 * math.pi * k / 48)}),
               set()) for k in range(48 * 3)]
    rest = [(d.axes_at({x: -0.05, y: 0.01}), set())] * 70
    out = still + circle + rest
    for b in ([wrong] if wrong else []) + [b for b in cal.CUE_DIRECTIONS for _ in range(2)]:
        axis, sign = cal.PUSHES[b]
        out += [rest[0], (d.axes_at({x: -0.05, y: 0.01, (x, y)[axis]: 0.9 * sign}), set())]
    return out


def _button_frames(d: _ScriptedController) -> list:
    return [f for b in ("S_A", "S_B") for f in [(d.axes_at({}), set()), (d.axes_at({}), {b})] * 2]


class _FakeClock:
    """64 frames a second, exactly: a frame's time is never rounded across a window's edge."""

    t = 0.0

    def tick(self) -> float:
        _FakeClock.t += 1 / 64
        return _FakeClock.t


@pytest.fixture
def screen(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setattr(cal, "_FrameClock", _FakeClock)
    pygame.init()
    pygame.event.clear()
    return pygame.Surface((900, 600))


# -- the calibration procedure ------------------------------------------------------------


def test_a_calibration_of_both_sticks_measures_each_and_confirms_each_control(config_home,
                                                                              screen):
    d = _ScriptedController()
    left, right = d.sticks
    d.frames = iter(_stick_frames(d, left, wrong="LEFT") + _stick_frames(d, right)
                    + _button_frames(d))
    result = cal.calibrate(d, screen, lambda: None, tuple(d.sticks))
    assert (result.confirmed, result.corrections) == (4 + 4 + 2, 1)
    path, measured = result.written[LEFT_STICK]
    assert (measured.x.rest, measured.x.low, measured.x.high, measured.x.drift) == (
        -0.05, -1.0, 1.0, 0.07)
    assert (measured.tuning.on, measured.tuning.off) == (cal.CAL_ON, cal.CAL_OFF)
    assert set(result.written) == {LEFT_STICK, RIGHT_STICK}
    assert files.newest("Scripted", LEFT_STICK) == (path, measured)
    assert path.with_suffix(".csv").exists() and "Scripted_LEFT-STICK_" in path.name
    assert left.calibration is None                 # the caller applies what was written
    from fmri_gym.checks import _pad_line
    line = _pad_line({"present": True, "device": "Scripted", **result.record()})
    assert "LEFT_STICK on past 0.50" in line and "10 controls confirmed, 1 wrong" in line


def test_one_stick_calibrated_alone_leaves_the_others_file_alone(config_home, screen):
    base = files.write(files.StickCalibration.from_dict(_set(FILE, "on", 0.4)))
    d = _ScriptedController()
    d.frames = iter(_stick_frames(d, d.stick(RIGHT_STICK)))     # no buttons: not every stick
    result = cal.calibrate(d, screen, lambda: None, (d.stick(RIGHT_STICK),))
    assert list(result.written) == [RIGHT_STICK] and result.confirmed == 4
    assert files.newest("Scripted", LEFT_STICK)[0] == base
    assert files.newest("Scripted", RIGHT_STICK) == result.written[RIGHT_STICK]


def test_a_calibration_stopped_part_way_writes_nothing(config_home, screen):
    d = _ScriptedController()
    d.frames = iter(_stick_frames(d, d.stick(LEFT_STICK))[:100])
    assert cal.calibrate(d, screen, lambda: None, (d.stick(LEFT_STICK),)) is None
    assert not config_home.exists()


def test_the_fit_raises_the_threshold_of_a_stick_that_dithers():
    stick = Stick(LEFT_STICK, *STICK_DIRECTIONS[LEFT_STICK])
    rows = ([{"t": k / 60, "phase": "cold", "cue": LEFT_STICK, LX: 0.0, LY: 0.0}
             for k in range(60)]
            + [{"t": 1 + k / 60, "phase": "roll", "cue": LEFT_STICK, LX: v, LY: -v}
               for k, v in enumerate((-0.5, 1.0, 0.2))]
            + [{"t": 2 + k / 60, "phase": "rest", "cue": LEFT_STICK, LX: 0.1 + 0.02 * (k % 2),
                LY: 0.0} for k in range(60)])
    fitted = cal.fit(rows, stick, "Scripted", datetime.now(timezone.utc))
    assert (fitted.x.rest, fitted.x.low, fitted.x.high, fitted.x.drift) == (0.11, -0.5, 1.0, 0.11)
    assert (fitted.device, fitted.stick) == ("Scripted", LEFT_STICK)
    # dither 0.01 of the 0.61 travel to the low side: ten times it is 0.164, under 0.5
    assert (fitted.tuning.on, fitted.tuning.off) == (0.5, 0.3)
    shaky = [{**r, LX: 0.1 + 0.1 * (k % 2)} if r["phase"] == "rest" else r
             for k, r in enumerate(rows)]
    tuning = cal.fit(shaky, stick, "Scripted", datetime.now(timezone.utc)).tuning
    assert tuning.on > cal.CAL_ON and tuning.off == round(0.6 * tuning.on, 3)


def test_a_check_controls_phase_refuses_pad_reps():
    from fmri_gym.config.phases import CheckControls
    assert any("pad_reps" in p for p in CheckControls.problems({"pad_reps": 3}))


# -- the sticks and the controllers ----------------------------------------------------------


def test_each_stick_reads_its_own_tuning():
    d = _calibrated(_ScriptedController(), _set(FILE, "dwell_s", 0), _set(RIGHT_FILE, "dwell_s", 0))
    d.frames = iter([(d.axes_at({LX: 0.55, RX: 0.55}), set()),    # past the left's 0.5, not the
                     (d.axes_at({LX: 0.55, RX: 0.65}), set())])   # right's 0.6
    assert d.handle([], 1.0) == [_down("LEFT_STICK_RIGHT")]
    assert d.handle([], 1.1) == [_down("RIGHT_STICK_RIGHT")]
    assert d.held() == {"LEFT_STICK_RIGHT", "RIGHT_STICK_RIGHT"}


def test_a_direction_is_pressed_once_it_has_dwelt_and_released_at_once():
    d = _calibrated(_ScriptedController(), FILE)                   # dwell 50 ms
    pushed, rest = (d.axes_at({LY: -0.9}), set()), (d.axes_at({}), set())
    d.frames = iter([pushed, pushed, pushed, rest])
    assert d.handle([], 1.0) == [] and d.handle([], 1.03) == []
    assert d.handle([], 1.05) == [_down("LEFT_STICK_UP")]
    assert d.handle([], 1.06) == [_up("LEFT_STICK_UP")]


def test_the_stick_rule_reads_one_direction_by_angle_and_keeps_the_one_held():
    one = StickTuning(on=0.5, off=0.3, diagonal=1.0, dwell_s=0)
    assert one.directions(0.2, 0.2, set()) == set()                 # inside on
    assert one.directions(0.3, 0.9, set()) == {"DOWN"}              # both axes moved: the angle
    assert one.directions(0.6, 0.65, {"RIGHT"}) == {"RIGHT"}        # held, and not clearly nearer
    assert one.directions(0.2, 0.3, {"DOWN"}) == {"DOWN"}           # past off while held
    assert one.directions(0.1, 0.1, {"DOWN"}) == set()
    both = StickTuning(on=0.5, off=0.3, diagonal=0.7, dwell_s=0)
    assert both.directions(0.9, 0.5, set()) == {"RIGHT"}            # the weaker under 0.7 of it
    assert both.directions(0.9, 0.7, set()) == {"RIGHT", "DOWN"}
    assert both.directions(-0.4, 0.0, {"LEFT"}) == {"LEFT"}


def test_a_controller_refuses_a_stick_the_curriculum_names_with_no_calibration(config_home):
    pad = _bare(CurrentDesignsGamepad3)
    with pytest.raises(CalibrationError, match="no calibration of 'Scripted'.s LEFT_STICK in "
                                               ".*rig check"):
        pad.require_calibrated_sticks_for(frozenset({"LEFT_STICK_UP", "BUTTON_A"}))
    pad.require_calibrated_sticks_for(frozenset({"BUTTON_A", "SPACE"}))    # buttons only
    assert "LEFT_STICK UNCALIBRATED" in pad.status()
    assert pad.stick(LEFT_STICK).tuning == UNCALIBRATED
    files.write(files.StickCalibration.from_dict(RIGHT_FILE))     # a stick it does not have
    pad.load_stick_calibrations()
    with pytest.raises(CalibrationError, match="LEFT_STICK"):
        pad.require_calibrated_sticks_for(frozenset({"LEFT_STICK_UP"}))
    path = files.write(files.StickCalibration.from_dict(FILE))
    pad.load_stick_calibrations()
    pad.require_calibrated_sticks_for(frozenset({"LEFT_STICK_UP"}))
    assert str(path) in pad.status() and pad.stick(LEFT_STICK).calibration_path == path


def _down(name: str) -> InputEvent:
    return InputEvent(name, Edge.DOWN)


def _up(name: str) -> InputEvent:
    return InputEvent(name, Edge.UP)


def _joy(type_: int, **fields) -> pygame.event.Event:
    return pygame.event.Event(type_, instance_id=7, **fields)


def test_the_gmpd3_names_its_buttons_as_an_xbox_pad_does_and_reads_its_stick():
    pad = _calibrated(_bare(CurrentDesignsGamepad3), _set(FILE, "dwell_s", 0))
    events = [_joy(pygame.JOYBUTTONDOWN, button=4), _joy(pygame.JOYBUTTONDOWN, button=9),
              pygame.event.Event(pygame.JOYBUTTONDOWN, instance_id=8, button=0),  # another's
              _joy(pygame.JOYAXISMOTION, axis=1, value=-0.9),
              _joy(pygame.JOYAXISMOTION, axis=3, value=-1.0)]            # pinned: no control
    assert pad.handle(events, 1.0) == [_down("BUTTON_A"), _down("LEFT_STICK_UP")]
    assert pad.held() == {"BUTTON_A", "LEFT_STICK_UP"}
    assert pad.handle([_joy(pygame.JOYBUTTONUP, button=4)], 1.1) == [_up("BUTTON_A")]
    assert pad.handle([], 1.2) == []                        # the stick held still stays pressed
    assert CurrentDesignsGamepad3.CONTROLS < XBoxController.CONTROLS


def test_an_xbox_controller_reads_its_mapping_and_not_the_joystick_twin():
    pad = _bare(XBoxController)
    a, trigger = pygame.CONTROLLER_BUTTON_A, pygame.CONTROLLER_AXIS_TRIGGERLEFT
    events = [_joy(pygame.CONTROLLERBUTTONDOWN, button=a), _joy(pygame.JOYBUTTONDOWN, button=0),
              _joy(pygame.CONTROLLERAXISMOTION, axis=trigger, value=12000)]   # under half
    assert pad.handle(events, 1.0) == [_down("BUTTON_A")]
    pull = [_joy(pygame.CONTROLLERAXISMOTION, axis=trigger, value=20000)]
    assert pad.handle(pull, 1.1) == [_down("LEFT_TRIGGER")]
    ease = [_joy(pygame.CONTROLLERAXISMOTION, axis=trigger, value=12000)]   # still past off
    assert pad.handle(ease, 1.2) == []
    let_go = [_joy(pygame.CONTROLLERAXISMOTION, axis=trigger, value=0)]
    assert pad.handle(let_go, 1.3) == [_up("LEFT_TRIGGER")]


def test_no_two_kinds_of_device_share_a_control_name():
    classes = list(inputs.INPUT_DEVICE_CLASSES_BY_NAME.values())
    for i, a in enumerate(classes):
        for b in classes[i + 1:]:
            assert not a.CONTROLS & b.CONTROLS or {a, b} == {CurrentDesignsGamepad3,
                                                               XBoxController}


# -- the keyboard and the run's inputs ---------------------------------------------------------


def _key(type_: int, key: int) -> pygame.event.Event:
    return pygame.event.Event(type_, key=key, mod=0, unicode="")


def test_the_keyboard_holds_what_its_events_press_until_the_window_loses_focus():
    keys = Keyboard()
    assert keys.handle([_key(pygame.KEYDOWN, pygame.K_EQUALS),
                        _key(pygame.KEYDOWN, pygame.K_LEFT)], 1.0) == [
        _down("EQUALS"), _down("LEFT")]
    assert keys.held() == {"EQUALS", "LEFT"}
    keys.handle([_key(pygame.KEYUP, pygame.K_EQUALS)], 1.1)
    assert keys.held() == {"LEFT"}
    assert keys.handle([pygame.event.Event(pygame.WINDOWFOCUSLOST)], 1.2) == [_up("LEFT")]
    assert keys.held() == frozenset()


def test_the_runs_inputs_take_the_queue_and_name_the_windows_close(screen):
    from fmri_gym.run import read_input_events
    pygame.event.post(_key(pygame.KEYDOWN, pygame.K_SPACE))
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    assert read_input_events(InputDevices([Keyboard()])) == [_down(QUIT), _down("SPACE")]
    assert pygame.event.peek() is False


@pytest.fixture
def sdl(monkeypatch):
    """SDL's joystick list, faked: ``sdl.names`` is what it sees; no SDL touched."""
    seen = types.SimpleNamespace(names=["Scripted", "Joystick Of Nobody's"])
    monkeypatch.setattr(inputs.controller, "start_sdl_joysticks_and_wait_for", lambda n: None)
    monkeypatch.setattr(inputs, "start_sdl_joysticks_and_wait_for", lambda n: None)
    monkeypatch.setattr(pygame.joystick, "get_count", lambda: len(seen.names))
    monkeypatch.setattr(pygame.joystick, "Joystick",
                        lambda i: types.SimpleNamespace(get_name=lambda: seen.names[i]))
    monkeypatch.setattr(inputs.controller.sdl_controller, "is_controller", lambda i: False)
    monkeypatch.setattr(CurrentDesignsGamepad3, "accepts",
                        classmethod(lambda cls, j, c: j.get_name() == "Scripted"))
    monkeypatch.setattr(CurrentDesignsGamepad3, "HANDLE",
                        staticmethod(lambda index: _FakeHandle(index, seen.names[index])))
    return seen


def test_a_controller_whose_handle_a_game_closed_is_given_a_new_one(sdl, capsys):
    pad = _bare(CurrentDesignsGamepad3)
    old = pad.joystick
    old.quit()                                              # what pygame.quit() does to it
    pad.handle([], 1.0)
    assert pad.joystick is not old and pad.is_open()
    sdl.names.clear()
    pad.joystick.quit()
    pad.handle([], 1.1)
    pad.handle([], 1.2)
    assert capsys.readouterr().err.count("is gone") == 1


def test_a_tap_inside_one_frame_counts_for_that_frame(screen):
    from fmri_gym.run import _poll_keys_until
    keys = InputDevices([Keyboard()])
    logged: list[dict] = []
    display = types.SimpleNamespace(idle=lambda deadline: None)
    clock = types.SimpleNamespace(run_time=lambda: 0.0)
    logger = types.SimpleNamespace(log=lambda **r: logged.append(r))
    pygame.event.post(_key(pygame.KEYDOWN, pygame.K_LEFT))
    pygame.event.post(_key(pygame.KEYUP, pygame.K_LEFT))
    pygame.event.post(_key(pygame.KEYDOWN, pygame.K_SPACE))
    pressed, quit_ = _poll_keys_until(keys, display, 0.0, clock, logger)
    assert (pressed, quit_, keys.held()) == (["LEFT", "SPACE"], False, {"SPACE"})
    assert [(r["name"], r["down"]) for r in logged] == [
        ("LEFT", True), ("LEFT", False), ("SPACE", True)]
    pygame.event.post(_key(pygame.KEYDOWN, pygame.K_z))     # turn-based: the wait ends at Z
    pygame.event.post(_key(pygame.KEYDOWN, pygame.K_x))
    pressed, _ = _poll_keys_until(keys, display, math.inf, clock, logger, stop_on={"Z"})
    assert pressed == ["Z"]


@pytest.mark.parametrize("names, says", [
    (["keyboard", "keyboard"], "expected distinct names"),
    (["gmpd3"], "expected distinct names"),
    (["xbox_controller"], 'lists "xbox_controller" under "inputs", but none is plugged in'),
])
def test_a_rigs_inputs_that_cannot_be_opened_stop_the_run(sdl, config_home, names, says):
    with pytest.raises(ValueError, match=re.escape(says)):
        inputs.open_input_devices(names)


def test_a_rigs_inputs_open_its_devices_and_say_which_joysticks_they_ignore(sdl, config_home):
    opened = inputs.open_input_devices(["keyboard", "current_designs_gamepad3"],
                                       frozenset({"BUTTON_A", "SPACE"}))
    assert [d.name for d in opened.devices] == ["keyboard", "current_designs_gamepad3"]
    assert "IGNORED \"Joystick Of Nobody's\"" in opened.status()
    assert opened.controls() == Keyboard.CONTROLS | CurrentDesignsGamepad3.CONTROLS
    with pytest.raises(CalibrationError, match="LEFT_STICK"):
        inputs.open_input_devices(["current_designs_gamepad3"], frozenset({"LEFT_STICK_LEFT"}))
    assert [d.name for d in inputs.open_input_devices(["current_designs_gamepad3"]).devices] == [
        "current_designs_gamepad3"]                   # no keyboard unless the rig lists one


# -- pad-calibration's window -------------------------------------------------------------------


@pytest.fixture
def scope(config_home, screen, monkeypatch):
    """The tool's window on the dummy display, with the scripted controller plugged in."""
    from fmri_gym import pad_scope
    monkeypatch.setattr(pad_scope, "first_controller_class_plugged_in",
                        lambda kinds: _ScriptedController)
    monkeypatch.setattr(_ScriptedController, "open",
                        classmethod(lambda cls, calibrated=True: cls(calibrated=calibrated)))
    return pad_scope._Scope(("current_designs_gamepad3",))


def test_the_keyboard_does_nothing_in_the_tool_but_esc(scope):
    path = files.write(files.StickCalibration.from_dict(copy.deepcopy(FILE)))
    scope.reload("")
    before = (scope.stick.tuning, scope.stick.name, sorted(files.calibrations_dir().iterdir()))
    for key in (pygame.K_s, pygame.K_d, pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT,
                pygame.K_RIGHT, pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET, pygame.K_TAB):
        pygame.event.post(_key(pygame.KEYDOWN, key))
    scope.events()
    assert (scope.stick.tuning, scope.stick.name,
            sorted(files.calibrations_dir().iterdir())) == before and scope.running
    assert scope.controller.stick(LEFT_STICK).calibration_path == path
    pygame.event.post(_key(pygame.KEYDOWN, pygame.K_ESCAPE))
    scope.events()
    assert not scope.running


@pytest.mark.parametrize("action", ["on-", "on+", "off-", "off+", "dwell-", "dwell+", "diag"])
def test_each_tuning_button_moves_one_field_within_bounds(action):
    from fmri_gym import pad_scope
    for start in (UNCALIBRATED, StickTuning(0.99, 0.99, 0.7, 0.3),
                  StickTuning(0.5, 0.01, 1.0, 0.0)):
        moved = pad_scope._retuned(start, action)
        assert sum(a != b for a, b in zip(vars(moved).values(), vars(start).values())) <= 1
        StickTuning.from_dict(moved.record(), "")      # what a file allows
    middle = StickTuning(0.5, 0.3, 1.0, 0.05)
    assert pad_scope._retuned(middle, action) != middle


def test_save_writes_a_new_newest_file_of_the_retuned_stick_and_leaves_its_source(scope):
    from fmri_gym import pad_scope
    source = files.write(files.StickCalibration.from_dict(copy.deepcopy(FILE)))
    right = files.write(files.StickCalibration.from_dict(copy.deepcopy(RIGHT_FILE)))
    for p in (source, right):
        os.utime(p, (1000, 1000))
    was = source.read_bytes()
    scope.reload("")
    pad_scope._ACTIONS["on+"](scope)
    scope.click(next(b.rect.center for b in scope.buttons if b.action == "save"))
    assert source.read_bytes() == was
    path, saved = files.newest("Scripted", LEFT_STICK)
    assert path != source and saved.tuned_from == source.name
    assert (saved.tuning.on, saved.made) == (0.51, files.read(source).made)
    assert files.newest("Scripted", RIGHT_STICK)[0] == right      # not retuned: not rewritten
    assert scope.controller.stick(LEFT_STICK).calibration_path == path


def test_the_watch_reloads_only_on_its_one_second_tick(scope, monkeypatch):
    reloads = []
    monkeypatch.setattr(scope, "reload", reloads.append)
    start = scope.next_watch
    files.write(files.StickCalibration.from_dict(copy.deepcopy(FILE)))   # a calibration added
    scope.watch(start - 0.5)
    assert reloads == []
    scope.watch(start)
    assert reloads == ["a calibration changed: reloaded"] and scope.next_watch == start + 1.0
    scope.draw()                                                     # and it draws


def test_a_click_is_mapped_back_to_the_picture_at_any_window_size(scope):
    from fmri_gym import pad_scope
    canvas = scope.canvas
    canvas.window.size = (900, 1200)                              # bars above and below
    assert canvas.window.size == (900, 1200)
    s = 900 / pad_scope.W
    top = (1200 - int(pad_scope.H * s)) // 2
    assert canvas.to_canvas((int(100 * s), top + int(40 * s))) == pytest.approx((100, 40), abs=2)
    clear = next(b for b in scope.buttons if b.action == "clear")
    scope.log.append((0.0, "a press", None))
    scope.click(canvas.to_canvas((int(clear.rect.centerx * s), top + int(clear.rect.centery * s))))
    assert not scope.log
