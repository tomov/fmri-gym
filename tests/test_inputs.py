"""The input devices with none plugged in: their events, their stick axes, their calibration.

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
import sys
import types
from datetime import datetime, timezone

import pygame
import pytest

import humans.inputs as inputs
from humans.standalone import gamepad_calibration as cal
from humans.inputs import (QUIT, CalibrationError, ControlNames, CurrentDesignsGamepad3, Edge,
                             InputDevices, InputEvent, Keyboard, XBoxController)
from humans.inputs import calibration as files
from humans.inputs.calibration import AxisCalibration
from humans.inputs.controller import AXIS_CONTROLS, FACE_BUTTONS, Controller, short_name

LX, LY = ControlNames.CONTROLLER_AXIS_LEFTX, ControlNames.CONTROLLER_AXIS_LEFTY
RX, RY = ControlNames.CONTROLLER_AXIS_RIGHTX, ControlNames.CONTROLLER_AXIS_RIGHTY
MADE = "2026-01-21T14:15:00+00:00"
AXIS = {"rest": 0.0, "low": -1.0, "high": 1.0, "drift": 0.01, "on": 0.5, "off": 0.3}
#: A calibration of the scripted controller's four axes, and of the GMPD3's two.
FILE = {"format": files.FORMAT, "device": "Scripted", "tuned_from": None, "made": MADE,
        "axes": {LX: AXIS, LY: AXIS, RX: {**AXIS, "on": 0.6}, RY: AXIS}}
GMPD3_FILE = {**FILE, "axes": {LX: AXIS, LY: AXIS}}


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


# -- the calibration file -----------------------------------------------------------------


def test_a_calibration_round_trips_through_its_file():
    loaded = files.ControllerCalibration.from_dict(copy.deepcopy(FILE))
    assert loaded.record() == FILE
    assert (loaded.axes[RX].on, loaded.axes[LX].on) == (0.6, 0.5)
    assert loaded.made == datetime(2026, 1, 21, 14, 15, tzinfo=timezone.utc)


@pytest.mark.parametrize("field, value, says", [
    (f"axes.{LX}.on", "0.5", f"axes.{LX}.on: expected a number in (0, 1), got '0.5'"),
    (f"axes.{LX}.off", 0.6, f"axes.{LX}.off: expected a number in (0, on=0.5], got 0.6"),
    (f"axes.{LX}.drift", True, f"axes.{LX}.drift: expected a number >= 0, got True"),
    (f"axes.{LX}.low", 0.5, f"axes.{LX}: expected low < rest < high"),
    (f"axes.{LX}.noise", 0.001, f"axes.{LX}.noise: unknown key"),
    (f"axes.{LY}", ..., None),
    ("axes", {}, "axes: expected each axis by its name, got {}"),
    ("made", "2026-10-21T14:15:00", "made: expected a time such as"),
    ("device", ..., "device: missing"),
    ("device", "", "device: expected a name, got ''"),
    ("tuned_from", 3, "tuned_from: expected null or a file name, got 3"),
    ("stick", "LEFT_STICK", "stick: unknown key"),
    ("format", ..., "format: expected 1, got None"),
    ("format", 0, "format: expected 1, got 0"),
    (f"axes.{LX}.drift", 1.0, f"axes.{LX}: its centre drifts 1, as far as its stop"),
])
def test_each_wrong_field_is_refused_by_its_file_and_name(tmp_path, field, value, says):
    path = tmp_path / "Scripted_20261021-141500.json"
    path.write_text(json.dumps(_set(FILE, field, value)))
    if says is None:                                # fewer axes is a file; a controller refuses it
        files.read(path)
        return
    with pytest.raises(CalibrationError, match=re.escape(f"{path}: {says}")):
        files.read(path)


def test_a_controller_is_read_through_its_newest_calibration(config_home):
    _write(config_home, "Scripted_a.json", _set(FILE, f"axes.{LX}.on", 0.31), 1000)
    newest = _write(config_home, "Scripted_b.json", _set(FILE, f"axes.{LX}.on", 0.33), 3000)
    _write(config_home, "Scripted_c.json", _set(FILE, f"axes.{LX}.on", 0.32), 2000)
    _write(config_home, "Other_d.json", {**FILE, "device": "Other"}, 4000)   # newer, another pad
    _write(config_home, "Other_e.json", {"device": "Other", "axes": {}}, 5000)  # bad, not ours
    path, found = files.newest("Scripted")
    assert (path, found.axes[LX].on) == (newest, 0.33)
    assert files.newest("Nobody") is None


def test_an_older_calibration_is_never_read(config_home):
    _write(config_home, "Scripted_a.json", {"device": "Scripted", "sticks": {}}, 1000)
    _write(config_home, "Scripted_b.json", _set(FILE, f"axes.{LX}.drift", 0.9), 2000)
    newest = _write(config_home, "Scripted_c.json", FILE, 3000)
    assert files.newest("Scripted")[0] == newest


def test_a_malformed_newest_calibration_stops_the_run_with_no_fallback(config_home):
    _write(config_home, "Scripted_a.json", FILE, 1000)
    bad = _write(config_home, "Scripted_b.json", _set(FILE, f"axes.{LX}.on", "0.5"), 2000)
    with pytest.raises(CalibrationError, match=re.escape(f"{bad}: axes.{LX}.on")):
        files.newest("Scripted")
    bad.write_text("{not json")
    with pytest.raises(CalibrationError, match="not JSON"):
        files.newest("Scripted")


# -- a controller with no SDL behind it -----------------------------------------------------


class _FakeHandle:
    """What a controller reads off SDL's handle: its name, the id its events carry, its axes
    at rest.

    A joystick's and a game controller's handles spell these differently; this one spells both.
    Like SDL's, ``id`` is the index it was opened at, not the instance id its events carry.
    """

    def __init__(self, index: int = 0, name: str = "Scripted", instance_id: int = 7,
                 axes: dict[int, float] | None = None) -> None:
        self.name, self.id, self.instance_id = name, index, instance_id
        self.axes = axes or {}
        self.open = True

    def get_name(self) -> str:
        return self.name

    def get_instance_id(self) -> int:
        return self.instance_id

    def as_joystick(self) -> _FakeHandle:
        return self

    def get_axis(self, axis: int) -> float:
        return self.axes.get(axis, 0.0)

    def get_init(self) -> bool:
        return self.open

    def quit(self) -> None:
        self.open = False


def _bare(cls: type[Controller], name: str = "Scripted") -> Controller:
    """A controller of ``cls`` as if SDL had opened it, with events of id 7, uncalibrated."""
    return cls(_FakeHandle(name=name), calibrated=False)


def _calibrated(controller: Controller, blob: dict) -> Controller:
    """``controller`` reading its axes through ``blob``."""
    controller.set_calibration((None, files.ControllerCalibration.from_dict(blob)))
    return controller


class _ScriptedController(Controller):
    """A two-stick, two-button controller whose readings are a script, one frame each; then
    ESC."""

    name = "scripted"
    AXIS_NAMES = (LX, LY, RX, RY)
    BUTTON_NAMES = ("S_A", "S_B")
    CONTROLS = frozenset({*BUTTON_NAMES, *(c for a in AXIS_NAMES for c in AXIS_CONTROLS[a])})
    HANDLE = _FakeHandle

    def __init__(self, handle=None, calibrated: bool = False, frames=()) -> None:
        self.frames = iter(frames)
        self.axes = self.axes_at({})
        super().__init__(handle or _FakeHandle(), calibrated)

    @classmethod
    def accepts(cls, joystick, is_controller: bool) -> bool:
        return True

    def attach(self, handle) -> None:
        self.device_handle, self.device_name = handle, handle.name
        self.instance_id = handle.get_instance_id()

    def _read_raw_value(self, axis_name: str) -> float:
        return self.axes[axis_name]

    def handle(self, events, now: float) -> list[InputEvent]:
        """Take the next scripted frame as the controller's reading, ignoring the events."""
        frame = next(self.frames, None)
        if frame is None:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
            frame = (self.axes_at({}), set())
        self.axes, buttons = frame
        axes = self.axes
        out = [e for b in self.BUTTON_NAMES
               for e in (self._press(b) if b in buttons else self._release(b))]
        return out + [e for name in self.AXIS_NAMES for e in self._move_axis(name, axes[name])]

    @classmethod
    def axes_at(cls, moved: dict[str, float]) -> dict[str, float]:
        return {name: moved.get(name, 0.0) for name in cls.AXIS_NAMES}


#: Where the scripted controller rests, cold and then warm.
COLD, REST = {LX: 0.02}, {LX: -0.05, LY: 0.01}


def _frames(d: _ScriptedController, wrong: str | None = None) -> list:
    """Hands off; each stick rolled twice round its edge; let go; each axis control
    twice (``wrong`` pushed once first); each button twice."""
    still = [(d.axes_at(COLD), set())] * 70
    turn = [(math.cos(2 * math.pi * k / 48), math.sin(2 * math.pi * k / 48)) for k in range(96)]
    stops = [(d.axes_at({**COLD, x: cx, y: cy}), set())
             for x, y in ((LX, LY), (RX, RY)) for cx, cy in turn]
    rest = [(d.axes_at(REST), set())] * 70
    out = still + stops + rest
    for control in ([wrong] if wrong else []) + [c for c in d.axis_controls() for _ in range(2)]:
        name = next(n for n, sides in AXIS_CONTROLS.items() if control in sides)
        sign = -1 if control == AXIS_CONTROLS[name][0] else 1
        out += [rest[0], (d.axes_at({**REST, name: 0.9 * sign}), set())]
    buttons = [f for b in d.BUTTON_NAMES for f in [(d.axes_at({}), set()), (d.axes_at({}), {b})] * 2]
    return out + buttons


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
    yield pygame.Surface((900, 600))
    pygame.event.clear()                    # the ESC a scripted controller posts at its end


# -- the calibration procedure ------------------------------------------------------------


def test_a_calibration_measures_each_axis_and_confirms_each_control(config_home, screen):
    d = _ScriptedController()
    d.frames = iter(_frames(d, wrong=ControlNames.CONTROLLER_AXIS_LEFTX_RIGHT))
    result = cal.calibrate(d, screen, lambda: None)
    assert (result.confirmed, result.corrections) == (8 + 2, 1)
    measured = result.calibration.axes[LX]
    assert (measured.rest, measured.low, measured.high, measured.drift) == (-0.05, -1.0, 1.0, 0.07)
    assert (measured.on, measured.off) == (cal.CAL_ON, cal.CAL_OFF)
    assert set(result.calibration.axes) == {LX, LY, RX, RY}
    assert files.newest("Scripted") == (result.path, result.calibration)
    assert result.path.with_suffix(".csv").exists() and "Scripted_" in result.path.name
    assert d.calibration is None                    # the caller applies what was written
    from humans.standalone.rig_setup.report import _pad_line
    line = _pad_line({"present": True, "device": "Scripted", **result.record()})
    assert f"{LX} on past 0.50" in line and "10 controls confirmed, 1 wrong" in line


def test_the_buttons_are_asked_for_in_order_and_the_system_ones_not():
    from humans.inputs.xbox_controller import XBoxController
    asked = cal.cue_order(XBoxController.BUTTON_NAMES)
    assert not set(asked) & cal.NOT_ASKED
    assert asked[:4] == list(FACE_BUTTONS) and len(asked) == len(XBoxController.BUTTON_NAMES) - 3
    names = [short_name(b) for b in asked[4:]]
    assert names == ["LEFTSTICK", "RIGHTSTICK", "DPAD_UP", "DPAD_DOWN", "DPAD_LEFT", "DPAD_RIGHT",
                     "LEFTSHOULDER", "RIGHTSHOULDER", "TRIGGERLEFT", "TRIGGERRIGHT"]


def test_a_calibration_stopped_part_way_writes_nothing(config_home, screen):
    d = _ScriptedController()
    d.frames = iter(_frames(d)[:100])
    assert cal.calibrate(d, screen, lambda: None) is None
    assert not config_home.exists()


def test_the_fit_raises_the_threshold_of_the_axis_that_dithers_and_no_other():
    rows = ([{"t": k / 60, "phase": "cold", "cue": "", LX: 0.0, LY: 0.0} for k in range(60)]
            + [{"t": 1 + k / 60, "phase": "stops", "cue": LX, LX: v, LY: -v}
               for k, v in enumerate((-0.5, 1.0, 0.2))]
            + [{"t": 2 + k / 60, "phase": "rest", "cue": "", LX: 0.1 + 0.02 * (k % 2), LY: 0.0}
               for k in range(60)])
    fitted = cal.fit(rows, (LX, LY), "Scripted", datetime.now(timezone.utc))
    x = fitted.axes[LX]
    assert (x.rest, x.low, x.high, x.drift) == (0.11, -0.5, 1.0, 0.11)
    # dither 0.01 of the 0.61 travel to the low side: ten times it is 0.164, under 0.5
    assert (x.on, x.off) == (0.5, 0.3) and fitted.device == "Scripted"
    shaky = [{**r, LX: 0.1 + 0.1 * (k % 2)} if r["phase"] == "rest" else r
             for k, r in enumerate(rows)]
    fitted = cal.fit(shaky, (LX, LY), "Scripted", datetime.now(timezone.utc))
    assert fitted.axes[LX].on > cal.CAL_ON and fitted.axes[LX].off == round(0.6 * fitted.axes[LX].on, 3)
    assert (fitted.axes[LY].on, fitted.axes[LY].off) == (cal.CAL_ON, cal.CAL_OFF)


# -- the axes and the controllers --------------------------------------------------------------


def test_each_axis_reads_its_own_thresholds():
    d = _calibrated(_ScriptedController(), FILE)
    d.frames = iter([(d.axes_at({LX: 0.55, RX: 0.55}), set()),    # past LX's 0.5, not RX's 0.6
                     (d.axes_at({LX: 0.55, RX: 0.65}), set())])
    assert d.handle([], 1.0) == [_down(ControlNames.CONTROLLER_AXIS_LEFTX_RIGHT)]
    assert d.handle([], 1.1) == [_down(ControlNames.CONTROLLER_AXIS_RIGHTX_RIGHT)]
    assert d.held() == {ControlNames.CONTROLLER_AXIS_LEFTX_RIGHT,
                        ControlNames.CONTROLLER_AXIS_RIGHTX_RIGHT}


def test_a_side_is_pressed_past_on_at_once_and_released_under_off():
    d = _calibrated(_ScriptedController(), FILE)
    d.frames = iter([(d.axes_at({LY: v}), set()) for v in (-0.4, -0.9, -0.4, -0.2, 0.9)])
    assert d.handle([], 1.0) == []                                  # short of on
    assert d.handle([], 1.01) == [_down(ControlNames.CONTROLLER_AXIS_LEFTY_UP)]
    assert d.handle([], 1.02) == []                                 # past off: still held
    assert d.handle([], 1.03) == [_up(ControlNames.CONTROLLER_AXIS_LEFTY_UP)]
    assert d.handle([], 1.04) == [_down(ControlNames.CONTROLLER_AXIS_LEFTY_DOWN)]


def test_a_diagonal_push_presses_a_side_of_each_axis():
    d = _calibrated(_ScriptedController(), FILE)
    d.frames = iter([(d.axes_at({LX: -0.9, LY: 0.6}), set())])
    assert d.handle([], 1.0) == [_down(ControlNames.CONTROLLER_AXIS_LEFTX_LEFT),
                                 _down(ControlNames.CONTROLLER_AXIS_LEFTY_DOWN)]


def test_a_stick_axis_has_a_value_and_each_side_how_far_it_is_pushed():
    d = _calibrated(_ScriptedController(), FILE)
    d.frames = iter([(d.axes_at({LX: -1.0, LY: 0.0, RX: 0.3}), {"S_A"})])
    d.handle([], 1.0)
    values = d.values()
    assert values[LX] == -1.0 and values[ControlNames.CONTROLLER_AXIS_LEFTX_LEFT] == 1.0
    assert 0.0 < values[RX] < 0.3 and values[RX] == values[ControlNames.CONTROLLER_AXIS_RIGHTX_RIGHT]
    assert "S_A" not in values                                  # a button's value is held()
    assert values[LY] == 0.0 and values[ControlNames.CONTROLLER_AXIS_LEFTX_RIGHT] == 0.0


def test_a_stick_within_its_drift_of_rest_reads_zero_and_its_stops_one():
    axis = files.AxisCalibration(rest=0.1, low=-0.9, high=0.6, drift=0.05, on=0.5, off=0.3)
    assert axis.value(0.1) == 0.0 and axis.value(0.14) == 0.0 and axis.value(0.06) == 0.0
    assert axis.value(0.6) == 1.0 and axis.value(-0.9) == -1.0
    assert axis.value(0.35) == pytest.approx((0.25 / 0.5 - 0.1) / 0.9)


def test_an_uncalibrated_axis_presses_nothing():
    d = _ScriptedController()
    d.frames = iter([(d.axes_at({LX: 1.0, LY: -1.0}), set())])
    assert d.handle([], 1.0) == [] and d.raw_value(LX) == 1.0
    assert d.values() == {}


def test_a_controller_records_the_calibration_it_reads_through(config_home):
    pad = _bare(CurrentDesignsGamepad3)
    assert pad.record()["calibration"] is None
    path = _write(config_home, "Scripted_a.json", GMPD3_FILE, 1000)
    pad.load_calibration()
    assert pad.record() == {"input": pad.name, "device_name": "Scripted",
                            "calibration_file": str(path), "calibration": GMPD3_FILE}


def test_a_controller_refuses_a_required_axis_with_no_calibration(config_home):
    pad = _bare(CurrentDesignsGamepad3)
    with pytest.raises(CalibrationError,
                       match="no calibration of 'Scripted' in .*stick controls are required"):
        pad.require_calibrated_axes_for(frozenset({ControlNames.CONTROLLER_AXIS_LEFTY_UP,
                                                   ControlNames.CONTROLLER_BUTTON_A}))
    buttons_only = frozenset({ControlNames.CONTROLLER_BUTTON_A, "SPACE"})
    pad.require_calibrated_axes_for(buttons_only)
    assert "axes UNCALIBRATED" in pad.status()
    other = _write(config_home, "Scripted_a.json", FILE, 1000)          # axes it does not have
    with pytest.raises(CalibrationError, match=re.escape(f"{other}: calibrates the axes")):
        pad.load_calibration()
    path = _write(config_home, "Scripted_b.json", GMPD3_FILE, 2000)
    pad.load_calibration()
    pad.require_calibrated_axes_for(frozenset({ControlNames.CONTROLLER_AXIS_LEFTY_UP}))
    assert str(path) in pad.status() and pad.calibration_path == path


def _down(name: str) -> InputEvent:
    return InputEvent(name, Edge.DOWN)


def _up(name: str) -> InputEvent:
    return InputEvent(name, Edge.UP)


def _joy(type_: int, **fields) -> pygame.event.Event:
    return pygame.event.Event(type_, instance_id=7, **fields)


def test_the_gmpd3_names_its_buttons_as_an_xbox_pad_does_and_reads_its_stick():
    pad = _calibrated(_bare(CurrentDesignsGamepad3), GMPD3_FILE)
    events = [_joy(pygame.JOYBUTTONDOWN, button=4), _joy(pygame.JOYBUTTONDOWN, button=9),
              pygame.event.Event(pygame.JOYBUTTONDOWN, instance_id=8, button=0),  # another's
              _joy(pygame.JOYAXISMOTION, axis=1, value=-0.9),
              _joy(pygame.JOYAXISMOTION, axis=3, value=-1.0)]            # pinned: no control
    assert pad.handle(events, 1.0) == [_down(ControlNames.CONTROLLER_BUTTON_A),
                                       _down(ControlNames.CONTROLLER_AXIS_LEFTY_UP)]
    assert pad.held() == {ControlNames.CONTROLLER_BUTTON_A, ControlNames.CONTROLLER_AXIS_LEFTY_UP}
    assert (pad.handle([_joy(pygame.JOYBUTTONUP, button=4)], 1.1)
            == [_up(ControlNames.CONTROLLER_BUTTON_A)])
    assert pad.handle([], 1.2) == []                        # the stick held still stays pressed
    assert CurrentDesignsGamepad3.CONTROLS < XBoxController.CONTROLS


def test_an_xbox_controller_reads_its_mapping_and_not_the_joystick_twin():
    pad = _bare(XBoxController)
    a, trigger = pygame.CONTROLLER_BUTTON_A, pygame.CONTROLLER_AXIS_TRIGGERLEFT
    events = [_joy(pygame.CONTROLLERBUTTONDOWN, button=a), _joy(pygame.JOYBUTTONDOWN, button=0),
              _joy(pygame.CONTROLLERAXISMOTION, axis=trigger, value=12000)]   # under half
    assert pad.handle(events, 1.0) == [_down(ControlNames.CONTROLLER_BUTTON_A)]
    pull = [_joy(pygame.CONTROLLERAXISMOTION, axis=trigger, value=20000)]
    assert pad.handle(pull, 1.1) == [_down(ControlNames.CONTROLLER_AXIS_TRIGGERLEFT)]
    ease = [_joy(pygame.CONTROLLERAXISMOTION, axis=trigger, value=12000)]   # still past off
    assert pad.handle(ease, 1.2) == []
    assert pad.values()[ControlNames.CONTROLLER_AXIS_TRIGGERLEFT] == 12000 / 32767
    let_go = [_joy(pygame.CONTROLLERAXISMOTION, axis=trigger, value=0)]
    assert pad.handle(let_go, 1.3) == [_up(ControlNames.CONTROLLER_AXIS_TRIGGERLEFT)]
    assert pad.values()[ControlNames.CONTROLLER_AXIS_TRIGGERLEFT] == 0.0


def test_an_xbox_controller_reads_each_stick_axis_through_its_calibration():
    pad = _calibrated(_bare(XBoxController), FILE)
    right_x = pygame.CONTROLLER_AXIS_RIGHTX
    assert pad.handle([_joy(pygame.CONTROLLERAXISMOTION, axis=right_x, value=-20000)], 1.0) == [
        _down(ControlNames.CONTROLLER_AXIS_RIGHTX_LEFT)]
    assert pad.raw_value(RX) == pytest.approx(-20000 / 32767)
    assert pad.values()[RX] == pytest.approx((20000 / 32767 - 0.01) / 0.99 * -1)


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


def test_each_read_that_makes_an_input_event_gives_the_triggers_every_control_held(screen):
    from humans.run import read_input_events
    held: list[frozenset[str]] = []
    triggers = types.SimpleNamespace(codes=types.SimpleNamespace(inputs={"LEFT": 1}),
                                     hold_input_codes=held.append)
    keys = InputDevices([Keyboard()])
    for events in ([_key(pygame.KEYDOWN, pygame.K_LEFT)], [],
                   [_key(pygame.KEYDOWN, pygame.K_UP), _key(pygame.KEYUP, pygame.K_LEFT)]):
        for event in events:
            pygame.event.post(event)
        read_input_events(keys, None, triggers)
    assert held == [{"LEFT"}, {"UP"}], "a read without an input event sends nothing"


def test_the_runs_inputs_take_the_queue_and_name_the_windows_close(screen):
    from humans.run import read_input_events
    pygame.event.post(_key(pygame.KEYDOWN, pygame.K_SPACE))
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    assert read_input_events(InputDevices([Keyboard()]), None) == [_down(QUIT), _down("SPACE")]
    assert pygame.event.peek() is False


def test_ctrl_escape_asks_to_stop_the_run_and_escape_alone_does_not(screen):
    from humans.run import read_input_events
    keys = InputDevices([Keyboard()])
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0))
    assert QUIT not in [name for name, _ in read_input_events(keys, None)]
    for ctrl in (pygame.KMOD_LCTRL, pygame.KMOD_RCTRL):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=ctrl))
        assert read_input_events(keys, None)[0] == _down(QUIT)


def test_ctrl_return_asks_to_end_the_run_and_says_so_and_return_alone_does_not(screen):
    from humans.run import read_input_events
    keys = InputDevices([Keyboard()])
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0))
    assert QUIT not in [name for name, _ in read_input_events(keys, None)]
    assert not keys.has_end_run_request
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN,
                                         mod=pygame.KMOD_LCTRL))
    assert read_input_events(keys, None)[0] == _down(QUIT)
    assert keys.has_end_run_request, "the stop it asks for ends this run alone"


def test_the_run_logs_each_event_read_as_pygame_gave_it_then_each_input_event(screen):
    from humans.run import read_input_events
    logged: list[tuple[str, float, dict]] = []
    logger = types.SimpleNamespace(log_run_event=lambda line_type, perf_time, **fields:
                                   logged.append((line_type, perf_time, fields)))
    events = [_key(pygame.KEYDOWN, pygame.K_LEFT),
              pygame.event.Event(pygame.JOYAXISMOTION, joy=0, instance_id=3, axis=1, value=-0.73),
              pygame.event.Event(pygame.MOUSEMOTION, pos=(3, 4), rel=(1, 1), buttons=(0, 0, 0)),
              pygame.event.Event(pygame.WINDOWFOCUSLOST),
              pygame.event.Event(pygame.QUIT)]
    for event in events:
        pygame.event.post(event)
    input_events = read_input_events(InputDevices([Keyboard()]), logger)
    assert input_events == [_down(QUIT), _down("LEFT"), _up("LEFT")]
    assert logged == [("pygame_event", logged[0][1], {"event_name": pygame.event.event_name(e.type),
                                                      "event": {"type": e.type, **e.dict}})
                      for e in events] + [
        ("input_event", logged[0][1], {"name": name, "down": edge is Edge.DOWN})
        for name, edge in input_events]


def test_a_value_a_log_line_cannot_carry_is_written_as_its_repr():
    from humans.run import _loggable
    window = object()
    assert _loggable({"key": 97, "pos": (1, 2), "window": window, "text": None}) == {
        "key": 97, "pos": (1, 2), "window": repr(window), "text": None}


def test_a_value_a_log_line_cannot_carry_inside_a_dict_is_written_as_its_repr():
    from humans.run import _loggable
    window = object()
    assert _loggable({"touch": {"x": 0.5, "window": window}}) == {
        "touch": {"x": 0.5, "window": repr(window)}}


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


@pytest.mark.parametrize("cls, button_down", [
    (CurrentDesignsGamepad3, lambda instance_id: pygame.event.Event(
        pygame.JOYBUTTONDOWN, instance_id=instance_id, button=4)),
    (XBoxController, lambda instance_id: pygame.event.Event(
        pygame.CONTROLLERBUTTONDOWN, instance_id=instance_id, button=pygame.CONTROLLER_BUTTON_A)),
])
def test_a_controller_whose_handle_a_game_closed_is_given_a_new_one(
        sdl, capsys, monkeypatch, cls, button_down):
    monkeypatch.setattr(cls, "accepts", classmethod(lambda c, j, i: j.get_name() == "Scripted"))
    monkeypatch.setattr(cls, "HANDLE", staticmethod(          # reopened: a new instance id
        lambda index: _FakeHandle(index, sdl.names[index], instance_id=8)))
    pad = _bare(cls)
    old = pad.device_handle
    old.quit()                                              # what pygame.quit() does to it
    assert pad.handle([button_down(7)], 1.0) == []          # the closed handle's id
    assert pad.device_handle is not old and pad.is_open()
    assert pad.handle([button_down(8)], 1.05) == [_down(ControlNames.CONTROLLER_BUTTON_A)]
    sdl.names.clear()
    pad.device_handle.quit()
    pad.handle([], 1.1)
    pad.handle([], 1.2)
    assert capsys.readouterr().err.count("is gone") == 1


def test_a_reopened_controller_forgets_its_values_and_reads_where_its_stick_is(sdl, monkeypatch):
    pad = _calibrated(_bare(CurrentDesignsGamepad3), GMPD3_FILE)
    pad.handle([_joy(pygame.JOYAXISMOTION, axis=0, value=0.9)], 1.0)
    assert {name for name, value in pad.values().items() if value} == {
        LX, ControlNames.CONTROLLER_AXIS_LEFTX_RIGHT}
    monkeypatch.setattr(CurrentDesignsGamepad3, "HANDLE", staticmethod(   # pushed up, no event
        lambda index: _FakeHandle(index, sdl.names[index], axes={1: -0.9})))
    pad.device_handle.quit()                                # what pygame.quit() does to it
    assert pad.handle([], 1.1) == [_up(ControlNames.CONTROLLER_AXIS_LEFTX_RIGHT),
                                   _down(ControlNames.CONTROLLER_AXIS_LEFTY_UP)]
    assert {name for name, value in pad.values().items() if value} == {
        LY, ControlNames.CONTROLLER_AXIS_LEFTY_UP}
    assert pad.values()[LY] < 0 and pad.raw_value(LY) == -0.9
    pad.close()
    assert pad.values() == {} and pad.raw_value(LY) == 0.0


class _SdlControllerHandle(_FakeHandle):
    """A handle that, like SDL's, raises when asked about itself after ``pygame.quit()``."""

    def get_init(self) -> bool:
        if not pygame.get_init():
            raise pygame.error("Controller system not initialized")
        return self.open


def test_closing_a_run_saves_every_frame_and_closes_a_controller_before_the_window(
        monkeypatch, tmp_path):
    import numpy as np
    from humans.run import Run
    from humans.save.logging import Logger, read_frames
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    pygame.init()                                           # the window's close quits it
    logger = Logger(str(tmp_path), "sub-test", [])
    block = str(tmp_path / logger.open_block(0, "gym", "CartPole-v1", {}, base_seed=0))
    for i in range(200):                                    # a backlog for the writer
        logger.log_frame({"frame": i}, frame=np.full((360, 640, 3), i % 256, dtype=np.uint8))
    run = object.__new__(Run)
    run.logger = logger
    run.inputs = InputDevices([Keyboard(), XBoxController(_SdlControllerHandle(),
                                                           calibrated=False)])
    run.triggers = run.audio = types.SimpleNamespace(close=lambda: None)
    run.display = types.SimpleNamespace(close=pygame.quit)  # what the window's close does
    try:
        run.close()
        assert not logger._process.is_alive() and logger._process.exitcode == 0
        assert len(read_frames(block)[0]) == 200
    finally:
        logger.close()      # when close raised: else pytest hangs at exit on the writer's queue


def test_a_tap_inside_one_frame_counts_for_that_frame(screen):
    from humans.run import _poll_keys_until
    keys = InputDevices([Keyboard()])
    logged: list[dict] = []
    display = types.SimpleNamespace(idle=lambda deadline: None)
    logger = types.SimpleNamespace(log_run_event=lambda line_type, perf_time, **fields:
                                   logged.append({"type": line_type, **fields}))
    pygame.event.post(_key(pygame.KEYDOWN, pygame.K_LEFT))
    pygame.event.post(_key(pygame.KEYUP, pygame.K_LEFT))
    pygame.event.post(_key(pygame.KEYDOWN, pygame.K_SPACE))
    pressed, quit_ = _poll_keys_until(keys, display, 0.0, logger)
    assert (pressed, quit_, keys.held()) == (["LEFT", "SPACE"], False, {"SPACE"})
    assert [(r["name"], r["down"]) for r in logged if r["type"] == "input_event"] == [
        ("LEFT", True), ("LEFT", False), ("SPACE", True)]
    pygame.event.post(_key(pygame.KEYDOWN, pygame.K_z))     # turn-based: the wait ends at Z
    pygame.event.post(_key(pygame.KEYDOWN, pygame.K_x))
    pressed, _ = _poll_keys_until(keys, display, math.inf, logger, stop_on={"Z"})
    assert pressed == ["Z"]


@pytest.mark.parametrize("names, says", [
    (["keyboard", "keyboard"], "expected distinct names"),
    (["gmpd3"], "expected distinct names"),
    (["xbox_controller"], 'no "xbox_controller" is plugged in'),
])
def test_a_rigs_inputs_that_cannot_be_opened_stop_the_run(sdl, config_home, names, says):
    with pytest.raises(ValueError, match=re.escape(says)):
        inputs.open_input_devices(names)


def test_a_rigs_inputs_open_its_devices_and_say_which_joysticks_they_ignore(sdl, config_home):
    opened = inputs.open_input_devices(["keyboard", "current_designs_gamepad3"],
                                       frozenset({ControlNames.CONTROLLER_BUTTON_A, "SPACE"}))
    assert [d.name for d in opened.devices] == ["keyboard", "current_designs_gamepad3"]
    assert "IGNORED \"Joystick Of Nobody's\"" in opened.status()
    assert opened.controls() == Keyboard.CONTROLS | CurrentDesignsGamepad3.CONTROLS
    with pytest.raises(CalibrationError, match="no calibration of 'Scripted'"):
        inputs.open_input_devices(["current_designs_gamepad3"],
                                  frozenset({ControlNames.CONTROLLER_AXIS_LEFTX_LEFT}))
    assert [d.name for d in inputs.open_input_devices(["current_designs_gamepad3"]).devices] == [
        "current_designs_gamepad3"]                   # no keyboard unless the rig lists one


def test_a_rigs_controller_is_read_through_the_calibration_file_given(sdl, config_home):
    given = _write(config_home, "Scripted_a.json", GMPD3_FILE, 1000)
    _write(config_home, "Scripted_b.json", {"device": "Scripted", "sticks": {}}, 2000)
    opened = inputs.open_input_devices(["current_designs_gamepad3"], frozenset(), [given])
    assert opened.devices[0].calibration_path == given
    other = _write(config_home, "Other_a.json", {**GMPD3_FILE, "device": "Other"}, 1000)
    with pytest.raises(CalibrationError, match=re.escape(f"{other}: a calibration of 'Other'")):
        inputs.open_input_devices(["current_designs_gamepad3"], frozenset(), [given, other])
    with pytest.raises(CalibrationError, match="a second calibration of 'Scripted'"):
        inputs.open_input_devices(["current_designs_gamepad3"], frozenset(), [given, given])


# -- gamepad-calibration's window ---------------------------------------------------------------


@pytest.fixture
def scope(config_home, screen, monkeypatch):
    """The tool's window on the dummy display, with the scripted controller plugged in."""
    from humans.standalone import gamepad_calibration
    monkeypatch.setattr(gamepad_calibration, "first_controller_class_plugged_in",
                        lambda kinds: _ScriptedController)
    monkeypatch.setattr(_ScriptedController, "open",
                        classmethod(lambda cls, calibrated=True: cls(calibrated=calibrated)))
    return gamepad_calibration._Scope(("current_designs_gamepad3",))


def test_the_keyboard_does_nothing_in_the_tool_but_esc(scope):
    path = files.write(files.ControllerCalibration.from_dict(copy.deepcopy(FILE)))
    scope.reload("")
    before = (scope.controller.calibration, scope.axis, sorted(files.calibrations_dir().iterdir()))
    for key in (pygame.K_s, pygame.K_d, pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT,
                pygame.K_RIGHT, pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET, pygame.K_TAB):
        pygame.event.post(_key(pygame.KEYDOWN, key))
    scope.events()
    assert (scope.controller.calibration, scope.axis,
            sorted(files.calibrations_dir().iterdir())) == before and scope.running
    assert scope.controller.calibration_path == path
    pygame.event.post(_key(pygame.KEYDOWN, pygame.K_ESCAPE))
    scope.events()
    assert not scope.running


@pytest.mark.parametrize("action", ["on-", "on+", "off-", "off+"])
def test_each_tuning_button_moves_one_field_within_bounds(action):
    from humans.standalone import gamepad_calibration
    for start in (AxisCalibration.from_dict(AXIS, ""), AxisCalibration(0, -1, 1, 0, 0.99, 0.99),
                  AxisCalibration(0, -1, 1, 0, 0.5, 0.01)):
        moved = gamepad_calibration._retuned(start, action)
        assert sum(a != b for a, b in zip(vars(moved).values(), vars(start).values())) <= 1
        AxisCalibration.from_dict(moved.record(), "")      # what a file allows
    middle = AxisCalibration(0, -1, 1, 0, 0.5, 0.3)
    assert gamepad_calibration._retuned(middle, action) != middle


def test_save_writes_a_new_newest_file_of_the_retuned_axis_and_leaves_its_source(scope):
    from humans.standalone import gamepad_calibration
    source = files.write(files.ControllerCalibration.from_dict(copy.deepcopy(FILE)))
    os.utime(source, (1000, 1000))
    was = source.read_bytes()
    scope.reload("")
    assert not scope.is_enabled("save")
    gamepad_calibration._ACTIONS[f"show-{LY}"](scope)
    gamepad_calibration._ACTIONS["on+"](scope)
    scope.click(next(b.rect.center for b in scope.buttons if b.action == "save"))
    assert source.read_bytes() == was
    path, saved = files.newest("Scripted")
    assert path != source and saved.tuned_from == source.name
    assert (saved.axes[LY].on, saved.axes[LX].on, saved.made) == (0.51, 0.5, files.read(source).made)
    assert scope.controller.calibration_path == path


def test_the_watch_reloads_only_on_its_one_second_tick(scope, monkeypatch):
    reloads = []
    monkeypatch.setattr(scope, "reload", reloads.append)
    start = scope.next_watch
    files.write(files.ControllerCalibration.from_dict(copy.deepcopy(FILE)))   # a calibration added
    scope.watch(start - 0.5)
    assert reloads == []
    scope.watch(start)
    assert reloads == ["a calibration changed: reloaded"] and scope.next_watch == start + 1.0
    scope.draw()                                                     # and it draws


def test_a_click_is_mapped_back_to_the_picture_at_any_window_size(scope):
    from humans.standalone import gamepad_calibration
    canvas = scope.canvas
    canvas.window.size = (900, 1200)                              # bars above and below
    assert canvas.window.size == (900, 1200)
    s = 900 / gamepad_calibration.W
    top = (1200 - int(gamepad_calibration.H * s)) // 2
    assert canvas.to_canvas((int(100 * s), top + int(40 * s))) == pytest.approx((100, 40), abs=2)
    clear = next(b for b in scope.buttons if b.action == "clear")
    scope.log.append((0.0, "a press", None))
    scope.click(canvas.to_canvas((int(clear.rect.centerx * s), top + int(clear.rect.centery * s))))
    assert not scope.log


def test_a_rig_measurement_lets_go_of_its_inputs_before_it_shuts_sdl_down():
    from humans.standalone.rig_setup.measure import RigMeasurement
    closed: list[str] = []
    devices = {name: types.SimpleNamespace(close=lambda name=name: closed.append(name))
               for name in ("inputs", "display", "audio", "triggers")}
    RigMeasurement.close(types.SimpleNamespace(**devices))
    assert closed.index("inputs") < closed.index("display")


# -- a session's recording: each device's own events, read from the OS --------------------------


linux_only = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the Linux reader")


@linux_only
def test_the_linux_reader_writes_each_kernel_record_at_the_kernels_time():
    from humans.standalone.session_logger import linux
    read_end, write_end = os.pipe()
    devices = object.__new__(linux.LinuxDevices)
    devices._devices = [{"name": "Trigger Box", "path": "/dev/input/event7"}]
    devices._fds = {read_end: 0}
    devices._fd_of_node = {"/dev/input/event7": read_end}
    devices._next_look = float("inf")
    os.write(write_end, linux._EVENT.pack(12345, 678901, 1, 13, 1)     # EV_KEY KEY_EQUAL down
             + linux._EVENT.pack(12345, 678901, 0, 0, 0))              # EV_SYN SYN_REPORT
    try:
        lines = devices.read(1.0)
    finally:
        devices.close()
        os.close(write_end)
    event = {"type": "device_event", "perf_time": 12345.678901, "device": 0}
    assert lines == [{**event, "event_type": 1, "code": 13, "value": 1},
                     {**event, "event_type": 0, "code": 0, "value": 0}]


@linux_only
def test_the_linux_reader_opens_every_node_and_stops_at_one_it_cannot_read(monkeypatch):
    from humans.standalone.session_logger import SessionLoggerError, linux
    present = [{"name": "Trigger Box", "path": "/dev/input/event7"},
               {"name": "Keyboard", "path": "/dev/input/event2"}]
    monkeypatch.setattr(linux, "_present_devices", lambda: present)

    def refuse(path, flags):
        raise PermissionError(path)
    monkeypatch.setattr(linux.os, "open", refuse)
    with pytest.raises(SessionLoggerError, match=re.escape(
            "/dev/input/event7 (Trigger Box) cannot be read by this user")):
        linux.LinuxDevices()


@linux_only
def test_the_linux_reader_stops_when_perf_counter_is_not_the_kernels_clock(monkeypatch):
    from humans.standalone.session_logger import SessionLoggerError, linux
    monkeypatch.setattr(linux.time, "get_clock_info",
                        lambda name: types.SimpleNamespace(implementation="something else"))
    with pytest.raises(SessionLoggerError, match="perf_counter reads something else"):
        linux.LinuxDevices()


@pytest.mark.skipif(sys.platform == "win32", reason="the Linux reader's nodes are POSIX files")
def test_the_linux_reader_adds_a_device_plugged_in_and_one_it_cannot_read(monkeypatch, capsys):
    from humans.standalone.session_logger import linux
    present: list[dict] = []
    readable: set[str] = set()
    pipes: list[int] = []
    clock = [100.0]
    monkeypatch.setattr(linux, "_present_devices", lambda: list(present))
    monkeypatch.setattr(linux.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(linux.fcntl, "ioctl", lambda fd, request, argument: 0)
    monkeypatch.setattr(linux.time, "get_clock_info", lambda name: types.SimpleNamespace(
        implementation="clock_gettime(CLOCK_MONOTONIC)"))

    def open_node(path, flags):
        if path not in readable:
            raise PermissionError(path)
        read_end, write_end = os.pipe()
        pipes.extend([read_end, write_end])
        return read_end
    monkeypatch.setattr(linux.os, "open", open_node)
    devices = linux.LinuxDevices()
    try:
        assert devices.record()["devices"] == []
        present.append({"name": "Pad", "path": "/dev/input/event9"})
        readable.add("/dev/input/event9")
        present.append({"name": "Box", "path": "/dev/input/event10"})
        clock[0] += linux.DEVICE_LOOK_S
        added = [line for line in devices.read(0.0) if line["type"] == "device_added"]
        assert [(line["device"], line["name"]) for line in added] == [(0, "Pad")]
        clock[0] += linux.DEVICE_LOOK_S
        assert devices.read(0.0) == [], "a node not readable yet is given time"
        clock[0] += linux.NEW_NODE_GRACE_S
        added = [line for line in devices.read(0.0) if line["type"] == "device_added"]
        assert [(line["device"], line["name"]) for line in added] == [(1, "Box")]
        assert "cannot be read by this user" in added[0]["error"]
        assert "as device 1" in capsys.readouterr().err
        clock[0] += linux.DEVICE_LOOK_S
        assert devices.read(0.0) == [], "a node refused once is not looked at again"
    finally:
        devices.close()
        for fd in pipes[1::2]:
            os.close(fd)


def test_a_rig_naming_device_events_is_refused():
    from humans.config import rig as rig_file
    assert any("unknown key 'device_events'" in problem
               for problem in rig_file.problems({"device_events": ["Trigger Box"]}))


#: Replaces each OS's reader in the recorder's process: devices from ``FAKE_DEVICES`` (JSON),
#: or the error ``FAKE_DEVICES_ERROR`` when it is set.
_FAKE_READERS = """
import importlib, json, os, sys, time
from humans.standalone.session_logger import SessionLoggerError

class FakeDevices:
    def __init__(self):
        if os.environ.get("FAKE_DEVICES_ERROR"):
            raise SessionLoggerError(os.environ["FAKE_DEVICES_ERROR"])

    def record(self):
        return {"reader": "fake", "clock": "perf_counter",
                "devices": json.loads(os.environ.get("FAKE_DEVICES", "[]"))}

    def read(self, timeout):
        time.sleep(timeout)
        return []

    def close(self):
        pass

# This OS's reader alone: another OS's cannot be imported here.
module, reader = {"darwin": ("macos", "MacDevices"), "win32": ("windows", "WindowsDevices")}.get(
    sys.platform, ("linux", "LinuxDevices"))
setattr(importlib.import_module("humans.standalone.session_logger." + module), reader,
        FakeDevices)
"""


@pytest.fixture
def fake_devices(tmp_path, monkeypatch):
    """The recorder's process reads fake devices, whatever this machine has plugged in."""
    folder = tmp_path / "fake_readers"
    folder.mkdir()
    (folder / "sitecustomize.py").write_text(_FAKE_READERS)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        [str(folder), *filter(None, [os.environ.get("PYTHONPATH")])]))
    return monkeypatch


def _rig(tmp_path) -> str:
    """A rig file of team ``lab`` whose data root is ``tmp_path/data``."""
    from humans.config import rig as rig_file
    path = tmp_path / "configs" / "lab" / "rigs" / "desk.json"
    rig_file.save({**{k: "x" for k in rig_file.FIELDS}, "site": "lab", "rig": "desk",
                   "pi": "MT", "modality": "meg", "notes": "",
                   "data_root": str(tmp_path / "data"),
                   "screen": {"geometry": {"width_cm": 30.0, "height_cm": 20.0,
                                           "distance_cm": 60.0}}}, str(path))
    return str(path)


def test_a_session_is_recorded_from_start_to_stop_with_its_marks(tmp_path, fake_devices,
                                                                  capsys):
    import json

    from humans.standalone import session_logger
    fake_devices.setenv("FAKE_DEVICES", json.dumps([{"name": "Pad"}, {"name": "Keyboard"}]))
    recording = session_logger.start_recording(_rig(tmp_path), "sub-01", 2, os.getpid())
    beh = tmp_path / "data" / "sub-01" / "ses-002" / "beh"
    try:
        assert os.path.dirname(recording) == str(beh), "straight into the session's beh/"
        assert os.path.basename(recording).startswith("sub-01_ses-002_session-")
        assert "every input device (2)" in capsys.readouterr().err
        assert session_logger.is_recording(recording)
        session_logger.mark_recording(recording, "run 1: the localizer, été")
    finally:
        assert session_logger.stop_recording(recording)
    assert os.listdir(beh) == [os.path.basename(recording)], "the events file alone is left"
    lines = [json.loads(line) for line in open(recording, encoding="utf-8")]
    first, last = lines[0], lines[-1]
    assert first["type"] == "recording_start" and last["type"] == "recording_end"
    assert first["devices"] == [{"name": "Pad"}, {"name": "Keyboard"}]
    assert first["subject"] == "sub-01" and first["wall_time"] > first["perf_time"]
    assert set(first["sources"]) == {"device_event", "device_added", "window_focus", "stall",
                                     "clock", "mark", "session_ended"}
    assert first["session_pid"] == os.getpid()
    marks = [line for line in lines if line["type"] == "mark"]
    assert [mark["text"] for mark in marks] == ["run 1: the localizer, été"]
    assert last["how"] == "stopped" and last["counts"]["mark"] == 1
    assert last["devices_added"] == [] and last["devices_gone"] == []
    assert not session_logger.is_recording(recording)
    from humans.save import bids
    assert bids.next_session(str(tmp_path / "data"), "sub-01") == 1
    os.makedirs(tmp_path / "data" / "sub-01" / "ses-001")
    assert bids.next_session(str(tmp_path / "data"), "sub-01") == 3, "a recording takes its session"
    assert session_logger.stop_recording(recording), "stopping a stopped recording is harmless"
    session_logger.mark_recording(recording, "too late")
    assert "has stopped, so this mark is not in it" in capsys.readouterr().err
    assert os.listdir(beh) == [os.path.basename(recording)], "a late mark writes nothing"


def test_a_recording_ends_itself_when_its_session_ends_without_stopping_it(tmp_path,
                                                                           fake_devices):
    import subprocess
    import time

    from humans.standalone import session_logger
    session = subprocess.Popen(["sleep", "60"])
    recording = session_logger.start_recording(_rig(tmp_path), "sub-01", 1, session.pid)
    session.kill()                              # killed: no stop will come
    session.wait()
    deadline = time.monotonic() + 10
    while session_logger.is_recording(recording) and time.monotonic() < deadline:
        time.sleep(0.1)
    lines = [json.loads(line) for line in open(recording)]
    assert lines[-2]["type"] == "session_ended" and lines[-2]["pid"] == session.pid
    assert lines[-1]["type"] == "recording_end" and lines[-1]["how"] == "session_ended"
    assert not session_logger.is_recording(recording), "its recorder has exited"
    assert session_logger.stop_recording(recording), "a stop after it is harmless"


def test_a_recording_does_not_start_for_a_session_that_is_not_running(tmp_path, fake_devices):
    import subprocess

    from humans.standalone import session_logger
    session = subprocess.Popen(["true"])
    session.wait()
    with pytest.raises(session_logger.SessionLoggerError, match="is not running"):
        session_logger.start_recording(_rig(tmp_path), "sub-01", 1, session.pid)
    assert not (tmp_path / "data").exists(), "nothing is written"


def test_a_recording_is_never_written_over(tmp_path, fake_devices):
    from humans.standalone import session_logger
    fake_devices.setattr(session_logger.time, "strftime", lambda _: "20261119T120000")
    rig = _rig(tmp_path)
    recording = session_logger.start_recording(rig, "sub-01", 1, os.getpid())
    assert session_logger.stop_recording(recording)
    beh = os.path.dirname(recording)

    def files_in_beh():
        return {name: open(os.path.join(beh, name), "rb").read() for name in os.listdir(beh)}
    before = files_in_beh()
    with pytest.raises(session_logger.SessionLoggerError, match="never written over"):
        session_logger.start_recording(rig, "sub-01", 1, os.getpid())
    assert files_in_beh() == before
    with pytest.raises(session_logger.SessionLoggerError, match="not a recording's events"):
        session_logger.stop_recording(beh)


def test_a_recording_does_not_start_when_a_device_cannot_be_read(tmp_path, fake_devices):
    from humans.standalone import session_logger
    fake_devices.setenv("FAKE_DEVICES_ERROR", "session-logger: the Pad cannot be read")
    with pytest.raises(session_logger.SessionLoggerError, match="the Pad cannot be read"):
        session_logger.start_recording(_rig(tmp_path), "sub-01", 1, os.getpid())
    beh = tmp_path / "data" / "sub-01" / "ses-001" / "beh"
    errors = [name for name in os.listdir(beh) if name.endswith("_errors.log")]
    assert errors and "the Pad cannot be read" in (beh / errors[0]).read_text(), \
        "what went wrong is kept"


def test_a_recording_writes_each_line_as_it_comes_between_a_first_and_a_last(tmp_path):
    import json
    import queue
    import threading

    from humans.standalone.session_logger.recorder import record_session_events
    path = tmp_path / "session_events.jsonl"
    stop = threading.Event()
    side_lines = queue.Queue()
    side_lines.put({"type": "mark", "perf_time": 2.0, "text": "run 1"})
    reads = iter([[{"type": "device_event", "perf_time": 1.0, "device": 0, "event_type": 1,
                    "code": 13, "value": 1}], []])

    class Box:
        closed = False

        def read(self, timeout):
            assert json.loads(path.read_text().splitlines()[0])["type"] == "recording_start", \
                "the first line is on disk before any device is read"
            stop.set()                          # one read, then the session ends
            return next(reads, [])

        def close(self):
            self.closed = True
    box = Box()
    record_session_events(box, str(path), {"type": "recording_start", "perf_time": 0.5},
                          side_lines, stop)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [line["type"] for line in lines] == [
        "recording_start", "device_event", "mark", "recording_end"]
    assert lines[-1]["counts"] == {"device_event": 1, "mark": 1} and box.closed
    with pytest.raises(FileExistsError):
        record_session_events(Box(), str(path), {"type": "recording_start"}, side_lines, stop)


def test_a_recording_lists_the_devices_plugged_in_and_unplugged_and_stop_says_so(tmp_path, capsys):
    import json
    import queue
    import threading

    from humans.standalone import session_logger
    from humans.standalone.session_logger.recorder import record_session_events
    stop = threading.Event()
    reads = iter([[{"type": "device_added", "perf_time": 2.0, "device": 1, "name": "Pad"},
                   {"type": "device_gone", "perf_time": 3.0, "device": 1}]])

    class Box:
        def read(self, timeout):
            stop.set()
            return next(reads, [])

        def close(self):
            pass
    first_line = {"type": "recording_start", "perf_time": 0.5, "wall_time": 1.7e9,
                  "devices": [{"name": "Trigger Box"}], "recorder_pid": os.getpid()}
    recording = tmp_path / "sub-01_ses-001_session-x_events.jsonl"
    record_session_events(Box(), str(recording), first_line, queue.Queue(), stop)
    last = json.loads(recording.read_text().splitlines()[-1])
    assert last["devices_added"] == [{"perf_time": 2.0, "device": 1, "name": "Pad"}]
    assert last["devices_gone"] == [{"device": 1, "perf_time": 3.0}]
    assert session_logger.stop_recording(str(recording))
    assert "'Pad' (device 1) was unplugged at" in capsys.readouterr().err


def test_a_stall_is_a_sleep_that_woke_late(monkeypatch):
    import queue
    import threading
    import time

    from humans.standalone.session_logger import recorder
    side_lines = queue.Queue()
    stop = threading.Event()
    sleeps, real_sleep = iter([0.001, 0.031]), time.sleep

    def sleep(seconds):
        time_slept = next(sleeps, None)
        if time_slept is None:
            stop.set()
            return
        real_sleep(time_slept)
    monkeypatch.setattr(recorder.time, "sleep", sleep)
    recorder.watch_stalls(side_lines, stop)
    stalls = [side_lines.get_nowait() for _ in range(side_lines.qsize())]
    assert [line["type"] for line in stalls] == ["stall"]
    assert 25.0 < stalls[0]["duration_ms"] < 200.0


def test_a_focus_line_is_written_each_time_another_application_has_the_focus(monkeypatch):
    import queue
    import threading

    from humans.standalone.session_logger import FrontApplication, recorder
    kitty, game = FrontApplication("kitty", 10), FrontApplication("python3", 20)
    answers = iter([kitty, kitty, None, game, game])
    side_lines = queue.Queue()
    stop = threading.Event()

    def front_application():
        answer = next(answers, None)
        if answer is None and not stop.is_set() and side_lines.qsize() == 2:
            stop.set()
        return answer
    monkeypatch.setattr(recorder, "FOCUS_POLL_S", 0.0)
    recorder.watch_focus(front_application, side_lines, stop)
    lines = [side_lines.get_nowait() for _ in range(side_lines.qsize())]
    assert [(line["process"], line["pid"]) for line in lines] == [("kitty", 10), ("python3", 20)]


def test_the_linux_front_application_is_the_active_windows_process(monkeypatch):
    from humans.standalone.session_logger import FrontApplication, linux
    answers = {("-root", "_NET_ACTIVE_WINDOW"): "_NET_ACTIVE_WINDOW(WINDOW): window id # 0x4a00007",
               ("-id", "0x4a00007", "_NET_WM_PID"): "_NET_WM_PID(CARDINAL) = 4242"}
    monkeypatch.setattr(linux, "_xprop", lambda *arguments: answers.get(arguments, ""))
    monkeypatch.setattr(linux, "_process_name", lambda pid: "python3")
    monkeypatch.setattr(linux, "_last_front_application", ("", None))
    assert linux.front_application() == FrontApplication("python3", 4242)
    answers[("-root", "_NET_ACTIVE_WINDOW")] = "_NET_ACTIVE_WINDOW(WINDOW): window id # 0x0"
    assert linux.front_application() is None, "no window is active"
    monkeypatch.setattr(linux, "_xprop", lambda *arguments: "")
    assert linux.front_application() is None, "no X display"


@pytest.mark.skipif(sys.platform != "darwin", reason="the macOS reader")
def test_the_macos_front_application_is_what_launchservices_puts_in_front(monkeypatch):
    from humans.standalone.session_logger import FrontApplication, macos
    answers = {("front",): "ASN:0x0-0x77077:\n",
               ("info", "-only", "name,pid", "ASN:0x0-0x77077:"):
                   '"LSDisplayName"="kitty"\n"pid"=1324\n'}
    monkeypatch.setattr(macos, "_lsappinfo", lambda *arguments: answers.get(arguments, ""))
    monkeypatch.setattr(macos, "_last_front_application", ("", None))
    assert macos.front_application() == FrontApplication("kitty", 1324)
    assert macos.front_application() == FrontApplication("kitty", 1324)

def _raw_input(kind, body: bytes) -> bytes:
    """A Windows ``RAWINPUT`` of ``kind`` (0 mouse, 1 keyboard, 2 HID) from device handle 7."""
    import ctypes

    from humans.standalone.session_logger import windows
    header = windows._RawInputHeader(type=kind, device=7, wparam=0)
    header.size = ctypes.sizeof(header) + len(body)
    return bytes(header) + body


def test_a_windows_keys_raw_input_is_a_line_with_its_scancode_and_whether_it_went_down():
    from humans.standalone.session_logger import windows

    def key(make_code, flags):
        return _raw_input(1, bytes(windows._RawKeyboard(make_code=make_code, flags=flags,
                                                        vkey=0x25)))
    keys_down = set()
    lines = [line for data in (key(0x4B, 2), key(0x4B, 2), key(0x4B, 3), key(0x0D, 0))
             for line in windows.lines_of_raw_input(data, 0, 5.0, None, keys_down)]
    assert [(line["make_code"], line["flags"], line["value"]) for line in lines] == [
        (0x4B, 2, 1), (0x4B, 2, 2), (0x4B, 3, 0), (0x0D, 0, 1)], "down, repeated, up, down"
    assert lines[0] == {"type": "device_event", "perf_time": 5.0, "device": 0,
                        "make_code": 0x4B, "flags": 2, "vkey": 0x25, "value": 1}
    assert keys_down == {(0, 0x0D, 0)}


def test_a_windows_mouse_and_hid_raw_input_are_lines_as_windows_gives_them():
    from humans.standalone.session_logger import windows
    mouse = _raw_input(0, bytes(windows._RawMouse(button_flags=1, last_x=-3, last_y=4)))
    [line] = windows.lines_of_raw_input(mouse, 1, 2.0, None, set())
    assert (line["button_flags"], line["last_x"], line["last_y"]) == (1, -3, 4)
    reports = bytes(windows._RawHid(report_size=3, report_count=2)) + bytes([1, 2, 3, 1, 2, 4])
    hid = _raw_input(2, reports)
    assert [line["report"] for line in windows.lines_of_raw_input(hid, 2, 3.0, None, set())] \
        == ["010203", "010204"], "a report no parser reads is kept whole"

    class Buttons:
        def lines(self, report, device, arrived):
            return [{"type": "device_event", "perf_time": arrived, "device": device,
                     "usage_page": 9, "usage": report[2], "value": 1}]
    assert [line["usage"] for line in windows.lines_of_raw_input(hid, 2, 3.0, Buttons(), set())] \
        == [3, 4]


@pytest.mark.parametrize("make_code, flags, name", [
    (0x0D, 0, "EQUAL"), (0x1E, 0, "A"), (0x4B, 2, "LEFT"), (0x4B, 0, "KP4"),
    (0x1D, 4, "PAUSE"), (0x64, 0, "F13"), (0x60, 0, None), (0x2A, 2, None),
])
def test_a_windows_key_is_named_by_its_evdev_code(make_code, flags, name):
    from humans.standalone.key_codes import EVDEV_KEYS, evdev_code_of_windows_key
    code = evdev_code_of_windows_key(make_code, flags)
    assert (None if code is None else EVDEV_KEYS[code][1]) == name

def test_the_control_file_passes_on_each_line_once_and_whole(tmp_path):
    import json
    import queue
    import threading

    from humans.standalone.session_logger.recorder import follow_control_file
    path = tmp_path / "control.jsonl"
    side_lines = queue.Queue()
    stop = threading.Event()
    follower = threading.Thread(target=follow_control_file, args=(str(path), side_lines, stop))
    follower.start()
    mark = json.dumps({"type": "mark", "perf_time": 1.0, "text": "café"}).encode()
    with open(path, "ab") as control:
        control.write(mark[:20])               # a line still being written
        control.flush()
        stop.wait(0.2)
        control.write(mark[20:] + b"\n" + json.dumps({"type": "stop"}).encode() + b"\n")
    follower.join(timeout=5)
    assert stop.is_set() and not follower.is_alive()
    assert [side_lines.get_nowait()["text"]] == ["café"] and side_lines.empty()
