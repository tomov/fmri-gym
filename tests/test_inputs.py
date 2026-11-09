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
import types
from datetime import datetime, timezone

import pygame
import pytest

import fmri_gym.inputs as inputs
from fmri_gym.standalone import gamepad_calibration as cal
from fmri_gym.inputs import (QUIT, CalibrationError, ControlNames, CurrentDesignsGamepad3, Edge,
                             InputDevices, InputEvent, Keyboard, XBoxController)
from fmri_gym.inputs import calibration as files
from fmri_gym.inputs.calibration import AxisCalibration
from fmri_gym.inputs.controller import AXIS_CONTROLS, Controller

LX, LY = ControlNames.CONTROLLER_AXIS_LEFTX, ControlNames.CONTROLLER_AXIS_LEFTY
RX, RY = ControlNames.CONTROLLER_AXIS_RIGHTX, ControlNames.CONTROLLER_AXIS_RIGHTY
MADE = "2026-01-21T14:15:00+00:00"
AXIS = {"rest": 0.0, "low": -1.0, "high": 1.0, "drift": 0.01, "on": 0.5, "off": 0.3}
#: A calibration of the scripted controller's four axes, and of the GMPD3's two.
FILE = {"device": "Scripted", "tuned_from": None, "made": MADE,
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


def test_a_malformed_newest_calibration_stops_the_run_with_no_fallback(config_home):
    _write(config_home, "Scripted_a.json", FILE, 1000)
    bad = _write(config_home, "Scripted_b.json", _set(FILE, f"axes.{LX}.on", "0.5"), 2000)
    with pytest.raises(CalibrationError, match=re.escape(f"{bad}: axes.{LX}.on")):
        files.newest("Scripted")
    bad.write_text("{not json")                    # names no device: it may be this one's
    with pytest.raises(CalibrationError, match="not JSON"):
        files.newest("Scripted")


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
        out = [e for b in self.BUTTON_NAMES for e in self._press(b, b in buttons)]
        return out + [e for name in self.AXIS_NAMES for e in self._move_axis(name, axes[name])]

    @classmethod
    def axes_at(cls, moved: dict[str, float]) -> dict[str, float]:
        return {name: moved.get(name, 0.0) for name in cls.AXIS_NAMES}


#: Where the scripted controller rests, cold and then warm.
COLD, REST = {LX: 0.02}, {LX: -0.05, LY: 0.01}


def _frames(d: _ScriptedController, wrong: str | None = None) -> list:
    """Hands off; each axis to each stop and back to the first; let go; each axis control
    twice (``wrong`` pushed once first); each button twice."""
    still = [(d.axes_at(COLD), set())] * 70
    sweep = ([-k / 10 for k in range(1, 11)] + [k / 10 for k in range(-9, 11)]
             + [k / 10 for k in range(9, -11, -1)])
    stops = [(d.axes_at({**COLD, name: v}), set()) for name in d.AXIS_NAMES for v in sweep]
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
    return pygame.Surface((900, 600))


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
    from fmri_gym.standalone.rig_setup.report import _pad_line
    line = _pad_line({"present": True, "device": "Scripted", **result.record()})
    assert f"{LX} on past 0.50" in line and "10 controls confirmed, 1 wrong" in line


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


def test_an_uncalibrated_axis_presses_nothing():
    d = _ScriptedController()
    d.frames = iter([(d.axes_at({LX: 1.0, LY: -1.0}), set())])
    assert d.handle([], 1.0) == [] and d.axis_values[LX] == 1.0
    assert d.calibrated_value(LX) is None


def test_a_controller_refuses_an_axis_the_curriculum_names_with_no_calibration(config_home):
    pad = _bare(CurrentDesignsGamepad3)
    with pytest.raises(CalibrationError,
                       match="no calibration of 'Scripted' in .*gamepad-calibration"):
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
    let_go = [_joy(pygame.CONTROLLERAXISMOTION, axis=trigger, value=0)]
    assert pad.handle(let_go, 1.3) == [_up(ControlNames.CONTROLLER_AXIS_TRIGGERLEFT)]


def test_an_xbox_controller_reads_each_stick_axis_through_its_calibration():
    pad = _calibrated(_bare(XBoxController), FILE)
    right_x = pygame.CONTROLLER_AXIS_RIGHTX
    assert pad.handle([_joy(pygame.CONTROLLERAXISMOTION, axis=right_x, value=-20000)], 1.0) == [
        _down(ControlNames.CONTROLLER_AXIS_RIGHTX_LEFT)]
    assert pad.calibrated_value(RX) == pytest.approx(-20000 / 32767)


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
                                       frozenset({ControlNames.CONTROLLER_BUTTON_A, "SPACE"}))
    assert [d.name for d in opened.devices] == ["keyboard", "current_designs_gamepad3"]
    assert "IGNORED \"Joystick Of Nobody's\"" in opened.status()
    assert opened.controls() == Keyboard.CONTROLS | CurrentDesignsGamepad3.CONTROLS
    with pytest.raises(CalibrationError, match="no calibration of 'Scripted'"):
        inputs.open_input_devices(["current_designs_gamepad3"],
                                  frozenset({ControlNames.CONTROLLER_AXIS_LEFTX_LEFT}))
    assert [d.name for d in inputs.open_input_devices(["current_designs_gamepad3"]).devices] == [
        "current_designs_gamepad3"]                   # no keyboard unless the rig lists one


# -- gamepad-calibration's window ---------------------------------------------------------------


@pytest.fixture
def scope(config_home, screen, monkeypatch):
    """The tool's window on the dummy display, with the scripted controller plugged in."""
    from fmri_gym.standalone import gamepad_calibration
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
    from fmri_gym.standalone import gamepad_calibration
    for start in (AxisCalibration.from_dict(AXIS, ""), AxisCalibration(0, -1, 1, 0, 0.99, 0.99),
                  AxisCalibration(0, -1, 1, 0, 0.5, 0.01)):
        moved = gamepad_calibration._retuned(start, action)
        assert sum(a != b for a, b in zip(vars(moved).values(), vars(start).values())) <= 1
        AxisCalibration.from_dict(moved.record(), "")      # what a file allows
    middle = AxisCalibration(0, -1, 1, 0, 0.5, 0.3)
    assert gamepad_calibration._retuned(middle, action) != middle


def test_save_writes_a_new_newest_file_of_the_retuned_axis_and_leaves_its_source(scope):
    from fmri_gym.standalone import gamepad_calibration
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
    from fmri_gym.standalone import gamepad_calibration
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
