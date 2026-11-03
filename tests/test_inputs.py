"""The input devices with none plugged in: their events, their sticks, their calibration.

A scripted device answers the calibration frame by frame, on a fake clock, so a
whole calibration plays in well under a second and writes into a temporary
folder. The controllers are fed the events SDL would send them.
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
from fmri_gym.inputs import (QUIT, CalibrationError, Edge, InputDevices, InputEvent, Joystick,
                             Keyboard)
from fmri_gym.inputs import calibration as files
from fmri_gym.inputs.calibration import Side, StickTuning, UNCALIBRATED
from fmri_gym.inputs.gamepad import Gamepad
from fmri_gym.inputs.gmpd3 import Gmpd3
from fmri_gym.inputs.stick import Stick, directions_of

MADE = "2026-01-21T14:15:00+00:00"
AXIS = {"rest": 0.0, "low": -1.0, "high": 1.0, "drift": 0.01}
STICK_ENTRY = {"made": MADE, "x": AXIS, "y": AXIS, "on": 0.5, "off": 0.3, "diagonal": 1.0,
               "dwell_s": 0.05}
FILE = {"device": "Scripted", "tuned_from": None,
        "sticks": {"left": STICK_ENTRY, "right": {**STICK_ENTRY, "on": 0.6}}}


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
    loaded = files.Calibration.from_dict(copy.deepcopy(FILE))
    assert loaded.record() == FILE
    assert loaded.sticks[Side.RIGHT].tuning.on == 0.6
    assert loaded.sticks[Side.LEFT].made == datetime(2026, 1, 21, 14, 15, tzinfo=timezone.utc)


@pytest.mark.parametrize("field, value, says", [
    ("sticks.left.on", "0.5", "sticks.left.on: expected a number in (0, 1), got '0.5'"),
    ("sticks.left.off", 0.6, "sticks.left.off: expected a number in (0, on=0.5], got 0.6"),
    ("sticks.left.diagonal", 0, "sticks.left.diagonal: expected a number in (0, 1], got 0"),
    ("sticks.left.dwell_s", 0.5, "sticks.left.dwell_s: expected a number of seconds in [0, 0.3]"),
    ("sticks.left.x.drift", True, "sticks.left.x.drift: expected a number >= 0, got True"),
    ("sticks.left.x.low", 0.5, "sticks.left.x: expected low < rest < high"),
    ("sticks.left.x.noise", 0.001, "sticks.left.x.noise: unknown key"),
    ("sticks.left.made", "2026-10-21T14:15:00", "sticks.left.made: expected a time such as"),
    ("sticks.left.y", ..., "sticks.left.y: missing"),
    ("sticks.middle", STICK_ENTRY, "sticks.middle: unknown stick (expected left, right)"),
    ("device", ..., "device: missing"),
    ("tuned_from", 3, "tuned_from: expected null or a file name, got 3"),
    ("sectors", {}, "sectors: unknown key"),
])
def test_each_wrong_field_is_refused_by_its_file_and_name(tmp_path, field, value, says):
    path = tmp_path / "Scripted_20261021-141500.json"
    path.write_text(json.dumps(_set(FILE, field, value)))
    with pytest.raises(CalibrationError, match=re.escape(f"{path}: {says}")):
        files.read(path)


def test_a_device_is_read_through_its_newest_calibration(config_home):
    _write(config_home, "Scripted_a.json", _set(FILE, "sticks.left.on", 0.31), 1000)
    newest = _write(config_home, "Scripted_b.json", _set(FILE, "sticks.left.on", 0.33), 3000)
    _write(config_home, "Scripted_c.json", _set(FILE, "sticks.left.on", 0.32), 2000)
    _write(config_home, "Other_d.json", {**FILE, "device": "Other"}, 4000)   # newer, another pad
    _write(config_home, "Other_e.json", {"device": "Other", "axes": {}}, 5000)  # bad, not ours
    path, found = files.newest("Scripted")
    assert (path, found.sticks[Side.LEFT].tuning.on) == (newest, 0.33)
    assert files.newest("Nobody") is None


def test_a_malformed_newest_calibration_stops_the_run_with_no_fallback(config_home):
    _write(config_home, "Scripted_a.json", FILE, 1000)
    bad = _write(config_home, "Scripted_b.json", _set(FILE, "sticks.left.on", "0.5"), 2000)
    with pytest.raises(CalibrationError, match=re.escape(f"{bad}: sticks.left.on")):
        files.newest("Scripted")
    bad.write_text("{not json")                    # names no device: it may be this one's
    with pytest.raises(CalibrationError, match="not JSON"):
        files.newest("Scripted")


# -- a device with no SDL behind it ---------------------------------------------------------


class _FakeHandle:
    """What a device asks of SDL's joystick handle: that it is open."""

    def get_init(self) -> bool:
        return True


def _bare(cls: type[Joystick], name: str = "Scripted", calibration=None) -> Joystick:
    """A device of ``cls`` as if SDL had opened it as joystick 0, with events of id 7."""
    device = cls.__new__(cls)
    device.index, device.instance_id, device.joystick = 0, 7, _FakeHandle()
    device.device_name = name
    device._buttons, device._warned, device._said_gone = set(), set(), False
    device.use(calibration)
    return device


class _ScriptedDevice(Joystick):
    """A two-stick, two-button device whose readings are a script, one frame each; then ESC."""

    name = "scripted"
    STICKS = {"S_LEFT": Side.LEFT, "S_RIGHT": Side.RIGHT}
    CONTROLS = frozenset({"S_A", "S_B", *directions_of("S_LEFT"), *directions_of("S_RIGHT")})

    def __init__(self, index: int = 0, calibration=None, frames=()) -> None:
        self.index, self.instance_id, self.joystick = index, 7, _FakeHandle()
        self.device_name, self.frames = "Scripted", iter(frames)
        self._buttons, self._warned, self._said_gone = set(), set(), False
        self.use(calibration)

    @classmethod
    def accepts(cls, index: int) -> bool:
        return True

    def _event(self, event, moved):
        return []

    def buttons(self) -> tuple[str, ...]:
        return ("S_A", "S_B")

    def read(self):
        frame = next(self.frames, None)
        if frame is None:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
            return self.axes_at({}), set()
        return frame

    def axes_at(self, moved: dict[int, float]) -> dict[int, float]:
        return {a: moved.get(a, 0.0) for s in self.sticks for a in s.axes}


def _stick_frames(d: _ScriptedDevice, side: Side, wrong: str | None = None) -> list:
    """Hands off, roll, let go, then each direction twice (``wrong`` pushed once first)."""
    x, y = side.axes
    still = [(d.axes_at({x: 0.02}), set())] * 70
    circle = [(d.axes_at({x: math.cos(2 * math.pi * k / 48), y: math.sin(2 * math.pi * k / 48)}),
               set()) for k in range(48 * 3)]
    rest = [(d.axes_at({x: -0.05, y: 0.01}), set())] * 70
    out = still + circle + rest
    for b in ([wrong] if wrong else []) + [b for b in cal.CUE_DIRECTIONS for _ in range(2)]:
        axis, sign = cal.DIRECTIONS[b]
        out += [rest[0], (d.axes_at({x: -0.05, y: 0.01, side.axes[axis]: 0.9 * sign}), set())]
    return out


def _button_frames(d: _ScriptedDevice) -> list:
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
    d = _ScriptedDevice()
    d.frames = iter(_stick_frames(d, Side.LEFT, wrong="LEFT") + _stick_frames(d, Side.RIGHT)
                    + _button_frames(d))
    result = cal.calibrate(d, screen, lambda: None, tuple(d.sticks))
    assert (result.confirmed, result.corrections) == (4 + 4 + 2, 1)
    left = result.calibration.sticks[Side.LEFT]
    assert (left.x.rest, left.x.low, left.x.high, left.x.drift) == (-0.05, -1.0, 1.0, 0.07)
    assert (left.tuning.on, left.tuning.off) == (cal.CAL_ON, cal.CAL_OFF)
    assert set(result.calibration.sticks) == {Side.LEFT, Side.RIGHT}
    assert files.newest("Scripted") == (result.path, result.calibration)
    assert result.path.with_suffix(".csv").exists()
    from fmri_gym.checks import _pad_line
    line = _pad_line({"present": True, **result.record()})
    assert "left stick on past 0.50" in line and "10 controls confirmed, 1 wrong" in line


def test_one_stick_calibrated_alone_keeps_the_other_as_it_was_measured(config_home, screen):
    old = files.Calibration.from_dict(_set(FILE, "sticks.left.on", 0.4))
    base = files.write(old)
    d = _ScriptedDevice()
    d.frames = iter(_stick_frames(d, Side.RIGHT))          # no buttons: not every stick
    result = cal.calibrate(d, screen, lambda: None, (Side.RIGHT,))
    assert result.path != base and base.read_text() == json.dumps(old.record(), indent=2) + "\n"
    assert result.calibration.sticks[Side.LEFT] == old.sticks[Side.LEFT]   # its made too
    assert result.calibration.sticks[Side.RIGHT].made > old.sticks[Side.RIGHT].made
    assert result.confirmed == 4


def test_a_calibration_stopped_part_way_writes_nothing(config_home, screen):
    d = _ScriptedDevice()
    d.frames = iter(_stick_frames(d, Side.LEFT)[:100])
    assert cal.calibrate(d, screen, lambda: None, (Side.LEFT,)) is None
    assert not config_home.exists()


def test_the_fit_raises_the_threshold_of_a_stick_that_dithers():
    rows = ([{"t": k / 60, "phase": "cold", "cue": "left", "ax0": 0.0, "ax1": 0.0}
             for k in range(60)]
            + [{"t": 1 + k / 60, "phase": "roll", "cue": "left", "ax0": v, "ax1": -v}
               for k, v in enumerate((-0.5, 1.0, 0.2))]
            + [{"t": 2 + k / 60, "phase": "rest", "cue": "left", "ax0": 0.1 + 0.02 * (k % 2),
                "ax1": 0.0} for k in range(60)])
    stick = cal.fit(rows, datetime.now(timezone.utc))[Side.LEFT]
    assert (stick.x.rest, stick.x.low, stick.x.high, stick.x.drift) == (0.11, -0.5, 1.0, 0.11)
    # dither 0.01 of the 0.61 travel to the low side: ten times it is 0.164, under 0.5
    assert (stick.tuning.on, stick.tuning.off) == (0.5, 0.3)
    shaky = [{**r, "ax0": 0.1 + 0.1 * (k % 2)} if r["phase"] == "rest" else r
             for k, r in enumerate(rows)]
    tuning = cal.fit(shaky, datetime.now(timezone.utc))[Side.LEFT].tuning
    assert tuning.on > cal.CAL_ON and tuning.off == round(0.6 * tuning.on, 3)


def test_a_check_controls_phase_refuses_pad_reps():
    from fmri_gym.config.phases import CheckControls
    assert any("pad_reps" in p for p in CheckControls.problems({"pad_reps": 3}))


# -- the sticks and the devices ------------------------------------------------------------


def test_each_stick_reads_its_own_tuning():
    blob = _set(_set(FILE, "sticks.left.dwell_s", 0), "sticks.right.dwell_s", 0)
    found = files.Calibration.from_dict(blob)
    left, right = (Stick(n, s, found.sticks[s]) for n, s in (("L", Side.LEFT), ("R", Side.RIGHT)))
    for stick in (left, right):
        stick.move(0, 0.55)                       # past the left's 0.5, not the right's 0.6
        stick.move(1, 0.0)
    assert left.resolve(1.0) == [_down("L_RIGHT")] and right.resolve(1.0) == []
    right.move(0, 0.65)
    assert right.resolve(1.1) == [_down("R_RIGHT")]
    assert left.last_held() == {"L_RIGHT"} and right.last_held() == {"R_RIGHT"}


def test_a_device_refuses_a_stick_the_curriculum_names_with_no_calibration(config_home):
    pad = _bare(Gmpd3)
    with pytest.raises(CalibrationError, match="no calibration of 'Scripted' in .*rig check"):
        pad.require(frozenset({"GMPD3_STICK_UP", "GMPD3_A"}))
    pad.require(frozenset({"GMPD3_A", "SPACE"}))            # buttons only: it starts
    assert "GMPD3_STICK UNCALIBRATED" in pad.status()
    assert pad.sticks[Side.LEFT].tuning == UNCALIBRATED
    path = files.write(files.Calibration.from_dict(_set(FILE, "sticks.left", ...)))
    pad.use(files.newest("Scripted"))
    with pytest.raises(CalibrationError, match="has no left stick"):
        pad.require(frozenset({"GMPD3_STICK_UP"}))
    pad.use((path, files.Calibration.from_dict(FILE)))
    pad.require(frozenset({"GMPD3_STICK_UP"}))
    assert str(path) in pad.status()


def _down(name: str) -> InputEvent:
    return InputEvent(name, Edge.DOWN)


def _up(name: str) -> InputEvent:
    return InputEvent(name, Edge.UP)


def _joy(type_: int, **fields) -> pygame.event.Event:
    return pygame.event.Event(type_, instance_id=7, **fields)


def test_the_gmpd3_names_its_buttons_and_reads_its_stick(capsys):
    pad = _bare(Gmpd3, calibration=(None, files.Calibration.from_dict(
        _set(FILE, "sticks.left.dwell_s", 0))))
    events = [_joy(pygame.JOYBUTTONDOWN, button=4), _joy(pygame.JOYBUTTONDOWN, button=9),
              pygame.event.Event(pygame.JOYBUTTONDOWN, instance_id=8, button=0),  # another's
              _joy(pygame.JOYAXISMOTION, axis=1, value=-0.9),
              _joy(pygame.JOYAXISMOTION, axis=3, value=-1.0)]            # pinned: no control
    assert pad.handle(events, 1.0) == [_down("GMPD3_A"), _down("GMPD3_STICK_UP")]
    assert pad.last_held() == {"GMPD3_A", "GMPD3_STICK_UP"}
    assert "button 9 pressed" in capsys.readouterr().err
    assert pad.handle([_joy(pygame.JOYBUTTONUP, button=4)], 1.1) == [_up("GMPD3_A")]


def test_a_gamepad_reads_its_mapping_and_not_the_joystick_twin():
    pad = _bare(Gamepad)
    a, trigger = pygame.CONTROLLER_BUTTON_A, pygame.CONTROLLER_AXIS_TRIGGERLEFT
    events = [_joy(pygame.CONTROLLERBUTTONDOWN, button=a), _joy(pygame.JOYBUTTONDOWN, button=0),
              _joy(pygame.CONTROLLERAXISMOTION, axis=trigger, value=12000)]   # under half
    assert pad.handle(events, 1.0) == [_down("CONTROLLER_BUTTON_A")]
    pull = [_joy(pygame.CONTROLLERAXISMOTION, axis=trigger, value=20000)]
    assert pad.handle(pull, 1.1) == [_down("CONTROLLER_TRIGGERLEFT")]
    ease = [_joy(pygame.CONTROLLERAXISMOTION, axis=trigger, value=12000)]   # still past off
    assert pad.handle(ease, 1.2) == []
    let_go = [_joy(pygame.CONTROLLERAXISMOTION, axis=trigger, value=0)]
    assert pad.handle(let_go, 1.3) == [_up("CONTROLLER_TRIGGERLEFT")]


# -- the keyboard and the run's inputs ---------------------------------------------------------


def _key(type_: int, key: int) -> pygame.event.Event:
    return pygame.event.Event(type_, key=key, mod=0, unicode="")


def test_the_keyboard_holds_what_its_events_press_until_the_window_loses_focus():
    keys = Keyboard()
    assert keys.handle([_key(pygame.KEYDOWN, pygame.K_EQUALS),
                        _key(pygame.KEYDOWN, pygame.K_LEFT)], 1.0) == [
        _down("EQUALS"), _down("LEFT")]
    assert keys.last_held() == {"EQUALS", "LEFT"}
    keys.handle([_key(pygame.KEYUP, pygame.K_EQUALS)], 1.1)
    assert keys.last_held() == {"LEFT"}
    keys.handle([pygame.event.Event(pygame.WINDOWFOCUSLOST)], 1.2)
    assert keys.last_held() == frozenset()


def test_the_runs_inputs_take_the_queue_and_name_the_windows_close(screen):
    pygame.event.post(_key(pygame.KEYDOWN, pygame.K_SPACE))
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    assert InputDevices([Keyboard()]).poll() == [_down(QUIT), _down("SPACE")]
    assert pygame.event.peek() is False


@pytest.fixture
def plugged(monkeypatch):
    """Two joysticks SDL sees, a GMPD3 at 0 and something else at 1; no SDL touched."""
    names = ["GMPD3", "Joystick Of Nobody's"]
    monkeypatch.setattr(inputs, "_start", lambda listed: None)
    monkeypatch.setattr(pygame.joystick, "get_count", lambda: len(names))
    monkeypatch.setattr(pygame.joystick, "Joystick",
                        lambda i: types.SimpleNamespace(get_name=lambda: names[i]))
    monkeypatch.setattr(Gmpd3, "accepts", classmethod(lambda cls, i: i == 0))
    monkeypatch.setattr(Gamepad, "accepts", classmethod(lambda cls, i: False))
    monkeypatch.setattr(inputs, "open_device",
                        lambda cls, i: _bare(cls, names[i]))


@pytest.mark.parametrize("names, says", [
    (["keyboard"], "expected distinct names"),
    (["gmpd3", "gmpd3"], "expected distinct names"),
    (["joypad"], "expected distinct names"),
    (["gamepad"], 'lists "gamepad" under "inputs", but none is plugged in'),
])
def test_a_rigs_inputs_that_cannot_be_opened_stop_the_run(plugged, names, says):
    with pytest.raises(ValueError, match=re.escape(says)):
        inputs.open_inputs(names)


def test_a_rigs_inputs_open_its_devices_and_say_which_joysticks_they_ignore(plugged):
    opened = inputs.open_inputs(["gmpd3"], needed=frozenset({"GMPD3_A", "SPACE"}))
    assert [d.name for d in opened.devices()] == ["keyboard", "gmpd3"]
    assert "IGNORED \"Joystick Of Nobody's\"" in opened.status()
    assert opened.controls() == Keyboard.CONTROLS | Gmpd3.CONTROLS
    with pytest.raises(CalibrationError, match="GMPD3_STICK"):
        inputs.open_inputs(["gmpd3"], needed=frozenset({"GMPD3_STICK_LEFT"}))


# -- pad-calibration's window -------------------------------------------------------------------


@pytest.fixture
def scope(config_home, screen, monkeypatch):
    """The tool's window on the dummy display, with the scripted device plugged in."""
    from fmri_gym import pad_scope
    monkeypatch.setattr(pad_scope, "find_device", lambda kinds: (_ScriptedDevice, 0))
    return pad_scope._Scope(("gmpd3",))


def test_the_keyboard_does_nothing_in_the_tool_but_esc(scope):
    path = files.write(files.Calibration.from_dict(copy.deepcopy(FILE)))
    scope.reload("")
    stick = scope.device.sticks[Side.LEFT]
    before = (stick.tuning, scope.stick, sorted(files.calibrations_dir().iterdir()))
    for key in (pygame.K_s, pygame.K_d, pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT,
                pygame.K_RIGHT, pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET, pygame.K_TAB):
        pygame.event.post(_key(pygame.KEYDOWN, key))
    scope.events()
    assert (scope.device.sticks[Side.LEFT].tuning, scope.stick,
            sorted(files.calibrations_dir().iterdir())) == before and scope.running
    assert scope.device.calibration[0] == path
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


def test_save_writes_a_new_newest_file_and_leaves_its_source(scope):
    from fmri_gym import pad_scope
    source = files.write(files.Calibration.from_dict(copy.deepcopy(FILE)))
    os.utime(source, (1000, 1000))
    was = source.read_bytes()
    scope.reload("")
    pad_scope._ACTIONS["on+"](scope)
    scope.click(next(b.rect.center for b in scope.buttons if b.action == "save"))
    assert source.read_bytes() == was
    path, saved = files.newest("Scripted")
    assert path != source and saved.tuned_from == source.name
    assert saved.sticks[Side.LEFT].tuning.on == 0.51
    assert saved.sticks[Side.LEFT].made == files.read(source).sticks[Side.LEFT].made
    assert scope.device.calibration[0] == path


def test_the_watch_reloads_only_on_its_one_second_tick(scope, monkeypatch):
    reloads = []
    monkeypatch.setattr(scope, "reload", reloads.append)
    start = scope.next_watch
    files.write(files.Calibration.from_dict(copy.deepcopy(FILE)))   # a calibration added
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
