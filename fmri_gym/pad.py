"""The controller: a gamepad or button box whose buttons become keypresses.

Some sites' response device is a game controller on an fMRI interface (a
Current Designs 932, say) that reports HID *joystick* buttons rather than
keystrokes. Nothing a phase's ``keys`` names is ever pressed there, so the game
does not move and the run records no responses, with nothing on screen to say
why. This module presses the keyboard key each control is bound to, so the run
sees a keyboard either way and a curriculum's ``keys`` stay in keyboard keys,
playable at the desk and in the scanner alike. :func:`pump` moves the events
across; it registers with :func:`fmri_gym.keys.get_events`, which runs it
before every read of the queue.

The controls are a gamepad's (:data:`CONTROLS`: each stick's and the D-pad's
four directions, Y B X A, LT RT, LB RB, START BACK), and a pad has the ones it
has. Which raw button or axis is which control is its :class:`Layout`: SDL's,
for a gamepad SDL knows (an Xbox controller, whose raw numbers differ by OS and
connection), or the Current Designs pad's for any other joystick. Which key each
control presses is the rig's (its file's ``"bindings"``, handed to :func:`init`),
one map for every game played on the rig. A stick is read through the newest
calibration of the pad plugged in: one file per measurement in
:func:`calibrations_dir`, outside the checkout since it describes this machine's
controller (:func:`newest`), made by :mod:`fmri_gym.pad_calibration`.

It turns itself on when a pad is plugged in, and a rig file's ``"has_pad": false`` leaves it off.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import pygame
from pygame._sdl2 import controller

from .keys import KEYCODE_BY_NAME, register_event_source, register_held_source


#: The directions of the left stick, the right stick and the D-pad, each up, down, left, right.
STICK = ("UP", "DOWN", "LEFT", "RIGHT")
RIGHT_STICK = ("RS_UP", "RS_DOWN", "RS_LEFT", "RS_RIGHT")
DPAD = ("DPAD_UP", "DPAD_DOWN", "DPAD_LEFT", "DPAD_RIGHT")
#: Every control a rig file's ``"bindings"`` can bind to a key; a pad has the ones it has.
CONTROLS = (*STICK, "Y", "B", "X", "LT", "A", "RT", "LB", "RB", "START", "BACK",
            *DPAD, *RIGHT_STICK)
STICK_DIRECTIONS = frozenset(STICK + RIGHT_STICK)
#: What a participant is told to press for a control: a button by its name, a
#: direction by what moves.
CONTROL_LABELS = {**{c: c for c in CONTROLS}, **{d: f"stick {d.lower()}" for d in STICK},
                  **{d: f"right stick {d[3:].lower()}" for d in RIGHT_STICK},
                  **{d: f"D-pad {d[5:].lower()}" for d in DPAD}}
_KEY_TAG = re.compile(r"\[([A-Z0-9_]+)\]")
#: Where each stick direction points, as an angle of push (y grows downwards).
_ANGLES: dict[str, float] = {
    **dict(zip(STICK, (-math.pi / 2, math.pi / 2, math.pi, 0.0))),
    **dict(zip(RIGHT_STICK, (-math.pi / 2, math.pi / 2, math.pi, 0.0))),
}
#: How much nearer another direction must be before the stick changes hands.
_SECTOR_KEEP = math.radians(15)
_DOMINANCE = 1.15           # with diagonals, the margin for one axis to take the stick over
#: The stick's centre is taken once, after it has first been used and then let
#: go: cold and warm are not the same place (the gimbal does not return where it
#: started) and the warm one moves between sessions. How far it may be taken
#: from the calibrated centre is what the calibration measured that movement to
#: be, so a stick held slightly off can never be mistaken for centre.
_SETTLE_S, _SETTLE_STILL, _SETTLE_SLACK = 1.0, 0.01, 1.5
#: How far an analog trigger is pulled to press, and let go to release, of its travel.
_TRIGGER_ON, _TRIGGER_OFF = 0.5, 0.3


class Stick(Enum):
    """A gamepad's sticks, by the name a calibration file gives each."""

    LEFT = "left"
    RIGHT = "right"

    @property
    def axes(self) -> tuple[int, int]:
        """Its x and y, by SDL's numbering of a gamepad's axes (up and left negative)."""
        return (0, 1) if self is Stick.LEFT else (2, 3)

    @property
    def directions(self) -> tuple[str, ...]:
        """Its controls: up, down, left, right."""
        return STICK if self is Stick.LEFT else RIGHT_STICK


#: What to do about a stick with no calibration, for the messages that say so.
HOW_TO_CALIBRATE = ("run pad-calibration (or Calibrate controller... in rig-config), or the "
                    "rig check")
#: The longest a direction may be made to wait before it is pressed.
MAX_DWELL_S = 0.3

class CalibrationError(ValueError):
    """A calibration file that is not one, or a pad a run needs calibrated that is not."""


@dataclass(frozen=True)
class AxisCalibration:
    """One stick axis, raw: where it rests, its two stops, and how far its centre moves."""

    rest: float
    low: float
    high: float
    drift: float

    @classmethod
    def from_dict(cls, d: Any, where: str) -> AxisCalibration:
        """An axis as a calibration file writes it.

        :raises CalibrationError: naming the field that is missing, unknown or out of range.
        """
        d = _checked(d, where, ("rest", "low", "high", "drift"))
        rest, low, high = (_number(d, k, where, math.isfinite, "a number")
                           for k in ("rest", "low", "high"))
        drift = _number(d, "drift", where, lambda v: 0 <= v < math.inf, "a number >= 0")
        if not low < rest < high:
            raise CalibrationError(f"{where}: expected low < rest < high, got low={low:g}, "
                                   f"rest={rest:g}, high={high:g}")
        return cls(rest, low, high, drift)

    def record(self) -> dict[str, float]:
        """The axis as a calibration file writes it."""
        return {"rest": self.rest, "low": self.low, "high": self.high, "drift": self.drift}


@dataclass(frozen=True)
class StickTuning:
    """How a stick's push is read, in fractions of its travel.

    It presses a direction past ``on`` and releases it under ``off``. ``diagonal``
    is how far a second direction must go, as a fraction of the first, to be
    pressed too: at 1.0 never, so a push is exactly one direction. A new direction
    is pressed once it has held for ``dwell_s``: a thumb leaving rest crosses a
    neighbouring direction for a frame or two (17 ms, against 254 ms for the
    shortest real push measured).
    """

    on: float
    off: float
    diagonal: float
    dwell_s: float

    @classmethod
    def from_dict(cls, d: dict, where: str) -> StickTuning:
        """The tuning in a stick's entry of a calibration file: its on, off, diagonal and dwell_s.

        :raises CalibrationError: naming the field that is out of range.
        """
        on = _number(d, "on", where, lambda v: 0 < v < 1, "a number in (0, 1)")
        off = _number(d, "off", where, lambda v: 0 < v <= on, f"a number in (0, on={on:g}]")
        diagonal = _number(d, "diagonal", where, lambda v: 0 < v <= 1, "a number in (0, 1]")
        dwell_s = _number(d, "dwell_s", where, lambda v: 0 <= v <= MAX_DWELL_S,
                          f"a number of seconds in [0, {MAX_DWELL_S:g}]")
        return cls(on, off, diagonal, dwell_s)

    def record(self) -> dict[str, float]:
        """The tuning as a stick's entry of a calibration file writes it."""
        return {"on": round(self.on, 3), "off": round(self.off, 3),
                "diagonal": round(self.diagonal, 2), "dwell_s": round(self.dwell_s, 3)}

    def rule(self) -> str:
        """How it reads a push, in words."""
        how = ("one direction at a time" if self.diagonal >= 1.0
               else f"diagonals past {self.diagonal:.2f}")
        return f"{self.on:.2f}/{self.off:.2f}, {how}, {self.dwell_s * 1000:.0f} ms dwell"


#: How a stick with no calibration is read.
UNCALIBRATED = StickTuning(on=0.2, off=0.2, diagonal=1.0, dwell_s=0.05)
_TUNING_KEYS = ("on", "off", "diagonal", "dwell_s")


@dataclass(frozen=True)
class StickCalibration:
    """One stick's measurement, made at ``made``, and how its push is read."""

    made: datetime
    x: AxisCalibration
    y: AxisCalibration
    tuning: StickTuning

    @classmethod
    def from_dict(cls, d: Any, where: str) -> StickCalibration:
        """A stick's entry of a calibration file, its tuning flattened into it.

        :raises CalibrationError: naming the field that is missing, unknown or wrong.
        """
        d = _checked(d, where, ("made", "x", "y", *_TUNING_KEYS))
        return cls(_made(d["made"], _at(where, "made")),
                   AxisCalibration.from_dict(d["x"], _at(where, "x")),
                   AxisCalibration.from_dict(d["y"], _at(where, "y")),
                   StickTuning.from_dict(d, where))

    def record(self) -> dict[str, Any]:
        """The stick's entry of a calibration file."""
        return {"made": self.made.isoformat(timespec="seconds"), "x": self.x.record(),
                "y": self.y.record(), **self.tuning.record()}


@dataclass(frozen=True)
class Calibration:
    """One file's calibration of a device: an entry per stick it has.

    ``tuned_from`` names the file this one was retuned from, None for a measurement.
    """

    device: str
    tuned_from: str | None
    sticks: dict[Stick, StickCalibration]

    @classmethod
    def from_dict(cls, d: Any, where: str = "") -> Calibration:
        """A calibration file's contents.

        :raises CalibrationError: naming the field that is missing, unknown or wrong.
        """
        d = _checked(d, where, ("device", "tuned_from", "sticks"))
        device, tuned_from, sticks = d["device"], d["tuned_from"], d["sticks"]
        if not isinstance(device, str) or not device:
            raise CalibrationError(f"{_at(where, 'device')}: expected the controller's name, "
                                   f"got {device!r}")
        if tuned_from is not None and not (isinstance(tuned_from, str) and tuned_from):
            raise CalibrationError(f"{_at(where, 'tuned_from')}: expected null or a file "
                                   f"name, got {tuned_from!r}")
        if not isinstance(sticks, dict):
            raise CalibrationError(f"{_at(where, 'sticks')}: expected {{stick: ...}}, "
                                   f"got {sticks!r}")
        names = [s.value for s in Stick]
        for name in sticks:
            if name not in names:
                raise CalibrationError(f"{_at(where, 'sticks')}.{name}: unknown stick "
                                       f"(expected {', '.join(names)})")
        return cls(device, tuned_from, {
            s: StickCalibration.from_dict(sticks[s.value], _at(where, f"sticks.{s.value}"))
            for s in Stick if s.value in sticks})

    def record(self) -> dict[str, Any]:
        """The calibration file's contents."""
        return {"device": self.device, "tuned_from": self.tuned_from,
                "sticks": {s.value: c.record() for s, c in self.sticks.items()}}


@dataclass(frozen=True)
class Layout:
    """Which of a joystick's numbered buttons and axes is which control."""

    name: str
    buttons: dict[int, str]
    axes: dict[int, int]            # its axis -> the stick axis it is (:attr:`Stick.axes`)


#: The Current Designs pad. SDL numbers buttons from 0, one less than the HID
#: report does. It also reports two axes pinned at -1.0, which are no control.
CURRENT_DESIGNS = Layout("Current Designs", {0: "Y", 1: "B", 2: "X", 3: "LT", 4: "A", 5: "RT"},
                         {0: 0, 1: 1})
#: A gamepad SDL knows the layout of (an Xbox controller, say), by SDL's names
#: for its controls: its raw numbers differ by pad, OS and connection.
GAMEPAD = Layout("SDL's gamepad layout", {
    pygame.CONTROLLER_BUTTON_Y: "Y", pygame.CONTROLLER_BUTTON_B: "B",
    pygame.CONTROLLER_BUTTON_X: "X", pygame.CONTROLLER_BUTTON_A: "A",
    pygame.CONTROLLER_BUTTON_LEFTSHOULDER: "LB", pygame.CONTROLLER_BUTTON_RIGHTSHOULDER: "RB",
    pygame.CONTROLLER_BUTTON_START: "START", pygame.CONTROLLER_BUTTON_BACK: "BACK",
    pygame.CONTROLLER_BUTTON_DPAD_UP: "DPAD_UP", pygame.CONTROLLER_BUTTON_DPAD_DOWN: "DPAD_DOWN",
    pygame.CONTROLLER_BUTTON_DPAD_LEFT: "DPAD_LEFT",
    pygame.CONTROLLER_BUTTON_DPAD_RIGHT: "DPAD_RIGHT",
}, {
    pygame.CONTROLLER_AXIS_LEFTX: 0, pygame.CONTROLLER_AXIS_LEFTY: 1,
    pygame.CONTROLLER_AXIS_RIGHTX: 2, pygame.CONTROLLER_AXIS_RIGHTY: 3,
})
#: A gamepad's analog triggers, which press their control past :data:`_TRIGGER_ON`.
GAMEPAD_TRIGGERS: dict[int, str] = {
    pygame.CONTROLLER_AXIS_TRIGGERLEFT: "LT", pygame.CONTROLLER_AXIS_TRIGGERRIGHT: "RT",
}
#: The controls a gamepad has, by the names in its SDL mapping.
_MAPPED: dict[str, tuple[str, ...]] = {
    "y": ("Y",), "b": ("B",), "x": ("X",), "a": ("A",), "lefttrigger": ("LT",),
    "righttrigger": ("RT",), "leftshoulder": ("LB",), "rightshoulder": ("RB",),
    "start": ("START",), "back": ("BACK",), "dpup": ("DPAD_UP",), "dpdown": ("DPAD_DOWN",),
    "dpleft": ("DPAD_LEFT",), "dpright": ("DPAD_RIGHT",),
    "leftx": ("LEFT", "RIGHT"), "lefty": ("UP", "DOWN"),
    "rightx": ("RS_LEFT", "RS_RIGHT"), "righty": ("RS_UP", "RS_DOWN"),
}
_GAMEPAD_EVENTS = (pygame.CONTROLLERBUTTONDOWN, pygame.CONTROLLERBUTTONUP,
                   pygame.CONTROLLERAXISMOTION)
PAD_EVENTS = (pygame.JOYBUTTONDOWN, pygame.JOYBUTTONUP, pygame.JOYAXISMOTION, *_GAMEPAD_EVENTS)
_DOWN_EVENTS = (pygame.JOYBUTTONDOWN, pygame.CONTROLLERBUTTONDOWN)


def _scaled(value: float) -> float:
    """A gamepad axis's reading, -32768..32767, as -1..1."""
    return max(-1.0, min(1.0, value / 32767))


class Pad:
    """One plugged-in pad: its SDL handles, and which of its numbers is which control."""

    def __init__(self, index: int) -> None:
        self.joystick = pygame.joystick.Joystick(index)
        self.gamepad = controller.Controller(index) if controller.is_controller(index) else None
        self.layout = CURRENT_DESIGNS if self.gamepad is None else GAMEPAD
        self.name = self.joystick.get_name()
        self.id = self.joystick.get_instance_id()

    def controls(self) -> tuple[str, ...]:
        """The controls this pad has, in :data:`CONTROLS` order."""
        if self.gamepad is None:
            have = {*STICK, *self.layout.buttons.values()}
        else:
            have = {c for key in self.gamepad.get_mapping() for c in _MAPPED.get(key, ())}
        return tuple(c for c in CONTROLS if c in have)

    def sticks(self) -> tuple[Stick, ...]:
        """The sticks this pad has."""
        return tuple(s for s in Stick if s.directions[0] in self.controls())

    def source(self, control: str) -> str:
        """Which of its raw buttons or axes a control is, as SDL numbers them ("b6", "+a2")."""
        if self.gamepad is not None:
            key = next((k for k, cs in _MAPPED.items() if control in cs), "")
            return self.gamepad.get_mapping().get(key, "")
        number = {c: b for b, c in self.layout.buttons.items()}.get(control)
        return "" if number is None else f"b{number}"

    def read(self) -> tuple[dict[int, float], set[str]]:
        """Its stick axes now (by :attr:`Stick.axes`, raw), and the buttons and triggers held."""
        if self.gamepad is None:
            j = self.joystick
            axes = {s: j.get_axis(a) for a, s in self.layout.axes.items() if a < j.get_numaxes()}
            return axes, {c for b, c in self.layout.buttons.items()
                          if b < j.get_numbuttons() and j.get_button(b)}
        g = self.gamepad
        axes = {s: _scaled(g.get_axis(a)) for a, s in self.layout.axes.items()}
        held = {c for b, c in self.layout.buttons.items() if g.get_button(b)}
        return axes, held | {c for a, c in GAMEPAD_TRIGGERS.items()
                             if _scaled(g.get_axis(a)) >= _TRIGGER_ON}


_enabled = False
_bindings: dict[str, str] = {}          # control -> the key it presses (the rig's)
_in_use: tuple[Path, Calibration] | None = None     # the calibration file read, and its contents
_axes: dict[int, AxisCalibration] = {}  # stick axis -> its calibration
_tunings: dict[Stick, StickTuning] = {}
_held: set[str] = set()                 # the controls the pad is holding down
_value: dict[int, float] = {}           # axis -> last reading, calibrated
_raw: dict[int, float] = {}             # axis -> last reading, untouched
_zero: dict[int, float] = {}            # axis -> where it is centred now
_pushed: dict[int, str | None] = {}     # axis -> its side, in diagonal mode
_since: dict[str, float] = {}           # direction -> when it first wanted pressing
_quiet: dict[int, tuple] = {}           # axis -> (since, low, high) while still
_settled: set[int] = set()              # axes whose centre has been taken
_warmed: set[Stick] = set()             # sticks used since their calibration was read
_pads: dict[int, Pad] = {}              # instance id -> the pad
_unmapped: set[tuple[str, int]] = set()  # (layout, button) already warned of
_status = 'off (the rig\'s "has_pad": false)'


def _at(where: str, key: str) -> str:
    """A field's name under ``where``."""
    return f"{where}.{key}" if where else key


def _checked(d: Any, where: str, keys: tuple[str, ...]) -> dict:
    """``d``, if it is a dict of exactly ``keys``.

    :raises CalibrationError: naming the first key missing or unknown.
    """
    if not isinstance(d, dict):
        raise CalibrationError(f"{where or 'the file'}: expected {{{', '.join(keys)}}}, "
                               f"got {d!r}")
    for key in keys:
        if key not in d:
            raise CalibrationError(f"{_at(where, key)}: missing")
    for key in d:
        if key not in keys:
            raise CalibrationError(f"{_at(where, key)}: unknown key "
                                   f"(expected {', '.join(keys)})")
    return d


def _number(d: dict, key: str, where: str, ok: Callable[[float], bool], expected: str) -> float:
    """``d[key]``, if it is a number that is ``ok``.

    :raises CalibrationError: naming the field, what it should be and what it is.
    """
    value = d[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not ok(value):
        raise CalibrationError(f"{_at(where, key)}: expected {expected}, got {value!r}")
    return float(value)


def _made(value: Any, where: str) -> datetime:
    """A time with its zone, as ISO 8601.

    :raises CalibrationError: when it is not one.
    """
    try:
        made = datetime.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        made = None
    if made is None or made.tzinfo is None:
        raise CalibrationError(f"{where}: expected a time such as 2026-10-21T14:15:00+00:00, "
                               f"got {value!r}")
    return made


def calibrations_dir() -> Path:
    """Where this machine's calibrations are kept, one file per measurement."""
    base = os.environ.get("XDG_CONFIG_HOME") or "~/.config"
    return Path(base).expanduser() / "fmri-gym" / "pads"


def read(path: Path) -> Calibration:
    """A calibration file's contents.

    :raises CalibrationError: naming the file and the field that is wrong.
    """
    try:
        d = json.loads(path.read_text())
    except ValueError as exc:
        raise CalibrationError(f"{path}: not JSON ({exc})") from None
    try:
        return Calibration.from_dict(d)
    except CalibrationError as exc:
        raise CalibrationError(f"{path}: {exc}") from None


def _device_in(path: Path) -> str | None:
    """The device a file names, or None when it names none."""
    try:
        d = json.loads(path.read_text())
    except ValueError:
        return None
    device = d.get("device") if isinstance(d, dict) else None
    return device if isinstance(device, str) else None


def newest(device: str) -> tuple[Path, Calibration] | None:
    """The newest calibration of ``device`` in :func:`calibrations_dir`, by file time; or None.

    :raises CalibrationError: when a file there is malformed and is, or may be, ``device``'s.
    """
    mine = []
    for path in sorted(calibrations_dir().glob("*.json")):
        try:
            calibration = read(path)
        except CalibrationError as exc:
            if _device_in(path) in (device, None):
                raise CalibrationError(f"{exc}; fix it, or move it out of "
                                       f"{calibrations_dir()}") from None
            continue
        if calibration.device == device:
            mine.append((path.stat().st_mtime_ns, path.name, path, calibration))
    if not mine:
        return None
    _, _, path, calibration = max(mine, key=lambda m: m[:2])
    return path, calibration


def _slug(device: str) -> str:
    """A device's name as a file name's start."""
    return re.sub(r"[^A-Za-z0-9]+", "-", device).strip("-") or "pad"


def write(calibration: Calibration, samples: list[dict] | None = None) -> Path:
    """Write ``calibration`` to a new file, never over another, and its recording beside it.

    :param samples: the frames it was measured from, written as the ``.csv`` of the same name.
    :return: the calibration's path: ``<device>_<YYYYmmdd-HHMMSS>.json`` in
        :func:`calibrations_dir`.
    """
    record = calibration.record()
    Calibration.from_dict(record)                   # what is written reads back
    folder = calibrations_dir()
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{_slug(calibration.device)}_{time.strftime('%Y%m%d-%H%M%S')}"
    path, n = folder / f"{stem}.json", 1
    while path.exists() or path.with_suffix(".csv").exists():
        n += 1
        path = folder / f"{stem}-{n}.json"
    with path.open("x") as fh:
        fh.write(json.dumps(record, indent=2) + "\n")
    if samples:
        with path.with_suffix(".csv").open("x", newline="") as fh:
            out = csv.DictWriter(fh, fieldnames=list(samples[0]))
            out.writeheader()
            out.writerows(samples)
    return path


def init(enabled: bool = True, bindings: dict[str, str] | None = None,
         is_calibration_required: bool = True) -> str:
    """Open whatever pad is plugged in, unless the rig file said not to, and read its 

    :param enabled: ``False`` to leave the pad off for this run.
    :param bindings: control (:data:`CONTROLS`) -> the key name it presses; a
        control left out presses nothing.
    :param is_calibration_required: refuse a pad with a bound stick that has no
        calibration; ``False`` reads such a stick uncalibrated.
    :return: a one-line status for the run's stderr banner and the manifest.
    :raises CalibrationError: when the pad's newest calibration file is malformed, or
        a bound stick has none and one is required.
    """
    global _status, _enabled, _bindings
    _bindings = dict(bindings or {})
    if not enabled:
        _status = 'off (the rig\'s "has_pad": false)'
        return _status
    if not _enabled:
        register_held_source(held)
        register_event_source(pump)
    _enabled = True
    pad = find()
    use(None if pad is None else newest(pad.name))
    if pad is None:
        _status = "no controller plugged in; the keyboard answers"
        return _status
    missing = _missing(pad, calibrations_dir())
    if missing and is_calibration_required:
        raise CalibrationError(
            f"{missing}: {HOW_TO_CALIBRATE} (a run reads each bound stick "
            "through the newest calibration of the pad plugged in)")
    pads = ", ".join(f"{p.name} ({p.layout.name})" for p in _pads.values())
    _status = f"{pads} -> {_bound()}; {_sticks_line(pad)}"
    return _status


def _missing(pad: Pad, folder: Path) -> str:
    """Which of the pad's bound sticks has no calibration, and where none was found; or ""."""
    for stick in pad.sticks():
        bound = [d for d in stick.directions if d in _bindings]
        if not bound or is_calibrated(stick):
            continue
        if _in_use is None:
            return f"no calibration of {pad.name!r} in {folder}"
        return (f"the calibration of {pad.name!r} in use, {_in_use[0]}, has no {stick.value} "
                f"stick, which the rig binds ({', '.join(bound)})")
    return ""


def _bound() -> str:
    if _bindings:
        return ", ".join(f"{c}={k}" for c, k in _bindings.items())
    return ('WARNING: the rig file has no "bindings", so its controls press no key (a '
            "controller that types keys itself needs none)")


def _sticks_line(pad: Pad) -> str:
    """Each stick's rule, and the calibration file read."""
    sticks = [f"{s.value} stick at {_tunings[s].rule()}" if is_calibrated(s)
              else f"{s.value} stick UNCALIBRATED" for s in pad.sticks()]
    cal = f"calibration {_in_use[0]}" if _in_use else f"no calibration of {pad.name!r}"
    return "; ".join([*sticks, cal])


def status() -> str:
    """The line :func:`init` last returned, for the manifest and the banner."""
    return _status

def label_keys(text: str, bindings: dict[str, str] | None) -> str:
    """``text`` with each ``[KEY]`` (a key name) named as the participant presses it.

    :param bindings: control -> key, the controller's; ``None`` for a keyboard.
    :return: each tag as the first control (:data:`CONTROLS` order, so the stick
        before the D-pad) bound to its key, or as the key's name when none is;
        ``[ARROWS]`` as what presses ``UP`` ("stick", "D-pad") or as ``ARROWS``.
        Brackets around anything but a key name are left as they are.
    """
    pressed_by: dict[str, str] = {}
    for control in CONTROLS:
        if bindings and control in bindings:
            pressed_by.setdefault(bindings[control], control)

    def label(tag: re.Match) -> str:
        key = tag.group(1)
        if key == "ARROWS":
            up = pressed_by.get("UP")
            return CONTROL_LABELS[up].rsplit(" ", 1)[0] if up in (*STICK, *DPAD, *RIGHT_STICK) \
                else "ARROWS"
        if key not in KEYCODE_BY_NAME:
            return tag.group(0)
        return CONTROL_LABELS[pressed_by[key]] if key in pressed_by else key
    return _KEY_TAG.sub(label, text)



def use(found: tuple[Path, Calibration] | None) -> None:
    """Read the sticks through ``found`` (a calibration file and its contents), or uncalibrated.

    A new calibration means a new centre to find: the one taken under the old one is dropped.
    """
    global _in_use
    _in_use = found
    sticks = {} if found is None else found[1].sticks
    _axes.clear()
    for stick, cal in sticks.items():
        _axes.update(zip(stick.axes, (cal.x, cal.y)))
    _tunings.clear()
    _tunings.update({s: sticks[s].tuning if s in sticks else UNCALIBRATED for s in Stick})
    for state in (_zero, _settled, _quiet, _warmed):
        state.clear()


def in_use() -> tuple[Path, Calibration] | None:
    """The calibration file the sticks are read through, and its contents; or None."""
    return _in_use


def tuning(stick: Stick) -> StickTuning:
    """How ``stick``'s push is read now."""
    return _tunings[stick]


def set_tuning(stick: Stick, value: StickTuning) -> None:
    """Read ``stick``'s push by ``value`` from now on."""
    _tunings[stick] = value


def is_calibrated(stick: Stick) -> bool:
    """Whether ``stick`` reads by a """
    return _in_use is not None and stick in _in_use[1].sticks


def _open() -> bool:
    """Open the pads, or reopen them after something reset the joystick subsystem.

    ``Display`` calls ``pygame.init()`` when the window goes up, which restarts
    every subsystem and quietly closes handles opened before it. Rather than
    depend on the order, they are checked here and taken again when stale.
    """
    if _pads and next(iter(_pads.values())).joystick.get_init():
        return True
    _pads.clear()
    if not pygame.joystick.get_init():
        pygame.joystick.init()
    if not controller.get_init():
        controller.init()
    for i in range(pygame.joystick.get_count()):
        pad = Pad(i)
        _pads[pad.id] = pad
    return bool(_pads)


def _arrived(seconds: float = 0.5) -> bool:
    """Wait for a pad SDL announces late, and open it.

    On macOS SDL sees a gamepad only with its display subsystem up (which opens
    no window), and the pad arrives as an event about 0.1 s after that.
    """
    pygame.display.init()
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        pygame.event.pump()
        if _open():
            return True
        time.sleep(0.05)
    return False


def find() -> Pad | None:
    """The first pad plugged in, opened, waiting briefly for one SDL announces late; or None."""
    return first() if _open() or _arrived() else None


def first() -> Pad | None:
    """The first pad opened, which is the one calibrated and shown; or None."""
    return next(iter(_pads.values()), None)


def binding(control: str) -> str | None:
    """The key the rig binds ``control`` to, or None."""
    return _bindings.get(control)


def held_controls() -> frozenset[str]:
    """The controls the pad is holding down, this frame."""
    return frozenset(_held)


def held() -> frozenset[str]:
    """The keys the pad's held controls are bound to, this frame."""
    return frozenset(_bindings[c] for c in _held if c in _bindings)


def centre(index: int) -> float:
    """Where axis ``index`` is currently taken to rest."""
    cal = _axes.get(index)
    return _zero.get(index, 0.0 if cal is None else cal.rest)


def normalise(index: int, value: float) -> float:
    """A raw reading as a fraction of that direction's own travel, -1..1.

    Each side is scaled by its own span, so a pad reaching -0.5 one way and +1.0
    the other still reports -1.0 and +1.0 at its stops. The stops do not move,
    so both spans are measured from wherever the stick is currently centred.
    """
    cal = _axes.get(index)
    if cal is None:
        return value
    rest = centre(index)
    span = (cal.high - rest) if value >= rest else (rest - cal.low)
    if span <= 1e-6:
        return 0.0
    return max(-1.0, min(1.0, (value - rest) / span))


def _settle(now: float) -> None:
    """Take each stick's centre, once, after it has been used and let go."""
    for stick in _warmed:
        if not set(stick.directions) & _held:
            for index in stick.axes:
                _settle_axis(index, now)


def _settle_axis(index: int, now: float) -> None:
    """Take one axis's centre once it has been still long enough."""
    if index in _settled or index not in _axes or index not in _raw:
        return
    value = _raw[index]
    since, low, high = _quiet.get(index, (now, value, value))
    low, high = min(low, value), max(high, value)
    if high - low > _SETTLE_STILL:
        _quiet[index] = (now, value, value)         # moved: start again
        return
    _quiet[index] = (since, low, high)
    if now - since < _SETTLE_S:
        return
    _settled.add(index)
    cal, found = _axes[index], (low + high) / 2
    if abs(found - cal.rest) <= max(cal.drift * _SETTLE_SLACK, _SETTLE_STILL):
        _zero[index] = found
        return
    print(f"pad: axis {index} settled at {found:+.3f}, {abs(found - cal.rest):.3f} from the "
          f"calibrated centre and further than the {cal.drift:.3f} it was measured to move; "
          "keeping the calibrated one. Calibrate it again.", file=sys.stderr)


def _press(control: str, down: bool) -> None:
    """Hold or release ``control``, and post the keypress of the key it is bound to."""
    if down:
        _held.add(control)
    else:
        _held.discard(control)
    key = _bindings.get(control)
    if key is None:
        return
    # `unicode` is what a "press <char> to go on" screen reads; `pad` marks the
    # event as ours for anything that wants to tell the two apart.
    char = " " if key == "SPACE" else key.lower() if len(key) == 1 else ""
    pygame.event.post(pygame.event.Event(
        pygame.KEYDOWN if down else pygame.KEYUP,
        key=KEYCODE_BY_NAME[key], mod=0, unicode=char if down else "",
        scancode=0, pad=True))


def _away(a: float, b: float) -> float:
    """How far apart two angles are, the short way round."""
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def _classified(stick: Stick) -> set[str]:
    """The one direction a stick is being pushed, or none.

    How far it is from centre says *whether* a direction is held, and the angle
    says *which*. Classifying the push as a whole is what keeps the horizontal
    axis from answering a push meant for down: that push does move both axes, a
    lot, but it points much closer to down than to left.
    """
    tuning, (x, y) = _tunings[stick], stick.axes
    nx, ny = _value.get(x, 0.0), _value.get(y, 0.0)
    current = set(stick.directions) & _held
    if math.hypot(nx, ny) < (tuning.off if current else tuning.on):
        return set()
    theta = math.atan2(ny, nx)
    best = min(stick.directions, key=lambda d: _away(theta, _ANGLES[d]))
    for name in current:                    # the one already pressed keeps it
        if _away(theta, _ANGLES[name]) - _away(theta, _ANGLES[best]) < _SECTOR_KEEP:
            return {name}
    return {best}


def _side(index: int, tuning: StickTuning) -> str | None:
    """Which way an axis is pushed, holding whatever it was doing in the gap."""
    value, was = _value.get(index, 0.0), _pushed.get(index)
    if value <= -tuning.on:
        return "-"
    if value >= tuning.on:
        return "+"
    return None if abs(value) < tuning.off else was


def _independent(stick: Stick) -> set[str]:
    """A stick's two axes read separately, for a game that wants diagonals."""
    tuning, (x, y) = _tunings[stick], stick.axes
    up, down, left, right = stick.directions
    sides = {(x, "-"): left, (x, "+"): right, (y, "-"): up, (y, "+"): down}
    wanted: dict[int, str] = {}
    for index in stick.axes:
        side = _side(index, tuning)
        _pushed[index] = side
        if side is not None:
            wanted[index] = sides[(index, side)]
    if x in wanted and y in wanted:
        strong, weak = (x, y) if abs(_value[x]) >= abs(_value[y]) else (y, x)
        if wanted[weak] in _held and wanted[strong] not in _held \
                and abs(_value[weak]) * _DOMINANCE > abs(_value[strong]):
            strong, weak = weak, strong
        lead = abs(_value[strong])
        ratio = abs(_value[weak]) / lead if lead > 1e-6 else 0.0
        if ratio < tuning.diagonal * (0.8 if wanted[weak] in _held else 1.0):
            del wanted[weak]
    return set(wanted.values())


def _dwelt(names: set[str], directions: tuple[str, ...], dwell: float, now: float) -> set[str]:
    """Of the directions a stick asks for, the ones that have asked long enough."""
    for name in [n for n in _since if n in directions and n not in names]:
        del _since[name]
    for name in names:
        _since.setdefault(name, now)
    return {n for n in names if n in _held or now - _since[n] >= dwell}


def _resolve() -> None:
    """Press what the sticks' positions now mean, and release what they do not."""
    now = time.monotonic()
    _settle(now)
    for index, value in _raw.items():
        _value[index] = normalise(index, value)
    names: set[str] = set()
    for stick in Stick:
        if not any(a in _raw for a in stick.axes):
            continue
        tuning = _tunings[stick]
        wanted = _classified(stick) if tuning.diagonal >= 1.0 else _independent(stick)
        names |= (_dwelt(wanted, stick.directions, tuning.dwell_s, now)
                  if tuning.dwell_s > 0 else wanted)
    for name in sorted(STICK_DIRECTIONS & (_held - names)):
        _press(name, False)
    for name in sorted(names - _held):
        _warmed.add(next(s for s in Stick if name in s.directions))
        _press(name, True)


def pump() -> None:
    """Move the pad's events into the keyboard ones (run by ``keys.get_events``).

    Takes only the joystick and gamepad events off the queue, so whatever else
    is waiting there is still the caller's to read. A gamepad's events come
    twice, as a joystick's and as a gamepad's; only the gamepad's are read.
    """
    if not (_enabled and _open()):
        return
    moved = False
    for event in pygame.event.get(PAD_EVENTS):
        pad = _pads.get(event.instance_id)
        if pad is None or (event.type in _GAMEPAD_EVENTS) != (pad.gamepad is not None):
            continue
        if event.type in (pygame.JOYAXISMOTION, pygame.CONTROLLERAXISMOTION):
            moved |= _axis(pad, event.axis, event.value)
        else:
            _button(pad, event.button, event.type in _DOWN_EVENTS)
    if moved:
        _resolve()


def _axis(pad: Pad, axis: int, value: float) -> bool:
    """Take an axis's reading. Returns whether it was a stick's."""
    if pad.gamepad is not None:
        value = _scaled(value)
        if axis in GAMEPAD_TRIGGERS:
            _trigger(GAMEPAD_TRIGGERS[axis], value)
            return False
    stick = pad.layout.axes.get(axis)
    if stick is None:
        return False
    _raw[stick] = value
    return True


def _trigger(control: str, value: float) -> None:
    """Press an analog trigger's control past :data:`_TRIGGER_ON`, release it under the off."""
    if control not in _held and value >= _TRIGGER_ON:
        _press(control, True)
    elif control in _held and value < _TRIGGER_OFF:
        _press(control, False)


def _button(pad: Pad, button: int, down: bool) -> None:
    """Press or release the control a button is, or warn once of one that is none."""
    control = pad.layout.buttons.get(button)
    if control is not None:
        _press(control, down)
    elif down and (pad.layout.name, button) not in _unmapped:
        _unmapped.add((pad.layout.name, button))
        print(f"pad: {pad.name}'s button {button} pressed, but it is no control of "
              f"{pad.layout.name} (fmri_gym/pad.py)", file=sys.stderr)
