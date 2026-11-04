"""One stick's calibration file: how it rests, how far it travels, and how its push is read.

One file per stick per measurement in :func:`calibrations_dir`, outside the
checkout since it describes this machine's controller, found by the name SDL
gives the device and the controller's name for the stick (:func:`newest`), and
never rewritten (:func:`write`). It is made by :mod:`fmri_gym.pad_calibration`.
Nothing here knows how many sticks a controller has; the rule that turns a
stick's position into its four directions is :meth:`AxisCalibration.normalise`
and :meth:`StickTuning.directions`.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

#: A stick's four directions, as :meth:`StickTuning.directions` names them.
UP, DOWN, LEFT, RIGHT = "UP", "DOWN", "LEFT", "RIGHT"
DIRECTIONS = (UP, DOWN, LEFT, RIGHT)
#: Where each direction points, as the angle of a push (y grows downwards).
_ANGLES = {UP: -math.pi / 2, DOWN: math.pi / 2, LEFT: math.pi, RIGHT: 0.0}
#: How much nearer another direction must point before the stick changes hands.
_SECTOR_KEEP = math.radians(15)
#: What to do about a stick with no calibration, for the messages that say so.
HOW_TO_CALIBRATE = ("run pad-calibration (or Calibrate controller... in rig-config), or the "
                    "rig check")
#: The longest a direction may be made to wait before it is pressed.
MAX_DWELL_S = 0.3


def _away(a: float, b: float) -> float:
    """How far apart two angles are, the short way round."""
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


class CalibrationError(ValueError):
    """A calibration file that is not one, or a stick a run needs calibrated that is not."""


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

    def normalise(self, value: float) -> float:
        """A raw reading as a fraction of the travel on its side of rest, -1..1.

        Each side is scaled by its own span, so a stick reaching -0.5 one way and
        +1.0 the other still reports -1.0 and +1.0 at its stops.
        """
        span = (self.high - self.rest) if value >= self.rest else (self.rest - self.low)
        if span <= 1e-6:
            return 0.0
        return max(-1.0, min(1.0, (value - self.rest) / span))


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

    def directions(self, x: float, y: float, held: set[str]) -> set[str]:
        """The directions (:data:`DIRECTIONS`) a calibrated position means, given those held.

        One at a time: how far the stick is from rest says *whether* a direction
        is pressed (past ``on``, or ``off`` while one is held), and the angle of
        the push says *which*; the one held keeps it unless another points
        clearly nearer. That is what keeps the horizontal axis from answering a
        push meant for down, which moves both axes, a lot, but points much closer
        to down than to left. With diagonals, each axis is read on its own.
        """
        if self.diagonal < 1.0:
            return self._each_axis(x, y, held)
        if math.hypot(x, y) < (self.off if held else self.on):
            return set()
        theta = math.atan2(y, x)
        best = min(DIRECTIONS, key=lambda d: _away(theta, _ANGLES[d]))
        for name in held:
            if _away(theta, _ANGLES[name]) - _away(theta, _ANGLES[best]) < _SECTOR_KEEP:
                return {name}
        return {best}

    def _each_axis(self, x: float, y: float, held: set[str]) -> set[str]:
        """Each axis past ``on`` (or ``off`` while held), the weaker dropped under ``diagonal``."""
        pushed: dict[str, float] = {}
        for value, (negative, positive) in ((x, (LEFT, RIGHT)), (y, (UP, DOWN))):
            name = negative if value < 0 else positive
            if abs(value) >= (self.off if name in held else self.on):
                pushed[name] = abs(value)
        if len(pushed) == 2:
            strong, weak = sorted(pushed, key=pushed.get, reverse=True)
            if pushed[weak] < self.diagonal * pushed[strong]:
                return {strong}
        return set(pushed)

    @classmethod
    def from_dict(cls, d: dict, where: str) -> StickTuning:
        """The tuning in a calibration file: its on, off, diagonal and dwell_s.

        :raises CalibrationError: naming the field that is out of range.
        """
        on = _number(d, "on", where, lambda v: 0 < v < 1, "a number in (0, 1)")
        off = _number(d, "off", where, lambda v: 0 < v <= on, f"a number in (0, on={on:g}]")
        diagonal = _number(d, "diagonal", where, lambda v: 0 < v <= 1, "a number in (0, 1]")
        dwell_s = _number(d, "dwell_s", where, lambda v: 0 <= v <= MAX_DWELL_S,
                          f"a number of seconds in [0, {MAX_DWELL_S:g}]")
        return cls(on, off, diagonal, dwell_s)

    def record(self) -> dict[str, float]:
        """The tuning as a calibration file writes it."""
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
_FILE_KEYS = ("device", "stick", "tuned_from", "made", "x", "y", *_TUNING_KEYS)


@dataclass(frozen=True)
class StickCalibration:
    """One calibration file: one stick of one device, measured at ``made``, and its tuning.

    ``tuned_from`` names the file this one was retuned from, None for a measurement.
    """

    #: the name SDL gives the device
    device: str
    #: the controller's name for the stick
    stick: str
    tuned_from: str | None
    made: datetime
    x: AxisCalibration
    y: AxisCalibration
    tuning: StickTuning

    @classmethod
    def from_dict(cls, d: Any, where: str = "") -> StickCalibration:
        """A calibration file's contents, its tuning flattened into it.

        :raises CalibrationError: naming the field that is missing, unknown or wrong.
        """
        d = _checked(d, where, _FILE_KEYS)
        for key in ("device", "stick"):
            if not isinstance(d[key], str) or not d[key]:
                raise CalibrationError(f"{_at(where, key)}: expected a name, got {d[key]!r}")
        tuned_from = d["tuned_from"]
        if tuned_from is not None and not (isinstance(tuned_from, str) and tuned_from):
            raise CalibrationError(f"{_at(where, 'tuned_from')}: expected null or a file "
                                   f"name, got {tuned_from!r}")
        return cls(d["device"], d["stick"], tuned_from, _made(d["made"], _at(where, "made")),
                   AxisCalibration.from_dict(d["x"], _at(where, "x")),
                   AxisCalibration.from_dict(d["y"], _at(where, "y")),
                   StickTuning.from_dict(d, where))

    def record(self) -> dict[str, Any]:
        """The calibration file's contents."""
        return {"device": self.device, "stick": self.stick, "tuned_from": self.tuned_from,
                "made": self.made.isoformat(timespec="seconds"), "x": self.x.record(),
                "y": self.y.record(), **self.tuning.record()}


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
    """Where this machine's calibrations are kept, one file per stick per measurement."""
    base = os.environ.get("XDG_CONFIG_HOME") or "~/.config"
    return Path(base).expanduser() / "fmri-gym" / "pads"


def read(path: Path) -> StickCalibration:
    """A calibration file's contents.

    :raises CalibrationError: naming the file and the field that is wrong.
    """
    try:
        d = json.loads(path.read_text())
    except ValueError as exc:
        raise CalibrationError(f"{path}: not JSON ({exc})") from None
    try:
        return StickCalibration.from_dict(d)
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


def newest(device: str, stick: str) -> tuple[Path, StickCalibration] | None:
    """The newest calibration of ``device``'s ``stick`` in :func:`calibrations_dir`, by file
    time; or None.

    :raises CalibrationError: when a file there is malformed and is, or may be, ``device``'s.
    """
    calibrations_of_stick = []
    for path in sorted(calibrations_dir().glob("*.json")):
        try:
            calibration = read(path)
        except CalibrationError as exc:
            if _device_in(path) in (device, None):
                raise CalibrationError(f"{exc}; fix it, or move it out of "
                                       f"{calibrations_dir()}") from None
            continue
        if calibration.device == device and calibration.stick == stick:
            calibrations_of_stick.append((path.stat().st_mtime_ns, path.name, path, calibration))
    if not calibrations_of_stick:
        return None
    _, _, path, calibration = max(calibrations_of_stick, key=lambda m: m[:2])
    return path, calibration


def _slug(name: str) -> str:
    """A name as part of a file name."""
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-") or "pad"


def write(calibration: StickCalibration, samples: list[dict] | None = None) -> Path:
    """Write ``calibration`` to a new file, never over another, and its recording beside it.

    :param samples: the frames it was measured from, written as the ``.csv`` of the same name.
    :return: the calibration's path: ``<device>_<stick>_<YYYYmmdd-HHMMSS>.json`` in
        :func:`calibrations_dir`.
    """
    record = calibration.record()
    StickCalibration.from_dict(record)              # what is written reads back
    folder = calibrations_dir()
    folder.mkdir(parents=True, exist_ok=True)
    stem = (f"{_slug(calibration.device)}_{_slug(calibration.stick)}_"
            f"{time.strftime('%Y%m%d-%H%M%S')}")
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
