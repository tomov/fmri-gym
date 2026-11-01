"""A controller's calibration file: how each of its sticks rests, travels and is read.

One file per measurement in :func:`calibrations_dir`, outside the checkout since
it describes this machine's controller, found by the name SDL gives the device
(:func:`newest`) and never rewritten (:func:`write`). It is made by
:mod:`fmri_gym.pad_calibration`.
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
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class Side(Enum):
    """A controller's sticks, by the name a calibration file gives each."""

    LEFT = "left"
    RIGHT = "right"

    @property
    def axes(self) -> tuple[int, int]:
        """Its x and y, by a calibration recording's numbering of the axes (``ax0`` ...)."""
        return (0, 1) if self is Side.LEFT else (2, 3)


#: What to do about a stick with no calibration, for the messages that say so.
HOW_TO_CALIBRATE = ("run pad-calibration (or Calibrate controller... in rig-config), or the "
                    "rig check")
#: The longest a direction may be made to wait before it is pressed.
MAX_DWELL_S = 0.3


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
    sticks: dict[Side, StickCalibration]

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
        names = [s.value for s in Side]
        for name in sticks:
            if name not in names:
                raise CalibrationError(f"{_at(where, 'sticks')}.{name}: unknown stick "
                                       f"(expected {', '.join(names)})")
        return cls(device, tuned_from, {
            s: StickCalibration.from_dict(sticks[s.value], _at(where, f"sticks.{s.value}"))
            for s in Side if s.value in sticks})

    def record(self) -> dict[str, Any]:
        """The calibration file's contents."""
        return {"device": self.device, "tuned_from": self.tuned_from,
                "sticks": {s.value: c.record() for s, c in self.sticks.items()}}


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
