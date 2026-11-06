"""A controller's calibration file: for each stick axis, how it rests, how far it travels, and
how far it must be pushed.

One file per controller per measurement in :func:`calibrations_dir`, outside the
checkout since it describes this machine's controller, found by the name SDL
gives the device (:func:`newest`), and never rewritten (:func:`write`). It is
made by :mod:`fmri_gym.pad_calibration`. Nothing here knows which axes a
controller has; each axis is read on its own (:meth:`AxisCalibration.normalise`).
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

#: What to do about a controller with no calibration, for the messages that say so.
HOW_TO_CALIBRATE = ("run pad-calibration (or Calibrate controller... in rig-config), or the "
                    "rig check")


class CalibrationError(ValueError):
    """A calibration file that is not one, or a controller a run needs calibrated that is not."""


@dataclass(frozen=True)
class AxisCalibration:
    """One stick axis: where it rests, its two stops and how far its centre moves, raw; and
    the fraction of its travel past which a side is pressed (``on``) and under which it is
    released (``off``)."""

    rest: float
    low: float
    high: float
    drift: float
    on: float
    off: float

    @classmethod
    def from_dict(cls, d: Any, where: str) -> AxisCalibration:
        """An axis as a calibration file writes it.

        :raises CalibrationError: naming the field that is missing, unknown or out of range.
        """
        d = _checked(d, where, ("rest", "low", "high", "drift", "on", "off"))
        rest, low, high = (_number(d, k, where, math.isfinite, "a number")
                           for k in ("rest", "low", "high"))
        drift = _number(d, "drift", where, lambda v: 0 <= v < math.inf, "a number >= 0")
        if not low < rest < high:
            raise CalibrationError(f"{where}: expected low < rest < high, got low={low:g}, "
                                   f"rest={rest:g}, high={high:g}")
        on = _number(d, "on", where, lambda v: 0 < v < 1, "a number in (0, 1)")
        off = _number(d, "off", where, lambda v: 0 < v <= on, f"a number in (0, on={on:g}]")
        return cls(rest, low, high, drift, on, off)

    def record(self) -> dict[str, float]:
        """The axis as a calibration file writes it."""
        return {"rest": self.rest, "low": self.low, "high": self.high, "drift": self.drift,
                "on": round(self.on, 3), "off": round(self.off, 3)}

    def normalise(self, value: float) -> float:
        """A raw reading as a fraction of the travel on its side of rest, -1..1.

        Each side is scaled by its own span, so an axis reaching -0.5 one way and
        +1.0 the other still reports -1.0 and +1.0 at its stops.
        """
        span = (self.high - self.rest) if value >= self.rest else (self.rest - self.low)
        if span <= 1e-6:
            return 0.0
        return max(-1.0, min(1.0, (value - self.rest) / span))


@dataclass(frozen=True)
class ControllerCalibration:
    """One calibration file: each stick axis of one device, measured at ``made``.

    ``tuned_from`` names the file this one was retuned from, None for a measurement.
    """

    #: the name SDL gives the device
    device: str
    tuned_from: str | None
    made: datetime
    #: each axis, by the controller's name for it
    axes: dict[str, AxisCalibration]

    @classmethod
    def from_dict(cls, d: Any, where: str = "") -> ControllerCalibration:
        """A calibration file's contents.

        :raises CalibrationError: naming the field that is missing, unknown or wrong.
        """
        d = _checked(d, where, ("device", "tuned_from", "made", "axes"))
        if not isinstance(d["device"], str) or not d["device"]:
            raise CalibrationError(f"{_at(where, 'device')}: expected a name, got {d['device']!r}")
        tuned_from = d["tuned_from"]
        if tuned_from is not None and not (isinstance(tuned_from, str) and tuned_from):
            raise CalibrationError(f"{_at(where, 'tuned_from')}: expected null or a file "
                                   f"name, got {tuned_from!r}")
        if not isinstance(d["axes"], dict) or not d["axes"]:
            raise CalibrationError(f"{_at(where, 'axes')}: expected each axis by its name, "
                                   f"got {d['axes']!r}")
        axes = {name: AxisCalibration.from_dict(axis, _at(_at(where, "axes"), name))
                for name, axis in d["axes"].items()}
        return cls(d["device"], tuned_from, _made(d["made"], _at(where, "made")), axes)

    def record(self) -> dict[str, Any]:
        """The calibration file's contents."""
        return {"device": self.device, "tuned_from": self.tuned_from,
                "made": self.made.isoformat(timespec="seconds"),
                "axes": {name: axis.record() for name, axis in self.axes.items()}}


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
    """Where this machine's calibrations are kept, one file per controller per measurement."""
    base = os.environ.get("XDG_CONFIG_HOME") or "~/.config"
    return Path(base).expanduser() / "fmri-gym" / "pads"


def read(path: Path) -> ControllerCalibration:
    """A calibration file's contents.

    :raises CalibrationError: naming the file and the field that is wrong.
    """
    try:
        d = json.loads(path.read_text())
    except ValueError as exc:
        raise CalibrationError(f"{path}: not JSON ({exc})") from None
    try:
        return ControllerCalibration.from_dict(d)
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


def newest(device: str) -> tuple[Path, ControllerCalibration] | None:
    """The newest calibration of ``device`` in :func:`calibrations_dir`, by file time; or None.

    :raises CalibrationError: when a file there is malformed and is, or may be, ``device``'s.
    """
    calibrations_of_device = []
    for path in sorted(calibrations_dir().glob("*.json")):
        try:
            calibration = read(path)
        except CalibrationError as exc:
            if _device_in(path) in (device, None):
                raise CalibrationError(f"{exc}; fix it, or move it out of "
                                       f"{calibrations_dir()}") from None
            continue
        if calibration.device == device:
            calibrations_of_device.append((path.stat().st_mtime_ns, path.name, path, calibration))
    if not calibrations_of_device:
        return None
    _, _, path, calibration = max(calibrations_of_device, key=lambda m: m[:2])
    return path, calibration


def _slug(name: str) -> str:
    """A name as part of a file name."""
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-") or "pad"


def write(calibration: ControllerCalibration, samples: list[dict] | None = None) -> Path:
    """Write ``calibration`` to a new file, never over another, and its recording beside it.

    :param samples: the frames it was measured from, written as the ``.csv`` of the same name.
    :return: the calibration's path: ``<device>_<YYYYmmdd-HHMMSS>.json`` in
        :func:`calibrations_dir`.
    """
    record = calibration.record()
    ControllerCalibration.from_dict(record)         # what is written reads back
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
