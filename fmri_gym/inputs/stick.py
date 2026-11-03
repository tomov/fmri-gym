"""One stick of a controller, read through its calibration into four directions it holds.

A device that has a stick owns one of these, feeds it the stick's two raw
axes (:meth:`Stick.move`) and asks it what that means (:meth:`Stick.resolve`):
``<name>_UP``, ``_DOWN``, ``_LEFT`` and ``_RIGHT``, pressed and released like
buttons. How far is a push, which way it points and how long it must last are
the stick's tuning; where it rests and how far it travels are its calibration.
"""

from __future__ import annotations

import math
import sys

from .base import Edge, InputEvent
from .calibration import UNCALIBRATED, AxisCalibration, Side, StickCalibration, StickTuning

#: A stick's directions, in the order of its controls' names.
DIRECTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
#: Where each direction points, as an angle of push (y grows downwards).
_ANGLES = dict(zip(DIRECTIONS, (-math.pi / 2, math.pi / 2, math.pi, 0.0)))
#: How much nearer another direction must be before the stick changes hands.
_SECTOR_KEEP = math.radians(15)
_DOMINANCE = 1.15           # with diagonals, the margin for one axis to take the stick over
#: The stick's centre is taken once, after it has first been used and then let
#: go: cold and warm are not the same place (the gimbal does not return where it
#: started) and the warm one moves between sessions. How far it may be taken
#: from the calibrated centre is what the calibration measured that movement to
#: be, so a stick held slightly off can never be mistaken for centre.
_SETTLE_S, _SETTLE_STILL, _SETTLE_SLACK = 1.0, 0.01, 1.5


def directions_of(name: str) -> tuple[str, str, str, str]:
    """The four direction controls of the stick called ``name``."""
    return tuple(f"{name}_{d}" for d in DIRECTIONS)


def _away(a: float, b: float) -> float:
    """How far apart two angles are, the short way round."""
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


class Stick:
    """One stick: its calibration and tuning, its two axes (0: x, 1: y) and what it holds."""

    def __init__(self, name: str, side: Side, calibration: StickCalibration | None) -> None:
        """A stick at rest, read through ``calibration``.

        :param name: its controls' prefix (``"GMPD3_STICK"``).
        :param side: which of the device's sticks it is, in its calibration file.
        :param calibration: its entry of the device's calibration; None reads it uncalibrated.
        """
        self.name, self.side, self.calibration = name, side, calibration
        self.directions = directions_of(name)
        self.base = dict(zip(self.directions, DIRECTIONS))
        #: how its push is read now; ``pad-calibration`` retunes it live
        self.tuning: StickTuning = UNCALIBRATED if calibration is None else calibration.tuning
        self._axes: dict[int, AxisCalibration] = (
            {} if calibration is None else {0: calibration.x, 1: calibration.y})
        self._held: set[str] = set()
        self._value: dict[int, float] = {}          # axis -> last reading, calibrated
        self._raw: dict[int, float] = {}            # axis -> last reading, untouched
        self._zero: dict[int, float] = {}           # axis -> where it is centred now
        self._pushed: dict[int, str | None] = {}    # axis -> its side, in diagonal mode
        self._since: dict[str, float] = {}          # direction -> when it first wanted pressing
        self._quiet: dict[int, tuple] = {}          # axis -> (since, low, high) while still
        self._settled: set[int] = set()             # axes whose centre has been taken
        self._warmed = False                        # used since its calibration was read

    @property
    def is_calibrated(self) -> bool:
        return self.calibration is not None

    def last_held(self) -> frozenset[str]:
        """The directions down as of the last :meth:`resolve`."""
        return frozenset(self._held)

    def move(self, axis: int, value: float) -> None:
        """Take a raw reading of axis ``axis`` (0: x, 1: y); :meth:`resolve` reads it."""
        self._raw[axis] = value

    def centre(self, axis: int) -> float:
        """Where axis ``axis`` is currently taken to rest."""
        cal = self._axes.get(axis)
        return self._zero.get(axis, 0.0 if cal is None else cal.rest)

    def normalise(self, axis: int, value: float) -> float:
        """A raw reading as a fraction of that direction's own travel, -1..1.

        Each side is scaled by its own span, so a stick reaching -0.5 one way and
        +1.0 the other still reports -1.0 and +1.0 at its stops. The stops do not
        move, so both spans are measured from wherever the stick is centred now.
        """
        cal = self._axes.get(axis)
        if cal is None:
            return value
        rest = self.centre(axis)
        span = (cal.high - rest) if value >= rest else (rest - cal.low)
        if span <= 1e-6:
            return 0.0
        return max(-1.0, min(1.0, (value - rest) / span))

    def resolve(self, now: float) -> list[InputEvent]:
        """Press what the stick's position now means, and release what it does not.

        :param now: the time, in seconds, on the clock every call uses.
        """
        self._settle(now)
        for axis, value in self._raw.items():
            self._value[axis] = self.normalise(axis, value)
        tuning = self.tuning
        wanted = self._classified() if tuning.diagonal >= 1.0 else self._independent()
        names = self._dwelt(wanted, tuning.dwell_s, now) if tuning.dwell_s > 0 else wanted
        out = [InputEvent(name, Edge.UP) for name in sorted(self._held - names)]
        out += [InputEvent(name, Edge.DOWN) for name in sorted(names - self._held)]
        self._warmed |= bool(names - self._held)
        self._held = set(names)
        return out

    def _settle(self, now: float) -> None:
        """Take the centre, once, after the stick has been used and let go."""
        if self._warmed and not self._held:
            for axis in (0, 1):
                self._settle_axis(axis, now)

    def _settle_axis(self, axis: int, now: float) -> None:
        """Take one axis's centre once it has been still long enough."""
        if axis in self._settled or axis not in self._axes or axis not in self._raw:
            return
        value = self._raw[axis]
        since, low, high = self._quiet.get(axis, (now, value, value))
        low, high = min(low, value), max(high, value)
        if high - low > _SETTLE_STILL:
            self._quiet[axis] = (now, value, value)         # moved: start again
            return
        self._quiet[axis] = (since, low, high)
        if now - since < _SETTLE_S:
            return
        self._settled.add(axis)
        cal, found = self._axes[axis], (low + high) / 2
        if abs(found - cal.rest) <= max(cal.drift * _SETTLE_SLACK, _SETTLE_STILL):
            self._zero[axis] = found
            return
        print(f"{self.name}: axis {axis} settled at {found:+.3f}, {abs(found - cal.rest):.3f} "
              f"from the calibrated centre and further than the {cal.drift:.3f} it was "
              "measured to move; keeping the calibrated one. Calibrate it again.",
              file=sys.stderr)

    def _classified(self) -> set[str]:
        """The one direction the stick is being pushed, or none.

        How far it is from centre says *whether* a direction is held, and the angle
        says *which*. Classifying the push as a whole is what keeps the horizontal
        axis from answering a push meant for down: that push does move both axes, a
        lot, but it points much closer to down than to left.
        """
        nx, ny = self._value.get(0, 0.0), self._value.get(1, 0.0)
        if math.hypot(nx, ny) < (self.tuning.off if self._held else self.tuning.on):
            return set()
        theta = math.atan2(ny, nx)
        best = min(self.directions, key=lambda d: _away(theta, _ANGLES[self.base[d]]))
        for name in self._held:                 # the one already pressed keeps it
            if (_away(theta, _ANGLES[self.base[name]])
                    - _away(theta, _ANGLES[self.base[best]]) < _SECTOR_KEEP):
                return {name}
        return {best}

    def _side(self, axis: int) -> str | None:
        """Which way an axis is pushed, holding whatever it was doing in the gap."""
        value, was = self._value.get(axis, 0.0), self._pushed.get(axis)
        if value <= -self.tuning.on:
            return "-"
        if value >= self.tuning.on:
            return "+"
        return None if abs(value) < self.tuning.off else was

    def _independent(self) -> set[str]:
        """The two axes read separately, for a game that wants diagonals."""
        up, down, left, right = self.directions
        sides = {(0, "-"): left, (0, "+"): right, (1, "-"): up, (1, "+"): down}
        wanted: dict[int, str] = {}
        for axis in (0, 1):
            side = self._side(axis)
            self._pushed[axis] = side
            if side is not None:
                wanted[axis] = sides[(axis, side)]
        if 0 in wanted and 1 in wanted:
            value = self._value
            strong, weak = (0, 1) if abs(value[0]) >= abs(value[1]) else (1, 0)
            if wanted[weak] in self._held and wanted[strong] not in self._held \
                    and abs(value[weak]) * _DOMINANCE > abs(value[strong]):
                strong, weak = weak, strong
            lead = abs(value[strong])
            ratio = abs(value[weak]) / lead if lead > 1e-6 else 0.0
            if ratio < self.tuning.diagonal * (0.8 if wanted[weak] in self._held else 1.0):
                del wanted[weak]
        return set(wanted.values())

    def _dwelt(self, names: set[str], dwell: float, now: float) -> set[str]:
        """Of the directions the stick asks for, the ones that have asked long enough."""
        for name in [n for n in self._since if n not in names]:
            del self._since[name]
        for name in names:
            self._since.setdefault(name, now)
        return {n for n in names if n in self._held or now - self._since[n] >= dwell}
