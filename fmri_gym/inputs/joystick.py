"""What every controller SDL opens as a joystick has: its handle, its sticks, its calibration.

Not a device of its own: :mod:`~fmri_gym.inputs.gmpd3` and
:mod:`~fmri_gym.inputs.gamepad` are, each naming its controls at the top of
its file. A stick's directions need its calibration
(:mod:`~fmri_gym.inputs.calibration`), read by the name SDL gives the device;
its buttons and raw axes do not.
"""

from __future__ import annotations

import sys
from abc import abstractmethod
from pathlib import Path
from typing import ClassVar

import pygame
from pygame._sdl2 import controller

from .base import Edge, InputDevice, InputEvent
from .calibration import HOW_TO_CALIBRATE, Calibration, CalibrationError, Side, calibrations_dir
from .stick import Stick


class Joystick(InputDevice):
    """A controller SDL opened as joystick ``index``, read through its newest calibration."""

    #: its sticks, by the prefix of their controls' names, and each one's side in a calibration
    STICKS: ClassVar[dict[str, Side]] = {}

    def __init__(self, index: int, calibration: tuple[Path, Calibration] | None) -> None:
        """Open joystick ``index``.

        :param calibration: the newest calibration of the device and its file, or None.
        """
        self._open(index)
        self.device_name = self.joystick.get_name()
        self._buttons: set[str] = set()
        self._warned: set[int] = set()          # buttons that are no control, said once
        self._said_gone = False
        self.use(calibration)

    @classmethod
    @abstractmethod
    def accepts(cls, index: int) -> bool:
        """Whether SDL's joystick ``index`` is a device of this class."""

    def _open(self, index: int) -> None:
        """Take SDL's handle of joystick ``index``, and the id its events carry."""
        self.index = index
        self.joystick = pygame.joystick.Joystick(index)
        self.instance_id = self.joystick.get_instance_id()

    def _reopen(self) -> None:
        """Take the handle again after something restarted SDL's joystick subsystem.

        ``pygame.quit()`` (a game's ``close``, say) closes every handle, and the
        device comes back under a new id, or not at all, which is said once.
        """
        start_subsystems()
        for i in range(pygame.joystick.get_count()):
            if self.accepts(i) and pygame.joystick.Joystick(i).get_name() == self.device_name:
                self._open(i)
                return
        if not self._said_gone:
            self._said_gone = True
            print(f"{self.name}: {self.device_name!r} is gone; its controls press nothing",
                  file=sys.stderr)

    def use(self, calibration: tuple[Path, Calibration] | None) -> None:
        """Read the sticks through ``calibration`` (a file and its contents), or uncalibrated.

        A new calibration means a new centre to find: the one taken under the old is dropped.
        """
        self.calibration = calibration
        sticks = {} if calibration is None else calibration[1].sticks
        self.sticks: dict[Side, Stick] = {side: Stick(name, side, sticks.get(side))
                                          for name, side in self.STICKS.items()}

    def handle(self, events: list[pygame.event.Event], now: float) -> list[InputEvent]:
        if not self.joystick.get_init():
            self._reopen()
        out: list[InputEvent] = []
        moved: set[Side] = set()
        for event in events:
            if getattr(event, "instance_id", None) == self.instance_id:
                out += self._event(event, moved)
        for side in sorted(moved, key=lambda s: s.value):
            out += self.sticks[side].resolve(now)
        return out

    @abstractmethod
    def _event(self, event: pygame.event.Event, moved: set[Side]) -> list[InputEvent]:
        """The edges one of the device's events makes; a stick's axis it adds to ``moved``."""

    def _button(self, name: str | None, number: int, down: bool) -> list[InputEvent]:
        """A button's press or release; one that is no control is said once on stderr."""
        if name is None:
            if down and number not in self._warned:
                self._warned.add(number)
                print(f"{self.name}: {self.device_name}'s button {number} pressed, but it is "
                      f"no control of fmri_gym/inputs/{self.name}.py", file=sys.stderr)
            return []
        if down:
            self._buttons.add(name)
        else:
            self._buttons.discard(name)
        return [InputEvent(name, Edge.DOWN if down else Edge.UP)]

    def last_held(self) -> frozenset[str]:
        return frozenset(self._buttons).union(*(s.last_held() for s in self.sticks.values()))

    @abstractmethod
    def buttons(self) -> tuple[str, ...]:
        """Its button controls (no axis, no stick direction), in the order its table lists them."""

    @abstractmethod
    def read(self) -> tuple[dict[int, float], set[str]]:
        """Its stick axes now, raw, by :attr:`Side.axes`; and the buttons down now.

        Read off the handle rather than the events, for a measurement that must
        see a stick at rest, which sends none.
        """

    def require(self, needed: frozenset[str]) -> None:
        """Refuse a control it lacks, and a stick direction whose stick has no calibration.

        :raises ValueError: naming the controls the device has not.
        :raises CalibrationError: naming the stick, and how to calibrate it.
        """
        mine = needed & self.CONTROLS
        missing = sorted(mine - self.controls())
        if missing:
            raise ValueError(f"{self.name}: {self.device_name!r} has no {', '.join(missing)}, "
                             "which the curriculum names; play it on a rig with a device "
                             "that has them")
        for stick in self.sticks.values():
            if not mine & set(stick.directions) or stick.is_calibrated:
                continue
            where = (f"no calibration of {self.device_name!r} in {calibrations_dir()}"
                     if self.calibration is None else
                     f"the calibration in use, {self.calibration[0]}, has no {stick.side.value} "
                     "stick")
            raise CalibrationError(f"{where}, and the curriculum names {stick.name}'s "
                                   f"directions: {HOW_TO_CALIBRATE}")

    def status(self) -> str:
        sticks = [f"{s.name} at {s.tuning.rule()}" if s.is_calibrated
                  else f"{s.name} UNCALIBRATED" for s in self.sticks.values()]
        cal = (f"calibration {self.calibration[0]}" if self.calibration
               else f"no calibration of {self.device_name!r}")
        return "; ".join([f"{self.name}: {self.device_name!r} (joystick {self.index})",
                          *sticks, *([cal] if self.sticks else [])])


def start_subsystems() -> None:
    """Start SDL's joystick and game-controller subsystems, if they are not."""
    if not pygame.joystick.get_init():
        pygame.joystick.init()
    if not controller.get_init():
        controller.init()
