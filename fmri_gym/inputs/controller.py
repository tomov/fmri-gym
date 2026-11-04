"""A game controller: buttons and thumbsticks, under the names every kind of controller shares.

A curriculum names a controller's controls as below, whatever the controller
(``BUTTON_A``, ``LEFT_TRIGGER``, ``LEFT_STICK_UP``), so one written for a
controller plays on any other that has those controls. A kind of controller is
a subclass that maps what SDL sends to these names; a stick is read through
its own calibration file (:mod:`fmri_gym.inputs.calibration`) into four
direction controls. SDL's joystick subsystem is started here, since only a
controller needs it.
"""

from __future__ import annotations

import sys
import time
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ClassVar

import pygame
from pygame._sdl2 import controller as sdl_controller

from .base import InputDevice, InputEvent
from .calibration import (DIRECTIONS, DOWN, HOW_TO_CALIBRATE, LEFT, RIGHT, UNCALIBRATED, UP,
                          CalibrationError, StickCalibration, StickTuning, calibrations_dir,
                          newest)

# -- the control names every controller shares ---------------------------------------
BUTTON_A, BUTTON_B, BUTTON_X, BUTTON_Y = "BUTTON_A", "BUTTON_B", "BUTTON_X", "BUTTON_Y"
BUTTON_BACK, BUTTON_GUIDE, BUTTON_START = "BUTTON_BACK", "BUTTON_GUIDE", "BUTTON_START"
LEFT_SHOULDER, RIGHT_SHOULDER = "LEFT_SHOULDER", "RIGHT_SHOULDER"
LEFT_TRIGGER, RIGHT_TRIGGER = "LEFT_TRIGGER", "RIGHT_TRIGGER"
LEFT_STICK_PRESS, RIGHT_STICK_PRESS = "LEFT_STICK_PRESS", "RIGHT_STICK_PRESS"
DPAD_UP, DPAD_DOWN, DPAD_LEFT, DPAD_RIGHT = "DPAD_UP", "DPAD_DOWN", "DPAD_LEFT", "DPAD_RIGHT"
LEFT_STICK, RIGHT_STICK = "LEFT_STICK", "RIGHT_STICK"
LEFT_STICK_UP, LEFT_STICK_DOWN = "LEFT_STICK_UP", "LEFT_STICK_DOWN"
LEFT_STICK_LEFT, LEFT_STICK_RIGHT = "LEFT_STICK_LEFT", "LEFT_STICK_RIGHT"
RIGHT_STICK_UP, RIGHT_STICK_DOWN = "RIGHT_STICK_UP", "RIGHT_STICK_DOWN"
RIGHT_STICK_LEFT, RIGHT_STICK_RIGHT = "RIGHT_STICK_LEFT", "RIGHT_STICK_RIGHT"
#: Each stick's four direction controls: up, down, left, right.
STICK_DIRECTIONS: dict[str, tuple[str, str, str, str]] = {
    LEFT_STICK: (LEFT_STICK_UP, LEFT_STICK_DOWN, LEFT_STICK_LEFT, LEFT_STICK_RIGHT),
    RIGHT_STICK: (RIGHT_STICK_UP, RIGHT_STICK_DOWN, RIGHT_STICK_LEFT, RIGHT_STICK_RIGHT),
}
#: The face buttons, in the order a tool asks for them: round the face under a thumb.
FACE_BUTTONS = (BUTTON_A, BUTTON_X, BUTTON_Y, BUTTON_B)
#: How long SDL is given to announce a controller it sees late (macOS: about 0.1 s).
_SDL_DEVICE_ARRIVAL_S = 0.5


def short_name(control: str) -> str:
    """A control's name as a tile or cue shows it: a button's without its ``BUTTON_``."""
    return control.removeprefix("BUTTON_")


def start_sdl_joysticks_and_wait_for(expected_count: int) -> None:
    """Start SDL's joystick and game-controller subsystems, and wait for the controllers.

    On macOS SDL sees a controller only with its display subsystem up (which
    opens no window), and it arrives as an event about 0.1 s after that; and a
    stick's position is read only once events have been pumped.

    :param expected_count: how many controllers to wait for; with none, SDL is not waited on.
    """
    if not pygame.joystick.get_init():
        pygame.joystick.init()
    if not sdl_controller.get_init():
        sdl_controller.init()
    if not expected_count:
        return
    pygame.display.init()
    end = time.monotonic() + _SDL_DEVICE_ARRIVAL_S
    while time.monotonic() < end and pygame.joystick.get_count() < expected_count:
        pygame.event.pump()
        time.sleep(0.05)
    pygame.event.pump()


def sdl_joystick_names() -> list[str]:
    """The name of every joystick SDL sees, by its index."""
    return [pygame.joystick.Joystick(i).get_name() for i in range(pygame.joystick.get_count())]


@dataclass
class Stick:
    """One thumbstick: its four direction controls, where it is, and how it is read."""

    #: the controller's name for it (:data:`STICK_DIRECTIONS`)
    name: str
    up: str
    down: str
    left: str
    right: str
    #: its position as last read, raw, -1..1
    x: float = 0.0
    y: float = 0.0
    #: the calibration file it is read through, and its contents; None: uncalibrated
    calibration_path: Path | None = None
    calibration: StickCalibration | None = None
    #: how its push is read now (its calibration's; ``pad-calibration`` retunes it live)
    tuning: StickTuning = UNCALIBRATED

    def directions(self) -> tuple[str, str, str, str]:
        """Its four direction controls, up, down, left, right."""
        return self.up, self.down, self.left, self.right

    def control_of(self, direction: str) -> str:
        """The control for one of :data:`~.calibration.DIRECTIONS`."""
        return {UP: self.up, DOWN: self.down, LEFT: self.left, RIGHT: self.right}[direction]

    def is_calibrated(self) -> bool:
        return self.calibration is not None

    def position(self) -> tuple[float, float]:
        """Where it is, each axis a fraction of its travel that way, -1..1; raw when
        uncalibrated."""
        if self.calibration is None:
            return self.x, self.y
        return self.calibration.x.normalise(self.x), self.calibration.y.normalise(self.y)


class Controller(InputDevice):
    """A game controller SDL opens: its buttons, and its sticks read through their calibrations.

    A subclass maps SDL's button and axis numbers to the shared names, opens
    the handle SDL gives it (:attr:`HANDLE`) and reads its events in its
    :meth:`handle`, pressing buttons with :meth:`_press`, moving sticks with
    :meth:`_move_stick` and ending with :meth:`_read_sticks`.
    """

    #: its buttons, in the order a tool asks for them
    BUTTON_NAMES: ClassVar[tuple[str, ...]]
    #: its sticks, by name (:data:`STICK_DIRECTIONS`)
    STICK_NAMES: ClassVar[tuple[str, ...]]
    #: what SDL opens it as, by SDL's joystick index
    HANDLE: ClassVar[Callable[[int], Any]]

    def __init__(self, handle: Any, calibrated: bool = True) -> None:
        """Read the controller SDL opened as ``handle``.

        :param calibrated: read each stick through its newest calibration file;
            ``False`` reads them uncalibrated, for a tool that shows the error itself.
        :raises CalibrationError: when a calibration file is malformed.
        """
        super().__init__()
        self.sticks = [Stick(name, *STICK_DIRECTIONS[name]) for name in self.STICK_NAMES]
        self._since: dict[str, float] = {}      # a direction -> when its stick first asked for it
        self._said_gone = False
        self.device_name = ""
        self.attach(handle)
        if calibrated:
            self.load_stick_calibrations()

    # -- SDL: which joystick it is -----------------------------------------------------
    @classmethod
    @abstractmethod
    def accepts(cls, joystick: pygame.joystick.Joystick, is_controller: bool) -> bool:
        """Whether SDL's ``joystick`` is a controller of this kind.

        :param is_controller: whether SDL has a game-controller mapping for it.
        """

    @classmethod
    def plugged_in_indices(cls) -> list[int]:
        """SDL's joystick indices of the controllers of this kind plugged in now."""
        start_sdl_joysticks_and_wait_for(1)
        return [i for i in range(pygame.joystick.get_count())
                if cls.accepts(pygame.joystick.Joystick(i), sdl_controller.is_controller(i))]

    @classmethod
    def open(cls, calibrated: bool = True) -> Controller:
        """The first controller of this kind plugged in.

        :raises ValueError: when none is.
        :raises CalibrationError: when a calibration file is malformed.
        """
        indices = cls.plugged_in_indices()
        if not indices:
            raise ValueError(f'the rig lists "{cls.name}" under "inputs", but none is plugged '
                             f"in (joysticks SDL sees: {sdl_joystick_names() or 'none'}): plug "
                             'it in, or take it out of the rig\'s "inputs"')
        return cls(cls.HANDLE(indices[0]), calibrated)

    @abstractmethod
    def attach(self, handle: Any) -> None:
        """Read through ``handle`` from now on: set :attr:`device_name`, remember the id
        its events carry, and read where its sticks are."""

    @abstractmethod
    def is_open(self) -> bool:
        """Whether the handle it reads through is still open."""

    def _reopen_if_closed(self) -> None:
        """A game that quits pygame closes every joystick handle: take a new one, or say
        once that the controller is gone."""
        if self.is_open():
            return
        for index in self.plugged_in_indices():
            if pygame.joystick.Joystick(index).get_name() == self.device_name:
                self.attach(self.HANDLE(index))
                return
        if not self._said_gone:
            self._said_gone = True
            print(f"{self.name}: {self.device_name!r} is gone; its controls press nothing",
                  file=sys.stderr)

    # -- the sticks and their calibrations ---------------------------------------------
    def stick(self, name: str) -> Stick:
        """The stick called ``name`` (:data:`STICK_DIRECTIONS`)."""
        return next(s for s in self.sticks if s.name == name)

    def load_stick_calibrations(self) -> None:
        """Read each stick through the newest calibration file of it, if there is one.

        :raises CalibrationError: when a calibration file is malformed.
        """
        for stick in self.sticks:
            self.set_stick_calibration(stick, newest(self.device_name, stick.name))

    def set_stick_calibration(self, stick: Stick,
                              calibration_file: tuple[Path, StickCalibration] | None) -> None:
        """Read ``stick`` through ``calibration_file`` (its path and its contents), or
        uncalibrated when None."""
        if calibration_file is None:
            stick.calibration_path, stick.calibration, stick.tuning = None, None, UNCALIBRATED
            return
        stick.calibration_path, stick.calibration = calibration_file
        stick.tuning = stick.calibration.tuning

    def require_calibrated_sticks_for(self, required_controls: frozenset[str]) -> None:
        """Refuse to play a curriculum that names a direction of an uncalibrated stick.

        :raises CalibrationError: naming the stick and how to calibrate it.
        """
        for stick in self.sticks:
            if stick.is_calibrated() or not required_controls & set(stick.directions()):
                continue
            raise CalibrationError(
                f"no calibration of {self.device_name!r}'s {stick.name} in "
                f"{calibrations_dir()}, and the curriculum names its directions: "
                f"{HOW_TO_CALIBRATE}")

    def status(self) -> str:
        """One line: the controller, and each stick's rule and calibration file."""
        sticks = [f"{s.name} at {s.tuning.rule()} ({s.calibration_path})" if s.is_calibrated()
                  else f"{s.name} UNCALIBRATED" for s in self.sticks]
        return "; ".join([f"{self.name}: {self.device_name!r}", *sticks])

    # -- for a subclass's handle() -----------------------------------------------------
    def _move_stick(self, stick_name: str, axis_name: str, value: float) -> None:
        """Take a raw reading of a stick's ``"x"`` or ``"y"``; :meth:`_read_sticks` reads it."""
        setattr(self.stick(stick_name), axis_name, value)

    def _read_sticks(self, now: float) -> list[InputEvent]:
        """Press and release each stick's directions by where it is now.

        A release is immediate; a press waits until the direction has been
        asked for through the tuning's dwell, so a thumb crossing a neighbouring
        direction on its way out presses nothing.
        """
        out: list[InputEvent] = []
        for stick in self.sticks:
            held = {d for d in DIRECTIONS if stick.control_of(d) in self._held}
            wanted = {stick.control_of(d)
                      for d in stick.tuning.directions(*stick.position(), held)}
            pressed = self._directions_pressed_after_dwell(wanted, set(stick.directions()),
                                                           stick.tuning.dwell_s, now)
            out += [e for n in stick.directions() for e in self._press(n, n in pressed)]
        return out

    def _directions_pressed_after_dwell(self, wanted_directions: set[str],
                                        directions_of_stick: set[str], dwell_s: float,
                                        now: float) -> set[str]:
        """Of ``wanted_directions``, those held already or asked for since ``dwell_s`` ago."""
        for name in [n for n in self._since if n in directions_of_stick
                     and n not in wanted_directions]:
            del self._since[name]
        for name in wanted_directions:
            self._since.setdefault(name, now)
        return {n for n in wanted_directions
                if n in self._held or now - self._since[n] >= dwell_s}
