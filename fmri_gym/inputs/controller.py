"""A game controller: buttons, triggers and stick axes, under the names every kind of controller
shares.

A curriculum names a controller's controls as below, whatever the controller
(``BUTTON_A``, ``LEFT_TRIGGER``, ``LEFT_STICK_UP``), so one written for a
controller plays on any other that has those controls. A kind of controller is
a subclass that maps what SDL sends to these names. Each stick axis is two
controls, one per side of rest, read like a trigger through the axis's own
calibration (:mod:`fmri_gym.inputs.calibration`), and independent of every
other axis. SDL's joystick subsystem is started here, since only a controller
needs it.
"""

from __future__ import annotations

import sys
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, ClassVar

import pygame
from pygame._sdl2 import controller as sdl_controller

from .base import InputDevice, InputEvent
from .calibration import (HOW_TO_CALIBRATE, AxisCalibration, CalibrationError,
                          ControllerCalibration, calibrations_dir, newest)

# -- the control names every controller shares ---------------------------------------
BUTTON_A, BUTTON_B, BUTTON_X, BUTTON_Y = "BUTTON_A", "BUTTON_B", "BUTTON_X", "BUTTON_Y"
BUTTON_BACK, BUTTON_GUIDE, BUTTON_START = "BUTTON_BACK", "BUTTON_GUIDE", "BUTTON_START"
LEFT_SHOULDER, RIGHT_SHOULDER = "LEFT_SHOULDER", "RIGHT_SHOULDER"
LEFT_TRIGGER, RIGHT_TRIGGER = "LEFT_TRIGGER", "RIGHT_TRIGGER"
LEFT_STICK_PRESS, RIGHT_STICK_PRESS = "LEFT_STICK_PRESS", "RIGHT_STICK_PRESS"
DPAD_UP, DPAD_DOWN, DPAD_LEFT, DPAD_RIGHT = "DPAD_UP", "DPAD_DOWN", "DPAD_LEFT", "DPAD_RIGHT"
LEFT_STICK_UP, LEFT_STICK_DOWN = "LEFT_STICK_UP", "LEFT_STICK_DOWN"
LEFT_STICK_LEFT, LEFT_STICK_RIGHT = "LEFT_STICK_LEFT", "LEFT_STICK_RIGHT"
RIGHT_STICK_UP, RIGHT_STICK_DOWN = "RIGHT_STICK_UP", "RIGHT_STICK_DOWN"
RIGHT_STICK_LEFT, RIGHT_STICK_RIGHT = "RIGHT_STICK_LEFT", "RIGHT_STICK_RIGHT"
# -- the stick axes, as a calibration file names them ----------------------------------
LEFT_STICK_X, LEFT_STICK_Y = "LEFT_STICK_X", "LEFT_STICK_Y"
RIGHT_STICK_X, RIGHT_STICK_Y = "RIGHT_STICK_X", "RIGHT_STICK_Y"
#: Each stick axis's two controls: the one its negative side presses, then its positive side's.
AXIS_CONTROLS: dict[str, tuple[str, str]] = {
    LEFT_STICK_X: (LEFT_STICK_LEFT, LEFT_STICK_RIGHT),
    LEFT_STICK_Y: (LEFT_STICK_UP, LEFT_STICK_DOWN),
    RIGHT_STICK_X: (RIGHT_STICK_LEFT, RIGHT_STICK_RIGHT),
    RIGHT_STICK_Y: (RIGHT_STICK_UP, RIGHT_STICK_DOWN),
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
    opens no window), and it arrives as an event about 0.1 s after that; and an
    axis's position is read only once events have been pumped.

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


class Controller(InputDevice):
    """A game controller SDL opens: its buttons, and its stick axes read through its calibration.

    A subclass maps SDL's button and axis numbers to the shared names, opens
    the handle SDL gives it (:attr:`HANDLE`) and reads its events in its
    :meth:`handle`, pressing buttons with :meth:`_press`, triggers with
    :meth:`_pull` and moving stick axes with :meth:`_move_axis`.
    """

    #: its buttons, in the order a tool asks for them
    BUTTON_NAMES: ClassVar[tuple[str, ...]]
    #: its stick axes (:data:`AXIS_CONTROLS`)
    AXIS_NAMES: ClassVar[tuple[str, ...]]
    #: what SDL opens it as, by SDL's joystick index
    HANDLE: ClassVar[Callable[[int], Any]]

    def __init__(self, handle: Any, calibrated: bool = True) -> None:
        """Read the controller SDL opened as ``handle``.

        :param calibrated: read the axes through its newest calibration file;
            ``False`` leaves it uncalibrated, for a tool that shows the error itself.
        :raises CalibrationError: when a calibration file is malformed.
        """
        super().__init__()
        #: each axis's last raw reading, -1..1
        self.axis_values = {name: 0.0 for name in self.AXIS_NAMES}
        #: the calibration file the axes are read through, and its contents; None: uncalibrated
        self.calibration_path: Path | None = None
        self.calibration: ControllerCalibration | None = None
        self._said_gone = False
        self.device_name = ""
        self.attach(handle)
        if calibrated:
            self.load_calibration()

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
        its events carry, and read where its axes are."""

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

    # -- the axes and their calibration ------------------------------------------------
    def axis_controls(self) -> tuple[str, ...]:
        """The two controls of each of its axes, in :attr:`AXIS_NAMES` order."""
        return tuple(c for name in self.AXIS_NAMES for c in AXIS_CONTROLS[name])

    def load_calibration(self) -> None:
        """Read the axes through the newest calibration file of this controller, if there is one.

        :raises CalibrationError: when a calibration file is malformed, or is not of these axes.
        """
        self.set_calibration(newest(self.device_name))

    def set_calibration(self,
                        calibration_file: tuple[Path, ControllerCalibration] | None) -> None:
        """Read the axes through ``calibration_file`` (its path and its contents), or
        uncalibrated when None.

        :raises CalibrationError: when it does not calibrate exactly this controller's axes.
        """
        if calibration_file is None:
            self.calibration_path, self.calibration = None, None
            return
        path, calibration = calibration_file
        if set(calibration.axes) != set(self.AXIS_NAMES):
            raise CalibrationError(f"{path}: calibrates the axes {sorted(calibration.axes)}, but "
                                   f"{self.device_name!r} has {sorted(self.AXIS_NAMES)}: "
                                   f"{HOW_TO_CALIBRATE}")
        self.calibration_path, self.calibration = path, calibration

    def require_calibrated_axes_for(self, required_controls: frozenset[str]) -> None:
        """Refuse to play a curriculum that names a control of an axis with no calibration.

        :raises CalibrationError: naming the controller and how to calibrate it.
        """
        if self.calibration is not None or not required_controls & set(self.axis_controls()):
            return
        raise CalibrationError(f"no calibration of {self.device_name!r} in {calibrations_dir()}, "
                               f"and the curriculum names its stick controls: {HOW_TO_CALIBRATE}")

    def calibrated_value(self, axis_name: str) -> float | None:
        """Where an axis is, as a fraction of its travel that way, -1..1; None uncalibrated."""
        if self.calibration is None:
            return None
        return self.calibration.axes[axis_name].normalise(self.axis_values[axis_name])

    def status(self) -> str:
        """One line: the controller, and the calibration its axes are read through."""
        if not self.AXIS_NAMES:
            return f"{self.name}: {self.device_name!r}"
        if self.calibration is None:
            return f"{self.name}: {self.device_name!r}; axes UNCALIBRATED"
        axes = ", ".join(f"{name} {axis.on:.2f}/{axis.off:.2f}"
                         for name, axis in self.calibration.axes.items())
        return f"{self.name}: {self.device_name!r}; {axes} ({self.calibration_path})"

    # -- for a subclass's handle() -----------------------------------------------------
    def _pull(self, name: str, value: float, on: float, off: float) -> list[InputEvent]:
        """Press ``name`` once ``value`` reaches ``on``, release it under ``off``."""
        if value >= on:
            return self._press(name, True)
        if value < off:
            return self._press(name, False)
        return []

    def _move_axis(self, axis_name: str, value: float) -> list[InputEvent]:
        """Take a raw reading of an axis, and press or release each of its sides by it.

        An uncalibrated axis presses nothing.
        """
        self.axis_values[axis_name] = value
        if self.calibration is None:
            return []
        axis: AxisCalibration = self.calibration.axes[axis_name]
        pushed = axis.normalise(value)
        negative, positive = AXIS_CONTROLS[axis_name]
        return (self._pull(negative, -pushed, axis.on, axis.off)
                + self._pull(positive, pushed, axis.on, axis.off))
