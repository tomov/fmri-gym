"""An Xbox-layout controller SDL has a mapping for (Xbox, PlayStation, 8BitDo...).

Read through SDL's game-controller layer, whose numbers are the same for every
such pad, OS and connection; the raw joystick events SDL sends beside them are
not this device's and go unread. Each analog trigger is a button, pressed past
half its travel.
"""

from __future__ import annotations

import pygame
from pygame._sdl2 import controller as sdl_controller

from .base import InputEvent
from .controller import (BUTTON_A, BUTTON_B, BUTTON_BACK, BUTTON_GUIDE, BUTTON_START, BUTTON_X,
                         BUTTON_Y, DPAD_DOWN, DPAD_LEFT, DPAD_RIGHT, DPAD_UP, LEFT_SHOULDER,
                         LEFT_STICK, LEFT_STICK_PRESS, LEFT_TRIGGER, RIGHT_SHOULDER, RIGHT_STICK,
                         RIGHT_STICK_PRESS, RIGHT_TRIGGER, STICK_DIRECTIONS, Controller)

#: SDL's button number -> the control.
BUTTONS = {
    pygame.CONTROLLER_BUTTON_A: BUTTON_A,
    pygame.CONTROLLER_BUTTON_B: BUTTON_B,
    pygame.CONTROLLER_BUTTON_X: BUTTON_X,
    pygame.CONTROLLER_BUTTON_Y: BUTTON_Y,
    pygame.CONTROLLER_BUTTON_BACK: BUTTON_BACK,
    pygame.CONTROLLER_BUTTON_GUIDE: BUTTON_GUIDE,
    pygame.CONTROLLER_BUTTON_START: BUTTON_START,
    pygame.CONTROLLER_BUTTON_LEFTSTICK: LEFT_STICK_PRESS,
    pygame.CONTROLLER_BUTTON_RIGHTSTICK: RIGHT_STICK_PRESS,
    pygame.CONTROLLER_BUTTON_LEFTSHOULDER: LEFT_SHOULDER,
    pygame.CONTROLLER_BUTTON_RIGHTSHOULDER: RIGHT_SHOULDER,
    pygame.CONTROLLER_BUTTON_DPAD_UP: DPAD_UP,
    pygame.CONTROLLER_BUTTON_DPAD_DOWN: DPAD_DOWN,
    pygame.CONTROLLER_BUTTON_DPAD_LEFT: DPAD_LEFT,
    pygame.CONTROLLER_BUTTON_DPAD_RIGHT: DPAD_RIGHT,
}
#: SDL's axis number -> the trigger it is, read as a button.
TRIGGERS = {
    pygame.CONTROLLER_AXIS_TRIGGERLEFT: LEFT_TRIGGER,
    pygame.CONTROLLER_AXIS_TRIGGERRIGHT: RIGHT_TRIGGER,
}
#: SDL's axis number -> the stick it is, and whether it is the stick's x or y.
STICK_AXES = {
    pygame.CONTROLLER_AXIS_LEFTX: (LEFT_STICK, "x"),
    pygame.CONTROLLER_AXIS_LEFTY: (LEFT_STICK, "y"),
    pygame.CONTROLLER_AXIS_RIGHTX: (RIGHT_STICK, "x"),
    pygame.CONTROLLER_AXIS_RIGHTY: (RIGHT_STICK, "y"),
}
#: How far a trigger is pulled to press, and let go to release, of its travel.
_TRIGGER_ON, _TRIGGER_OFF = 0.5, 0.3


def _scaled(value: float) -> float:
    """A game-controller axis's reading, -32768..32767, as -1..1."""
    return max(-1.0, min(1.0, value / 32767))


class XBoxController(Controller):
    """An Xbox-layout controller: its buttons, its triggers as buttons, and both sticks."""

    name = "xbox_controller"
    BUTTON_NAMES = (*BUTTONS.values(), *TRIGGERS.values())
    STICK_NAMES = (LEFT_STICK, RIGHT_STICK)
    CONTROLS = frozenset({*BUTTON_NAMES, *STICK_DIRECTIONS[LEFT_STICK],
                          *STICK_DIRECTIONS[RIGHT_STICK]})
    HANDLE = sdl_controller.Controller

    @classmethod
    def accepts(cls, joystick: pygame.joystick.Joystick, is_controller: bool) -> bool:
        """Any joystick SDL has a game-controller mapping for."""
        return is_controller

    def attach(self, handle: sdl_controller.Controller) -> None:
        self.controller, self.instance_id, self.device_name = handle, handle.id, handle.name
        for axis, (stick_name, axis_name) in STICK_AXES.items():
            self._move_stick(stick_name, axis_name, _scaled(handle.get_axis(axis)))

    def is_open(self) -> bool:
        return self.controller.get_init()

    def handle(self, events: list[pygame.event.Event], now: float) -> list[InputEvent]:
        self._reopen_if_closed()
        out: list[InputEvent] = []
        for event in events:
            if getattr(event, "instance_id", None) != self.instance_id:
                continue
            if event.type == pygame.CONTROLLERBUTTONDOWN and event.button in BUTTONS:
                out += self._press(BUTTONS[event.button], True)
            elif event.type == pygame.CONTROLLERBUTTONUP and event.button in BUTTONS:
                out += self._press(BUTTONS[event.button], False)
            elif event.type == pygame.CONTROLLERAXISMOTION and event.axis in TRIGGERS:
                out += self._pull_trigger(TRIGGERS[event.axis], _scaled(event.value))
            elif event.type == pygame.CONTROLLERAXISMOTION and event.axis in STICK_AXES:
                stick_name, axis_name = STICK_AXES[event.axis]
                self._move_stick(stick_name, axis_name, _scaled(event.value))
        return out + self._read_sticks(now)

    def _pull_trigger(self, name: str, value: float) -> list[InputEvent]:
        """Press a trigger past :data:`_TRIGGER_ON`, release it under :data:`_TRIGGER_OFF`."""
        if value >= _TRIGGER_ON:
            return self._press(name, True)
        if value < _TRIGGER_OFF:
            return self._press(name, False)
        return []

    def close(self) -> None:
        if self.controller.get_init():
            self.controller.quit()
