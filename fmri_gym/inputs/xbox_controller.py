"""An Xbox-layout controller SDL has a mapping for (Xbox, PlayStation, 8BitDo...).

Read through SDL's game-controller layer, whose numbers are the same for every
such pad, OS and connection; the raw joystick events SDL sends beside them are
not this device's and go unread. Each analog trigger is a button, pressed past
half its travel; each stick axis is two controls, read through its calibration.
"""

from __future__ import annotations

import pygame
from pygame._sdl2 import controller as sdl_controller

from .base import InputEvent
from .controller import (AXIS_CONTROLS, BUTTON_A, BUTTON_B, BUTTON_BACK, BUTTON_GUIDE,
                         BUTTON_START, BUTTON_X, BUTTON_Y, DPAD_DOWN, DPAD_LEFT, DPAD_RIGHT,
                         DPAD_UP, LEFT_SHOULDER, LEFT_STICK_PRESS, LEFT_STICK_X, LEFT_STICK_Y,
                         LEFT_TRIGGER, RIGHT_SHOULDER, RIGHT_STICK_PRESS, RIGHT_STICK_X,
                         RIGHT_STICK_Y, RIGHT_TRIGGER, Controller)

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
#: SDL's axis number -> the stick axis it is.
STICK_AXES = {
    pygame.CONTROLLER_AXIS_LEFTX: LEFT_STICK_X,
    pygame.CONTROLLER_AXIS_LEFTY: LEFT_STICK_Y,
    pygame.CONTROLLER_AXIS_RIGHTX: RIGHT_STICK_X,
    pygame.CONTROLLER_AXIS_RIGHTY: RIGHT_STICK_Y,
}
#: How far a trigger is pulled to press, and let go to release, of its travel.
_TRIGGER_ON, _TRIGGER_OFF = 0.5, 0.3


def _scaled(value: float) -> float:
    """A game-controller axis's reading, -32768..32767, as -1..1."""
    return max(-1.0, min(1.0, value / 32767))


class XBoxController(Controller):
    """An Xbox-layout controller: its buttons, its triggers as buttons, and both sticks' axes."""

    name = "xbox_controller"
    BUTTON_NAMES = (*BUTTONS.values(), *TRIGGERS.values())
    AXIS_NAMES = tuple(STICK_AXES.values())
    CONTROLS = frozenset({*BUTTON_NAMES, *(c for a in AXIS_NAMES for c in AXIS_CONTROLS[a])})
    HANDLE = sdl_controller.Controller

    @classmethod
    def accepts(cls, joystick: pygame.joystick.Joystick, is_controller: bool) -> bool:
        """Any joystick SDL has a game-controller mapping for."""
        return is_controller

    def attach(self, handle: sdl_controller.Controller) -> None:
        self.controller, self.instance_id, self.device_name = handle, handle.id, handle.name
        for axis, axis_name in STICK_AXES.items():
            self.axis_values[axis_name] = _scaled(handle.get_axis(axis))

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
                out += self._pull(TRIGGERS[event.axis], _scaled(event.value),
                                  _TRIGGER_ON, _TRIGGER_OFF)
            elif event.type == pygame.CONTROLLERAXISMOTION and event.axis in STICK_AXES:
                out += self._move_axis(STICK_AXES[event.axis], _scaled(event.value))
        return out

    def close(self) -> None:
        if self.controller.get_init():
            self.controller.quit()
