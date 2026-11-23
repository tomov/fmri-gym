"""An Xbox-layout controller SDL has a mapping for (Xbox, PlayStation, 8BitDo...).

Read through SDL's game-controller layer, whose numbers are the same for every
such pad, OS and connection; the raw joystick events SDL sends beside them are
not this device's and go unread. Each analog trigger is a button, pressed past
half its travel, whose value is how far it is pulled; each stick axis is two
controls and the axis itself, read through its calibration.
"""

from __future__ import annotations

import pygame
from pygame._sdl2 import controller as sdl_controller

from .base import InputEvent
from .control_names import ControlNames
from .controller import AXIS_CONTROLS, Controller

#: SDL's button number -> the control.
BUTTONS = {
    pygame.CONTROLLER_BUTTON_A: ControlNames.CONTROLLER_BUTTON_A,
    pygame.CONTROLLER_BUTTON_B: ControlNames.CONTROLLER_BUTTON_B,
    pygame.CONTROLLER_BUTTON_X: ControlNames.CONTROLLER_BUTTON_X,
    pygame.CONTROLLER_BUTTON_Y: ControlNames.CONTROLLER_BUTTON_Y,
    pygame.CONTROLLER_BUTTON_BACK: ControlNames.CONTROLLER_BUTTON_BACK,
    pygame.CONTROLLER_BUTTON_GUIDE: ControlNames.CONTROLLER_BUTTON_GUIDE,
    pygame.CONTROLLER_BUTTON_START: ControlNames.CONTROLLER_BUTTON_START,
    pygame.CONTROLLER_BUTTON_LEFTSTICK: ControlNames.CONTROLLER_BUTTON_LEFTSTICK,
    pygame.CONTROLLER_BUTTON_RIGHTSTICK: ControlNames.CONTROLLER_BUTTON_RIGHTSTICK,
    pygame.CONTROLLER_BUTTON_LEFTSHOULDER: ControlNames.CONTROLLER_BUTTON_LEFTSHOULDER,
    pygame.CONTROLLER_BUTTON_RIGHTSHOULDER: ControlNames.CONTROLLER_BUTTON_RIGHTSHOULDER,
    pygame.CONTROLLER_BUTTON_DPAD_UP: ControlNames.CONTROLLER_BUTTON_DPAD_UP,
    pygame.CONTROLLER_BUTTON_DPAD_DOWN: ControlNames.CONTROLLER_BUTTON_DPAD_DOWN,
    pygame.CONTROLLER_BUTTON_DPAD_LEFT: ControlNames.CONTROLLER_BUTTON_DPAD_LEFT,
    pygame.CONTROLLER_BUTTON_DPAD_RIGHT: ControlNames.CONTROLLER_BUTTON_DPAD_RIGHT,
}
#: SDL's axis number -> the trigger it is, read as a button.
TRIGGERS = {
    pygame.CONTROLLER_AXIS_TRIGGERLEFT: ControlNames.CONTROLLER_AXIS_TRIGGERLEFT,
    pygame.CONTROLLER_AXIS_TRIGGERRIGHT: ControlNames.CONTROLLER_AXIS_TRIGGERRIGHT,
}
#: SDL's axis number -> the stick axis it is.
STICK_AXES = {
    pygame.CONTROLLER_AXIS_LEFTX: ControlNames.CONTROLLER_AXIS_LEFTX,
    pygame.CONTROLLER_AXIS_LEFTY: ControlNames.CONTROLLER_AXIS_LEFTY,
    pygame.CONTROLLER_AXIS_RIGHTX: ControlNames.CONTROLLER_AXIS_RIGHTX,
    pygame.CONTROLLER_AXIS_RIGHTY: ControlNames.CONTROLLER_AXIS_RIGHTY,
}
#: The stick axis -> SDL's number for it.
_SDL_AXIS_BY_NAME = {name: axis for axis, name in STICK_AXES.items()}
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
    CONTROLS = frozenset({*BUTTON_NAMES, *AXIS_NAMES,
                          *(c for a in AXIS_NAMES for c in AXIS_CONTROLS[a])})
    HANDLE = sdl_controller.Controller

    @classmethod
    def accepts(cls, joystick: pygame.joystick.Joystick, is_controller: bool) -> bool:
        """Any joystick SDL has a game-controller mapping for."""
        return is_controller

    def attach(self, handle: sdl_controller.Controller) -> None:
        self.device_handle, self.device_name = handle, handle.name
        # handle.id is SDL's device index; the events carry the joystick's instance id.
        self.instance_id = handle.as_joystick().get_instance_id()

    def _read_axis(self, axis_name: str) -> float:
        return _scaled(self.device_handle.get_axis(_SDL_AXIS_BY_NAME[axis_name]))

    def _take_axis_readings(self) -> list[InputEvent]:
        """The sticks' readings (:meth:`Controller._take_axis_readings`), then the triggers'."""
        return super()._take_axis_readings() + [
            e for axis, trigger in TRIGGERS.items()
            for e in self._move_trigger(trigger, _scaled(self.device_handle.get_axis(axis)))]

    def _move_trigger(self, trigger: str, axis_reading: float) -> list[InputEvent]:
        """Set a trigger's value, how far it is pulled (0..1), and press or release it."""
        pulled = max(0.0, axis_reading)
        self._set(trigger, pulled)
        return self._pull(trigger, pulled, _TRIGGER_ON, _TRIGGER_OFF)

    def handle(self, events: list[pygame.event.Event], now: float) -> list[InputEvent]:
        out = self._reopen_if_closed()
        for event in events:
            if getattr(event, "instance_id", None) != self.instance_id:
                continue
            if event.type == pygame.CONTROLLERBUTTONDOWN and event.button in BUTTONS:
                out += self._press(BUTTONS[event.button])
            elif event.type == pygame.CONTROLLERBUTTONUP and event.button in BUTTONS:
                out += self._release(BUTTONS[event.button])
            elif event.type == pygame.CONTROLLERAXISMOTION and event.axis in TRIGGERS:
                out += self._move_trigger(TRIGGERS[event.axis], _scaled(event.value))
            elif event.type == pygame.CONTROLLERAXISMOTION and event.axis in STICK_AXES:
                out += self._move_axis(STICK_AXES[event.axis], _scaled(event.value))
        return out
