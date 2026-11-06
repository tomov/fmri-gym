"""Current Designs' GMPD3 fMRI gamepad on its 932 interface in joystick mode.

Four face buttons, two index-finger buttons and one stick, under the names an
Xbox controller gives the same places. SDL knows no mapping for it, so it is
the joystick SDL does not take for a game controller, with at least the buttons
and axes below. SDL numbers its buttons from 0, one less than the HID report.
It also reports two axes pinned at -1.0, which are no control. In the
interface's keyboard mode the pad types keys, and is the keyboard.
"""

from __future__ import annotations

import pygame

from .base import InputEvent
from .control_names import ControlNames
from .controller import AXIS_CONTROLS, Controller

#: SDL's button number -> the control.
BUTTONS = {
    0: ControlNames.CONTROLLER_BUTTON_Y,
    1: ControlNames.CONTROLLER_BUTTON_B,
    2: ControlNames.CONTROLLER_BUTTON_X,
    3: ControlNames.CONTROLLER_AXIS_TRIGGERLEFT,
    4: ControlNames.CONTROLLER_BUTTON_A,
    5: ControlNames.CONTROLLER_AXIS_TRIGGERRIGHT,
}
#: SDL's axis number -> the stick axis it is.
STICK_AXES = {
    0: ControlNames.CONTROLLER_AXIS_LEFTX,
    1: ControlNames.CONTROLLER_AXIS_LEFTY
}


class CurrentDesignsGamepad3(Controller):
    """The GMPD3: four face buttons, two index-finger buttons, one stick's two axes."""

    name = "current_designs_gamepad3"
    BUTTON_NAMES = tuple(BUTTONS.values())
    AXIS_NAMES = tuple(STICK_AXES.values())
    CONTROLS = frozenset({*BUTTON_NAMES, *(c for a in AXIS_NAMES for c in AXIS_CONTROLS[a])})
    HANDLE = pygame.joystick.Joystick

    @classmethod
    def accepts(cls, joystick: pygame.joystick.Joystick, is_controller: bool) -> bool:
        """A joystick SDL has no mapping for, with enough buttons and axes."""
        return (not is_controller and joystick.get_numbuttons() >= len(BUTTONS)
                and joystick.get_numaxes() >= len(STICK_AXES))

    def attach(self, handle: pygame.joystick.Joystick) -> None:
        self.joystick, self.instance_id = handle, handle.get_instance_id()
        self.device_name = handle.get_name()
        for axis, axis_name in STICK_AXES.items():
            self.axis_values[axis_name] = handle.get_axis(axis)

    def is_open(self) -> bool:
        return self.joystick.get_init()

    def handle(self, events: list[pygame.event.Event], now: float) -> list[InputEvent]:
        self._reopen_if_closed()
        out: list[InputEvent] = []
        for event in events:
            if getattr(event, "instance_id", None) != self.instance_id:
                continue
            if event.type == pygame.JOYBUTTONDOWN and event.button in BUTTONS:
                out += self._press(BUTTONS[event.button], True)
            elif event.type == pygame.JOYBUTTONUP and event.button in BUTTONS:
                out += self._press(BUTTONS[event.button], False)
            elif event.type == pygame.JOYAXISMOTION and event.axis in STICK_AXES:
                out += self._move_axis(STICK_AXES[event.axis], event.value)
        return out

    def close(self) -> None:
        if self.joystick.get_init():
            self.joystick.quit()
