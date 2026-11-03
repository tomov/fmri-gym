"""A gamepad SDL has a mapping for (Xbox, PlayStation, 8BitDo...), by SDL's names for its controls.

Read through SDL's game-controller layer, whose numbers are the same for every
such pad, OS and connection; the raw joystick events SDL sends beside them are
ignored. Each analog trigger is also a button, pressed past half its travel.
"""

from __future__ import annotations

import pygame
from pygame._sdl2 import controller

from .base import Edge, InputEvent
from .calibration import Side
from .joystick import Joystick
from .stick import directions_of

BUTTONS = {
    pygame.CONTROLLER_BUTTON_A: "CONTROLLER_BUTTON_A",
    pygame.CONTROLLER_BUTTON_B: "CONTROLLER_BUTTON_B",
    pygame.CONTROLLER_BUTTON_X: "CONTROLLER_BUTTON_X",
    pygame.CONTROLLER_BUTTON_Y: "CONTROLLER_BUTTON_Y",
    pygame.CONTROLLER_BUTTON_BACK: "CONTROLLER_BUTTON_BACK",
    pygame.CONTROLLER_BUTTON_GUIDE: "CONTROLLER_BUTTON_GUIDE",
    pygame.CONTROLLER_BUTTON_START: "CONTROLLER_BUTTON_START",
    pygame.CONTROLLER_BUTTON_LEFTSTICK: "CONTROLLER_BUTTON_LEFTSTICK",
    pygame.CONTROLLER_BUTTON_RIGHTSTICK: "CONTROLLER_BUTTON_RIGHTSTICK",
    pygame.CONTROLLER_BUTTON_LEFTSHOULDER: "CONTROLLER_BUTTON_LEFTSHOULDER",
    pygame.CONTROLLER_BUTTON_RIGHTSHOULDER: "CONTROLLER_BUTTON_RIGHTSHOULDER",
    pygame.CONTROLLER_BUTTON_DPAD_UP: "CONTROLLER_BUTTON_DPAD_UP",
    pygame.CONTROLLER_BUTTON_DPAD_DOWN: "CONTROLLER_BUTTON_DPAD_DOWN",
    pygame.CONTROLLER_BUTTON_DPAD_LEFT: "CONTROLLER_BUTTON_DPAD_LEFT",
    pygame.CONTROLLER_BUTTON_DPAD_RIGHT: "CONTROLLER_BUTTON_DPAD_RIGHT",
}
AXES = {
    pygame.CONTROLLER_AXIS_LEFTX: "CONTROLLER_AXIS_LEFTX",
    pygame.CONTROLLER_AXIS_LEFTY: "CONTROLLER_AXIS_LEFTY",
    pygame.CONTROLLER_AXIS_RIGHTX: "CONTROLLER_AXIS_RIGHTX",
    pygame.CONTROLLER_AXIS_RIGHTY: "CONTROLLER_AXIS_RIGHTY",
    pygame.CONTROLLER_AXIS_TRIGGERLEFT: "CONTROLLER_AXIS_TRIGGERLEFT",
    pygame.CONTROLLER_AXIS_TRIGGERRIGHT: "CONTROLLER_AXIS_TRIGGERRIGHT",
}
#: The sticks: the prefix of each one's four directions (``CONTROLLER_LEFTSTICK_UP`` ...).
LEFTSTICK, RIGHTSTICK = "CONTROLLER_LEFTSTICK", "CONTROLLER_RIGHTSTICK"
#: Each stick axis: its stick, and whether it is the stick's x (0) or y (1).
STICK_AXES = {
    pygame.CONTROLLER_AXIS_LEFTX: (Side.LEFT, 0), pygame.CONTROLLER_AXIS_LEFTY: (Side.LEFT, 1),
    pygame.CONTROLLER_AXIS_RIGHTX: (Side.RIGHT, 0), pygame.CONTROLLER_AXIS_RIGHTY: (Side.RIGHT, 1),
}
#: The analog triggers as buttons.
TRIGGERS = {
    pygame.CONTROLLER_AXIS_TRIGGERLEFT: "CONTROLLER_TRIGGERLEFT",
    pygame.CONTROLLER_AXIS_TRIGGERRIGHT: "CONTROLLER_TRIGGERRIGHT",
}
#: How far a trigger is pulled to press, and let go to release, of its travel.
_TRIGGER_ON, _TRIGGER_OFF = 0.5, 0.3


def _scaled(value: float) -> float:
    """A gamepad axis's reading, -32768..32767, as -1..1."""
    return max(-1.0, min(1.0, value / 32767))


class Gamepad(Joystick):
    """A gamepad SDL knows: its buttons, axes, triggers as buttons, and two sticks' directions."""

    name = "gamepad"
    CONTROLS = frozenset({*BUTTONS.values(), *AXES.values(), *TRIGGERS.values(),
                          *directions_of(LEFTSTICK), *directions_of(RIGHTSTICK)})
    STICKS = {LEFTSTICK: Side.LEFT, RIGHTSTICK: Side.RIGHT}

    @classmethod
    def accepts(cls, index: int) -> bool:
        return controller.is_controller(index)

    def _open(self, index: int) -> None:
        super()._open(index)
        self.gamepad = controller.Controller(index)

    def _event(self, event: pygame.event.Event, moved: set[Side]) -> list[InputEvent]:
        if event.type in (pygame.CONTROLLERBUTTONDOWN, pygame.CONTROLLERBUTTONUP):
            return self._button(BUTTONS.get(event.button), event.button,
                                event.type == pygame.CONTROLLERBUTTONDOWN)
        if event.type != pygame.CONTROLLERAXISMOTION:
            return []
        value = _scaled(event.value)
        if event.axis in TRIGGERS:
            return self._trigger(TRIGGERS[event.axis], value)
        if event.axis in STICK_AXES:
            side, axis = STICK_AXES[event.axis]
            self.sticks[side].move(axis, value)
            moved.add(side)
        return []

    def _trigger(self, name: str, value: float) -> list[InputEvent]:
        """Press a trigger's button past :data:`_TRIGGER_ON`, release it under the off."""
        if name not in self._buttons and value >= _TRIGGER_ON:
            self._buttons.add(name)
            return [InputEvent(name, Edge.DOWN)]
        if name in self._buttons and value < _TRIGGER_OFF:
            self._buttons.discard(name)
            return [InputEvent(name, Edge.UP)]
        return []

    def axes(self) -> dict[str, float]:
        return {name: _scaled(self.gamepad.get_axis(a)) for a, name in AXES.items()}

    def buttons(self) -> tuple[str, ...]:
        return (*BUTTONS.values(), *TRIGGERS.values())

    def read(self) -> tuple[dict[int, float], set[str]]:
        g = self.gamepad
        axes = {side.axes[k]: _scaled(g.get_axis(a)) for a, (side, k) in STICK_AXES.items()}
        held = {name for b, name in BUTTONS.items() if g.get_button(b)}
        return axes, held | {name for a, name in TRIGGERS.items()
                             if _scaled(g.get_axis(a)) >= _TRIGGER_ON}
