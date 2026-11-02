"""Current Designs' GMPD3 gamepad on its 932 interface in joystick mode: six buttons, one stick.

SDL knows no mapping for it, so it is the joystick SDL does not take for a
gamepad, with at least the buttons and axes below. SDL numbers its buttons from
0, one less than the HID report. It also reports two axes pinned at -1.0,
which are no control. In the interface's keyboard mode the pad types keys, and
is the keyboard.
"""

from __future__ import annotations

import pygame
from pygame._sdl2 import controller

from .base import Press
from .calibration import Side
from .joystick import Joystick
from .stick import directions_of

BUTTONS = {0: "GMPD3_Y", 1: "GMPD3_B", 2: "GMPD3_X", 3: "GMPD3_LT", 4: "GMPD3_A", 5: "GMPD3_RT"}
AXES = {0: "GMPD3_STICK_X", 1: "GMPD3_STICK_Y"}
#: Its stick: the prefix of its four directions (``GMPD3_STICK_UP`` ...).
STICK = "GMPD3_STICK"


class Gmpd3(Joystick):
    """The GMPD3: its buttons, its stick's two axes, and the stick's four directions."""

    name = "gmpd3"
    CONTROLS = frozenset({*BUTTONS.values(), *AXES.values(), *directions_of(STICK)})
    STICKS = {STICK: Side.LEFT}

    @classmethod
    def accepts(cls, index: int) -> bool:
        joystick = pygame.joystick.Joystick(index)
        return (not controller.is_controller(index)
                and joystick.get_numbuttons() >= len(BUTTONS)
                and joystick.get_numaxes() >= len(AXES))

    def _event(self, event: pygame.event.Event, moved: set[Side]) -> list[Press]:
        if event.type == pygame.JOYAXISMOTION and event.axis in AXES:
            self.sticks[Side.LEFT].move(event.axis, event.value)
            moved.add(Side.LEFT)
        elif event.type in (pygame.JOYBUTTONDOWN, pygame.JOYBUTTONUP):
            return self._button(BUTTONS.get(event.button), event.button,
                                event.type == pygame.JOYBUTTONDOWN)
        return []

    def axes(self) -> dict[str, float]:
        return {name: self.joystick.get_axis(i) for i, name in AXES.items()}

    def buttons(self) -> tuple[str, ...]:
        return tuple(BUTTONS.values())

    def read(self) -> tuple[dict[int, float], set[str]]:
        j = self.joystick
        return ({axis: j.get_axis(axis) for axis in AXES},
                {name for b, name in BUTTONS.items() if j.get_button(b)})
