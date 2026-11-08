"""The keyboard: every key a curriculum may name, by pygame's constant without its ``K_``.

A rig lists it like any device. Escape is a control but no curriculum's: it
ends a run. What it holds is read from the key events it handles, so a press
posted to the queue (a test, the rig measurement) is held like a typed one.
"""

from __future__ import annotations

import pygame

from .base import InputDevice, InputEvent
from .control_names import ControlNames

#: pygame's keycode -> the control.
NAMES: dict[int, ControlNames] = {
    pygame.K_UP: ControlNames.K_UP,
    pygame.K_DOWN: ControlNames.K_DOWN,
    pygame.K_LEFT: ControlNames.K_LEFT,
    pygame.K_RIGHT: ControlNames.K_RIGHT,
    pygame.K_SPACE: ControlNames.K_SPACE,
    pygame.K_RETURN: ControlNames.K_RETURN,
    pygame.K_ESCAPE: ControlNames.K_ESCAPE,
    pygame.K_BACKSPACE: ControlNames.K_BACKSPACE,
    pygame.K_TAB: ControlNames.K_TAB,
    pygame.K_LSHIFT: ControlNames.K_LSHIFT,
    pygame.K_RSHIFT: ControlNames.K_RSHIFT,
    pygame.K_LCTRL: ControlNames.K_LCTRL,
    pygame.K_RCTRL: ControlNames.K_RCTRL,
    pygame.K_LALT: ControlNames.K_LALT,
    pygame.K_RALT: ControlNames.K_RALT,
    pygame.K_COMMA: ControlNames.K_COMMA,
    pygame.K_PERIOD: ControlNames.K_PERIOD,
    pygame.K_SLASH: ControlNames.K_SLASH,
    pygame.K_SEMICOLON: ControlNames.K_SEMICOLON,
    pygame.K_QUOTE: ControlNames.K_QUOTE,
    pygame.K_MINUS: ControlNames.K_MINUS,
    pygame.K_EQUALS: ControlNames.K_EQUALS,
    pygame.K_BACKQUOTE: ControlNames.K_BACKQUOTE,
    pygame.K_LEFTBRACKET: ControlNames.K_LEFTBRACKET,
    pygame.K_RIGHTBRACKET: ControlNames.K_RIGHTBRACKET,
    pygame.K_BACKSLASH: ControlNames.K_BACKSLASH,
    pygame.K_INSERT: ControlNames.K_INSERT,
    pygame.K_DELETE: ControlNames.K_DELETE,
    pygame.K_HOME: ControlNames.K_HOME,
    pygame.K_END: ControlNames.K_END,
    pygame.K_PAGEUP: ControlNames.K_PAGEUP,
    pygame.K_PAGEDOWN: ControlNames.K_PAGEDOWN,
    pygame.K_F1: ControlNames.K_F1,
    pygame.K_F2: ControlNames.K_F2,
    pygame.K_F3: ControlNames.K_F3,
    pygame.K_F4: ControlNames.K_F4,
    pygame.K_F5: ControlNames.K_F5,
    pygame.K_F6: ControlNames.K_F6,
    pygame.K_F7: ControlNames.K_F7,
    pygame.K_F8: ControlNames.K_F8,
    pygame.K_F9: ControlNames.K_F9,
    pygame.K_F10: ControlNames.K_F10,
    pygame.K_F11: ControlNames.K_F11,
    pygame.K_F12: ControlNames.K_F12,
    pygame.K_KP0: ControlNames.K_KP0,
    pygame.K_KP1: ControlNames.K_KP1,
    pygame.K_KP2: ControlNames.K_KP2,
    pygame.K_KP3: ControlNames.K_KP3,
    pygame.K_KP4: ControlNames.K_KP4,
    pygame.K_KP5: ControlNames.K_KP5,
    pygame.K_KP6: ControlNames.K_KP6,
    pygame.K_KP7: ControlNames.K_KP7,
    pygame.K_KP8: ControlNames.K_KP8,
    pygame.K_KP9: ControlNames.K_KP9,
    pygame.K_KP_ENTER: ControlNames.K_KP_ENTER,
    pygame.K_KP_PLUS: ControlNames.K_KP_PLUS,
    pygame.K_KP_MINUS: ControlNames.K_KP_MINUS,
    pygame.K_KP_MULTIPLY: ControlNames.K_KP_MULTIPLY,
    pygame.K_KP_DIVIDE: ControlNames.K_KP_DIVIDE,
    pygame.K_KP_PERIOD: ControlNames.K_KP_PERIOD,
    pygame.K_a: ControlNames.K_a,
    pygame.K_b: ControlNames.K_b,
    pygame.K_c: ControlNames.K_c,
    pygame.K_d: ControlNames.K_d,
    pygame.K_e: ControlNames.K_e,
    pygame.K_f: ControlNames.K_f,
    pygame.K_g: ControlNames.K_g,
    pygame.K_h: ControlNames.K_h,
    pygame.K_i: ControlNames.K_i,
    pygame.K_j: ControlNames.K_j,
    pygame.K_k: ControlNames.K_k,
    pygame.K_l: ControlNames.K_l,
    pygame.K_m: ControlNames.K_m,
    pygame.K_n: ControlNames.K_n,
    pygame.K_o: ControlNames.K_o,
    pygame.K_p: ControlNames.K_p,
    pygame.K_q: ControlNames.K_q,
    pygame.K_r: ControlNames.K_r,
    pygame.K_s: ControlNames.K_s,
    pygame.K_t: ControlNames.K_t,
    pygame.K_u: ControlNames.K_u,
    pygame.K_v: ControlNames.K_v,
    pygame.K_w: ControlNames.K_w,
    pygame.K_x: ControlNames.K_x,
    pygame.K_y: ControlNames.K_y,
    pygame.K_z: ControlNames.K_z,
    pygame.K_0: ControlNames.K_0,
    pygame.K_1: ControlNames.K_1,
    pygame.K_2: ControlNames.K_2,
    pygame.K_3: ControlNames.K_3,
    pygame.K_4: ControlNames.K_4,
    pygame.K_5: ControlNames.K_5,
    pygame.K_6: ControlNames.K_6,
    pygame.K_7: ControlNames.K_7,
    pygame.K_8: ControlNames.K_8,
    pygame.K_9: ControlNames.K_9,
}
#: The keycode of each name: the table above, read the other way.
KEYCODES: dict[str, int] = {name: code for code, name in NAMES.items()}


class Keyboard(InputDevice):
    """The keyboard, and anything that types keys: a button box in a key mode, a scanner."""

    name = "keyboard"
    CONTROLS = frozenset(NAMES.values())

    @classmethod
    def open(cls) -> Keyboard:
        return cls()

    def handle(self, events: list[pygame.event.Event], now: float) -> list[InputEvent]:
        out: list[InputEvent] = []
        for event in events:
            if event.type == pygame.KEYDOWN and event.key in NAMES:
                out += self._press(NAMES[event.key], True)
            elif event.type == pygame.KEYUP and event.key in NAMES:
                out += self._press(NAMES[event.key], False)
            elif event.type == pygame.WINDOWFOCUSLOST:  # its key-ups go to the window in focus
                out += [e for name in sorted(self._held) for e in self._press(name, False)]
        return out
