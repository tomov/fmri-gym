"""The keyboard: every key a curriculum may name, by pygame's constant without its ``K_``.

A rig lists it like any device. ``ESCAPE`` is a control but no curriculum's:
it ends a run. What it holds is read from the key events it handles, so a press
posted to the queue (a test, the rig check) is held like a typed one.
"""

from __future__ import annotations

import pygame

from .base import InputDevice, InputEvent

# pygame constants: single letters are lowercase (K_a), everything else is
# uppercase (K_UP, K_SPACE, K_RETURN, K_0). Getting this wrong makes a key
# undetectable.
NAMES: dict[int, str] = {
    pygame.K_UP: "UP",
    pygame.K_DOWN: "DOWN",
    pygame.K_LEFT: "LEFT",
    pygame.K_RIGHT: "RIGHT",
    pygame.K_SPACE: "SPACE",
    pygame.K_RETURN: "RETURN",
    pygame.K_ESCAPE: "ESCAPE",
    pygame.K_BACKSPACE: "BACKSPACE",
    pygame.K_TAB: "TAB",
    pygame.K_LSHIFT: "LSHIFT",
    pygame.K_RSHIFT: "RSHIFT",
    pygame.K_LCTRL: "LCTRL",
    pygame.K_RCTRL: "RCTRL",
    pygame.K_LALT: "LALT",
    pygame.K_RALT: "RALT",
    pygame.K_COMMA: "COMMA",
    pygame.K_PERIOD: "PERIOD",
    pygame.K_SLASH: "SLASH",
    pygame.K_SEMICOLON: "SEMICOLON",
    pygame.K_QUOTE: "QUOTE",
    pygame.K_MINUS: "MINUS",
    pygame.K_EQUALS: "EQUALS",
    pygame.K_BACKQUOTE: "BACKQUOTE",
    pygame.K_LEFTBRACKET: "LEFTBRACKET",
    pygame.K_RIGHTBRACKET: "RIGHTBRACKET",
    pygame.K_BACKSLASH: "BACKSLASH",
    pygame.K_INSERT: "INSERT",
    pygame.K_DELETE: "DELETE",
    pygame.K_HOME: "HOME",
    pygame.K_END: "END",
    pygame.K_PAGEUP: "PAGEUP",
    pygame.K_PAGEDOWN: "PAGEDOWN",
    pygame.K_F1: "F1",
    pygame.K_F2: "F2",
    pygame.K_F3: "F3",
    pygame.K_F4: "F4",
    pygame.K_F5: "F5",
    pygame.K_F6: "F6",
    pygame.K_F7: "F7",
    pygame.K_F8: "F8",
    pygame.K_F9: "F9",
    pygame.K_F10: "F10",
    pygame.K_F11: "F11",
    pygame.K_F12: "F12",
    pygame.K_KP0: "KP0",
    pygame.K_KP1: "KP1",
    pygame.K_KP2: "KP2",
    pygame.K_KP3: "KP3",
    pygame.K_KP4: "KP4",
    pygame.K_KP5: "KP5",
    pygame.K_KP6: "KP6",
    pygame.K_KP7: "KP7",
    pygame.K_KP8: "KP8",
    pygame.K_KP9: "KP9",
    pygame.K_KP_ENTER: "KP_ENTER",
    pygame.K_KP_PLUS: "KP_PLUS",
    pygame.K_KP_MINUS: "KP_MINUS",
    pygame.K_KP_MULTIPLY: "KP_MULTIPLY",
    pygame.K_KP_DIVIDE: "KP_DIVIDE",
    pygame.K_KP_PERIOD: "KP_PERIOD",
    pygame.K_a: "A",
    pygame.K_b: "B",
    pygame.K_c: "C",
    pygame.K_d: "D",
    pygame.K_e: "E",
    pygame.K_f: "F",
    pygame.K_g: "G",
    pygame.K_h: "H",
    pygame.K_i: "I",
    pygame.K_j: "J",
    pygame.K_k: "K",
    pygame.K_l: "L",
    pygame.K_m: "M",
    pygame.K_n: "N",
    pygame.K_o: "O",
    pygame.K_p: "P",
    pygame.K_q: "Q",
    pygame.K_r: "R",
    pygame.K_s: "S",
    pygame.K_t: "T",
    pygame.K_u: "U",
    pygame.K_v: "V",
    pygame.K_w: "W",
    pygame.K_x: "X",
    pygame.K_y: "Y",
    pygame.K_z: "Z",
    pygame.K_0: "0",
    pygame.K_1: "1",
    pygame.K_2: "2",
    pygame.K_3: "3",
    pygame.K_4: "4",
    pygame.K_5: "5",
    pygame.K_6: "6",
    pygame.K_7: "7",
    pygame.K_8: "8",
    pygame.K_9: "9",
}
#: The keycode of each name: the table above, read the other way.
KEYCODES: dict[str, int] = {name: code for code, name in NAMES.items()}
#: The key that ends a run, which a curriculum may not name.
ESCAPE = NAMES[pygame.K_ESCAPE]


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
