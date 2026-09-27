"""Shared keyboard helpers for the display loop and adapters.

Maps pygame keycodes to the upper-case NAMES a phase's ``keys`` are written in
("LEFT", "SPACE", "Z", "F1", "KP_ENTER", ...). One definition so every caller
agrees, and so a config can be checked against it before the window opens.
"""

from __future__ import annotations

import pygame

#: pygame's ``K_*`` names without the prefix, upper-cased. Letters are the one
#: trap: their constants are lower-case (``K_a``), everything else is upper
#: (``K_UP``, ``K_SPACE``, ``K_0``); getting that wrong makes a key undetectable.
_NAMES = [
    "UP", "DOWN", "LEFT", "RIGHT", "SPACE", "RETURN", "BACKSPACE", "TAB",
    "LSHIFT", "RSHIFT", "LCTRL", "RCTRL", "LALT", "RALT",
    "COMMA", "PERIOD", "SLASH", "SEMICOLON", "QUOTE", "MINUS", "EQUALS", "BACKQUOTE",
    "LEFTBRACKET", "RIGHTBRACKET", "BACKSLASH",
    "INSERT", "DELETE", "HOME", "END", "PAGEUP", "PAGEDOWN",
    *(f"F{i}" for i in range(1, 13)),
    *(f"KP{i}" for i in range(10)),
    "KP_ENTER", "KP_PLUS", "KP_MINUS", "KP_MULTIPLY", "KP_DIVIDE", "KP_PERIOD",
    *"ABCDEFGHIJKLMNOPQRSTUVWXYZ", *"0123456789",
]
_PYGAME_KEY_NAMES: dict[int, str] = {
    getattr(pygame, "K_" + (n.lower() if n.isalpha() and len(n) == 1 else n)): n for n in _NAMES}
#: Every name a phase's ``keys`` may use.
KEY_NAMES: frozenset[str] = frozenset(_NAMES)


def held_key_names() -> frozenset[str]:
    """Return the currently held keys as NAMES (``"LEFT"``, ``"SPACE"``, ...).

    :return: frozenset of the pressed keys' names.
    """
    pressed = pygame.key.get_pressed()
    return frozenset(name for code, name in _PYGAME_KEY_NAMES.items() if pressed[code])


def key_name(keycode: int) -> str | None:
    """Map a pygame keycode to its NAME, or ``None`` if it is not a game key.

    Used for turn-based games, which step on discrete KEYDOWN events rather than
    polling held keys.

    :param keycode: a ``pygame.K_*`` constant.
    :return: the key NAME, or ``None``.
    """
    return _PYGAME_KEY_NAMES.get(keycode)
