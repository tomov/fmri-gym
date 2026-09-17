"""The key NAME vocabulary, and its binding to the current window backend.

Key NAMES ("LEFT", "SPACE", "Z", ...) are this package's own contract: both
curriculum "keys" blocks and KeySpec combos are written in them, so a name has
to keep meaning the same thing even if the window backend changes. pygame is
one binding of that vocabulary, not the vocabulary itself -- a port to another
toolkit replaces the codes in ``_CODES`` and the two lookups below, and nothing
under configs/ or adapters/ has to move.

Names outside :data:`KEY_NAMES` are rejected by :func:`validate_name` rather
than silently never matching a held key. ESCAPE is deliberately absent:
session.py reserves it for quitting, so an action bound to it could never fire.
"""

from __future__ import annotations

from difflib import get_close_matches
from string import ascii_uppercase

import pygame

# Our NAME -> the backend's keycode. Add a row to make a new key available to
# keymaps; KEY_NAMES and the reverse lookup follow from it.
_CODES: dict[str, int] = {
    "UP": pygame.K_UP,
    "DOWN": pygame.K_DOWN,
    "LEFT": pygame.K_LEFT,
    "RIGHT": pygame.K_RIGHT,
    "SPACE": pygame.K_SPACE,
    "RETURN": pygame.K_RETURN,
    "TAB": pygame.K_TAB,
    "LSHIFT": pygame.K_LSHIFT,
    "COMMA": pygame.K_COMMA,
    "PERIOD": pygame.K_PERIOD,
    # pygame spells single letters lowercase (K_a) and everything else
    # uppercase (K_UP, K_0), so these two rows are easy to hand-write wrong.
    **{letter: getattr(pygame, f"K_{letter.lower()}") for letter in ascii_uppercase},
    **{digit: getattr(pygame, f"K_{digit}") for digit in "0123456789"},
}

#: Every NAME a keymap may use: the vocabulary configs and adapters share.
KEY_NAMES: frozenset[str] = frozenset(_CODES)

_NAMES: dict[int, str] = {code: name for name, code in _CODES.items()}
if len(_NAMES) != len(_CODES):
    raise RuntimeError("two key NAMES share a keycode; key_name() would be ambiguous")


def validate_name(name: str) -> str:
    """Check ``name`` against the vocabulary, suggesting near misses.

    Called wherever a keymap is built, so a name nothing can ever press is an
    error at construction instead of a game that quietly ignores the subject.

    :param name: an upper-case key NAME, from an adapter keymap or a
        curriculum ``keys`` block.
    :return: ``name``, unchanged.
    :raises ValueError: if ``name`` is not in :data:`KEY_NAMES`.
    """
    if name in KEY_NAMES:
        return name
    close = get_close_matches(name.upper(), KEY_NAMES, n=3)
    suggestion = f" -- did you mean {', '.join(sorted(close))}?" if close else ""
    raise ValueError(f"unknown key name {name!r}{suggestion}"
                     " (add it to _CODES in fmri_gym/keys.py)")


def held_key_names(keys: dict[int, str] | None = None) -> frozenset[str]:
    """Return currently-held keys as NAMES (``"LEFT"``, ``"SPACE"``, …).

    Kept here so every adapter/display shares one definition of "held keys".

    :param keys: optional ``{keycode: NAME}`` map to poll; defaults to the
        whole vocabulary (avoids scanning all 512 codes).
    :return: frozenset of currently pressed key NAMES.
    """
    pressed = pygame.key.get_pressed()
    if keys is None:
        keys = _NAMES
    out = set()
    for code, name in keys.items():
        if pressed[code]:
            out.add(name)
    return frozenset(out)


def key_name(keycode: int) -> str | None:
    """Map a backend keycode to its NAME (``"LEFT"``, …).

    Used for turn-based games, which step on discrete KEYDOWN events rather than
    polling held keys.

    :param keycode: a keycode as reported by the window backend.
    :return: the key NAME, or ``None`` if the code is outside the vocabulary.
    """
    return _NAMES.get(keycode)
