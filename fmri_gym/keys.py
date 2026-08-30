"""Shared keyboard helpers for the display loop and adapters.

Maps pygame keycodes to the upper-case NAMES used in KeySpec combos
("LEFT", "SPACE", "Z", ...). One definition so every caller agrees.
"""

from __future__ import annotations

import pygame

# pygame constants: single letters are lowercase (K_a), everything else is
# uppercase (K_UP, K_SPACE, K_RETURN, K_0). Getting this wrong makes
# arrows/space undetectable -> every arrow/space action becomes NOOP.
_PYGAME_KEY_NAMES = {
    pygame.K_UP: "UP",
    pygame.K_DOWN: "DOWN",
    pygame.K_LEFT: "LEFT",
    pygame.K_RIGHT: "RIGHT",
    pygame.K_SPACE: "SPACE",
    pygame.K_RETURN: "RETURN",
    pygame.K_TAB: "TAB",
    pygame.K_LSHIFT: "LSHIFT",
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


def held_key_names(keys: dict[int, str] | None = None) -> frozenset[str]:
    """Return currently-held keys as upper-case names (``"LEFT"``, ``"SPACE"``, …).

    Kept here so every adapter/display shares one definition of "held keys".

    :param keys: optional ``{pygame_keycode: NAME}`` map to poll; defaults to
        the shared game-key table (avoids scanning all 512 codes).
    :return: frozenset of currently pressed key NAMES.
    """
    pressed = pygame.key.get_pressed()
    if keys is None:
        # Scan the common game keys; cheap and avoids enumerating all 512 codes.
        keys = _PYGAME_KEY_NAMES
    out = set()
    for code, name in keys.items():
        if pressed[code]:
            out.add(name)
    return frozenset(out)


def key_name(keycode: int) -> str | None:
    """Map a pygame keycode to its upper-case NAME (``"LEFT"``, …).

    Used for turn-based games, which step on discrete KEYDOWN events rather than
    polling held keys.

    :param keycode: a ``pygame.K_*`` constant.
    :return: the key NAME, or ``None`` if the code is not in the shared table.
    """
    return _PYGAME_KEY_NAMES.get(keycode)
