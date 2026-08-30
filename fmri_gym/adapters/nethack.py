"""NetHack adapter (nle, the NetHack Learning Environment).

Distinct from the `minihack` backend: base NLE (`NetHack*-v0`) does NOT provide a
`pixel` observation -- only the ASCII terminal (`tty_chars` / `tty_colors`, a
24x80 grid) plus `glyphs`/`blstats`/`message`, with a Discrete(23) action space
whose values are ASCII keycodes (NetHack's vi-keys: k/l/j/h = N/E/S/W, etc.).

NetHack is a terminal game, so we render the TTY buffer to a pixel frame (a
monospace text grid) for display -- faithful to how the game actually looks.
Arrow keys map to the 4 cardinal movement actions; `blstats` (score, HP, depth,
...) are logged as analysis variables. No pixel obs and no savestate over the
gym API -> reconstruction is seed + action replay.

Requires: `pip install nle` (already present if minihack is installed) and
`setuptools<81` (pkg_resources).
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from .base import EnvAdapter, FrameState, KeySpec

# NetHack TTY palette (16 colors), indexed by tty_colors (0..15).
_TTY_PALETTE = np.array([
    (0, 0, 0), (170, 0, 0), (0, 170, 0), (170, 85, 0),
    (0, 0, 170), (170, 0, 170), (0, 170, 170), (170, 170, 170),
    (85, 85, 85), (255, 85, 85), (85, 255, 85), (255, 255, 85),
    (85, 85, 255), (255, 85, 255), (85, 255, 255), (255, 255, 255),
], dtype=np.uint8)

# Arrow-key names -> the ASCII keycode NLE uses for that compass move
# (vi-keys: h=west 104, j=south 106, k=north 107, l=east 108).
_ARROW_TO_KEYCODE = {"UP": 107, "RIGHT": 108, "DOWN": 106, "LEFT": 104}


class NetHackAdapter(EnvAdapter):
    name = "nethack"

    def make(self, spec):
        import nle  # noqa: F401  (registers NetHack*-v0 env ids)
        env = gym.make(spec.get("game", "NetHackScore-v0"))
        # Map each arrow's target keycode to its Discrete action index (the
        # action list holds the keycodes as its values/enum).
        actions = list(env.unwrapped.actions)
        code_to_idx = {int(a): i for i, a in enumerate(actions)}
        self._key_to_action = {}
        for name, code in _ARROW_TO_KEYCODE.items():
            if code in code_to_idx:
                self._key_to_action[name] = code_to_idx[code]
        # ENTER (13) is handy for menus/prompts.
        if 13 in code_to_idx:
            self._key_to_action["RETURN"] = code_to_idx[13]
        self._cell = spec.get("cell_px", 10)  # pixel size of one TTY cell
        self._last = None
        return env

    def keymap(self, env) -> KeySpec:
        combos = {frozenset([k]): v for k, v in self._key_to_action.items()}
        # noop: NLE has no true no-op; default to the first action.
        return KeySpec(combos=combos, noop=0)

    def reset(self, env, seed, spec):
        obs, info = env.reset(seed=seed)
        self._last = obs
        return obs, info

    def render(self, env):
        return _tty_to_rgb(self._last, self._cell)

    def capture(self, env, obs, info, want_blob=True) -> FrameState:
        self._last = obs
        variables = {}
        for k in ("blstats", "glyphs", "message"):
            if isinstance(obs, dict) and k in obs:
                variables[k] = np.asarray(obs[k])
        return FrameState(blob=None, variables=variables)


def _tty_to_rgb(obs, cell):
    """Render NLE's (24,80) tty_chars/tty_colors grid to an RGB image."""
    if not isinstance(obs, dict) or "tty_chars" not in obs:
        return np.zeros((240, 800, 3), dtype=np.uint8)
    import pygame
    if not pygame.font.get_init():
        pygame.font.init()
    chars = np.asarray(obs["tty_chars"])
    colors = np.asarray(obs["tty_colors"])
    rows, cols = chars.shape
    # Size a monospace font by cell height, then MEASURE its actual glyph box
    # and lay the grid out on exactly that pitch -- so cells never overlap
    # (the previous bug) and there are no gaps. Terminal cells are taller than
    # wide, which the measured (gw, gh) naturally reproduces.
    font = _mono_font(pygame, int(cell * 2))
    gw, gh = font.size("W")
    surf = pygame.Surface((cols * gw, rows * gh))
    surf.fill((0, 0, 0))
    for r in range(rows):
        for c in range(cols):
            ch = int(chars[r, c])
            if ch in (0, 32):  # null / space
                continue
            col = _TTY_PALETTE[int(colors[r, c]) & 0x0F]
            glyph = font.render(chr(ch), True, tuple(int(x) for x in col))
            # center the glyph horizontally in its cell (glyph width may be < gw)
            surf.blit(glyph, (c * gw + (gw - glyph.get_width()) // 2, r * gh))
    return pygame.surfarray.array3d(surf).transpose(1, 0, 2)


def _mono_font(pygame, size):
    """A real fixed-width font (falls back gracefully across systems)."""
    for name in ("dejavusansmono", "liberationmono", "couriernew", "monospace"):
        try:
            f = pygame.font.SysFont(name, size)
            if f is not None:
                return f
        except Exception:
            continue
    return pygame.font.Font(pygame.font.get_default_font(), size)
