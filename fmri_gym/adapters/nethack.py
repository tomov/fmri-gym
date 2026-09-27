"""NetHack adapter (nle, the NetHack Learning Environment).

Distinct from the `minihack` backend: base NLE (`NetHack*-v0`) does NOT provide a
`pixel` observation -- only the ASCII terminal (`tty_chars` / `tty_colors`, a
24x80 grid) plus `glyphs`/`blstats`/`message`, with a Discrete(23) action space
whose values are ASCII keycodes (NetHack's vi-keys: k/l/j/h = N/E/S/W, etc.). A
phase's ``keys`` are indices into that list (``env.unwrapped.actions``; for
NetHackScore-v0: 0 = MORE (Enter), 1 = N, 2 = E, 3 = S, 4 = W). Play it
``turn_based``: NetHack has no no-op.

NetHack is a terminal game, so we render the TTY buffer to a pixel frame (a
monospace text grid) for display -- faithful to how the game actually looks.
`blstats` (score, HP, depth, ...) are logged as analysis variables. No pixel obs
and no savestate over the gym API -> reconstruction is seed + action replay.

Requires: `pip install nle` (already present if minihack is installed) and
`setuptools<81` (pkg_resources).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym

from .base import Env, EnvAdapter, FrameState

# NetHack TTY palette (16 colors), indexed by tty_colors (0..15).
_TTY_PALETTE = np.array([
    (0, 0, 0), (170, 0, 0), (0, 170, 0), (170, 85, 0),
    (0, 0, 170), (170, 0, 170), (0, 170, 170), (170, 170, 170),
    (85, 85, 85), (255, 85, 85), (85, 255, 85), (255, 255, 85),
    (85, 85, 255), (255, 85, 255), (85, 255, 255), (255, 255, 255),
], dtype=np.uint8)


class NetHackAdapter(EnvAdapter):
    name: str = "nethack"

    def _make(self, spec: dict) -> Env:
        import nle  # noqa: F401  (registers NetHack*-v0 env ids)
        env = gym.make(spec.get("game", "NetHackScore-v0"))
        self._cell = spec.get("cell_px", 10)  # pixel size of one TTY cell
        self._last = None
        return env

    def reset(self, seed: int | None) -> tuple[Any, dict]:
        obs, info = self.env.reset(seed=seed)
        self._last = obs
        return obs, info

    def render(self) -> np.ndarray:
        return _tty_to_rgb(self._last, self._cell)

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> FrameState:
        self._last = obs
        variables = {}
        for k in ("blstats", "glyphs", "message"):
            if isinstance(obs, dict) and k in obs:
                variables[k] = np.asarray(obs[k])
        return FrameState(blob=None, variables=variables)


def _tty_to_rgb(obs: Any, cell: int) -> np.ndarray:
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


def _mono_font(pygame: Any, size: int) -> Any:
    """A real fixed-width font (falls back gracefully across systems)."""
    for name in ("dejavusansmono", "liberationmono", "couriernew", "monospace"):
        try:
            f = pygame.font.SysFont(name, size)
            if f is not None:
                return f
        except Exception:
            continue
    return pygame.font.Font(pygame.font.get_default_font(), size)
