"""NetHack adapter (nle, the NetHack Learning Environment).

Distinct from the `minihack` backend: base NLE (`NetHack*-v0`) does NOT provide a
`pixel` observation -- only the ASCII terminal (`tty_chars` / `tty_colors`, a
24x80 grid) plus `glyphs`/`blstats`/`message`, with a Discrete(23) action space
whose values are ASCII keycodes (NetHack's vi-keys: k/l/j/h = N/E/S/W, etc.). A
phase's ``keys`` are indices into that list (``env.unwrapped.actions``; for
NetHackScore-v0: 0 = MORE (Enter), 1 = N, 2 = E, 3 = S, 4 = W). Play it
``is_turn_based``: NetHack has no no-op.

NetHack is a terminal game, so we render the TTY buffer to a pixel frame (a
monospace text grid) for display -- faithful to how the game actually looks.
`blstats` (score, HP, depth, ...) are logged as analysis variables. NLE has no
savestate API, so the savestate is ``replay_state_gym.ReplayState``'s (seed
and actions, restored by replay), exact because ``minihack_levels.NLESeeding``
seeds NetHack's own RNGs on every reset and ``fix_moon_phase`` takes the real
date out of the game. One date remains: the character's birthday, which full
NetHack reads for shop prices, ant species and zoo stocking.

Requires: `pip install nle` (already present if minihack is installed) and
`setuptools<81` (pkg_resources).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from .base import EnvAdapter, FrameState

# NetHack TTY palette (16 colors), indexed by tty_colors (0..15).
_TTY_PALETTE = np.array([
    (0, 0, 0), (170, 0, 0), (0, 170, 0), (170, 85, 0),
    (0, 0, 170), (170, 0, 170), (0, 170, 170), (170, 170, 170),
    (85, 85, 85), (255, 85, 85), (85, 255, 85), (255, 255, 85),
    (85, 85, 255), (255, 85, 255), (85, 255, 255), (255, 255, 255),
], dtype=np.uint8)


class NetHackAdapter(EnvAdapter):
    name: str = "nethack"

    def _make(self, spec: dict) -> gym.Env:
        import nle  # noqa: F401  (registers NetHack*-v0 env ids)
        from minihack_levels import NLESeeding
        from replay_state_gym import ReplayState
        env = gym.make(spec.get("game", "NetHackScore-v0"), fix_moon_phase=True)
        self._cell = spec.get("cell_px", 10)  # pixel size of one TTY cell
        self._last = None
        return ReplayState(NLESeeding(env))

    def reset(self, seed: int) -> tuple[Any, dict]:
        obs, info = super().reset(seed)
        self._last = obs
        return obs, info

    def render(self) -> np.ndarray:
        return _tty_to_rgb(self._last, self._cell)
    

    def outcome(self, terminated: bool, truncated: bool) -> tuple[str, str]:
        # NLE's end_status: TASK_SUCCESSFUL, DEATH, or ABORTED at the step limit.
        status = getattr(self.last_info.get("end_status"), "name", "")
        if status == "TASK_SUCCESSFUL":
            return "won", "Task complete"
        if status == "DEATH":
            return "lost", "You died"
        if status == "ABORTED":
            return "truncated", "Timeout"
        return super().outcome(terminated, truncated)

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> FrameState:
        self._last = obs
        variables = {}
        if isinstance(obs, dict):
            for k, v in obs.items():
                if k in ("tty_chars", "tty_colors"):
                    continue
                # NLE reuses its observation buffers, and the logger pickles a
                # line after the next step: a view here logs the next frame.
                variables[k] = np.asarray(v).copy()
        variables.update(info or {})
        return FrameState(variables=variables)

    def restore(self, blob: bytes) -> None:
        self._last, _ = self.env.set_state(blob)

    def block_extra(self) -> dict:
        return {"state": self.env.get_state()}


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
