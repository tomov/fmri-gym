"""Rush Hour adapter (chrplr/Rush-Hour) -- the DBP "puzzle" pick.

Slide cars out of a 6x6 grid to free the red car. The env (rushhour_gym) wraps a
Go engine binary and renders only ANSI (a letter grid), with a Discrete(n)
"pick a move" action space (no arrow semantics). So, like a board puzzle, we:
  * render the ANSI grid to a COLORED pixel board (each car = a colored block;
    the red player car and the exit are highlighted);
  * play turn-based, mapping number keys 1..9,0 to move indices 0..9 (Discrete
    is larger; extend via a per-phase "keys" map). Each move slides one car.

Needs the Go binary `rushhour-env` built from the checkout:
    go build -o rushhour-env ./cmd/rushhour-env
Point at it via the phase "binary" field or the RUSHHOUR_ENV_BIN env var; this
adapter also auto-finds the vendored copy under vendor/rush-hour-src/.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .base import EnvAdapter, FrameState, KeySpec

# Distinct colors for car letters; 'A' (red player car) and exit are special.
_PALETTE: list[tuple[int, int, int]] = [
    (220, 60, 60), (70, 130, 220), (80, 190, 90), (230, 190, 60),
    (170, 90, 200), (230, 140, 60), (90, 200, 200), (230, 120, 170),
    (150, 110, 70), (120, 160, 90), (200, 200, 120), (110, 200, 160),
]
_VENDOR_BIN: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "vendor", "rush-hour-src", "rushhour-env")
_NUM_KEYS: list[str] = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]


class RushHourAdapter(EnvAdapter):
    name: str = "rushhour"

    def make(self, spec: dict) -> Any:
        import gymnasium as gym
        import rushhour_gym  # noqa: F401  (registers RushHour*-v0)
        binary = spec.get("binary") or os.environ.get("RUSHHOUR_ENV_BIN")
        if not binary and os.path.exists(_VENDOR_BIN):
            binary = _VENDOR_BIN
        if binary:
            os.environ["RUSHHOUR_ENV_BIN"] = binary
        self._last_ansi = ""
        return gym.make(spec.get("game", "RushHour-Easy-v0"), render_mode="ansi")

    def keymap(self, env: Any) -> KeySpec:
        n = int(getattr(env.action_space, "n", 10))
        combos = {frozenset([k]): i for i, k in enumerate(_NUM_KEYS) if i < n}
        return KeySpec(combos=combos, noop=0)

    def reset(self, env: Any, seed: int | None, spec: dict) -> tuple[Any, dict]:
        obs, info = env.reset(seed=seed)
        self._last_ansi = env.render() or ""
        return obs, info

    def step(self, env: Any, action: Any) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = env.step(int(action))
        self._last_ansi = env.render() or ""
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self, env: Any) -> np.ndarray:
        return _board_to_rgb(self._last_ansi)

    def capture(
        self, env: Any, obs: Any, info: dict, want_blob: bool = True
    ) -> FrameState:
        variables = {}
        if isinstance(info, dict) and "slot" in info:
            try:
                variables["slot"] = int(info["slot"])
            except Exception:
                pass
        return FrameState(blob=None, variables=variables)


def _board_to_rgb(ansi: str, cell: int = 64) -> np.ndarray:
    """Render the ANSI letter grid to a colored pixel board."""
    rows = [r for r in (ansi or "").split("\n") if r != ""]
    if not rows:
        return np.zeros((cell * 6, cell * 6, 3), dtype=np.uint8)
    h = len(rows)
    w = max(len(r) for r in rows)
    img = np.full((h * cell, w * cell, 3), 30, dtype=np.uint8)
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            y0, x0 = r * cell, c * cell
            if ch in (" ", "o"):
                color = (45, 45, 45)          # empty
            elif ch == "<":
                color = (255, 255, 255)       # exit marker
            elif ch == "A":
                color = (230, 40, 40)         # red player car
            elif ch.isalpha():
                color = _PALETTE[(ord(ch.upper()) - ord("A")) % len(_PALETTE)]
            else:
                color = (60, 60, 60)
            # draw a padded block so grid lines show
            img[y0 + 2:y0 + cell - 2, x0 + 2:x0 + cell - 2] = color
    return img
