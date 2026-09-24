"""stable-retro adapter (NES / SNES / Genesis / GB / ... via libretro).

Maps stable-retro behind the standard EnvAdapter interface:
- keymap: keyboard -> the game's console buttons (MultiBinary action vector);
- per-frame exact savestate via em.get_state()/set_state() (bit-exact, verified);
- state variables: the console RAM plus the game's decoded `info` variables
  (score/lives/... from the integration's data.json), surfaced uniformly.

Notes verified against stable_retro 1.0.1:
- The emulator object is env.unwrapped.em; the libretro RAM view must be
  refreshed with data.update_ram() before get_ram() after a bare set_state.
- Named levels load via env.unwrapped.load_state(name) then reset().
- retro allows only ONE emulator per process; the session opens/closes one env
  per block, so this is respected as long as blocks don't overlap.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import stable_retro as retro

from .keyspec import MultiKeySpec
from .base import EnvAdapter, FrameState

# Keyboard -> console button. Same scheme as the interactive retro player.
# We map by button NAME; each game reports its own button ordering via
# env.buttons, so the adapter builds the action vector for that ordering.
_KEY_TO_BUTTON = {
    "Z": ("BUTTON", "A"), "X": ("B",), "C": ("C",),
    "A": ("X",), "S": ("Y",), "D": ("Z",),
    "Q": ("L",), "W": ("R",),
    "UP": ("UP",), "DOWN": ("DOWN",), "LEFT": ("LEFT",), "RIGHT": ("RIGHT",),
    "RETURN": ("START", "RESET"), "TAB": ("MODE", "SELECT"),
}


class RetroAdapter(EnvAdapter):
    name: str = "retro"

    def _make(self, spec: dict) -> gym.Env:
        # save_pixels accepted for interface symmetry; retro frames are already
        # reconstructable from the per-frame state, so pixels aren't stored.
        self.save_pixels = bool(spec.get("save_pixels", False))
        return retro.make(
            game=spec["game"], scenario=spec.get("scenario"),
            render_mode="rgb_array")

    def _keyspec(self) -> MultiKeySpec:
        buttons = list(self.env.unwrapped.buttons)   # e.g. ["B","A","MODE",...,"C"]
        btn_index = {b: i for i, b in enumerate(buttons)}

        def action_for(held_key: str) -> list[int]:
            vec = [0] * len(buttons)
            for target in _KEY_TO_BUTTON.get(held_key, ()):
                if target in btn_index:
                    vec[btn_index[target]] = 1
            return vec

        # Console buttons need true simultaneity, so combo values are button
        # vectors that MultiKeySpec ORs together: holding RIGHT+Z fires while
        # moving. Combo values are vectors already, hence no button_map.
        combos = {}
        for key in _KEY_TO_BUTTON:
            vec = action_for(key)
            if any(vec):
                combos[frozenset([key])] = vec
        return MultiKeySpec(combos=combos, noop=[0] * len(buttons))

    def reset(self, seed: int | None) -> tuple[Any, dict]:
        state = self.spec.get("state")
        if state:
            self.env.unwrapped.load_state(state)
        return self.env.reset()

    def capture(
        self, obs: Any, info: dict, want_blob: bool = True
    ) -> FrameState:
        u = self.env.unwrapped
        u.data.update_ram()
        variables = {"ram": u.get_ram().copy()}
        # Surface the game's decoded integration variables (score/lives/...).
        for k, v in (info or {}).items():
            variables[f"info_{k}"] = v
        # em.get_state() is ~1 MB for Genesis; only snapshot on stride frames.
        blob = u.em.get_state() if want_blob else None
        return FrameState(blob=blob, variables=variables)

    def restore(self, blob: bytes) -> None:
        u = self.env.unwrapped
        u.em.set_state(blob)
        u.data.update_ram()

    def rich_state(self, obs: Any, info: dict) -> dict | None:
        """Thin wrapper so Session (hook lookup by name) finds this --
        see :meth:`get_rich_state`."""
        return self.get_rich_state(obs, info)

    def get_rich_state(self, obs: Any, info: dict) -> dict | None:
        """``info`` is already every declared integration variable
        (``data.lookup_all()``); what's left is static console metadata:
        buttons, players, game/state file."""
        u = self.env.unwrapped
        return {
            "game": u.gamename, "state": u.statename,
            "players": u.players, "buttons": list(u.buttons),
            "variables": dict(info or {}),
        }
