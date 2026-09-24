"""ALE (Atari 2600) EnvAdapter.

Exposes the Atari-specific bits behind the standard EnvAdapter interface:
- keymap from the game's own action meanings;
- per-frame exact savestate via clone_state (restorable, determinism-free);
- state variables: the 128-byte console RAM (+ optional lossless indexed pixels)
  surfaced through FrameState.variables.
"""

from __future__ import annotations

import pickle
from typing import Any

import gymnasium as gym
import numpy as np

from .keyspec import SingleKeySpec
from .base import EnvAdapter, FrameState

_DIRECTIONS = {
    "UP": ("UP",), "DOWN": ("DOWN",), "LEFT": ("LEFT",), "RIGHT": ("RIGHT",),
    "UPRIGHT": ("UP", "RIGHT"), "UPLEFT": ("UP", "LEFT"),
    "DOWNRIGHT": ("DOWN", "RIGHT"), "DOWNLEFT": ("DOWN", "LEFT"),
}


class ALEAdapter(EnvAdapter):
    name: str = "ale"

    def _make(self, spec: dict) -> gym.Env:
        import ale_py
        gym.register_envs(ale_py)
        # Set "save_pixels": true in the phase for lossless indexed-pixel logging.
        self.save_pixels = bool(spec.get("save_pixels", False))
        # Block-wide palette for that logging.
        self._palette = np.zeros((256, 3), dtype=np.uint8)
        self._palette_seen = np.zeros(256, dtype=bool)
        return gym.make(
            spec["game"], render_mode="rgb_array",
            frameskip=1, repeat_action_probability=0.0)

    def _keyspec(self) -> SingleKeySpec:
        combos = {}
        for action, meaning in enumerate(self.env.unwrapped.get_action_meanings()):
            if meaning == "NOOP":
                continue
            fire = meaning.endswith("FIRE")
            direction = meaning[:-4] if fire and meaning != "FIRE" else meaning
            keys = _DIRECTIONS.get(direction, ())
            if fire:
                keys = keys + ("SPACE",)
            if keys:
                combos[frozenset(keys)] = action
        return SingleKeySpec(combos=combos, noop=0)

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> FrameState:
        ale = self.env.unwrapped.ale
        variables = {"ram": ale.getRAM().copy()}
        if self.save_pixels:
            idx = ale.getScreen()  # (210,160) uint8 palette indices
            new = np.unique(idx)
            new = new[~self._palette_seen[new]]
            if new.size:
                flat_i, flat_c = idx.reshape(-1), obs.reshape(-1, 3)
                for i in new:
                    self._palette[i] = flat_c[flat_i == i][0]
                    self._palette_seen[i] = True
            variables["screen_index"] = idx.copy()
        blob = (pickle.dumps(self.env.unwrapped.clone_state(include_rng=True))
                if want_blob else None)
        return FrameState(blob=blob, variables=variables)

    def restore(self, blob: bytes) -> None:
        self.env.unwrapped.restore_state(pickle.loads(blob))

    def rich_state(self, obs: Any, info: dict) -> dict | None:
        """Thin wrapper so Session (which calls ``adapter.rich_state(obs,
        info)`` by name) finds this hook -- see :meth:`get_rich_state`."""
        return self.get_rich_state(obs, info)

    def get_rich_state(self, obs: Any, info: dict) -> dict | None:
        """Everything the ALE interface can report beyond the RAM bytes
        capture() already logs: lives remaining, the engine's own frame
        counters (total vs. this episode's), and the two ALE-level
        termination flags (``game_over``/``game_truncated``) -- ``ale``
        tracks these itself, they aren't derivable from RAM alone without
        knowing the game's own memory layout.

        :param obs: unused -- matches :meth:`capture`'s signature.
        :param info: unused -- everything here comes from ``self.env.
            unwrapped.ale`` directly.
        """
        ale = self.env.unwrapped.ale
        return {
            "lives": ale.lives(),
            "frame_number": ale.getFrameNumber(),
            "episode_frame_number": ale.getEpisodeFrameNumber(),
            "game_over": bool(ale.game_over()),
            "game_truncated": bool(ale.game_truncated()),
        }

    def block_extra(self) -> dict | None:
        """Block-level arrays merged into the npz (the palette, if save_pixels)."""
        if self.save_pixels:
            return {"palette": self._palette}
        return None
