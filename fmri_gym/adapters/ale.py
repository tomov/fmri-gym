"""ALE (Atari 2600) adapter.

Exposes the Atari-specific bits behind the standard EnvAdapter interface:
- keymap from the game's own action meanings;
- per-frame exact savestate via clone_state (restorable, determinism-free);
- state variables: the 128-byte console RAM (+ optional lossless indexed pixels)
  surfaced through FrameState.variables.
"""

from __future__ import annotations

import pickle

import gymnasium as gym
import numpy as np

from .base import EnvAdapter, FrameState, KeySpec

_DIRECTIONS = {
    "UP": ("UP",), "DOWN": ("DOWN",), "LEFT": ("LEFT",), "RIGHT": ("RIGHT",),
    "UPRIGHT": ("UP", "RIGHT"), "UPLEFT": ("UP", "LEFT"),
    "DOWNRIGHT": ("DOWN", "RIGHT"), "DOWNLEFT": ("DOWN", "LEFT"),
}


class ALEAdapter(EnvAdapter):
    name = "ale"

    def __init__(self, save_pixels: bool = False):
        self.save_pixels = save_pixels
        # Block-wide palette for lossless indexed-pixel logging.
        self._palette = np.zeros((256, 3), dtype=np.uint8)
        self._palette_seen = np.zeros(256, dtype=bool)
        import ale_py
        gym.register_envs(ale_py)

    def make(self, spec):
        return gym.make(
            spec["game"], render_mode="rgb_array",
            frameskip=1, repeat_action_probability=0.0)

    def keymap(self, env) -> KeySpec:
        combos = {}
        for action, meaning in enumerate(env.unwrapped.get_action_meanings()):
            if meaning == "NOOP":
                continue
            fire = meaning.endswith("FIRE")
            direction = meaning[:-4] if fire and meaning != "FIRE" else meaning
            keys = _DIRECTIONS.get(direction, ())
            if fire:
                keys = keys + ("SPACE",)
            if keys:
                combos[frozenset(keys)] = action
        return KeySpec(combos=combos, noop=0,
                       help="Arrow keys move, SPACE fires.")

    def capture(self, env, obs, info) -> FrameState:
        ale = env.unwrapped.ale
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
        blob = pickle.dumps(env.unwrapped.clone_state(include_rng=True))
        return FrameState(blob=blob, variables=variables)

    def restore(self, env, blob):
        env.unwrapped.restore_state(pickle.loads(blob))

    def block_extra(self):
        """Block-level arrays merged into the npz (the palette, if save_pixels)."""
        if self.save_pixels:
            return {"palette": self._palette}
        return None
