"""Crafter adapter (danijar/crafter).

Crafter uses the OLD gym API shape -- reset() returns obs only, step() returns a
4-tuple (obs, reward, done, info) with no `truncated`, and it doesn't register
cleanly under gymnasium. We wrap crafter.Env directly and normalize it to the
gymnasium contract the session loop expects.

The observation IS the RGB frame (default 64x64x3; bump via
env_kwargs.size), so render() just returns obs. Crafter has no savestate
API; reconstruction is via seed + action replay (deterministic given
crafter.Env(seed=...)). The 17 discrete actions get a movement keymap;
the rest (do/place/make) are reachable via the curriculum "keys" override.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from .keyspec import SingleKeySpec
from .base import EnvAdapter, FrameState

# Default keyboard mapping over crafter's Discrete(17) space:
#   0=noop, 1=move_left, 2=move_right, 3=move_up, 4=move_down, 5=do,
#   6=sleep, 7=place_stone, 8=place_table, 9=place_furnace, 10=place_plant,
#   11=make_wood_pickaxe, 12=make_stone_pickaxe, 13=make_iron_pickaxe,
#   14=make_wood_sword, 15=make_stone_sword, 16=make_iron_sword.
# Place/make actions are reachable via a curriculum "keys" override.
# Source: https://github.com/danijar/crafter/blob/master/crafter/data.yaml
_DEFAULT_KEYMAP = {"LEFT": 1, "RIGHT": 2, "UP": 3, "DOWN": 4, "SPACE": 5, "S": 6}


class CrafterAdapter(EnvAdapter):
    name: str = "crafter"

    def _make(self, spec: dict) -> gym.Env:
        import crafter
        # crafter.Env seeds at construction; size/view/area/length via env_kwargs.
        # Default size is 64x64 (RL-benchmark pixel art); bump size for a
        # sharper on-screen render (textures are redrawn at the new tile size).
        self._last_obs = None
        return crafter.Env(**spec.get("env_kwargs", {}))

    def _keyspec(self) -> SingleKeySpec:
        combos = {frozenset([k]): v for k, v in _DEFAULT_KEYMAP.items()}
        return SingleKeySpec(combos=combos, noop=0)

    def reset(self, seed: int | None) -> tuple[Any, dict]:
        # crafter.Env has no reset(seed=...) -- its RNG is fixed at construction
        # (crafter.Env(..., seed=...)) and never changes after. Re-seeding an
        # episode is done by rebuilding the env
        import crafter
        self.env.close()
        env_kwargs = dict(self.spec.get("env_kwargs", {}))
        if seed is not None:
            env_kwargs["seed"] = seed
        self.env = crafter.Env(**env_kwargs)
        obs = self.env.reset()
        if isinstance(obs, tuple):  # be tolerant if a newer crafter returns (obs, info)
            obs, info = obs
        else:
            info = {}
        self._last_obs = np.asarray(obs)
        return self._last_obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, done, info = self.env.step(int(action))
        self._last_obs = np.asarray(obs)
        # Map old-gym `done` onto gymnasium (terminated, truncated).
        return self._last_obs, reward, bool(done), False, info

    def render(self) -> np.ndarray:
        # obs is the RGB frame; avoids a second render call.
        return self._last_obs

    def capture(
        self, obs: Any, info: dict, want_blob: bool = True
    ) -> FrameState:
        # No savestate API -> rely on seed + action replay. Log the achievements
        # dict (crafter's semantic progress signal) when present.
        variables = {}
        if isinstance(info, dict) and "achievements" in info:
            variables["achievements"] = list(info["achievements"].values())
        return FrameState(blob=None, variables=variables)
