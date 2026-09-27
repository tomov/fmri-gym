"""Crafter adapter (danijar/crafter), via ``crafter-gym``.

``crafter_gym`` (``vendor/crafter/``) is the Gymnasium env: ``crafter.Env``
speaks the old ``gym`` API and is seeded at construction only, and that
package puts the Gymnasium contract in front of it (``reset(seed=)`` rebuilds
the world). The observation IS the RGB frame (64x64 by default; sharper with
``env_kwargs.size``), so ``render()`` returns it.

A phase's ``keys`` index crafter's Discrete(17) space (crafter/data.yaml):
0 = noop, 1 = move_left, 2 = move_right, 3 = move_up, 4 = move_down, 5 = do,
6 = sleep, 7 = place_stone, 8 = place_table, 9 = place_furnace,
10 = place_plant, 11 = make_wood_pickaxe, 12 = make_stone_pickaxe,
13 = make_iron_pickaxe, 14 = make_wood_sword, 15 = make_stone_sword,
16 = make_iron_sword. ``env_kwargs`` are ``crafter.Env``'s (``size``, ``area``,
``view``, ``length``, ...). No savestate -> seed + action replay; the
achievements are logged each frame.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from .base import EnvAdapter, FrameState


class CrafterAdapter(EnvAdapter):
    name: str = "crafter"

    def _make(self, spec: dict) -> gym.Env:
        from crafter_gym import CrafterEnv

        return CrafterEnv(**spec.get("env_kwargs", {}))

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> FrameState:
        # The achievements dict is crafter's semantic progress signal.
        variables = {}
        if "achievements" in info:
            variables["achievements"] = list(info["achievements"].values())
        return FrameState(blob=None, variables=variables)
