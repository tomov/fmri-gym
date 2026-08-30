"""Baba Is You adapter (baba-is-ai) -- the DBP "language" pick.

A Baba-Is-You-style puzzle where you push word blocks to rewrite the rules.
baba-is-ai (nacloos/baba-is-ai) uses the OLD gym API (obs-only reset, 4-tuple
step) and is created via baba.make("env/<id>"); render("rgb_array") gives a
256x256 frame. Actions are Discrete(5) via BabaIsYouEnv.Actions:
idle=0, up=1, right=2, down=3, left=4.

We normalize the old-gym shape to the gymnasium contract the loop expects and
display the rendered frame. No savestate -> seed + action replay.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .keyspec import SingleKeySpec
from .base import EnvAdapter, FrameState

# BabaIsYouEnv.Actions (baba/grid.py): idle=0, up=1, right=2, down=3, left=4.
# (baba.envs.ACTIONS is a separate name->delta dict used for planning, not indices.)
_DEFAULT_KEYMAP: dict[str, int] = {"UP": 1, "RIGHT": 2, "DOWN": 3, "LEFT": 4}


class BabaAdapter(EnvAdapter):
    name: str = "baba"

    def make(self, spec: dict) -> Any:
        import baba
        self._env = baba.make(spec.get("game", "env/make_win"))
        return self._env

    def keymap(self, env: Any) -> SingleKeySpec:
        combos = {frozenset([k]): v for k, v in _DEFAULT_KEYMAP.items()}
        return SingleKeySpec(combos=combos, noop=0)

    def reset(self, env: Any, seed: int | None, spec: dict) -> tuple[Any, dict]:
        try:
            out = env.reset(seed=seed)
        except TypeError:
            out = env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        return obs, {}

    def step(self, env: Any, action: Any) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, done, info = env.step(int(action))
        return obs, float(reward), bool(done), False, info

    def render(self, env: Any) -> np.ndarray:
        return np.asarray(env.render("rgb_array"))

    def capture(
        self, env: Any, obs: Any, info: dict, want_blob: bool = True
    ) -> FrameState:
        return FrameState(blob=None, variables={})
