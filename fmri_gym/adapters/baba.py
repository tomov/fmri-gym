"""Baba Is You adapter (baba-is-ai) -- the DBP "language" pick.

A Baba-Is-You-style puzzle where you push word blocks to rewrite the rules.
baba-is-ai (nacloos/baba-is-ai) uses the OLD gym API (obs-only reset, 4-tuple
step) and is created via baba.make("env/<id>"); render("rgb_array") gives a
256x256 frame. Actions are Discrete(5): move up/down/left/right + idle.

We normalize the old-gym shape to the gymnasium contract the loop expects and
display the rendered frame. No savestate -> seed + action replay.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import EnvAdapter, FrameState, SingleKeySpec

# baba.envs.ACTIONS order: up, down, left, right (idx 0..3); 4 = idle.
_KEYS: dict[str, int] = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}


class BabaAdapter(EnvAdapter):
    name: str = "baba"

    def make(self, spec: dict) -> Any:
        import baba
        self._env = baba.make(spec.get("game", "env/make_win"))
        return self._env

    def keymap(self, env: Any) -> SingleKeySpec:
        combos = {frozenset([k]): v for k, v in _KEYS.items()}
        return SingleKeySpec(combos=combos, noop=4)

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
