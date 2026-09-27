"""Baba Is You adapter (baba-is-ai) -- the DBP "language" pick.

A Baba-Is-You-style puzzle where you push word blocks to rewrite the rules.
baba-is-ai (nacloos/baba-is-ai) uses the OLD gym API (obs-only reset, 4-tuple
step) and is created via baba.make("env/<id>"); render("rgb_array") gives a
256x256 frame. Actions are Discrete(5) via BabaIsYouEnv.Actions (baba/grid.py):
idle=0, up=1, right=2, down=3, left=4 -- what a phase's ``keys`` index.

We normalize the old-gym shape to the gymnasium contract the loop expects and
display the rendered frame. No savestate -> seed + action replay.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import EnvAdapter, FrameState


class BabaAdapter(EnvAdapter):
    name: str = "baba"

    def _make(self, spec: dict) -> Any:
        import baba
        return baba.make(spec.get("game", "env/make_win"))

    def reset(self, seed: int | None) -> tuple[Any, dict]:
        try:
            out = self.env.reset(seed=seed)
        except TypeError:
            out = self.env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        return obs, {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        obs, reward, done, info = self.env.step(int(action))
        return obs, float(reward), bool(done), False, info

    def render(self) -> np.ndarray:
        return np.asarray(self.env.render("rgb_array"))

    def capture(
        self, obs: Any, info: dict, want_blob: bool = True
    ) -> FrameState:
        return FrameState(blob=None, variables={})
