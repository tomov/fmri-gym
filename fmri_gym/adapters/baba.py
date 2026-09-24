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

import json
from typing import Any

import numpy as np

from .keyspec import SingleKeySpec
from .base import EnvAdapter, FrameState

# BabaIsYouEnv.Actions (baba/grid.py): idle=0, up=1, right=2, down=3, left=4.
# (baba.envs.ACTIONS is a separate name->delta dict used for planning, not indices.)
_DEFAULT_KEYMAP: dict[str, int] = {"UP": 1, "RIGHT": 2, "DOWN": 3, "LEFT": 4}


class BabaAdapter(EnvAdapter):
    name: str = "baba"

    def _make(self, spec: dict) -> Any:
        import baba
        return baba.make(spec.get("game", "env/make_win"))

    def _keyspec(self) -> SingleKeySpec:
        combos = {frozenset([k]): v for k, v in _DEFAULT_KEYMAP.items()}
        return SingleKeySpec(combos=combos, noop=0)

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

    def rich_state(self, obs: Any, info: dict) -> dict | None:
        """Thin wrapper so Session (hook lookup by name) finds this --
        see :meth:`get_rich_state`."""
        return self.get_rich_state(obs, info)

    def get_rich_state(self, obs: Any, info: dict) -> dict | None:
        """Agent position/facing/carrying, the win condition, the live
        ruleset (recomputed from the word-block layout each step -- Baba's
        core mechanic), and a sparse listing of every non-empty grid cell.
        """
        env = self.env
        grid = [
            {"x": x, "y": y, "type": cell.type}
            for y in range(env.height) for x in range(env.width)
            if (cell := env.grid.get(x, y)) is not None
        ]
        # get_ruleset() returns a Ruleset wrapper object, not a plain dict --
        # the actual {condition: {object: value}} mapping lives on .ruleset_dict.
        ruleset = json.loads(json.dumps(dict(env.get_ruleset().ruleset_dict), default=str))
        return {
            "agent_pos": [int(v) for v in env.agent_pos],
            "agent_dir": int(env.agent_dir),
            "carrying": getattr(env.carrying, "type", None),
            "win_rule": env.win_rule, "win_obj": env.win_obj,
            "ruleset": ruleset,
            "grid": grid,
        }
