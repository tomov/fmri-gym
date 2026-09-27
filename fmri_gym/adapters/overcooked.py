"""Overcooked adapter (overcooked_ai) -- the DBP "social" pick.

Overcooked is a 2-cook cooperative game. We let the participant control cook 0
with the keyboard and have the partner (cook 1) idle (STAY) by default, so it's
playable solo; set "partner": "random" for a moving partner. Frames are rendered
with overcooked's StateVisualizer (a pygame surface -> RGB). Reward is the
sparse soup-delivery reward; deliveries are logged.

overcooked_ai's env is not a Gymnasium env, so this wraps OvercookedEnv
directly (old-style 4-tuple step, joint actions). Its actions are the moves as
``(dx, dy)`` and ``"interact"``, so a phase writes ``"keys": {"UP": [0, -1],
"DOWN": [0, 1], "LEFT": [-1, 0], "RIGHT": [1, 0], "SPACE": "interact"}`` and
``"noop": [0, 0]`` (stay).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import EnvAdapter, FrameState


class OvercookedAdapter(EnvAdapter):
    name: str = "overcooked"

    def _make(self, spec: dict) -> Any:
        from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
        from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
        from overcooked_ai_py.mdp.actions import Action
        from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
        self._Action = Action
        layout = spec.get("game", "cramped_room")
        # accept a bare layout or an "overcooked/<layout>" style id
        if "/" in layout:
            layout = layout.split("/")[-1]
        self._mdp = OvercookedGridworld.from_layout_name(layout)
        self._viz = StateVisualizer()
        self._partner = spec.get("partner", "stay")
        return OvercookedEnv.from_mdp(self._mdp, horizon=spec.get("horizon", 1000))

    def reset(self, seed: int | None) -> tuple[Any, dict]:
        self.env.reset()
        return self.env.state, {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        A = self._Action
        partner = A.STAY
        if self._partner == "random":
            import random
            partner = random.choice(A.ALL_ACTIONS)
        if isinstance(action, list):  # a move from the JSON keys, as the hashable tuple
            action = tuple(action)
        next_state, reward, done, info = self.env.step((action, partner))
        return next_state, float(reward), bool(done), False, info

    def render(self) -> np.ndarray:
        import pygame
        surf = self._viz.render_state(self.env.state, grid=self._mdp.terrain_mtx)
        return pygame.surfarray.array3d(surf).transpose(1, 0, 2)

    def capture(
        self, obs: Any, info: dict, want_blob: bool = True
    ) -> FrameState:
        variables = {}
        if isinstance(info, dict):
            shaped = info.get("shaped_r_by_agent")
            if shaped is not None:
                variables["shaped_reward"] = float(np.sum(shaped))
        return FrameState(blob=None, variables=variables)
