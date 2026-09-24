"""Overcooked adapter (overcooked_ai) -- the DBP "social" pick.

Overcooked is a 2-cook cooperative game. We let the participant control cook 0
with the keyboard and have the partner (cook 1) idle (STAY) by default, so it's
playable solo; set "partner": "random" for a moving partner. Frames are rendered
with overcooked's StateVisualizer (a pygame surface -> RGB). Reward is the
sparse soup-delivery reward; deliveries are logged.

overcooked_ai's env is not a Gymnasium env, so this wraps OvercookedEnv
directly (old-style 4-tuple step, joint actions).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .keyspec import SingleKeySpec
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

    def _keyspec(self) -> SingleKeySpec:
        A = self._Action
        combos = {frozenset(["UP"]): (0, -1), frozenset(["DOWN"]): (0, 1),
                  frozenset(["LEFT"]): (-1, 0), frozenset(["RIGHT"]): (1, 0),
                  frozenset(["SPACE"]): "interact"}
        return SingleKeySpec(combos=combos, noop=(0, 0))

    def reset(self, seed: int | None) -> tuple[Any, dict]:
        self.env.reset()
        return self.env.state, {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        A = self._Action
        partner = A.STAY
        if self._partner == "random":
            import random
            partner = random.choice(A.ALL_ACTIONS)
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

    def rich_state(self, obs: Any, info: dict) -> dict | None:
        """Thin wrapper so Session (which calls ``adapter.rich_state(obs,
        info)`` by name) finds this hook -- see :meth:`get_rich_state`."""
        return self.get_rich_state(obs, info)

    def get_rich_state(self, obs: Any, info: dict) -> dict | None:
        """The complete ground-truth kitchen state: ``obs`` here already
        *is* the ``OvercookedState`` object (``self.env.state`` -- see
        :meth:`reset`/:meth:`step`), and it ships its own ``to_dict()``
        covering both players (position/orientation/held item), every
        object on the map (ingredients, plated/cooking soups, dishes) with
        position, and the order list (``bonus_orders``/``all_orders``) --
        everything :meth:`capture` doesn't already log (only the scalar
        shaped-reward signal).

        :param obs: the current ``OvercookedState`` (unused-looking, but
            it IS the state -- this backend has no separate array obs).
        :param info: unused.
        :return: ``OvercookedState.to_dict()``, or ``None`` before reset.
        """
        return obs.to_dict() if obs is not None else None
