"""Native NetHackDS desktop port: up/right/down/left/confirm/back/commands/inventory (0..7).

Observations contain RGB pixels and native tile/glyph maps; actions are Discrete(8).
Supported env_kwargs: runtime, game_mode (classic/modern), initial_state (an encoded
immutable campaign checkpoint), and render_mode (rgb_array only). Experiment runs
require isolated seeded episodes; persistent player_directory saves belong to the
standalone player. The engine is silent and reward is zero.

Constructor arguments live in env_kwargs. The environment owns all rendering,
menus and input translation, and ReplayState replays the controller actions.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from .base import EnvAdapter, FrameState


class NetHackGamepadAdapter(EnvAdapter):
    name = "nethack_gamepad"

    def _make(self, spec: dict) -> gym.Env:
        kwargs = spec.get("env_kwargs", {})
        if kwargs.get("player_directory") is not None:
            raise ValueError("fMRI recordings require isolated seeded episodes; omit player_directory")
        return gym.make(spec["game"], **kwargs)

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> FrameState:
        variables = {key: np.asarray(value).copy() for key, value in obs.items()
                     if key != "pixels"}
        state = self.env.get_wrapper_attr("get_state")() if want_blob else None
        return FrameState(blob=state, variables={**variables, **info})

    def restore(self, blob: bytes) -> None:
        _, self.last_info = self.env.get_wrapper_attr("set_state")(blob)
        self.last_reward = 0.0

    def block_extra(self) -> dict:
        return {"state": self.env.get_wrapper_attr("get_state")(),
                "continuation_state": self.env.get_wrapper_attr("get_continuation_state")()}
