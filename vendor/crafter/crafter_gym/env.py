"""Crafter (danijar/crafter) as a Gymnasium env.

``crafter.Env`` is written against the old ``gym`` API -- ``reset`` returns the
observation alone, ``step`` a 4-tuple -- and takes its seed at construction,
never at ``reset``. This env holds one ``crafter.Env`` and presents the
Gymnasium contract in front of it: ``reset(seed=)`` rebuilds the game with
that seed, so an episode replays from its seed and actions. The game itself is
untouched.

The observation IS the RGB frame (``size``, 64x64 by default), so ``render``
returns the latest one. Actions are crafter's Discrete(17) (crafter/data.yaml):
0 = noop, 1 = move_left, 2 = move_right, 3 = move_up, 4 = move_down, 5 = do,
6 = sleep, 7 = place_stone, 8 = place_table, 9 = place_furnace,
10 = place_plant, 11 = make_wood_pickaxe, 12 = make_stone_pickaxe,
13 = make_iron_pickaxe, 14 = make_wood_sword, 15 = make_stone_sword,
16 = make_iron_sword. ``info`` carries ``inventory``, ``achievements``,
``player_pos``, ``semantic`` (the map around the player) and ``discount``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CrafterEnv(gym.Env):
    """One Crafter world. Wraps the Crafter (legacy) gym env.

    :param kwargs: ``crafter.Env``'s own: ``area`` (world size, default
        ``(64, 64)``), ``view`` (tiles shown, ``(9, 9)``), ``size`` (frame in
        pixels, ``(64, 64)``), ``reward`` (``True``), ``length`` (steps per
        episode, 10000) and ``seed``, which ``reset(seed=)`` replaces.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"]}

    def __init__(self, **kwargs: Any) -> None:
        import crafter

        self._kwargs = kwargs
        self.game = crafter.Env(**kwargs)
        self.action_space = spaces.Discrete(int(self.game.action_space.n))
        shape = tuple(self.game.observation_space.shape)
        self.observation_space = spaces.Box(0, 255, shape, dtype=np.uint8)
        self._frame: np.ndarray | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        """Generate a new world.

        :param seed: if given, the game is rebuilt with it (crafter seeds at
            construction only); the same seed gives the same world.
        :param options: unused.
        :return: ``(frame, info)``.
        """
        super().reset(seed=seed)
        if seed is not None:
            import crafter

            self.game = crafter.Env(**{**self._kwargs, "seed": seed})
        self._frame = self.game.reset()
        return self._frame, {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply one action.

        :param action: an index into crafter's 17 actions.
        :return: ``(frame, reward, terminated, truncated, info)``.
            ``terminated`` is the game's ``done``: dead, or ``length`` steps.
        """
        self._frame, reward, done, info = self.game.step(int(action))
        return self._frame, float(reward), bool(done), False, info

    def render(self) -> np.ndarray | None:
        """Return the latest frame -- the observation is the picture."""
        return self._frame

    def close(self) -> None:
        """Nothing to close: crafter draws into arrays."""
