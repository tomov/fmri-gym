"""Baba Is You (nacloos/baba-is-ai) as a Gymnasium env.

``baba`` is written against the old ``gym`` API: ``reset`` returns the
observation alone, ``step`` returns ``(obs, reward, done, info)`` and
``render`` takes a mode. This env holds one ``baba`` env and presents the
Gymnasium contract in front of it; the game itself is untouched.

The observation is the grid's own encoding (``(H, W, 3)`` uint8); the picture
is ``render()``, a 256x256 RGB frame. The action space is
``BabaIsYouEnv.Actions`` (baba/grid.py): 0 = idle, 1 = up, 2 = right,
3 = down, 4 = left. There is no savestate: an episode replays from its seed
and actions.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BabaEnv(gym.Env):
    """One Baba Is You puzzle.

    :param game: a ``baba`` env id -- ``env/make_win``, ``env/goto_win``,
        ``env/you_win``, ``env/two_room-make_win``, and the distractor
        variants (``baba.registration.registry`` lists them all).
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, game: str = "env/make_win") -> None:
        import baba

        self.game = baba.make(game)
        self.action_space = spaces.Discrete(int(self.game.action_space.n))
        box = self.game.observation_space
        self.observation_space = spaces.Box(box.low, box.high, dtype=box.dtype)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        """Generate a new grid.

        :param seed: seeds the grid generator; the same seed gives the same puzzle.
        :param options: unused.
        :return: ``(obs, info)``.
        """
        super().reset(seed=seed)
        return self.game.reset(seed=seed), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply one action.

        :param action: an index into ``BabaIsYouEnv.Actions``.
        :return: ``(obs, reward, terminated, truncated, info)``. ``terminated``
            is the game's ``done``: won, lost, or out of steps.
        """
        obs, reward, done, info = self.game.step(int(action))
        return obs, float(reward), bool(done), False, info

    def render(self) -> np.ndarray:
        """Return the current picture of the grid, ``(256, 256, 3)`` uint8."""
        return np.asarray(self.game.render("rgb_array"))

    def close(self) -> None:
        """Close the game's own window, if it ever opened one."""
        self.game.close()
