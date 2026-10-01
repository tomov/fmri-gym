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
``player_pos``, ``semantic`` (the map around the player) and ``discount``,
plus ``sleeping``, which crafter tracks but does not report (see :meth:`CrafterEnv.step`).
"""

from __future__ import annotations

import sys
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

_MISSING = object()


def import_crafter() -> Any:
    """Import ``crafter`` with fmri-gym's own ``gym/`` directory out of the way.

    Old ``gym`` is optional to crafter, which falls back to namedtuples for its
    spaces when it is absent, but that fallback is guarded by ``ImportError``
    alone. fmri-gym keeps its Gymnasium envs in a directory called ``gym/``, so
    anything started from the repo root (``python fmri_play.py ...``) puts that
    directory first on ``sys.path``, and with no real ``gym`` installed -- the
    ``crafter`` extra alone pulls none -- ``import gym`` succeeds as an empty
    namespace package and crafter raises ``AttributeError`` on ``gym.spaces``
    instead of taking its fallback. Blocking ``gym`` for the length of the
    import restores the branch crafter means to take. An installed ``gym`` is
    untouched, and never shadowed in the first place: a real package beats a
    namespace portion wherever the two sit on the path.

    :return: the ``crafter`` module.
    """
    if "crafter" in sys.modules:
        return sys.modules["crafter"]
    saved = sys.modules.get("gym", _MISSING)
    # None in sys.modules is the documented way to make an import raise.
    sys.modules["gym"] = None
    try:
        import crafter
    finally:
        if saved is _MISSING:
            del sys.modules["gym"]
        else:
            sys.modules["gym"] = saved
    return crafter


class CrafterEnv(gym.Env):
    """One Crafter world. Wraps the Crafter (legacy) gym env.

    :param kwargs: ``crafter.Env``'s own: ``area`` (world size, default
        ``(64, 64)``), ``view`` (tiles shown, ``(9, 9)``), ``size`` (frame in
        pixels, ``(64, 64)``), ``reward`` (``True``), ``length`` (steps per
        episode, 10000) and ``seed``, which ``reset(seed=)`` replaces.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"]}

    def __init__(self, **kwargs: Any) -> None:
        crafter = import_crafter()

        self._kwargs = kwargs
        self.game = crafter.Env(**kwargs)
        self.action_space = spaces.Discrete(int(self.game.action_space.n))
        shape = tuple(self.game.observation_space.shape)
        self.observation_space = spaces.Box(0, 255, shape, dtype=np.uint8)
        self._frame: np.ndarray | None = None

    @property
    def action_names(self) -> list[str]:
        """crafter's own names for the action ids, in id order."""
        return list(self.game.action_names)

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
            self.game = import_crafter().Env(**{**self._kwargs, "seed": seed})
        self._frame = self.game.reset()
        return self._frame, {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply one action.

        crafter's ``done`` is two endings in one flag (``env.py``: ``done =
        dead or over``, where ``over`` is its ``length`` step cap), and
        Gymnasium separates them: ``terminated`` is the game ending, dying,
        and ``truncated`` is the step limit stopping an episode that had not
        ended. Which one it was decides what the subject is told at the end of
        the episode and what a block's ``advancing_outcomes`` reads, so it is
        split here from the health the game reports anyway.

        ``info`` gains ``sleeping``, which crafter keeps on the player and
        reports nowhere. It is the one state in which the game discards the
        action it was given: ``Player.update`` overwrites it with ``sleep``
        until energy is full (``crafter/objects.py``), so a press made while it
        is True never reaches the world. A caller that asks a person for one
        press per step needs to know that, or it will ask for presses the game
        throws away.

        :param action: an index into crafter's 17 actions.
        :return: ``(frame, reward, terminated, truncated, info)``.
        """
        self._frame, reward, done, info = self.game.step(int(action))
        dead = info["inventory"]["health"] <= 0
        info["sleeping"] = bool(self.game._player.sleeping)
        return (self._frame, float(reward),
                bool(done and dead), bool(done and not dead), info)

    def render(self) -> np.ndarray | None:
        """Return the latest frame -- the observation is the picture."""
        return self._frame

    def redraw(self) -> np.ndarray:
        """Draw the current state again, replacing the latest frame.

        For a caller that changes what the game looks like between a reset and
        the frame the player is shown: ``crafter.Env.reset`` generates the
        world and renders it in the same call, so the frame a level wrapper
        inherits was drawn before its rules existed
        (:mod:`crafter_gym.levels`).

        Only ever at the start of an episode. A render is not free of the
        game's randomness: the engine mixes per-pixel noise into the view at
        night, drawn from ``world.random``, the stream the creatures share
        (``engine.LocalView._light``), so a redraw mid-episode would shift
        every later draw and the episode would stop replaying from its seed.
        The noise is drawn only below daylight 0.5, which the engine's clock
        reaches between steps 148 and 272 of each 300-step day; at step 0
        daylight is 0.797 and a redraw there costs nothing.

        :return: the new frame, which :meth:`render` now returns too.
        """
        self._frame = self.game.render()
        return self._frame

    def close(self) -> None:
        """Nothing to close: crafter draws into arrays."""
