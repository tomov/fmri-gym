"""Crafter as a Gymnasium environment.

``gym.make("Crafter-v0")`` or :func:`make_plain`. See :mod:`crafter_gym.env`.
``gym.make("CrafterMenu-v0")`` is the same game through the eight-button
interface of :mod:`crafter_gym.menu`. Either takes ``level=``, one of the four
rule variants the scanner paradigm plays (:mod:`crafter_gym.levels`).
"""

from __future__ import annotations

import gymnasium as gym

from .env import CrafterEnv, import_crafter
from .levels import (LEVELS, LevelWrapper, level_of, reachable_achievements,
                     with_level)
from .menu import MenuWrapper

__all__ = ["LEVELS", "CrafterEnv", "LevelWrapper", "MenuWrapper",
           "import_crafter", "level_of", "make_menu", "make_plain",
           "reachable_achievements", "with_level"]


def make_plain(**kwargs) -> gym.Env:
    """Build a :class:`~crafter_gym.env.CrafterEnv`, in a level if asked.

    :param kwargs: ``CrafterEnv``'s, plus ``level`` for
        :class:`~crafter_gym.levels.LevelWrapper`.
    :return: the env, wrapped if ``level`` was given.
    """
    level = kwargs.pop("level", None)
    return with_level(CrafterEnv(**kwargs), level)


def make_menu(**kwargs) -> MenuWrapper:
    """Build a :class:`~crafter_gym.env.CrafterEnv` behind the eight-button menu.

    A level goes on first, under the menu: the rules are the game, the buttons
    are how it is played.

    :param kwargs: ``CrafterEnv``'s, plus ``level`` for
        :class:`~crafter_gym.levels.LevelWrapper` and ``direct`` for
        :class:`~crafter_gym.menu.MenuWrapper`.
    :return: the wrapped env.
    """
    direct = kwargs.pop("direct", None)
    env = make_plain(**kwargs)
    return MenuWrapper(env) if direct is None else MenuWrapper(env, direct)


gym.register(id="Crafter-v0", entry_point="crafter_gym:make_plain",
             disable_env_checker=True)
gym.register(id="CrafterMenu-v0", entry_point="crafter_gym:make_menu",
             disable_env_checker=True)
