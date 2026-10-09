"""Crafter as a Gymnasium environment.

``gym.make("Crafter-v0")`` or :func:`make_plain`. See :mod:`crafter_gym.env`.
``gym.make("CrafterMenu-v0")`` is the same game through the eight-button
interface of :mod:`crafter_gym.menu`. Either takes ``level=``, one of the four
rule variants the scanner paradigm plays (:mod:`crafter_gym.levels`), which on
the levels that ask for tasks also names one at a time
(:mod:`crafter_gym.tasks`).

The three wrappers stack in one order, from the inside out: what the world does
(the level), how it is played (the menu), what it is being played for (the
tasks). The level is innermost because the rules are the game; the tasks are
outermost because the skip entry is a menu press that the game never sees.
"""

from __future__ import annotations

import gymnasium as gym

from .env import CrafterEnv, import_crafter
from .levels import (LEVELS, LevelWrapper, level_of, reachable_achievements,
                     require_level, with_level)
from .menu import MenuWrapper, menu_of
from .tasks import (SKIP_ENTRY, TASK_CHAIN, TASK_DONE_TEXT, TASK_LABELS, TaskWrapper,
                    chain_for, has_tasks, next_task_text, task_text, tracker_of,
                    with_tasks)

__all__ = ["LEVELS", "SKIP_ENTRY", "TASK_CHAIN", "TASK_DONE_TEXT", "TASK_LABELS",
           "CrafterEnv", "LevelWrapper", "MenuWrapper", "TaskWrapper", "chain_for",
           "has_tasks", "import_crafter", "level_of", "make_menu", "make_plain",
           "menu_of", "next_task_text", "reachable_achievements", "require_level",
           "task_text", "tracker_of", "with_level", "with_tasks"]


def make_plain(**kwargs) -> gym.Env:
    """Build a :class:`~crafter_gym.env.CrafterEnv`, in a level if asked.

    With no menu there is nowhere to put the entry that asks for a task to be
    put off, so a task level built this way is the chain without its escape
    hatch: an entry the world has made impossible for now stays the one named.

    :param kwargs: ``CrafterEnv``'s, plus ``level`` for
        :class:`~crafter_gym.levels.LevelWrapper` and, on the levels that ask
        for them, :class:`~crafter_gym.tasks.TaskWrapper`.
    :return: the env, wrapped if ``level`` was given.
    """
    level = kwargs.pop("level", None)
    return with_tasks(with_level(CrafterEnv(**kwargs), level), level)


def make_menu(**kwargs) -> gym.Env:
    """Build a :class:`~crafter_gym.env.CrafterEnv` behind the eight-button menu.

    :param kwargs: ``CrafterEnv``'s, plus ``level`` for
        :class:`~crafter_gym.levels.LevelWrapper` and
        :class:`~crafter_gym.tasks.TaskWrapper`, and ``direct`` for
        :class:`~crafter_gym.menu.MenuWrapper`.
    :return: the wrapped env.
    """
    direct = kwargs.pop("direct", None)
    level = kwargs.pop("level", None)
    env = with_level(CrafterEnv(**kwargs), level)
    env = MenuWrapper(env) if direct is None else MenuWrapper(env, direct)
    return with_tasks(env, level)


gym.register(id="Crafter-v0", entry_point="crafter_gym:make_plain",
             disable_env_checker=True)
gym.register(id="CrafterMenu-v0", entry_point="crafter_gym:make_menu",
             disable_env_checker=True)
