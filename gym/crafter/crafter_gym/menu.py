"""Crafter's seventeen actions behind eight buttons.

A button box in a scanner has far fewer buttons than crafter has actions, so
the ten that are not movement have to share. This wrapper is that sharing: six
actions keep a button of their own (move in four directions, ``do``, ``sleep``)
and the rest sit in a list the player steps through with ``cycle`` and fires
with ``confirm``. It is a wrapper rather than a frontend detail because the
cursor is game state: which action ``confirm`` fires depends on how many times
``cycle`` was pressed, so an agent evaluated on the same interface has to see
the same cursor, and a replay has to reproduce it (see the repo's Rule 1).

The action space is ``Discrete(n + 2)``: crafter's own ``n`` actions in their
own order, then ``cycle`` and ``confirm``. ``info`` gains ``env_action`` (what
the engine was actually given), ``menu_idx`` (where the cursor ended up),
``menu_sel`` (the name it is on) and ``menu_extra`` (the extra entry this press
fired, or ``""``), so a log keeps both what was pressed and what it did.

``extra`` entries follow the actions in the menu: names with no crafter action
behind them, for a wrapper above this one to act on (the task chain's
``skip_task``). Confirming one steps the engine with ``noop`` and names it in
``menu_extra``.

``cycle`` spends a turn: the engine is stepped with ``noop``. Otherwise a look
through the menu would be free and the player could reach any of the ten for
the price of one press, which is not the interface the rig gives a subject.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces

#: Actions that keep a button of their own, so the menu is everything else:
#: noop (no key held), the four moves, ``do`` and ``sleep``.
DEFAULT_DIRECT = (0, 1, 2, 3, 4, 5, 6)


class MenuWrapper(gym.Wrapper):
    """Crafter with a cycling menu for the actions that have no button.

    :param env: a :class:`~crafter_gym.env.CrafterEnv` (or anything with
        crafter's action space and ``action_names``).
    :param direct: the action ids that keep their own button. The menu is
        whatever the action space has that is not in here, in ascending order,
        so a crafter that gained an action would put it in the menu rather
        than drop it.
    :param extra: entry names after the actions, with no action behind them.
    """

    def __init__(self, env: gym.Env, direct: tuple[int, ...] = DEFAULT_DIRECT,
                 extra: tuple[str, ...] = ()) -> None:
        super().__init__(env)
        n = int(env.action_space.n)
        self.menu = [i for i in range(n) if i not in set(direct)]
        if not self.menu:
            raise ValueError(f"nothing left for the menu: {direct} covers all "
                             f"{n} actions")
        if set(extra) & set(env.action_names):
            raise ValueError(f"extra menu entries {extra} reuse crafter action names")
        self.extra = tuple(extra)
        self.cycle, self.confirm = n, n + 1
        self.action_space = spaces.Discrete(n + 2)
        self.menu_idx = 0
        self.showing = False

    @property
    def action_names(self) -> list[str]:
        """Crafter's action names, with the two meta-actions after them."""
        return list(self.env.action_names) + ["cycle", "confirm"]

    @property
    def menu_names(self) -> list[str]:
        """The menu's entries, in the order ``cycle`` steps through them."""
        return [self.env.action_names[i] for i in self.menu] + list(self.extra)

    @property
    def selected(self) -> str:
        """The entry the cursor is on."""
        return self.menu_names[self.menu_idx]

    def reset(self, **kwargs: Any) -> tuple[Any, dict]:
        """Reset the game, with the cursor on the menu's first entry.

        An episode is then fixed by its seed and actions alone, so it replays
        without the episodes before it.

        :param kwargs: passed through (``seed``, ``options``).
        :return: ``(obs, info)`` with the menu fields added.
        """
        obs, info = self.env.reset(**kwargs)
        self.menu_idx = 0
        self.showing = False
        return obs, self._with_menu(info, 0)

    def env_action(self, action: Any) -> int:
        """What the engine would be given for this press, cursor as it stands.

        Read-only, and separate from :meth:`step` so a caller that has to know
        before the world moves -- an adapter reading the tile a press is about
        to meet, say -- asks rather than reimplements the translation.

        :param action: an index into :attr:`action_space`.
        :return: a crafter action id. ``cycle`` is ``noop``: a look costs a
            turn, or the player could reach any menu action for one press.
        """
        action = int(action)
        if action == self.cycle:
            return 0
        if action == self.confirm:
            return self.menu[self.menu_idx] if self.menu_idx < len(self.menu) else 0
        return action

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Apply a button press, translating the two meta-actions.

        :param action: an index into :attr:`action_space`.
        :return: ``(obs, reward, terminated, truncated, info)``; ``info``
            carries ``env_action``, ``menu_idx``, ``menu_sel`` and ``menu_extra``.
        """
        action = int(action)
        self.showing = action in (self.cycle, self.confirm)
        env_action = self.env_action(action)
        fired = (self.selected if action == self.confirm and self.menu_idx >= len(self.menu)
                 else "")
        if action == self.cycle:
            self.menu_idx = (self.menu_idx + 1) % len(self.menu_names)
        obs, reward, terminated, truncated, info = self.env.step(env_action)
        return obs, reward, terminated, truncated, self._with_menu(info, env_action, fired)

    def _with_menu(self, info: dict, env_action: int, fired: str = "") -> dict:
        """Add the menu fields to an ``info`` dict.

        :param info: the wrapped env's info.
        :param env_action: what the engine was given.
        :param fired: the extra entry this press fired, or ``""``.
        :return: the same dict, with the four fields set.
        """
        info = dict(info)
        info["env_action"] = env_action
        info["menu_extra"] = fired
        info["menu_idx"] = self.menu_idx
        info["menu_sel"] = self.selected
        return info
