"""Keymap: a phase's ``keys`` -> the action ``step`` gets, one class per action space.

``keys`` maps a rig key (:data:`fmri_gym.rig.CONTROLS`, the controller's
buttons), or several joined with ``"+"``, to the action to send, written as the
env takes it. The empty name ``""`` is the action for no key held. There is no
default map and nothing is merged in: the file states all of it.

:func:`make_keymap` picks the class from the env's action space:

- :class:`MultiBinaryKeymap`: a value is the index of the button that key holds
  down, and every held key sets its button, so keys combine as on a controller.
  Nothing held is every button up, so ``""`` is refused.
- :class:`DiscreteKeymap`, :class:`BoxKeymap`: a value is an action of the space
  (an index, a list). The most specific combo whose keys are all held wins, and
  ``""`` -- the empty combo, held whenever nothing else is -- is what a frame
  with no key gets. No action of these spaces means "do nothing" everywhere (0
  is FrozenLake's LEFT, MiniHack's "move N"), so a real-time phase has to write
  it; a ``turn_based`` phase steps only on presses and needs none.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Keymap(ABC):
    """The combos and what they resolve to; subclasses say what a value means.

    :param keys: the phase's ``keys`` (``validate_config`` has already checked
        that its names are real keys).
    :param space: the env's action space, which the values are checked against.
    :raises ValueError: a value that is not an action of the space.
    """

    def __init__(self, keys: dict[str, Any], space: gym.Space) -> None:
        self.combos = {frozenset(combo.split("+") if combo else ()):
                       self._action(space, combo, value) for combo, value in keys.items()}

    @abstractmethod
    def _action(self, space: gym.Space, combo: str, value: Any) -> Any:
        """The action ``value`` stands for in ``space``; raises if it is not one."""

    @abstractmethod
    def resolve(self, held: frozenset[str]) -> Any:
        """The action for the keys held this frame.

        :param held: the pressed keys' NAMES.
        :return: the action to send to the env this frame.
        """

    def _matched(self, held: frozenset[str]) -> list[frozenset[str]]:
        """The combos whose keys are all held (``""`` always is)."""
        return [combo for combo in self.combos if combo <= held]

    def turn_actions(self) -> dict[str, Any]:
        """``{key: action}`` for the single-key entries: turn-based play steps on one press."""
        return {next(iter(combo)): self.resolve(combo) for combo in self.combos if len(combo) == 1}


class MultiBinaryKeymap(Keymap):
    """``MultiBinary(n)``: each key holds one button down; held keys combine."""

    def __init__(self, keys: dict[str, Any], space: spaces.MultiBinary) -> None:
        self.n = int(space.n)
        super().__init__(keys, space)

    def _action(self, space: spaces.MultiBinary, combo: str, value: Any) -> int:
        if combo == "":
            raise ValueError(f'keys: "": {value!r}: the env\'s action is MultiBinary({self.n}), '
                             "whose no-key action is every button up; remove it")
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < self.n:
            raise ValueError(f"keys: {combo!r}: {value!r}: the env's action is "
                             f"MultiBinary({self.n}), so each value is the index "
                             f"(0..{self.n - 1}) of the button that key holds down")
        return value

    def resolve(self, held: frozenset[str]) -> list[int]:
        vec = [0] * self.n
        for combo in self._matched(held):
            vec[self.combos[combo]] = 1
        return vec


class DiscreteKeymap(Keymap):
    """``Discrete``: a value is an action index; the most specific held combo wins."""

    def _action(self, space: spaces.Discrete, combo: str, value: Any) -> int:
        if isinstance(value, bool) or not space.contains(value):
            raise ValueError(f"keys: {combo!r}: {value!r} is not an action of the env's {space}")
        return value

    def resolve(self, held: frozenset[str]) -> Any:
        return self.combos[max(self._matched(held), key=len)]


class BoxKeymap(DiscreteKeymap):
    """``Box``: a value is a list, sent as an array of the space's dtype."""

    def _action(self, space: spaces.Box, combo: str, value: Any) -> np.ndarray:
        action = np.asarray(value, dtype=space.dtype)
        if not space.contains(action):
            raise ValueError(f"keys: {combo!r}: {value!r} is not an action of the env's {space}")
        return action


def make_keymap(spec: dict, space: gym.Space) -> Keymap:
    """The phase's :class:`Keymap`, of the class its env's action space calls for.

    :param spec: the game-phase config (``keys``, ``turn_based``).
    :param space: the env's action space.
    :raises TypeError: an action space other than MultiBinary, Discrete or Box.
    :raises ValueError: a real-time Discrete / Box phase whose ``keys`` lack ``""``.
    """
    keys = spec["keys"]
    if isinstance(space, spaces.MultiBinary):
        return MultiBinaryKeymap(keys, space)
    if isinstance(space, spaces.Discrete):
        cls: type[Keymap] = DiscreteKeymap
    elif isinstance(space, spaces.Box):
        cls = BoxKeymap
    else:
        raise TypeError(f"keys: the env's action space is {space}; a keymap drives "
                        "MultiBinary, Discrete or Box only")
    if "" not in keys and not spec.get("turn_based", False):
        raise ValueError(f'keys: "" missing; the env\'s action space is {space}, so name the '
                         'action sent on a frame with no key held ("": 0, say), or set '
                         '"turn_based": true to step only on key presses')
    return cls(keys, space)
