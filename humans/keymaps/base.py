"""Keymap: the base of each keymap, a phase's combos and what they resolve to."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym


class Keymap(ABC):
    """The combos and what they resolve to; subclasses say what a value means.

    :param controls: the phase's ``controls``, whose names are real controls.
    :param space: the env's action space, which the values are checked against.
    :raises ValueError: a value that is not an action of the space.
    """

    def __init__(self, controls: dict[str, Any], space: gym.Space) -> None:
        self.combos = {frozenset(combo.split("+") if combo else ()):
                       self._action(space, combo, value) for combo, value in controls.items()}

    @abstractmethod
    def _action(self, space: gym.Space, combo: str, value: Any) -> Any:
        """The action ``value`` stands for in ``space``; raises if it is not one."""

    @abstractmethod
    def resolve(self, held: frozenset[str], values: dict[str, float]) -> Any:
        """The action for the controls of this frame; a keymap reads either argument or both.

        :param held: the NAMES of the controls held at the tick and of those pressed
            during the frame, so a tap shorter than a frame counts for one.
        :param values: each control with a range by its value, 0.0 at rest
            (:meth:`~humans.inputs.InputDevice.values`).
        :return: the action to send to the env this frame.
        """

    def _matched(self, held: frozenset[str]) -> list[frozenset[str]]:
        """The combos whose controls are all held (``""`` always is)."""
        return [combo for combo in self.combos if combo <= held]

    def turn_actions(self) -> dict[str, Any]:
        """``{control: action}`` for the single-control entries: turn-based play steps on one press."""
        return {next(iter(combo)): self.resolve(combo, {})
                for combo in self.combos if len(combo) == 1}
