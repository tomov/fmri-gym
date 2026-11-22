"""Keymap: a phase's ``controls`` -> the action ``step`` gets, one class per action space.

``controls`` maps a control NAME (:mod:`humans.inputs`), or several joined with ``"+"``,
to the action to send, written as the env takes it. The empty name ``""`` is the
action for no key held. There is no default map and nothing is merged in: which
key does what differs from site to site, so the file states all of it.

:func:`make_keymap` picks the class from the env's action space:

- :class:`MultiBinaryKeymap`: a value is the index of the button that key holds
  down, and every held control sets its button, so they combine as on a controller.
  Nothing held is every button up, so ``""`` is refused.
- :class:`DiscreteKeymap`, :class:`BoxKeymap`: a value is an action of the space
  (an index, a list). The most specific combo whose controls are all held wins, and
  ``""`` -- the empty combo, held whenever nothing else is -- is what a frame
  with no control gets. In a ``Box`` phase a control's value may instead be one
  dimension of the action, which the control's value sets (:class:`DimensionBinding`). No action of these spaces means "do nothing" everywhere (0
  is FrozenLake's LEFT, MiniHack's "move N"), so a real-time phase has to write
  it; an ``is_turn_based`` phase steps only on presses and needs none.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, NamedTuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..inputs import value_range


class DimensionBinding(NamedTuple):
    """One dimension of a ``Box`` action that one control's value sets: written as the
    dimension's index, which takes the value as it is, or as ``{"dimension": d, "range": [a,
    b]}``, which maps the control's range (:func:`~humans.inputs.value_range`) onto ``a..b``
    (``[1, -1]`` flips it)."""

    dimension: int
    #: what the dimension takes at the control's least and greatest value; None: the value
    value_range: tuple[float, float] | None

    @classmethod
    def from_dict(cls, control: str, value: Any, space: spaces.Box) -> DimensionBinding:
        """The binding ``value`` writes for ``control``.

        :raises ValueError: a combo, a dimension the space has not, or a range that is not
            two numbers.
        """
        where = f"controls: {control!r}: {value!r}"
        if "+" in control or control == "":
            raise ValueError(f"{where}: a dimension of the action is set by one control")
        dimension = value.get("dimension") if isinstance(value, dict) else value
        if isinstance(value, dict) and set(value) != {"dimension", "range"}:
            raise ValueError(f'{where}: expected {{"dimension": <index>, "range": [<at the '
                             'least value>, <at the greatest>]}')
        if (isinstance(dimension, bool) or not isinstance(dimension, int) or len(space.shape) != 1
                or not 0 <= dimension < space.shape[0]):
            raise ValueError(f"{where}: expected a dimension (0..{space.shape[0] - 1}) of the "
                             f"env's {space}")
        if not isinstance(value, dict):
            return cls(dimension, None)
        ends = value["range"]
        if not (isinstance(ends, list) and len(ends) == 2 and all(
                isinstance(end, (int, float)) and not isinstance(end, bool) for end in ends)):
            raise ValueError(f"{where}: range: expected two numbers, got {ends!r}")
        return cls(dimension, (float(ends[0]), float(ends[1])))

    def dimension_value(self, control: str, value: float) -> float:
        """What the dimension takes when ``control``'s value is ``value``."""
        if self.value_range is None:
            return value
        least, greatest = value_range(control)
        at_least, at_greatest = self.value_range
        return at_least + (value - least) / (greatest - least) * (at_greatest - at_least)


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
        """The action for the controls of this frame.

        :param held: the NAMES of the controls held at the tick and of those pressed
            during the frame, so a tap shorter than a frame counts for one.
        :param values: each control's value at the tick, a control at rest left out
            (:meth:`~humans.inputs.InputDevice.values`).
        :return: the action to send to the env this frame.
        """

    def _matched(self, held: frozenset[str]) -> list[frozenset[str]]:
        """The combos whose controls are all held (``""`` always is)."""
        return [combo for combo in self.combos if combo <= held]

    def turn_actions(self) -> dict[str, Any]:
        """``{control: action}`` for the single-control entries: turn-based play steps on one press."""
        return {next(iter(combo)): self.resolve(combo, dict.fromkeys(combo, 1.0))
                for combo in self.combos if len(combo) == 1}


class MultiBinaryKeymap(Keymap):
    """``MultiBinary(n)``: each control holds one button down; held controls combine."""

    def __init__(self, controls: dict[str, Any], space: spaces.MultiBinary) -> None:
        self.n = int(space.n)
        super().__init__(controls, space)

    def _action(self, space: spaces.MultiBinary, combo: str, value: Any) -> int:
        if combo == "":
            raise ValueError(f'controls: "": {value!r}: the env\'s action is '
                             f"MultiBinary({self.n}), whose no-control action is every button "
                             "up")
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < self.n:
            raise ValueError(f"controls: {combo!r}: {value!r}: the env's action is "
                             f"MultiBinary({self.n}), so each value is the index "
                             f"(0..{self.n - 1}) of the button that control holds down")
        return value

    def resolve(self, held: frozenset[str], values: dict[str, float]) -> list[int]:
        vec = [0] * self.n
        for combo in self._matched(held):
            vec[self.combos[combo]] = 1
        return vec


class DiscreteKeymap(Keymap):
    """``Discrete``: a value is an action index; the most specific held combo wins."""

    def _action(self, space: spaces.Discrete, combo: str, value: Any) -> int:
        if isinstance(value, bool) or not space.contains(value):
            raise ValueError(f"controls: {combo!r}: {value!r} is not an action of the env's "
                             f"{space}")
        return value

    def resolve(self, held: frozenset[str], values: dict[str, float]) -> Any:
        return self.combos[max(self._matched(held), key=len)]


class BoxKeymap(DiscreteKeymap):
    """``Box``: a list is the whole action, sent as an array of the space's dtype; anything
    else binds one dimension (:class:`DimensionBinding`), which the control's value sets over
    the whole action, clipped to the space."""

    def __init__(self, controls: dict[str, Any], space: spaces.Box) -> None:
        self.low, self.high = space.low, space.high
        self.dimensions = {control: DimensionBinding.from_dict(control, value, space)
                           for control, value in controls.items() if not isinstance(value, list)}
        controls_by_dimension: dict[int, str] = {}
        for control, binding in self.dimensions.items():
            if binding.dimension in controls_by_dimension:
                raise ValueError(f"controls: {control!r} and "
                                 f"{controls_by_dimension[binding.dimension]!r} both set "
                                 f"dimension {binding.dimension}; one control sets a dimension")
            controls_by_dimension[binding.dimension] = control
        if self.dimensions and "" not in controls:
            raise ValueError('controls: "" missing: it is the action whose dimensions no '
                             "control sets")
        super().__init__({combo: value for combo, value in controls.items()
                          if isinstance(value, list)}, space)

    def resolve(self, held: frozenset[str], values: dict[str, float]) -> np.ndarray:
        action = super().resolve(held, values)
        if not self.dimensions:
            return action
        action = action.copy()
        for control, binding in self.dimensions.items():
            action[binding.dimension] = binding.dimension_value(control, values.get(control, 0.0))
        return np.clip(action, self.low, self.high).astype(action.dtype)

    def turn_actions(self) -> dict[str, Any]:
        return {**super().turn_actions(),
                **{control: self.resolve(frozenset({control}), {control: 1.0})
                   for control in self.dimensions}}

    def _action(self, space: spaces.Box, combo: str, value: Any) -> np.ndarray:
        action = np.asarray(value, dtype=space.dtype)
        if not space.contains(action):
            raise ValueError(f"controls: {combo!r}: {value!r} is not an action of the env's "
                             f"{space}")
        return action


def make_keymap(spec: dict, space: gym.Space) -> Keymap:
    """The phase's :class:`Keymap`, of the class its env's action space calls for.

    :param spec: the game-phase config (``controls``, ``is_turn_based``).
    :param space: the env's action space.
    :raises TypeError: an action space other than MultiBinary, Discrete or Box.
    :raises ValueError: a real-time Discrete / Box phase whose ``controls`` lack ``""``.
    """
    controls = spec["controls"]
    if isinstance(space, spaces.MultiBinary):
        return MultiBinaryKeymap(controls, space)
    if isinstance(space, spaces.Discrete):
        cls: type[Keymap] = DiscreteKeymap
    elif isinstance(space, spaces.Box):
        cls = BoxKeymap
    else:
        raise TypeError(f"controls: the env's action space is {space}; a keymap drives "
                        "MultiBinary, Discrete or Box only")
    if "" not in controls and not spec.get("is_turn_based", False):
        raise ValueError(f'controls: "" missing: the env\'s action space is {space}, and a '
                         "phase that is not turn-based needs the action sent on a frame with "
                         "no control held")
    return cls(controls, space)
