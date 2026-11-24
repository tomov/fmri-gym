"""ContinuousBoxKeymap: a ``Box`` phase where a control's value sets a dimension of the action."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from gymnasium import spaces

from ..inputs import value_range
from .base import Keymap


class DimensionBinding(NamedTuple):
    """One dimension of a ``Box`` action, set by one control's value.

    Written as the dimension's index, which takes the value as it is, or as
    ``{"dimension": d, "range": [a, b]}``, which maps the control's range
    (:func:`~humans.inputs.value_range`) onto ``a..b`` (``[1, -1]`` flips it).
    """

    dimension: int
    #: what the action's dimension takes at the control's least and greatest value; None: the value
    action_range: tuple[float, float] | None

    @classmethod
    def from_dict(cls, control: str, value: Any, space: spaces.Box) -> DimensionBinding:
        """The binding ``value`` writes for ``control``.

        :param control: the control's name, as ``controls`` writes it.
        :param value: what ``controls`` maps it to: an index, or a dimension and a range.
        :param space: the env's action space.
        :return: the binding.
        :raises ValueError: a combo, a dimension the space has not, or a range that is not
            two numbers.
        """
        where = f"controls: {control!r}: {value!r}"
        if "+" in control or control == "":
            raise ValueError(f"{where}: a dimension of the action is set by one control")
        if isinstance(value, dict) and set(value) != {"dimension", "range"}:
            raise ValueError(f'{where}: expected {{"dimension": <index>, "range": [<at the '
                             'least value>, <at the greatest>]}')
        dimension = value["dimension"] if isinstance(value, dict) else value
        if (isinstance(dimension, bool) or not isinstance(dimension, int)
                or len(space.shape) != 1 or not 0 <= dimension < space.shape[0]):
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
        """What the dimension takes when ``control``'s value is ``value``.

        :param control: the control bound.
        :param value: its value, within :func:`~humans.inputs.value_range`.
        :return: the value as it is, or mapped onto :attr:`action_range`.
        """
        if self.action_range is None:
            return value
        least, greatest = value_range(control)
        at_least, at_greatest = self.action_range
        return at_least + (value - least) / (greatest - least) * (at_greatest - at_least)


class ContinuousBoxKeymap(Keymap):
    """``Box`` with dimensions: the most specific held combo's list (``""`` with none) is the
    action, and each control bound to a dimension sets it from its value
    (:class:`DimensionBinding`), clipped to the space.

    :raises ValueError: ``""`` missing, a binding it cannot read, or two controls on one
        dimension.
    """

    def __init__(self, controls: dict[str, Any], space: spaces.Box) -> None:
        if "" not in controls:
            raise ValueError('controls: "" missing: it is the action whose dimensions no '
                             "control sets")
        self.low, self.high = space.low, space.high
        super().__init__(controls, space)
        controls_by_dimension: dict[int, str] = {}
        for control, binding in self._dimension_bindings():
            if binding.dimension in controls_by_dimension:
                raise ValueError(f"controls: {control!r} and "
                                 f"{controls_by_dimension[binding.dimension]!r} both set "
                                 f"dimension {binding.dimension}; one control sets a dimension")
            controls_by_dimension[binding.dimension] = control

    def _action(self, space: spaces.Box, combo: str, value: Any) -> Any:
        if not isinstance(value, list):
            return DimensionBinding.from_dict(combo, value, space)
        action = np.asarray(value, dtype=space.dtype)
        if not space.contains(action):
            raise ValueError(f"controls: {combo!r}: {value!r} is not an action of the env's "
                             f"{space}")
        return action

    def _dimension_bindings(self) -> list[tuple[str, DimensionBinding]]:
        """Each control bound to a dimension, and its binding."""
        return [(next(iter(combo)), binding) for combo, binding in self.combos.items()
                if isinstance(binding, DimensionBinding)]

    def _matched(self, held: frozenset[str]) -> list[frozenset[str]]:
        """The held combos whose value is a whole action (``""`` always is)."""
        return [combo for combo in super()._matched(held)
                if not isinstance(self.combos[combo], DimensionBinding)]

    def resolve(self, held: frozenset[str], values: dict[str, float]) -> np.ndarray:
        """The most specific held combo's action, each bound dimension set by its control.

        :param held: as :meth:`Keymap.resolve`.
        :param values: as :meth:`Keymap.resolve`. A control not in it (a key or button) reads
            1.0 when held and 0.0 when not.
        :return: the action to send to the env this frame.
        """
        action = self.combos[max(self._matched(held), key=len)].copy()
        for control, binding in self._dimension_bindings():
            value = values.get(control, float(control in held))
            action[binding.dimension] = binding.dimension_value(control, value)
        return np.clip(action, self.low, self.high).astype(action.dtype)
