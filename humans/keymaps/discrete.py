"""DiscreteKeymap: a value is an action index; the most specific held combo wins."""

from __future__ import annotations

from typing import Any

from gymnasium import spaces

from .base import Keymap


class DiscreteKeymap(Keymap):
    """``Discrete``: a value is an action index; the most specific held combo wins."""

    def _action(self, space: spaces.Discrete, combo: str, value: Any) -> int:
        if isinstance(value, bool) or not space.contains(value):
            raise ValueError(f"controls: {combo!r}: {value!r} is not an action of the env's "
                             f"{space}")
        return value

    def resolve(self, held: frozenset[str], _: dict[str, float]) -> Any:
        return self.combos[max(self._matched(held), key=len)]
