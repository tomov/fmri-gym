"""MultiBinaryKeymap: a value is the index of the button that control holds down."""

from __future__ import annotations

from typing import Any

from gymnasium import spaces

from .base import Keymap


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

    def resolve(self, held: frozenset[str], _: dict[str, float]) -> list[int]:
        vec = [0] * self.n
        for combo in self._matched(held):
            vec[self.combos[combo]] = 1
        return vec
