"""DiscreteBoxKeymap: a ``Box`` phase whose every value is a whole action, a list."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from .discrete import DiscreteKeymap


class DiscreteBoxKeymap(DiscreteKeymap):
    """``Box``, a whole action per combo: a value is a list, sent as an array of the space's
    dtype."""

    def _action(self, space: spaces.Box, combo: str, value: Any) -> np.ndarray:
        action = np.asarray(value, dtype=space.dtype)
        if not space.contains(action):
            raise ValueError(f"controls: {combo!r}: {value!r} is not an action of the env's "
                             f"{space}")
        return action
