"""The keymaps: a phase's ``controls`` -> the action ``step`` gets, one class per action space.

``controls`` maps a control NAME (:mod:`humans.inputs`), or several joined with ``"+"``,
to the action to send, written as the env takes it. The empty name ``""`` is the
action for no key held. There is no default map and nothing is merged in: which
key does what differs from site to site, so the file states all of it.

:func:`make_keymap` picks the class from the env's action space:

- :class:`MultiBinaryKeymap`: a value is the index of the button that key holds
  down, and every held control sets its button, so they combine as on a controller.
  Nothing held is every button up, so ``""`` is refused.
- :class:`DiscreteKeymap`, :class:`DiscreteBoxKeymap`: a value is an action of the
  space (an index, a list). The most specific combo whose controls are all held wins,
  and ``""`` -- the empty combo, held whenever nothing else is -- is what a frame
  with no control gets. No action of these spaces means "do nothing" everywhere (0
  is FrozenLake's LEFT, MiniHack's "move N"), so a real-time phase has to write
  it; an ``is_turn_based`` phase steps only on presses and needs none.
- :class:`ContinuousBoxKeymap`, for a ``Box`` phase where a control's value is a
  dimension, not a list: that control's value sets the dimension over the action the
  held combos give (:class:`DimensionBinding`).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces

from ..inputs import STICK_AXIS_NAMES
from .base import Keymap
from .continuous_box import ContinuousBoxKeymap, DimensionBinding
from .discrete import DiscreteKeymap
from .discrete_box import DiscreteBoxKeymap
from .multi_binary import MultiBinaryKeymap

__all__ = ["ContinuousBoxKeymap", "DimensionBinding", "DiscreteBoxKeymap", "DiscreteKeymap",
           "Keymap", "MultiBinaryKeymap", "make_keymap"]


def make_keymap(spec: dict, space: gym.Space) -> Keymap:
    """The phase's :class:`Keymap`, of the class its env's action space calls for.

    :param spec: the game-phase config (``controls``, ``is_turn_based``).
    :param space: the env's action space.
    :raises TypeError: an action space other than MultiBinary, Discrete or Box.
    :raises ValueError: a real-time Discrete / Box phase whose ``controls`` lack ``""``, a
        value that is not an action of the space, or a whole stick axis bound to anything but
        a dimension: it is never held.
    """
    keymap = _keymap_of_space(spec, space)
    for combo, action in keymap.combos.items():
        if combo & STICK_AXIS_NAMES and not isinstance(action, DimensionBinding):
            raise ValueError(f"controls: {'+'.join(sorted(combo))!r}: a whole stick axis is "
                             "never held, so it only sets a dimension of a Box action; bind "
                             "its sides to press")
    return keymap


def _keymap_of_space(spec: dict, space: gym.Space) -> Keymap:
    """The :class:`Keymap` of the class ``space`` calls for, built from the phase's controls."""
    controls = spec["controls"]
    if isinstance(space, spaces.MultiBinary):
        return MultiBinaryKeymap(controls, space)
    if isinstance(space, spaces.Discrete):
        cls: type[Keymap] = DiscreteKeymap
    elif isinstance(space, spaces.Box) and _binds_dimensions(controls):
        cls = ContinuousBoxKeymap
    elif isinstance(space, spaces.Box):
        cls = DiscreteBoxKeymap
    else:
        raise TypeError(f"controls: the env's action space is {space}; a keymap drives "
                        "MultiBinary, Discrete or Box only")
    if "" not in controls and not spec.get("is_turn_based", False):
        raise ValueError(f'controls: "" missing: the env\'s action space is {space}, and a '
                         "phase that is not turn-based needs the action sent on a frame with "
                         "no control held")
    return cls(controls, space)


def _binds_dimensions(controls: dict[str, Any]) -> bool:
    """Whether a control maps to a dimension rather than a whole action."""
    return any(not isinstance(value, list) for combo, value in controls.items() if combo)
