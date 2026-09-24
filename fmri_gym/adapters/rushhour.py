"""Rush Hour adapter -- the DBP "puzzle" pick, via ``rushhour-gym``.

Slide cars out of a 6x6 grid to free the red car. The rules run in a Go engine
(https://github.com/chrplr/Rush-Hour); ``rushhour-gym`` (PyPI) wraps it and
fetches the engine binary of its matching release on first use.

The env this adapter drives is ``RushHourHuman-v0``: the experiment program's
own interface -- the eight meta-actions of its button scheme (choose a car,
slide it), its picture (``rgb_array``: board, white outline and legal-slide
arrows on the chosen car, status line) and the columns of its results file in
``info``. All of that lives in the package; this adapter is the keymap plus
the ``info`` fields to log.

One game block is one puzzle: the phase's ``puzzle`` (``"p07"``; the library
is numbered easiest first) or ``puzzle_indices`` / ``min_moves_range`` picks
it, and the block ends when it is solved (``mode: "episode"``,
``n_episodes: 1``). The program's trial flow -- a self-paced ready screen, a
blank interval, a "PUZZLE SOLVED!" hold -- is not the env's business here: the
curriculum lists each as a ``message`` phase around the game phase, and
session.py runs them (see ``configs/dbp_games/rushhour__complete.json``).

Other phase fields passed to the env: ``movable_only`` (default true) and
``binary``, an engine binary of your own (else ``$RUSHHOUR_ENV_BIN``, else the
package's own lookup). A ``game`` other than ``RushHourHuman-v0`` (say
``RushHour-Easy-v0``, ``Discrete(32)``) has no keymap a participant can use
and is refused. Use ``turn_based: true``: a move is a key press.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import EnvAdapter, FrameState
from .keyspec import SingleKeySpec

_ENV_ID = "RushHourHuman-v0"
_ENV_KWARGS = ("puzzle", "puzzle_indices", "min_moves_range", "movable_only", "binary")
# info fields logged as per-frame variables: the program's results-file
# columns, plus what ties a row to the engine (env_action) and to the UI.
_LOGGED = ("event", "phase", "puzzle", "puzzle_index", "min_moves",
           "car", "orientation", "from_row", "from_col", "to_row", "to_col",
           "n_slides", "solved", "t_ms", "trial_ms",
           "env_action", "selected", "slot", "moved", "illegal")


class RushHourAdapter(EnvAdapter):
    name: str = "rushhour"

    def _make(self, spec: dict) -> Any:
        import gymnasium as gym
        import rushhour_gym  # noqa: F401  (registers RushHour*-v0)
        game = spec.get("game", _ENV_ID)
        if game != _ENV_ID:
            raise ValueError(
                f"rushhour backend: game must be {_ENV_ID!r} (a person's interface), "
                f"not {game!r}; the agent ids have no keymap a participant can use")
        kwargs = {k: spec[k] for k in _ENV_KWARGS if k in spec}
        return gym.make(_ENV_ID, render_mode="rgb_array", **kwargs)

    def _keyspec(self) -> SingleKeySpec:
        from rushhour_gym.human import DEFAULT_KEYS, NOOP
        combos = {frozenset([k]): v for k, v in DEFAULT_KEYS.items()}
        return SingleKeySpec(combos=combos, noop=NOOP)

    def capture(
        self, obs: Any, info: dict, want_blob: bool = True
    ) -> FrameState:
        info = info if isinstance(info, dict) else {}
        return FrameState(blob=None, variables={k: info.get(k) for k in _LOGGED if k in info})

    def rich_state(self, obs: Any, info: dict) -> dict | None:
        """Thin wrapper so Session (which calls ``adapter.rich_state(obs,
        info)`` by name) finds this hook -- see :meth:`get_rich_state`."""
        return self.get_rich_state(obs, info)

    def get_rich_state(self, obs: Any, info: dict) -> dict | None:
        """The actual board layout underneath the human-interface event
        columns :meth:`capture` logs (``_LOGGED`` -- car selected, slide
        direction, whether it was legal, ...): ``obs`` is the engine's own
        per-slot ``(row, col, length, horizontal)`` array (default
        ``obs_mode="cars"``), decoded per labeled car via ``info["labels"]``,
        plus the full legal-action mask.

        :param obs: the per-slot car-geometry array.
        :param info: info dict from the latest :meth:`step`/:meth:`reset`.
        :return: ``None`` if ``obs`` isn't array-like (not yet reset).
        """
        info = info if isinstance(info, dict) else {}
        if obs is None:
            return None
        board = np.asarray(obs)
        labels = info.get("labels", "")
        cars = [
            {"label": label, "row": int(row), "col": int(col),
             "length": int(length), "horizontal": bool(horizontal)}
            for label, (row, col, length, horizontal) in zip(labels, board.tolist())
            if length > 0
        ]
        return {
            "board": board.tolist(), "cars": cars, "labels": labels,
            "n_cars": info.get("n_cars"),
            "action_mask": (np.asarray(info["action_mask"]).tolist()
                            if "action_mask" in info else None),
        }
