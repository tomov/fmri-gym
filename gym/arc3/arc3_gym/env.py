"""ARC-AGI-3 (arcprize.org) games as a Gymnasium env, via the ``arc-agi`` toolkit.

Each game is a turn-based puzzle on a 64x64 grid of 16 colours, in several
levels, that ends in ``WIN`` or ``GAME_OVER``. The toolkit is not a Gymnasium
env (``Arcade.make(game_id, seed=)`` returns a wrapper whose ``step`` takes a
``GameAction`` and returns a ``FrameDataRaw``); this env puts the contract in
front of it and leaves the games untouched.

The toolkit runs in ``OFFLINE`` mode: it plays the game files already in
``environments_dir`` and makes no network call and needs no API key. The
files are downloaded once, by :func:`fetch` (``python -m arc3_gym ls20``),
which does need the network; the folder names carry the game's version
(``ls20/9607627b``), and ``info["game_id"]`` reports it.

The observation is the last frame layer of a step (``(64, 64)`` uint8, cell
values 0-15); a step that animates returns several layers and only the final
one is kept, and a step that ends the game returns none, so the previous grid
stays. ``render()`` colours that grid with the toolkit's own palette at
``cell_px`` pixels per cell, from the grid the step produced.

The action space is ``Discrete(k)`` over the game's simple actions in
ascending order, ``ACTION1..ACTION5`` then ``ACTION7`` (undo), keeping only the
ones the game offers; ``RESET`` is never an action. By convention ACTION1-4 are
up, down, left, right and ACTION5 is the game's own interact action. ``ACTION6``
is a click at an x, y position; a game that offers it is refused, because a
button box has no pointer. ``reward`` is the number of levels completed by the
step; ``terminated`` is ``WIN`` or ``GAME_OVER``. The game is deterministic:
an episode replays from its seed and actions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

#: fmri-gym's ``external/arc3``; this file is ``gym/arc3/arc3_gym/env.py``.
DEFAULT_GAMES_DIR = Path(__file__).resolve().parents[3] / "external" / "arc3"
_RESET, _CLICK = 0, 6
_ENDED = ("WIN", "GAME_OVER")
_LOG = logging.getLogger("arc3_gym")
_LOG.setLevel(logging.WARNING)


def fetch(game_ids: list[str], environments_dir: str | Path = DEFAULT_GAMES_DIR) -> None:
    """Download games' files into ``environments_dir``; the only call that needs the network.

    :param game_ids: four-letter game ids (``"ls20"``).
    :param environments_dir: where the toolkit keeps ``<game>/<version>/``.
    :raises RuntimeError: if the toolkit cannot make one of the games.
    """
    from arc_agi import Arcade

    arcade = Arcade(environments_dir=str(environments_dir), logger=_LOG)
    for game_id in game_ids:
        if arcade.make(game_id) is None:
            raise RuntimeError(f"could not download ARC-AGI-3 game {game_id!r}")


def _palette() -> np.ndarray:
    from arc_agi.rendering import COLOR_MAP, hex_to_rgb

    return np.array([hex_to_rgb(COLOR_MAP[i]) for i in range(16)], dtype=np.uint8)


class Arc3Env(gym.Env):
    """One ARC-AGI-3 game, played from level 1 to ``WIN`` or ``GAME_OVER``.

    :param game: four-letter game id (``"ls20"``), or ``"ls20-9607627b"`` for one version.
    :param environments_dir: the folder :func:`fetch` filled. Defaults to ``external/arc3``
        under the fmri-gym root.
    :param cell_px: pixels per grid cell in ``render()``.
    :param render_mode: ``"rgb_array"`` or ``None``; ``render()`` always returns the picture.
    :raises RuntimeError: if the game's files are not in ``environments_dir``, or the toolkit
        is not in offline mode.
    :raises ValueError: if the game offers ACTION6 (a click), or ``render_mode`` is unknown.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self, game: str = "ls20", *, environments_dir: str | None = None, cell_px: int = 8,
        render_mode: str | None = None,
    ) -> None:
        from arc_agi import Arcade, OperationMode

        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"render_mode must be 'rgb_array' or None, not {render_mode!r}")
        self.render_mode = render_mode
        self.game = game
        self.cell_px = cell_px
        self._dir = str(environments_dir or DEFAULT_GAMES_DIR)
        self._arcade = Arcade(
            operation_mode=OperationMode.OFFLINE, environments_dir=self._dir, logger=_LOG)
        if self._arcade.operation_mode != OperationMode.OFFLINE:
            raise RuntimeError(
                f"arc3 needs offline mode but the toolkit is in {self._arcade.operation_mode}; "
                "unset the OPERATION_MODE environment variable")
        self._palette = _palette()
        self._env = self._open(0)
        self._actions = self._simple_actions(self._env.reset())
        self.action_space = spaces.Discrete(len(self._actions))
        self.observation_space = spaces.Box(0, 15, (64, 64), np.uint8)
        self._grid = np.zeros((64, 64), np.uint8)
        self._levels = 0

    def _open(self, seed: int) -> Any:
        env = self._arcade.make(self.game, seed=seed)
        if env is None:
            raise RuntimeError(
                f"no ARC-AGI-3 game {self.game!r} in {self._dir}; download it once, with the "
                f"network on: python -m arc3_gym {self.game.split('-')[0]}")
        return env

    def _simple_actions(self, frame: Any) -> list[int]:
        offered = sorted(int(a) for a in frame.available_actions)
        if _CLICK in offered:
            raise ValueError(
                f"ARC-AGI-3 game {self.game!r} needs ACTION6 (a click); a button box has no "
                "pointer. Pick a game whose actions are ACTION1-5 only")
        return [a for a in offered if a != _RESET]

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        """Start the game over from level 1.

        :param seed: handed to the toolkit's ``make``.
        :param options: unused.
        :return: ``(obs, info)``.
        """
        super().reset(seed=seed)
        self._env = self._open(seed or 0)
        frame = self._env.reset()
        self._levels = 0
        return self._read(frame), self._info(frame)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply one action.

        :param action: an index into this game's simple actions.
        :return: ``(obs, reward, terminated, truncated, info)``; the reward is the number of
            levels the step completed.
        """
        from arcengine import GameAction

        frame = self._env.step(GameAction.from_id(self._actions[int(action)]))
        if frame is None:
            raise RuntimeError(f"ARC-AGI-3 game {self.game!r} refused action {int(action)}")
        gained = frame.levels_completed - self._levels
        self._levels = frame.levels_completed
        info = self._info(frame)
        return self._read(frame), float(gained), info["state"] in _ENDED, False, info

    def _read(self, frame: Any) -> np.ndarray:
        if len(frame.frame):
            self._grid = np.asarray(frame.frame[-1]).astype(np.uint8)
        return self._grid.copy()

    def _info(self, frame: Any) -> dict:
        return {
            "game_id": frame.game_id, "state": frame.state.name,
            "levels_completed": int(frame.levels_completed), "win_levels": int(frame.win_levels),
        }

    def render(self) -> np.ndarray:
        """Return the current grid in the toolkit's colours, ``(64*cell_px, 64*cell_px, 3)``."""
        rgb = self._palette[self._grid]
        return rgb.repeat(self.cell_px, axis=0).repeat(self.cell_px, axis=1)
