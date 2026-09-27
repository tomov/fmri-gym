"""VGDL games (the ``dbp`` fork of ccolas/language_and_experience) as a standard Gymnasium env.

The fork's ``VGDLEnv`` is a ``gymnasium.Env`` already, ported from old gym,
but with habits of its own: it is built from game and level FILES, its
``reset`` and ``step`` take a ``with_img`` flag and no seed (seeding is
``game.set_seed``), ``render`` opens a pygame window (``display.set_mode`` on
whatever window the process already has, shrinking it to the game's size), and
``close`` quits pygame altogether. This wrapper takes a game name, a level and
a checkout, seeds on ``reset(seed=)``, draws on an offscreen surface, and
leaves the display alone.

Games live at ``<repo>/games/<game>_v0/<game>.txt`` and ``<game>_lvl<level>.txt``
(``aliens``, ``beesAndBirds``, ``avoidGeorge``, ``jaws``, ``missile_command``,
``plaqueAttack``, ``portals``, ``preconditions``, ``pushBoulders``,
``relational``); ``repo`` defaults to ``$VGDL_REPO`` and is put on ``sys.path``
(the fork's package is ``src.vgdl``). The observation is the fork's "objects"
observation; ``info`` carries the symbolic per-cell ``state``, the collision
``events_triggered``, ``won`` and ``lose``. Actions are the fork's fixed order:
0 = UP, 1 = DOWN, 2 = LEFT, 3 = RIGHT, 4 = NO_OP, 5 = SPACE.
:meth:`get_state` / :meth:`set_state` are the game's exact savestate.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np


class VGDLEnv(gym.Wrapper):
    """One VGDL game at one level.

    :param game: game name, the ``<game>_v0`` directory under ``games/``.
    :param level: level index, ``<game>_lvl<level>.txt``.
    :param repo: the language_and_experience checkout. Defaults to
        ``$VGDL_REPO``.
    :param block_size: pixels per grid cell in the rendered frame.
    :raises RuntimeError: no checkout path.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, game: str, *, level: int = 0, repo: str | None = None,
                 block_size: int = 25) -> None:
        repo = repo or os.environ.get("VGDL_REPO")
        if not repo:
            raise RuntimeError(
                "VGDL env needs the language_and_experience checkout; pass repo= or set VGDL_REPO")
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from src.vgdl.interfaces.gym.env import VGDLEnv as ForkEnv
        from src.vgdl.render import PygameRenderer

        files = Path(repo) / "games" / f"{game}_v0"
        env = ForkEnv(game_file=str(files / f"{game}.txt"),
                      level_file=str(files / f"{game}_lvl{level}.txt"),
                      obs_type="objects", block_size=block_size)
        env.renderer = _offscreen(PygameRenderer(env.game, block_size))
        super().__init__(env)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict]:
        """Restart the level.

        :param seed: seeds the game's RNG (sprite spawns, enemy moves).
        :param options: unused.
        :return: ``(obs, info)``.
        """
        if seed is not None:
            self.env.game.set_seed(int(seed))
        return self.env.reset(with_img=False)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Advance one game tick.

        :param action: an index into the fork's six actions.
        :return: ``(obs, reward, terminated, truncated, info)``.
        """
        return self.env.step(int(action), with_img=False)

    def render(self) -> np.ndarray:
        """Draw the board on the offscreen surface and return it, ``(H, W, 3)`` uint8."""
        renderer = self.env.renderer
        renderer.draw_all()
        return renderer.get_image()

    def close(self) -> None:
        """Nothing to close: the fork's ``close`` would quit pygame, which is the caller's."""

    def get_state(self) -> Any:
        """The game's exact, picklable state; :meth:`set_state` takes it back."""
        return self.env.get_state(return_orientation=True)

    def set_state(self, state: Any) -> None:
        """Restore a state from :meth:`get_state`."""
        self.env.set_state(state)


def _offscreen(renderer: Any) -> Any:
    """Give the fork's renderer a plain Surface to draw on instead of a window.

    ``PygameRenderer.init_screen`` would call ``pygame.display.set_mode``; this
    sets the same attributes by hand, so ``draw_all`` and ``get_image`` work
    and the display is never touched.
    """
    import pygame

    renderer.headless = True
    renderer.screen = pygame.Surface(renderer.screen_dims)
    renderer.screen.fill((255, 255, 255))
    renderer.background = renderer.screen.copy()
    return renderer
