"""Gymnasium interface to the complete native NetHackDS engine and window callbacks."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np

from .build import default_home
from .client import NativeSession
from .controller import Controller
from .rendering import Renderer


class _NativeEngine(gym.Env):
    """Expose controller presses; character selection and all menus belong to NetHack.

    :param runtime: explicitly built native runtime directory.
    :param player_directory: persistent saves; None isolates each reset for experiments.
    :param render_mode: only rgb_array is supported.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, runtime: str | None = None, player_directory: str | None = None,
                 render_mode: str = "rgb_array", game_mode: str = "classic") -> None:
        if render_mode != "rgb_array":
            raise ValueError("The native desktop port supports render_mode='rgb_array'")
        self.render_mode = render_mode
        directory = Path(runtime) if runtime else default_home() / "runtime"
        self.session = NativeSession(directory, Path(player_directory) if player_directory else None,
                                     game_mode=game_mode)
        self.renderer = Renderer(directory)
        self.controller: Controller | None = None
        self.action_space = gym.spaces.Discrete(8)
        self.observation_space = gym.spaces.Dict({
            "tiles": gym.spaces.Box(-1, 32767, (21, 79), dtype=np.int32),
            "glyphs": gym.spaces.Box(-1, 32767, (21, 79), dtype=np.int32),
            "pixels": gym.spaces.Box(0, 255, (720, 1280, 3), dtype=np.uint8),
        })
        self._frame: np.ndarray | None = None
        self.seed_value: int | None = None
        self._replaying = False
        self._replay_observation = {
            "tiles": np.full((21, 79), -1, dtype=np.int32),
            "glyphs": np.full((21, 79), -1, dtype=np.int32),
            "pixels": np.zeros((720, 1280, 3), dtype=np.uint8)}

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[dict, dict]:
        """Reset to the native name/character/restore flow, never a fixed role."""
        super().reset(seed=seed)
        if options:
            raise ValueError("No native reset options are defined")
        self.seed_value = seed
        self.session.start(seed)
        self.controller = Controller(self.session)
        return self._result()

    def _result(self) -> tuple[dict, dict]:
        if self._replaying:
            return self._replay_observation, {}
        self._frame = self.renderer.draw(self.session, self.controller)
        observation = {"tiles": np.asarray(self.session.tiles, dtype=np.int32),
                       "glyphs": np.asarray(self.session.glyphs, dtype=np.int32),
                       "pixels": self._frame}
        info = {"native": self.session.snapshot(), "controller": self.controller.state(),
                "engine_reply": self.controller.last_reply,
                "engine_stepped": self.controller.last_reply is not None,
                "engine": "NetHackDS 3.6.0 desktop-4-cursor", "seed": self.seed_value,
                "calendar_epoch": None if self.seed_value is None else 1609459200,
                "build": self.session.manifest.copy()}
        return observation, info

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        """Advance to the next native input boundary or update local menu focus."""
        if self.controller is None or self.session.finished:
            raise RuntimeError("Call reset() before stepping a native episode")
        if not self.action_space.contains(action):
            raise ValueError(f"Expected {self.action_space}, got {action}")
        self.session.game_events = []
        self.controller.press(int(action))
        observation, info = self._result()
        return observation, 0.0, self.session.finished, False, info

    def render(self) -> np.ndarray:
        """Return the cached image that is also present in the observation."""
        if self._frame is None:
            raise RuntimeError("Call reset() before rendering")
        return self._frame

    def close(self) -> None:
        """Stop this environment's native process."""
        self.session.close()


def NetHackGamepad(*, initial_state: str | None = None, **kwargs):
    """Build native gameplay with shared seed/action replay and optional campaign start."""
    from .campaign import Campaign
    return Campaign(_NativeEngine(**kwargs), initial_state=initial_state)
