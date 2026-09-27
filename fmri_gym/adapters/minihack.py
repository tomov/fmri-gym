"""MiniHack adapter (facebookresearch/minihack, NetHack Learning Environment).

MiniHack's default observation is ASCII/tty and env.render() returns None, so we
request a pixel observation and display that. By default we show `pixel_crop` --
a 144x144 square window centered on the agent -- because the full `pixel`
observation is the entire 80-column NetHack terminal (336x1264, ~3.8:1) in which
a small room fills only a few percent of the frame, so aspect-fitting it makes
the game look tiny. Set the phase field "full_screen": true to display the whole
`pixel` frame instead. Actions are 8 compass directions (N,E,S,W,NE,SE,SW,NW ->
Discrete(8), indices 0..7), which a phase's ``keys`` index; some tasks append
more (Eat: 29 = eat). Play ``turn_based``: there is no no-op (0 is "move N").

No savestate API -> reconstruction is via seed + action replay (deterministic
under reset(seed=)). The full obs dict (glyphs, blstats, message, ...) is the
analysis state; we log the compact non-pixel fields and keep the pixel frame
for display only.

Requires `pkg_resources` (install setuptools<81) as minihack imports it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym

from .base import EnvAdapter, FrameState


class MiniHackAdapter(EnvAdapter):
    name: str = "minihack"

    def _make(self, spec: dict) -> gym.Env:
        import minihack  # noqa: F401  (registers MiniHack-* env ids)
        # Prefer the agent-centered square crop for display; the full terminal
        # ("pixel") only looks good with "full_screen": true.
        self._pixel_key = "pixel" if spec.get("full_screen") else "pixel_crop"
        keys = tuple(spec.get("observation_keys",
                              (self._pixel_key, "glyphs", "blstats", "message")))
        if self._pixel_key not in keys:
            keys = (self._pixel_key,) + keys
        env = gym.make(spec["game"], observation_keys=keys)
        self._last = None
        return env

    def reset(self, seed: int | None) -> tuple[Any, dict]:
        # Explicitly seed NetHack's core and dispersion RNGs for reproducible
        # action replay. This avoids relying solely on Gymnasium's reset seed,
        # whose propagation to NLE/NetHack is version-dependent.
        if seed is not None:
            self.env.unwrapped.seed(core=seed, disp=seed, reseed=False)
        obs, info = self.env.reset(seed=seed)
        self._last = obs
        return obs, info

    def render(self) -> np.ndarray:
        # Display the pixel observation (env.render() is None for MiniHack).
        return np.asarray(self._last[self._pixel_key])

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> FrameState:
        self._last = obs
        variables = {}
        # Compact symbolic fields make good analysis regressors; skip the big
        # pixel array (it's reconstructable via seed + action replay).
        for k in ("blstats", "glyphs", "message"):
            if isinstance(obs, dict) and k in obs:
                variables[k] = np.asarray(obs[k]).copy()
        return FrameState(blob=None, variables=variables)
