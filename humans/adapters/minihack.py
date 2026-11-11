"""MiniHack adapter (facebookresearch/minihack, NetHack Learning Environment).

MiniHack's default observation is ASCII/tty and env.render() returns None, so we
request the ``pixel_crop`` observation and display that. What it shows is the
env id's to say, so that an agent trained on the id sees what the participant
sees: MiniHack's own ids crop a 9x9-tile window centred on the agent (the agent
holds still, the level slides), and ``minihack_levels`` (gym/minihack) registers
for each an ``-Allocentric-v0`` twin whose ``pixel_crop`` is the level's map,
fixed (the level holds still, the agent moves), alongside its own layouts:
``minihack_levels:MiniHack-Room-5x5-Allocentric-v0``. The whole-terminal
``pixel`` observation is never shown: the 80-column terminal (336x1264, ~3.8:1)
makes a small room a few percent of the picture.

Actions are 8 compass directions (N,E,S,W,NE,SE,SW,NW ->
Discrete(8), indices 0..7), which a phase's ``keys`` index; some tasks append
more (Eat: 29 = eat). Play ``is_turn_based``: there is no no-op (0 is "move N").

NLE has no savestate API and its game lives in a NetHack library loaded per
env, out of Python's reach, so the savestate is ``replay_state_gym.ReplayState``'s:
the seed and the actions since, restored by replay. That is exact only because
``minihack_levels.NLESeeding`` seeds NetHack's own RNGs on every reset and refuses one
where that did not take. The full obs dict (glyphs, blstats, message, ...) is
the analysis state; we log the compact non-pixel fields and keep the pixel
frame for display only.

Requires `pkg_resources` (install setuptools<81) as minihack imports it.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from .base import EnvAdapter, FrameState


class MiniHackAdapter(EnvAdapter):
    name: str = "minihack"

    def _make(self, spec: dict) -> gym.Env:
        import minihack  # noqa: F401  (registers MiniHack-* env ids)
        from minihack_levels import NLESeeding
        from replay_state_gym import ReplayState
        keys = tuple(spec.get("observation_keys", ("pixel_crop", "glyphs", "blstats", "message")))
        if "pixel_crop" not in keys:
            keys = ("pixel_crop",) + keys
        env = gym.make(spec["game"], observation_keys=keys, fix_moon_phase=True)
        self._last = None
        return ReplayState(NLESeeding(env))

    def reset(self, seed: int) -> tuple[Any, dict]:
        obs, info = super().reset(seed)
        self._last = obs
        return obs, info

    def render(self) -> np.ndarray:
        # Display the pixel observation (env.render() is None for MiniHack).
        return np.asarray(self._last["pixel_crop"])

    def outcome(self, terminated: bool, truncated: bool) -> tuple[str, str]:
        # NLE's end_status: TASK_SUCCESSFUL, DEATH, or ABORTED at the step limit.
        status = getattr(self.last_info.get("end_status"), "name", "")
        if status == "TASK_SUCCESSFUL":
            return "won", "Task complete"
        if status == "DEATH":
            return "lost", "You died"
        if status == "ABORTED":
            return "truncated", "Timeout"
        return super().outcome(terminated, truncated)

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> FrameState:
        self._last = obs
        variables = {}
        if isinstance(obs, dict):
            for k, v in obs.items():
                if k in ("pixel", "pixel_crop"):
                    continue
                variables[k] = np.asarray(v).copy()
        variables.update(info or {})
        return FrameState(variables=variables)

    def restore(self, blob: bytes) -> None:
        self._last, _ = self.env.set_state(blob)

    def block_extra(self) -> dict:
        return {"state": self.env.get_state()}
