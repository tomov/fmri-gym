"""Default adapter -- works with ANY Gymnasium environment.

No engine-specific savestate is assumed. Reconstruction relies on the env being
deterministic under a fixed seed + action sequence (true for most gym envs); we
store the seed and per-frame actions, and the observation itself as the
analysis "state" (for many envs, e.g. CartPole, the observation IS the full
state). The same assumption is the savestate a block ends on, so a run that
stops mid-episode can be resumed: ``replay_state_gym.ReplayState``'s seed and
actions, replayed on restore. No frame carries one, so the log stays small. A phase's "controls" are written as the env's action space takes them: an
index for Discrete, a list for Box.

For old-`gym` (pre-Gymnasium) envs, pass them through shimmy -- see
make_via_shimmy() -- and everything else here still applies.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from .base import EnvAdapter, FrameState


class DefaultAdapter(EnvAdapter):
    name: str = "gym"

    def _make(self, spec: dict) -> gym.Env:
        # Many third-party envs only register their ids as a side effect of
        # importing their package (crafter, minihack, tile_match_gym, ...).
        # A curriculum can name that module via "import_module".
        import_mod = spec.get("import_module")
        if import_mod:
            import importlib
            importlib.import_module(import_mod)
        from replay_state_gym import ReplayState

        kwargs = dict(spec.get("env_kwargs", {}))
        kwargs.setdefault("render_mode", "rgb_array")
        if spec.get("legacy_gym"):
            return ReplayState(_make_via_shimmy(spec["game"], **kwargs))
        return ReplayState(gym.make(spec["game"], **kwargs))

    def capture(
        self, obs: Any, info: dict, want_blob: bool = True
    ) -> FrameState:
        variables = {"obs": np.asarray(obs), **(info or {})}
        return FrameState(blob=None, variables=variables)

    def restore(self, blob: bytes) -> None:
        self.env.set_state(blob)

    def block_extra(self) -> dict:
        return {"state": self.env.get_state()}


def _make_via_shimmy(game_id: str, **kwargs: Any) -> gym.Env:
    """Wrap an old-`gym` env id as a Gymnasium env using shimmy."""
    import shimmy  # noqa: F401  (registers compatibility envs on import)
    # Gymnasium exposes the v0.21 compat entrypoint once shimmy is installed.
    return gym.make("GymV21Environment-v0", env_id=game_id, **kwargs)
