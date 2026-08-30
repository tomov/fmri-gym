"""Default adapter -- works with ANY Gymnasium environment.

No engine-specific savestate is assumed. Reconstruction relies on the env being
deterministic under a fixed seed + action sequence (true for most gym envs); we
store the seed and per-frame actions, and the observation itself as the
analysis "state" (for many envs, e.g. CartPole, the observation IS the full
state). Discrete and Box action spaces both get a sensible default keymap.

For old-`gym` (pre-Gymnasium) envs, pass them through shimmy -- see
make_via_shimmy() -- and everything else here still applies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .base import EnvAdapter, FrameState, KeySpec


class DefaultAdapter(EnvAdapter):
    name: str = "gym"

    def make(self, spec: dict) -> gym.Env:
        # Many third-party envs only register their ids as a side effect of
        # importing their package (crafter, minihack, tile_match_gym, ...).
        # A curriculum can name that module via "import_module".
        import_mod = spec.get("import_module")
        if import_mod:
            import importlib
            importlib.import_module(import_mod)
        kwargs = dict(spec.get("make_kwargs", {}))
        kwargs.setdefault("render_mode", "rgb_array")
        if spec.get("legacy_gym"):
            return _make_via_shimmy(spec["game"], **kwargs)
        return gym.make(spec["game"], **kwargs)

    def keymap(self, env: gym.Env) -> KeySpec:
        # Allow a curriculum to hand-specify a mapping: {"keys": {"LEFT": 0, ...}}
        # or {"keys": {"LEFT+SPACE": 2}} for combos.
        space = env.action_space
        combos, noop = {}, None

        if isinstance(space, spaces.Discrete):
            n = int(space.n)
            noop = 0
            # Generic arrows->first-N-actions mapping; games with meaningful
            # action semantics should override via the curriculum "keys" field.
            arrows = ["LEFT", "RIGHT", "UP", "DOWN"]
            for i, key in enumerate(arrows):
                if i < n:
                    combos[frozenset([key])] = i
        elif isinstance(space, spaces.Box):
            # Map arrow keys to +/- on the first (up to 2) continuous dims.
            lo, hi = np.asarray(space.low), np.asarray(space.high)
            noop = np.zeros(space.shape, dtype=space.dtype)
            def vec(dim: int, sign: int) -> np.ndarray:
                v = np.zeros(space.shape, dtype=space.dtype)
                v[dim] = (hi[dim] if sign > 0 else lo[dim])
                return v
            if space.shape[0] >= 1:
                combos[frozenset(["RIGHT"])] = vec(0, +1)
                combos[frozenset(["LEFT"])] = vec(0, -1)
            if space.shape[0] >= 2:
                combos[frozenset(["UP"])] = vec(1, +1)
                combos[frozenset(["DOWN"])] = vec(1, -1)
        else:
            raise TypeError(f"DefaultAdapter can't map action space {space!r}; "
                            "provide a custom adapter.")
        return KeySpec(combos=combos, noop=noop)

    def capture(
        self, env: gym.Env, obs: Any, info: dict, want_blob: bool = True
    ) -> FrameState:
        # No universal savestate: blob=None -> reconstruction is via seed+replay.
        # The observation is the analysis state for most gym envs.
        variables = {"obs": np.asarray(obs)}
        return FrameState(blob=None, variables=variables)


def _make_via_shimmy(game_id: str, **kwargs: Any) -> gym.Env:
    """Wrap an old-`gym` env id as a Gymnasium env using shimmy."""
    import shimmy  # noqa: F401  (registers compatibility envs on import)
    # Gymnasium exposes the v0.21 compat entrypoint once shimmy is installed.
    return gym.make("GymV21Environment-v0", env_id=game_id, **kwargs)
