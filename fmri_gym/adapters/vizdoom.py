"""ViZDoom adapter -- Doom action-shooter scenarios (the COOM engine).

COOM (the DBP "action/shooter" pick) is a continual-RL suite built on ViZDoom,
but COOM itself pins gymnasium 0.28 which conflicts with the other backends
(minihack/nle need 1.2). ViZDoom -- the same Doom engine COOM uses -- ships
Gymnasium environments that work with our gymnasium and cover the same
action-shooter category (DefendCenter, DeadlyCorridor, HealthGathering,
TakeCover, MyWayHome, Deathmatch, and full Doom E1M1..). So this backend uses
ViZDoom directly.

The env's observation is a dict {"screen": (H,W,3) uint8, "gamevariables": ...};
`env.render()` (rgb_array) returns the screen for display, and we log
gamevariables (health, ammo, ...) as an analysis variable. Actions are a small
Discrete(n) button set per scenario; arrows/space map to the first n actions.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from .base import EnvAdapter, FrameState, KeySpec

_KEYS = ["LEFT", "RIGHT", "UP", "DOWN", "SPACE", "Z"]


class VizDoomAdapter(EnvAdapter):
    name = "vizdoom"

    def make(self, spec):
        from vizdoom import gymnasium_wrapper  # noqa: F401  (registers Vizdoom*-v1)
        return gym.make(spec["game"], render_mode="rgb_array",
                        **spec.get("make_kwargs", {}))

    def keymap(self, env) -> KeySpec:
        n = int(getattr(env.action_space, "n", 3))
        # Best-effort: bind the first n keys to actions 0..n-1. Doom scenarios
        # differ (turn/move/attack); override per game with a "keys" map.
        combos = {frozenset([k]): i for i, k in enumerate(_KEYS[:n])}
        return KeySpec(combos=combos, noop=0,
                       help="arrows + SPACE map to this scenario's buttons "
                            "(override with a per-phase 'keys' map)",
                       controls=[("ARROWS / SPACE", "scenario buttons "
                                  "(move / turn / attack, varies by scenario)")])

    def render(self, env):
        return np.asarray(env.render())

    def capture(self, env, obs, info, want_blob=True) -> FrameState:
        variables = {}
        if isinstance(obs, dict) and "gamevariables" in obs:
            variables["gamevariables"] = np.asarray(obs["gamevariables"])
        return FrameState(blob=None, variables=variables)
