"""Crafter adapter (danijar/crafter).

Crafter uses the OLD gym API shape -- reset() returns obs only, step() returns a
4-tuple (obs, reward, done, info) with no `truncated`, and it doesn't register
cleanly under gymnasium. We wrap crafter.Env directly and normalize it to the
gymnasium contract the session loop expects.

The observation IS the 64x64x3 RGB frame, so render() just returns obs. Crafter
has no savestate API; reconstruction is via seed + action replay (deterministic
given crafter.Env(seed=...)). The 17 discrete actions get a movement keymap;
the rest (do/place/make) are reachable via the curriculum "keys" override.
"""

from __future__ import annotations

import numpy as np

from .base import EnvAdapter, FrameState, KeySpec

# Default keyboard mapping over crafter's 17-action Discrete space.
# 0 noop,1 move_left,2 move_right,3 move_up,4 move_down,5 do,6 sleep,...
_KEYS = {"LEFT": 1, "RIGHT": 2, "UP": 3, "DOWN": 4, "SPACE": 5, "S": 6}


class CrafterAdapter(EnvAdapter):
    name = "crafter"

    def make(self, spec):
        import crafter
        seed = spec.get("seed")
        # crafter.Env seeds at construction; area/length are optional tunables.
        self._env = crafter.Env(seed=seed) if seed is not None else crafter.Env()
        self._last_obs = None
        return self._env

    def keymap(self, env) -> KeySpec:
        combos = {frozenset([k]): v for k, v in _KEYS.items()}
        return KeySpec(combos=combos, noop=0,
                       help="place/make actions via a keys override",
                       controls=[("LEFT/RIGHT/UP/DOWN", "move"),
                                 ("SPACE", "interact / do"),
                                 ("S", "sleep")])

    def reset(self, env, seed, spec):
        # Old-gym reset(): obs only. Re-seed per episode if supported.
        try:
            obs = env.reset(seed=seed) if seed is not None else env.reset()
        except TypeError:
            obs = env.reset()
        if isinstance(obs, tuple):  # be tolerant if a newer crafter returns (obs, info)
            obs, info = obs
        else:
            info = {}
        self._last_obs = np.asarray(obs)
        return self._last_obs, info

    def step(self, env, action):
        obs, reward, done, info = env.step(int(action))
        self._last_obs = np.asarray(obs)
        # Map old-gym `done` onto gymnasium (terminated, truncated).
        return self._last_obs, reward, bool(done), False, info

    def render(self, env):
        # obs is the RGB frame; avoids a second render call.
        return self._last_obs

    def capture(self, env, obs, info, want_blob=True) -> FrameState:
        # No savestate API -> rely on seed + action replay. Log the achievements
        # dict (crafter's semantic progress signal) when present.
        variables = {}
        if isinstance(info, dict) and "achievements" in info:
            variables["achievements"] = list(info["achievements"].values())
        return FrameState(blob=None, variables=variables)
