"""Baba Is You adapter -- the DBP "language" pick, via ``baba-gym``.

A Baba-Is-You-style puzzle where you push word blocks to rewrite the rules.
``baba_gym`` (``vendor/baba/``) is the Gymnasium env: nacloos/baba-is-ai speaks
the old ``gym`` API, and that package puts the Gymnasium contract in front of
it. The env's ``render()`` is a 256x256 frame of the grid.

A phase's ``keys`` index ``BabaIsYouEnv.Actions`` (baba/grid.py): 0 = idle,
1 = up, 2 = right, 3 = down, 4 = left. The phase's ``game`` is a ``baba`` env
id (``env/make_win``, ``env/goto_win``, ``env/you_win``, and the distractor
variants). No savestate -> seed + action replay.
"""

from __future__ import annotations

import gymnasium as gym

from .base import EnvAdapter


class BabaAdapter(EnvAdapter):
    name: str = "baba"

    def _make(self, spec: dict) -> gym.Env:
        from baba_gym import BabaEnv

        return BabaEnv(spec.get("game", "env/make_win"))
