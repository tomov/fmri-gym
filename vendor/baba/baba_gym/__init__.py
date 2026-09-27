"""Baba Is You as a Gymnasium environment.

``gym.make("Baba-v0", game="env/make_win")`` or :class:`BabaEnv` directly.
See :mod:`baba_gym.env`.
"""

from __future__ import annotations

import gymnasium as gym

from .env import BabaEnv

__all__ = ["BabaEnv"]

gym.register(id="Baba-v0", entry_point="baba_gym.env:BabaEnv", disable_env_checker=True)
