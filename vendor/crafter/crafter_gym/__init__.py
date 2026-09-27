"""Crafter as a Gymnasium environment.

``gym.make("Crafter-v0")`` or :class:`CrafterEnv` directly. See
:mod:`crafter_gym.env`.
"""

from __future__ import annotations

import gymnasium as gym

from .env import CrafterEnv

__all__ = ["CrafterEnv"]

gym.register(id="Crafter-v0", entry_point="crafter_gym.env:CrafterEnv",
             disable_env_checker=True)
