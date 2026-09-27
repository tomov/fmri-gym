"""COOM scenarios as a Gymnasium environment.

``gym.make("COOM-v0", scenario="pitfall")`` or :class:`COOMEnv` directly.
Needs a TTomilin/COOM checkout (``repo=`` or ``COOM_REPO``) and does not
import the COOM package. See :mod:`coom_gym.env`.
"""

from __future__ import annotations

import gymnasium as gym

from .env import COOMEnv

__all__ = ["COOMEnv"]

gym.register(id="COOM-v0", entry_point="coom_gym.env:COOMEnv", disable_env_checker=True)
