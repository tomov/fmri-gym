"""ARC-AGI-3 games as a Gymnasium environment.

``gym.make("Arc3-v0", game="ls20")`` or :class:`Arc3Env` directly, after
:func:`fetch` has put the game's files on disk. See :mod:`arc3_gym.env`.
"""

from __future__ import annotations

import gymnasium as gym

from .env import Arc3Env, fetch

__all__ = ["Arc3Env", "fetch"]

gym.register(id="Arc3-v0", entry_point="arc3_gym.env:Arc3Env", disable_env_checker=True)
