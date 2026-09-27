"""VGDL games as standard Gymnasium environments.

``gym.make("VGDL-v0", game="aliens", level=0)`` or :class:`VGDLEnv` directly.
Needs a language_and_experience checkout (``repo=`` or ``VGDL_REPO``). See
:mod:`vgdl_gym.env`.
"""

from __future__ import annotations

import gymnasium as gym

from .env import VGDLEnv

__all__ = ["VGDLEnv"]

gym.register(id="VGDL-v0", entry_point="vgdl_gym.env:VGDLEnv", disable_env_checker=True)
