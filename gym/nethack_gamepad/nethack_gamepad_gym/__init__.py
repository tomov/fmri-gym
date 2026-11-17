"""A native NetHack window port registered lazily by Gymnasium."""

from __future__ import annotations

from gymnasium.envs.registration import register

register(id="NetHackGamepad-v0", entry_point="nethack_gamepad_gym.env:NetHackGamepad")
