"""Boing! adapter -- Pong against the computer, via ``boing-gym``.

Boing! is the Pong of *Code the Classics* (Raspberry Pi Press); ``boing-gym``
(PyPI, https://github.com/chrplr/boing-gym) is the book's game with Pygame
Zero taken out, as a Gymnasium environment: one step per frame of its 60 Hz
loop, the book's sprites as the picture, the book's computer player on the
other bat, and the book's sound effects mixed into a per-frame PCM block.

The env this adapter drives is ``BoingHuman-v0``: ``Discrete(3)`` (stay, up,
down -- what the held key says this frame), ``rgb_array`` rendering, no step
budget (a match ends when a score passes 9). All of that lives in the package;
this adapter is the keymap, the audio hand-off and the ``info`` fields to log.

One game block is one match (``mode: "episode"``, ``n_episodes: 1``) or as
many as fit (``mode: "duration"``); the block must run real-time at
``fps: 60`` (speeds are pixels per frame), not turn-based. Instructions and
"game over" are ``message`` phases around it. The phase's ``audio`` (default
true) turns the sound on, played through the session's audio output; set it
false without an audio device.

A ``game`` other than ``BoingHuman-v0`` (``Boing-v0`` has a 100 s step
budget that would cut a participant off) is refused.
"""

from __future__ import annotations

from typing import Any

from .base import EnvAdapter, FrameState, Sound
from .keyspec import SingleKeySpec

_ENV_ID = "BoingHuman-v0"
# info fields logged as per-frame variables: what happened, where things are,
# the score.
_LOGGED = ("event", "ball_x", "ball_y", "ball_dx", "ball_dy", "ball_speed",
           "bat0_y", "bat1_y", "score0", "score1", "scorer", "n_hits", "game_over")


class BoingAdapter(EnvAdapter):
    name: str = "boing"

    def _make(self, spec: dict) -> Any:
        import boing_gym  # noqa: F401  (registers Boing*-v0)
        import gymnasium as gym
        game = spec.get("game", _ENV_ID)
        if game != _ENV_ID:
            raise ValueError(
                f"boing backend: game must be {_ENV_ID!r} (no step budget), not {game!r}")
        return gym.make(_ENV_ID, render_mode="rgb_array", audio=bool(spec.get("audio", True)))

    def _keyspec(self) -> SingleKeySpec:
        from boing_gym import DEFAULT_KEYS, NOOP
        combos = {frozenset([k]): v for k, v in DEFAULT_KEYS.items()}
        return SingleKeySpec(combos=combos, noop=NOOP)

    def sound(self) -> Sound | None:
        """The frame's mixed effects, or ``None`` with ``audio`` off.

        :return: a ``(rate/60, 2)`` int16 block at the env's rate, or ``None``.
        """
        env = self.env.unwrapped
        pcm = env.get_audio_buffer()
        if pcm is None:
            return None
        return Sound(pcm, env.get_audio_sampling_rate())

    def capture(
        self, obs: Any, info: dict, want_blob: bool = True
    ) -> FrameState:
        info = info if isinstance(info, dict) else {}
        return FrameState(blob=None, variables={k: info.get(k) for k in _LOGGED if k in info})
