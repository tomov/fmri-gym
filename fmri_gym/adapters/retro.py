"""stable-retro adapter (NES / SNES / Genesis / GB / ... via libretro).

Maps stable-retro behind the standard EnvAdapter interface:
- the action is MultiBinary over the console's buttons, so a phase's ``keys``
  are button indices in the core's order (``env.unwrapped.buttons``; Genesis:
  B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z; NES / Game Boy: B, -,
  SELECT, START, UP, DOWN, LEFT, RIGHT, A) and held keys combine;
- per-frame exact savestate via em.get_state()/set_state() (bit-exact, verified);
- state variables: the console RAM plus the game's decoded `info` variables
  (score/lives/... from the integration's data.json), surfaced uniformly.
- native PCM through sound(), played unless the phase sets "audio": false
  (retro does not log it). Match fps to the core's frame rate (Genesis: 59.92).

Notes verified against stable_retro 1.0.1:
- The emulator object is env.unwrapped.em; the libretro RAM view must be
  refreshed with data.update_ram() before get_ram() after a bare set_state.
- Named levels load via env.unwrapped.load_state(name) then reset().
- retro allows only ONE emulator per process; the session opens/closes one env
  per block, so this is respected as long as blocks don't overlap.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import stable_retro as retro

from .base import Env, EnvAdapter, FrameState, Sound


class RetroAdapter(EnvAdapter):
    name: str = "retro"

    def _make(self, spec: dict) -> Env:
        # save_pixels accepted for interface symmetry; retro frames are already
        # reconstructable from the per-frame state, so pixels aren't stored.
        self.save_pixels = bool(spec.get("save_pixels", False))
        return retro.make(
            game=spec["game"], scenario=spec.get("scenario"),
            render_mode="rgb_array")

    def reset(self, seed: int | None) -> tuple[Any, dict]:
        state = self.spec.get("state")
        if state:
            self.env.unwrapped.load_state(state)
        return self.env.reset()

    def capture(
        self, obs: Any, info: dict, want_blob: bool = True
    ) -> FrameState:
        u = self.env.unwrapped
        u.data.update_ram()
        variables = {"ram": u.get_ram().copy()}
        # Surface the game's decoded integration variables (score/lives/...).
        for k, v in (info or {}).items():
            variables[f"info_{k}"] = v
        # em.get_state() is ~1 MB for Genesis; only snapshot on stride frames.
        blob = u.em.get_state() if want_blob else None
        return FrameState(blob=blob, variables=variables)

    def native_fps(self) -> float:
        """The core's own frame rate (59.92 for Genesis, 60.10 for NES), one step per frame."""
        return self.env.unwrapped.em.get_screen_rate()

    def sound(self) -> Sound | None:
        """Return native emulator PCM for the session's audio output.

        :return: stereo PCM with its unrounded sample rate, or ``None``.
        """
        em = self.env.unwrapped.em
        return Sound(np.asarray(em.get_audio()), em.get_audio_rate())

    def restore(self, blob: bytes) -> None:
        u = self.env.unwrapped
        u.em.set_state(blob)
        u.data.update_ram()
