"""COOM adapter -- Doom continual-RL scenarios (TTomilin/COOM).

``coom_gym`` (``vendor/coom/``) is the Gymnasium env. COOM's own package pins
gymnasium 0.28, so the env never imports it: it drives ``vizdoom.DoomGame``
against a checkout's scenario assets (``conf.cfg`` + ``<task>.wad`` under
``<repo>/COOM/env/scenarios/<scenario>/``, from the phase ``repo`` field or
``COOM_REPO``). This adapter builds one env per block and logs its sound and
game variables.

A phase's ``keys`` are the env's Discrete(12) indices (turn x move x execute;
see ``coom_gym``): 0 = noop, 1 = execute, 2 = forward, 3 = forward + execute,
4 = right, 6 = right + forward, 8 = left, 10 = left + forward. "Execute" is
the scenario's 4th button (JUMP for pitfall and parkour, ATTACK for chainsaw
and run_and_gun, SPEED for the rest but raise_the_roof, whose is USE).

``step()`` returns ViZDoom's raw reward. Game variables (health, ammo,
position, ...) are logged as an analysis variable, mirroring the ``vizdoom``
backend's ``gamevariables``. Opt-in native audio uses one 44.1 kHz stereo
buffer per Doom tic, so the phase must run at 35 fps. Playback uses the shared
Audio output; capture logs independent PCM copies even when playback is muted.
Terminal states have no audio buffer: log a marked zero placeholder, never
replay the previous step's sound.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import Env, EnvAdapter, FrameState, Sound


class COOMAdapter(EnvAdapter):
    name: str = "coom"
    _game: Any

    def _make(self, spec: dict) -> Env:
        """Build one COOM scenario env for this block.

        :param spec: game-phase config dict; ``game`` is the scenario name
            (e.g. "pitfall"), ``env_kwargs.task`` picks the WAD variant
            (default "default").
        :return: a :class:`coom_gym.COOMEnv`.
        :raises ValueError: audio is on, but the phase is not one tic per step.
        """
        from coom_gym import COOMEnv

        options = spec.get("env_kwargs", {})
        audio = options.get("audio_buffer_enabled", False)
        if audio and (spec.get("fps", 30) != 35 or spec.get("turn_based", False)):
            raise ValueError("COOM audio needs fps=35 and turn_based=false "
                             "(one Doom tic per step)")
        env = COOMEnv(
            spec["game"],
            repo=spec.get("repo"),
            task=options.get("task", "default"),
            seed=options.get("seed", 0),
            audio_buffer_enabled=audio,
            audio_efx=options.get("audio_efx", False),
        )
        self._game = env.game
        return env

    def sound(self) -> Sound | None:
        """Return this tic's native PCM, or None for disabled/terminal audio."""
        if not self._game.is_audio_buffer_enabled():
            return None
        state = self._game.get_state()
        if state is None:
            return None
        return Sound(state.audio_buffer, self._game.get_audio_sampling_rate())

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> FrameState:
        """Copy per-step variables/PCM; terminal placeholders keep arrays aligned."""
        state = self._game.get_state()
        variables = {"game_variables": state.game_variables.copy() if state is not None
                     else np.full(len(self._game.get_available_game_variables()), np.nan)}
        if self._game.is_audio_buffer_enabled():
            variables["audio_valid"] = state is not None
            variables["audio"] = (state.audio_buffer.copy() if state is not None else
                                  np.zeros((self._game.get_audio_sampling_rate() // 35, 2),
                                           dtype=np.int16))
        return FrameState(blob=None, variables=variables)

    def block_extra(self) -> dict | None:
        """Describe the logged PCM format and the engine's reverb setting."""
        if not self._game.is_audio_buffer_enabled():
            return None
        return {"audio_sampling_rate": self._game.get_audio_sampling_rate(),
                "audio_buffer_tics": 1,
                "audio_efx": self.spec.get("env_kwargs", {}).get("audio_efx", False)}
