"""COOM adapter -- Doom continual-RL scenarios (TTomilin/COOM).

``coom_gym`` (``gym/coom/``) is the Gymnasium env. COOM's own package pins
gymnasium 0.28, so the env never imports it: it drives ``vizdoom.DoomGame``
against a checkout's scenario assets (``conf.cfg`` + ``<task>.wad`` under
``<repo>/COOM/env/scenarios/<scenario>/``, from the phase ``repo`` field or
``COOM_REPO``).  This adapter builds one env per block and logs its sound,
game variables, and every object and label ``coom_gym`` turns on. The level's
sector geometry is logged once per episode, on its first frame: a level has
over a thousand sectors, and writing them on every frame costs more than the
frame budget at 35 fps.

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

Savestate: the scenario's logic is ACS inside the WAD and the env keeps no
state of its own, so the game is lockstep-deterministic from ``reset(seed=)``;
``DoomGame.save()`` would run two extra tics on every save (see the vizdoom
adapter), so the savestate is ``replay_state_gym.ReplayState``'s instead:
the seed and the actions since, restored by replay.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from .base import EnvAdapter, FrameState, Sound


class COOMAdapter(EnvAdapter):
    name: str = "coom"
    _game: Any
    _sectors_pending: bool = False

    def _make(self, spec: dict) -> gym.Env:
        """Build one COOM scenario env for this block.

        :param spec: game-phase config dict; ``game`` is the scenario name
            (e.g. "pitfall"), ``env_kwargs.task`` picks the WAD variant
            (default "default").
        :return: a :class:`coom_gym.COOMEnv`.
        :raises ValueError: audio is on, but the phase is not one tic per step.
        """
        from coom_gym import COOMEnv
        from replay_state_gym import ReplayState

        options = spec.get("env_kwargs", {})
        audio = options.get("audio_buffer_enabled", False)
        if audio and (spec.get("fps", 30) != 35 or spec.get("is_turn_based", False)):
            raise ValueError("COOM audio needs fps=35 and is_turn_based=false "
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
        return ReplayState(env)

    def reset(self, seed: int) -> tuple[Any, dict]:
        out = super().reset(seed)
        self._sectors_pending = True
        return out

    def sound(self) -> Sound | None:
        """Return this tic's native PCM, or None for disabled/terminal audio."""
        if not self._game.is_audio_buffer_enabled():
            return None
        state = self.env.unwrapped.state
        if state is None:
            return None
        return Sound(state.audio_buffer, self._game.get_audio_sampling_rate())

    def outcome(self, terminated: bool, truncated: bool) -> tuple[str, str]:
        # Doom ends an episode on the player's death or the scenario's tic
        # limit; a scenario with a goal (a vest to reach, a monster to kill) ends
        # there too, but the engine does not say which, so only death is read.
        if terminated and self._game.is_player_dead():
            return "lost", "You died"
        return super().outcome(terminated, truncated)

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> FrameState:
        state = self.env.unwrapped.state
        variables = {"game_variables": state.game_variables.copy() if state is not None
                     else np.full(len(self._game.get_available_game_variables()), np.nan)}
        variables["tic"] = state.tic if state is not None else -1
        variables["number"] = state.number if state is not None else -1
        variables["objects"] = _objects(state)
        variables["labels"] = _labels(state)
        if self._sectors_pending and state is not None:
            variables["sectors"] = _sectors(state)
            self._sectors_pending = False
        if self._game.is_audio_buffer_enabled():
            variables["audio_valid"] = state is not None
            variables["audio"] = (state.audio_buffer.copy() if state is not None else
                                  np.zeros((self._game.get_audio_sampling_rate() // 35, 2),
                                           dtype=np.int16))
        return FrameState(variables=variables)

    def restore(self, blob: bytes) -> None:
        self.env.set_state(blob)

    def block_extra(self) -> dict:
        """The block's end state, and the logged PCM format and reverb setting when sound is on."""
        if not self._game.is_audio_buffer_enabled():
            return {"state": self.env.get_state()}
        return {"state": self.env.get_state(),
                "audio_sampling_rate": self._game.get_audio_sampling_rate(),
                "audio_buffer_tics": 1,
                "audio_efx": self.spec.get("env_kwargs", {}).get("audio_efx", False)}


_OBJECT_FIELDS = ("id", "name", "position_x", "position_y", "position_z",
                   "angle", "pitch", "roll", "velocity_x", "velocity_y", "velocity_z")
_LABEL_FIELDS = ("object_id", "object_name", "object_category", "x", "y", "width", "height",
                  "value", "object_position_x", "object_position_y", "object_position_z")
_LINE_FIELDS = ("x1", "y1", "x2", "y2", "is_blocking")


def _objects(state: Any) -> list[dict]:
    if state is None or state.objects is None:
        return []
    return [{f: getattr(o, f) for f in _OBJECT_FIELDS} for o in state.objects]


def _labels(state: Any) -> list[dict]:
    if state is None or state.labels is None:
        return []
    return [{f: getattr(lb, f) for f in _LABEL_FIELDS} for lb in state.labels]


def _sectors(state: Any) -> list[dict]:
    if state is None or state.sectors is None:
        return []
    return [{"floor_height": s.floor_height, "ceiling_height": s.ceiling_height,
             "lines": [{f: getattr(ln, f) for f in _LINE_FIELDS} for ln in s.lines]}
            for s in state.sectors]
