"""Baba Is Auto adapter -- Baba Is You's levels on the baba-is-auto C++ engine, via ``baba-auto-gym``.

utilForever/baba-is-auto simulates the real game's ruleset -- pushes, word
tiles parsed into rules each turn, SINK / HOT-MELT / DEFEAT / OPEN-SHUT,
transformations, MOVE -- on the original levels' text maps, where ``baba``
(nacloos/baba-is-ai, the other Baba backend) generates small synthetic
puzzles. ``baba_auto_gym`` (``gym/baba_auto/``) is the Gymnasium env over
its ``pyBaba`` module, drawing with the engine's own GUI sprites (24 px tiles;
the display scales the frame up). The engine is built from, and the maps and
sprites read from, the baba-is-auto checkout at ``external/baba_auto`` (README
"External checkouts"), or the phase's ``repo``.

A phase's ``keys`` index ``Discrete(5)``: 0 = wait, 1 = up, 2 = down,
3 = left, 4 = right. The phase's ``game`` is a map name under the checkout's
``Resources/Maps`` (``baba_is_you``, ``out_of_reach``, ``off_limits``, ...) or
a map file's path. Logged per turn: ``play_state`` (0 PLAYING, 1 WON, 2 LOST)
and the ``(16, H, W)`` state tensor. ``pyBaba.Game`` has no serializer (its
map, object ids, noun history and RNG are private C++), so the savestate is
``replay_state_gym.ReplayState``'s: the seed and the moves since, restored
by replay, at a few ms per move.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from .base import EnvAdapter, FrameState

_PLAY_STATE = {"PLAYING": 0, "WON": 1, "LOST": 2}


class BabaAutoAdapter(EnvAdapter):
    name: str = "baba_auto"

    def _make(self, spec: dict) -> gym.Env:
        from baba_auto_gym import BabaAutoEnv
        from replay_state_gym import ReplayState

        return ReplayState(BabaAutoEnv(spec["game"], repo=spec.get("repo")))

    def outcome(self, terminated: bool, truncated: bool) -> tuple[str, str]:
        # The engine's PlayState, WON or LOST, is what ended the episode.
        if terminated:
            won = self.last_info.get("play_state") == "WON"
            return ("won", "Game won") if won else ("lost", "Game lost")
        return super().outcome(terminated, truncated)

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> FrameState:
        return FrameState(variables={"play_state": _PLAY_STATE[info["play_state"]],
                                     "state": np.asarray(obs).copy()})

    def restore(self, blob: bytes) -> None:
        self.env.set_state(blob)

    def block_extra(self) -> dict:
        return {"state": self.env.get_state()}
