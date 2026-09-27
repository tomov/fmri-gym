"""VGDL adapter (Video Game Description Language games), via ``vgdl-gym``.

``vgdl_gym`` (``vendor/vgdl/``) is the Gymnasium env: it wraps the ``VGDLEnv``
of tomov/language_and_experience's ``dbp`` branch (the fork ported from old
gym to gymnasium, so it runs in this same numpy-2 env) and gives it the
standard contract -- a game name and level in, ``reset(seed=)``, an offscreen
``render()`` that leaves the fMRI window alone, and ``get_state``/``set_state``
for an exact savestate. The checkout is found through the phase's ``repo`` or
``VGDL_REPO`` and must be importable (``PYTHONPATH``).

A phase's ``keys`` index the fork's fixed action order: 0 = UP, 1 = DOWN,
2 = LEFT, 3 = RIGHT, 4 = NO_OP, 5 = SPACE; the no-key entry is ``"": 4``.

Phase fields (backend "vgdl"): ``game`` (``aliens``, ``beesAndBirds``, ...;
the ``games/<game>_v0/`` directory), ``level`` (default 0), ``block_size``
(pixels per cell, default 25), ``repo``. Logged each frame: the symbolic
per-cell object grid (``symbolic_state``), collision ``events``, and the
pickled savestate.
"""

from __future__ import annotations

import pickle
from typing import Any

import gymnasium as gym

from .base import EnvAdapter, FrameState


class VGDLAdapter(EnvAdapter):
    name: str = "vgdl"

    def _make(self, spec: dict) -> gym.Env:
        from vgdl_gym import VGDLEnv

        return VGDLEnv(spec["game"], level=spec.get("level", 0), repo=spec.get("repo"),
                       block_size=spec.get("block_size", 25))

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> FrameState:
        variables = {}
        if "state" in info:
            variables["symbolic_state"] = info["state"]
        if "events_triggered" in info:
            variables["events"] = info["events_triggered"]
        blob = pickle.dumps(self.env.get_state()) if want_blob else None
        return FrameState(blob=blob, variables=variables)

    def restore(self, blob: bytes) -> None:
        self.env.set_state(pickle.loads(blob))
