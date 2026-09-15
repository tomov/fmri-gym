"""SuperTuxKart adapter (stk_gym) -- the current game, in a gym env that returns the screen.

``stk_gym`` is the Python client of the chrplr/stk-code fork: SuperTuxKart with a
``--gym`` server built in, stepped over a pipe. It is chosen over ``pystk2`` (the
``supertuxkart`` backend) for the current engine, tracks and physics, and because
the same env object is what models train against.

Why this shape: with ``render_mode="rgb_array"`` the game renders into a window
that is created hidden (never mapped, so it neither steals focus nor gets resized
by the window manager) and every ``step`` brings the frame back, HUD included, at
the requested ``screensize``; ``display.py`` scales it to the screen. With
``action_mode="keys"`` the env's action *is* the set of keys held -- a
``MultiBinary(8)`` over left, right, up, down, nitro, skid, fire, rescue -- and
the game's own player controller turns it into controls, so steering ramps and
skids latch exactly as they do for a keyboard. The participant and a model are
therefore in front of the same env, and a block replays from ``episode_seeds`` +
``actions`` (with the same launch ``seed``: see ``stk_gym``'s README).

Timing: the game is lock-stepped, one ``step`` per fmri-gym frame, so
``fps`` here must equal the game's physics rate divided by ``frame_skip``
(120 / 2 = 60 by default); ``_make`` refuses a config where they disagree.
Measured on an Intel Arc laptop at 640x360 with four karts: 2.5 ms per step
mean, 2.9 ms p95 -- well inside 16.7 ms. If steps did run long, session.py
catches up rather than drops frames, and ``session_time`` records it.

Needs a real OpenGL display (the frame is the game's own rendering) and the
fork's binary: ``STK_ENV_BIN``, or the fork's ``build/bin`` found by ``stk_gym``.
Not supported: the pystk2 backend's ``num_kart`` spelling (it is ``num_karts``
here, the game's), and a different track per episode (one process, one track).
"""

from __future__ import annotations

from typing import Any

from .base import EnvAdapter, FrameState
from .keyspec import MultiKeySpec

# One pygame key per entry of stk_gym's KEYS (left, right, up, down, nitro,
# skid, fire, rescue), in that order: SuperTuxKart's own default bindings.
_KEYS = ["LEFT", "RIGHT", "UP", "DOWN", "N", "V", "SPACE", "BACKSPACE"]


class STKGymAdapter(EnvAdapter):
    name: str = "stk_gym"

    def _make(self, spec: dict) -> Any:
        import stk_gym

        env = stk_gym.StkEnv(
            track=spec.get("track", "hacienda"),
            laps=spec.get("laps"),
            num_karts=spec.get("num_karts"),
            difficulty=spec.get("difficulty"),
            obs_mode="vector",
            action_mode="keys",
            render_mode="rgb_array",
            frame_skip=int(spec.get("frame_skip", 2)),
            screensize=spec.get("screensize", "640x360"),
            seed=spec.get("seed"),
            binary=spec.get("binary"),
        )
        # session.py paces the loop at fps; the game advances frame_skip ticks
        # per step. If the two disagree the race runs in slow or fast motion
        # relative to the scanner clock, silently.
        fps = int(spec.get("fps", 30))
        if env.metadata["render_fps"] != fps:
            env.close()
            raise ValueError(
                f"fps={fps} but the game steps at {env.metadata['render_fps']} Hz "
                f"(physics {env.engine.meta['physics_fps']} Hz / frame_skip "
                f"{env.engine.meta['frames']}); set frame_skip so they match"
            )
        return env

    def _keyspec(self) -> MultiKeySpec:
        # Held keys combine (steer while accelerating), so combo values are 0/1
        # vectors that MultiKeySpec ORs together -- the env's own action.
        n = len(_KEYS)
        combos = {frozenset([k]): [int(i == j) for j in range(n)]
                  for i, k in enumerate(_KEYS)}
        return MultiKeySpec(combos=combos, noop=[0] * n)

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> FrameState:
        # Every SAMPLE_FIELDS column every frame (NaN when the game did not
        # report), including the controls the kart applied.
        return FrameState(blob=None, variables=self.env.sample())
