"""SuperTuxKart adapter (``stk``) -- the current game, in a gym env that returns the screen.

The backend is ``stk``; the package it drives is ``stk_gym``, the client half of
``supertuxkart-gym`` (don't confuse the two). That is SuperTuxKart with a
``--gym`` server built in, stepped over a pipe. It is chosen over ``pystk2`` (the
``supertuxkart`` backend) for the current engine, tracks and physics, and because
the same env object is what models train against.

Why this shape: with ``render_mode="rgb_array"`` the game renders into a window
that is created hidden (never mapped, so it neither steals focus nor gets resized
by the window manager) and every ``step`` brings the frame back, HUD included, at
the requested ``screensize``; ``display.py`` scales it to the screen. With
``action_mode="keys"`` the env's action *is* the set of keys held -- a
``MultiBinary(8)`` over left, right, up, down, nitro, skid, fire, rescue, the
indices a phase's ``keys`` name -- and the game's own player controller turns
it into controls, so steering ramps and skids latch exactly as they do for a
keyboard. The participant and a model are therefore in front of the same env.

Replay: a run reproduces bit for bit from a fresh launch, but a restart of the
race in the same process leaves the physics world a little off a fresh one
(``stk_gym``'s README measures it: millimetres, growing to centimetres over a
few hundred steps). So every episode but the block's first gets a fresh
launch (``reset``), about half a second of track loading between races, and an
episode is then exact from its launch ``seed``, its episode seed and its
actions alone.

No savestate, so no ``--resume`` of an episode a run stopped: the game keeps
none of its own, and one by replay (``replay_state_gym.ReplayState``) is a
fresh launch and the whole episode played again, about 6 s plus a third of
the time already played -- minutes of a frozen screen, charged to the block,
after a long race. A run that tries refuses to start (:func:`fmri_gym.save.bids
.last_position`).

Timing: the game is lock-stepped, one ``step`` per fmri-gym frame, so
``fps`` here must equal the game's physics rate divided by ``frame_skip``
(120 / 2 = 60 by default); ``_make`` refuses a config where they disagree.
Measured on an Intel Arc laptop at 640x360 with four karts: 2.5 ms per step
mean, 2.9 ms p95 -- well inside 16.7 ms. If steps did run long, run.py
catches up rather than drops frames, and ``run_time`` records it.

Needs a real OpenGL display (the frame is the game's own rendering) and the
game itself, which ``pip install supertuxkart-gym`` brings: the wheel fetches a
prebuilt binary and a trimmed asset pack once, for its own version.
``STK_ENV_BIN`` overrides that binary.
Not supported: the pystk2 backend's ``num_kart`` spelling (it is ``num_karts``
here, the game's), and a different track per episode (one process, one track):
a curriculum that wants several tracks uses one game phase per track. Arenas
and soccer fields need game modes this adapter never sets.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from .base import EnvAdapter, FrameState


class STKAdapter(EnvAdapter):
    name: str = "stk"
    _played: bool = False

    def _make(self, spec: dict) -> gym.Env:
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
        # run.py paces the loop at fps; the game advances frame_skip ticks
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

    def reset(self, seed: int) -> tuple[Any, dict]:
        """Relaunch the game, unless it is unused since its launch, and start a race."""
        if self._played:
            self.env.close()
            self.env = self._make(self.spec)
        self._played = True
        return super().reset(seed)

    def outcome(self, terminated: bool, truncated: bool) -> tuple[str, str]:
        # The env's is_success: the kart crossed the finish line.
        if terminated and self.last_info.get("is_success"):
            return "won", "Race finished"
        return super().outcome(terminated, truncated)

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> FrameState:
        # Every SAMPLE_FIELDS column every frame (NaN when the game did not
        # report), including the controls the kart applied.
        return FrameState(variables=self.env.unwrapped.sample())
