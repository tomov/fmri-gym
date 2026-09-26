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

A hidden window is not portable, though: it is a real X11 window that is never
mapped, and whether the driver draws into one is the driver's business. Mesa
(Intel, AMD) renders it via DRI3, so the frame is live; the NVIDIA proprietary
driver does not, and every frame comes back byte-identical to the first one -- a
frozen picture, which is what the participant sees. So ``_make`` *probes* the
hidden window once per process (:func:`_renders_hidden`) and, if it is frozen,
relaunches the game with its window on the screen, where every driver renders,
then puts our own window back in front of it and takes the keyboard focus back
(:func:`_reclaim_focus`). The compositor keeps rendering the game's window while
it is covered, so the frames stay live and only our display is visible. A visible
window is vsync'd, which would cost a whole refresh per step on top of our own
flip, so that launch gets a private STK config with vsync off
(:func:`_config_home`). ``"hidden": true`` / ``false`` in the phase decides it by
hand and skips the probe.

Timing: the game is lock-stepped, one ``step`` per fmri-gym frame, so
``fps`` here must equal the game's physics rate divided by ``frame_skip``
(120 / 2 = 60 by default); ``_make`` refuses a config where they disagree.
Measured on an Intel Arc laptop at 640x360 with four karts: 2.5 ms per step
mean, 2.9 ms p95 -- well inside 16.7 ms. On an NVIDIA Quadro T2000 through the
visible-window path above: 1.6-3.2 ms per step. If steps did run long, run.py
catches up rather than drops frames, and ``run_time`` records it.

Needs a real OpenGL display (the frame is the game's own rendering) and the
game itself, which ``pip install supertuxkart-gym`` brings: the wheel fetches a
prebuilt binary and a trimmed asset pack once, and a fork checkout beside it is
preferred over that download. ``STK_ENV_BIN`` overrides both.
Not supported: the pystk2 backend's ``num_kart`` spelling (it is ``num_karts``
here, the game's), and a different track per episode (one process, one track).
"""

from __future__ import annotations

import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .base import EnvAdapter, FrameState
from .keyspec import MultiKeySpec

# One pygame key per entry of stk_gym's KEYS (left, right, up, down, nitro,
# skid, fire, rescue), in that order: SuperTuxKart's own default bindings.
_KEYS = ["LEFT", "RIGHT", "UP", "DOWN", "N", "V", "SPACE", "BACKSPACE"]

# Does --gym-hidden actually render on this machine? Probed on the first block
# that asks for a hidden window and remembered, because finding out costs a
# launch of the game (half a second) and the answer is a property of the driver.
_HIDDEN_RENDERS: bool | None = None


def _config_home() -> Path:
    """XDG_CONFIG_HOME for the game, holding a config of ours with vsync off.

    Two reasons not to use the player's own config. Vsync is the one that forces
    it: a mapped window swaps on the refresh, so a step would block for up to
    16.7 ms *before* our own flip and the block could not hold 60 Hz
    (``--gym-hidden`` turns vsync off itself, so this only matters for the
    visible window). The other is that the game's own key bindings live in that
    directory, and a player who has rebound them would otherwise change what
    ``action_mode="keys"`` does.
    """
    home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return home / "fmri-gym" / "stk_gym-config"


def _vsync_off(config_home: Path) -> None:
    """Rewrite ``swap-interval-vsync`` to 0 in the game's config, if it exists.

    The game writes that file itself, so on the very first run there is nothing
    to patch; the launch that follows creates it and every later one is
    corrected. Nothing here creates or validates the file -- a missing or
    unexpected config just leaves vsync as the game's default.
    """
    for path in config_home.glob("supertuxkart/config-*/config.xml"):
        text = path.read_text()
        patched = re.sub(r'swap-interval-vsync="[^"]*"',
                         'swap-interval-vsync="0"', text)
        if patched != text:
            path.write_text(patched)


@contextmanager
def _own_config(config_home: Path) -> Iterator[None]:
    """Run the block with ``XDG_CONFIG_HOME`` pointed at ``config_home``.

    ``stk_gym`` spawns the game with our environment, so this is how the child
    is told where its config is; it is restored immediately after the spawn.
    """
    config_home.mkdir(parents=True, exist_ok=True)
    _vsync_off(config_home)
    before = os.environ.get("XDG_CONFIG_HOME")
    os.environ["XDG_CONFIG_HOME"] = str(config_home)
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("XDG_CONFIG_HOME", None)
        else:
            os.environ["XDG_CONFIG_HOME"] = before


def _renders_hidden(env: Any) -> bool:
    """True if the game's hidden window really is being drawn into.

    Steps the race a few ticks with no keys held and compares the frames: the
    intro camera sweeps over the track during the countdown, so a live window
    changes every frame, while a driver that does not render an unmapped window
    keeps handing back the first frame forever, byte for byte.
    """
    env.reset(seed=0)
    first = env.render()
    noop = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    for _ in range(8):
        env.step(noop)
    return not np.array_equal(first, env.render())


def _reclaim_focus() -> None:
    """Raise our display window over the game's and take the keyboard back.

    The game's window is mapped last, so the window manager puts it on top and
    gives it the focus: measured, our window goes ``focused=False`` the moment
    the game starts, which would cost us every key the participant presses.
    ``Window.focus()`` (SDL raise + set input focus) puts both back. Best-effort
    -- if there is no window yet the run still works, with the game's window in
    the way.
    """
    try:
        import pygame

        if pygame.display.get_init():
            pygame.Window.from_display_module().focus()
    except Exception:  # a window trick is never worth failing a run over
        pass


class STKGymAdapter(EnvAdapter):
    name: str = "stk_gym"

    def _make(self, spec: dict) -> Any:
        global _HIDDEN_RENDERS

        hidden = spec.get("hidden")
        if hidden is None:
            # Not yet probed -> try hidden, which is the cheap and tidy path;
            # known broken -> go straight to the window.
            hidden = _HIDDEN_RENDERS is not False
        env = self._launch(spec, hidden=bool(hidden))

        if hidden and _HIDDEN_RENDERS is None and spec.get("hidden") is None:
            _HIDDEN_RENDERS = _renders_hidden(env)
            if not _HIDDEN_RENDERS:
                print(
                    "WARNING stk_gym: this driver does not draw into the game's "
                    "hidden window (every frame came back identical, i.e. a frozen "
                    "picture). Relaunching the game with its window on the screen, "
                    "behind ours, which does render; set \"hidden\": true in the "
                    "game phase to refuse that and keep the hidden window.",
                    file=sys.stderr,
                )
                env.close()
                env = self._launch(spec, hidden=False)

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

    def _launch(self, spec: dict, *, hidden: bool) -> Any:
        """Build the env with the game's window hidden or on the screen."""
        import stk_gym

        def build() -> Any:
            return stk_gym.StkEnv(
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
                hidden=hidden,
            )

        # Both launches use the config of our own, so that the one the hidden
        # probe writes is already there -- patched -- if the visible relaunch
        # needs it. Otherwise the first visible block of the first session
        # would be the one that creates the config, and would run vsync'd.
        with _own_config(_config_home()):
            env = build()
        if not hidden:
            _reclaim_focus()
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
