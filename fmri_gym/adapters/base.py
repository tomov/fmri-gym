"""EnvAdapter -- the seam that makes the fMRI loop engine-agnostic.

The experiment loop (session.py) NEVER touches `env.unwrapped`, a specific
emulator, or any engine-specific API. Everything engine-specific lives behind
an EnvAdapter. To support a new backend you write one small adapter subclass;
the rest of the framework is unchanged.

An adapter is responsible for four things:

    make(spec)         -> create the gym.Env for one game block
    keymap(env)        -> how held keyboard keys become an env action
    capture(env)       -> the per-frame state we log (opaque blob + named vars)
    restore(env, blob) -> put the env back into a captured state (if supported)

State is returned in a STANDARD shape (a FrameState) so the logger and any
downstream analysis code are identical across ALE / stable-retro / plain gym.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class KeySpec:
    """A keyboard->action mapping for one game.

    combos: maps a frozenset of pressed key NAMES (pygame key names without the
            "K_" prefix, upper-case: "LEFT", "SPACE", "Z", ...) to the action to
            send. The most specific fully-held combo wins (see resolve()).
    noop:   the action to send when no combo matches.
    help:   optional human-readable control description shown on instructions.
    """

    combos: dict[frozenset[str], Any]
    noop: Any
    help: str = ""

    def resolve(self, held: frozenset[str]) -> Any:
        best, best_len = self.noop, -1
        for keys, action in self.combos.items():
            if keys <= held and len(keys) > best_len:
                best, best_len = action, len(keys)
        return best


@dataclass
class FrameState:
    """Everything an adapter exposes about the env at one frame.

    blob:      opaque bytes that `restore()` can turn back into this exact state
               (e.g. pickled ALE clone_state, retro em.get_state()). None if the
               backend has no in-memory savestate -- then reconstruction relies
               on seed + action replay instead.
    variables: named, analysis-friendly scalars/arrays surfaced uniformly via
               `info`, so the loop never calls getRAM()/get_ram() itself. Keys
               are backend-defined but SHOULD include "ram" when available.
    """

    blob: Optional[bytes] = None
    variables: dict[str, Any] = field(default_factory=dict)


class EnvAdapter:
    """Base class. Subclasses override the four hooks below."""

    #: short id used in filenames / manifest, e.g. "ale", "retro", "gym"
    name: str = "base"

    def make(self, spec: dict):
        """Return a gym-compatible env for one game block.

        `spec` is the game phase dict from the curriculum (already validated for
        the keys this adapter cares about). Must render RGB frames via
        env.render() with render_mode="rgb_array".
        """
        raise NotImplementedError

    def keymap(self, env) -> KeySpec:
        """Return the keyboard->action mapping for this env."""
        raise NotImplementedError

    def reset(self, env, seed: Optional[int], spec: dict):
        """Reset the env for a new episode. Return (obs, info).

        Adapters may use `spec` for per-episode setup (e.g. retro load_state).
        """
        return env.reset(seed=seed)

    def step(self, env, action):
        """Advance one frame. Return (obs, reward, terminated, truncated, info).

        Default is the Gymnasium contract; adapters with non-standard
        signatures (e.g. VGDL's step(a, with_img=)) override this.
        """
        return env.step(action)

    def capture(self, env, obs, info, want_blob: bool = True) -> FrameState:
        """Return the FrameState to log for the current frame.

        Called once per step. `obs`/`info` are the latest step() outputs so
        adapters can fold observation-derived state in without re-querying.
        When `want_blob` is False the caller does not need the (often expensive)
        savestate this frame, so adapters SHOULD skip computing blob and leave
        it None -- the cheap analysis variables should still be filled.
        """
        return FrameState()

    def restore(self, env, blob: bytes) -> None:
        """Inverse of capture().blob. Raise if the backend has no savestate."""
        raise NotImplementedError(f"{self.name} adapter has no in-memory savestate")

    def render(self, env):
        """Return the current RGB frame (H,W,3) uint8 for display.

        Default assumes the Gymnasium contract (env.render() with the env made
        using render_mode="rgb_array"). Adapters for non-standard envs override
        this (e.g. old-gym's env.render(mode="rgb_array")).
        """
        return env.render()

    def close(self, env) -> None:
        env.close()


def held_key_names(pygame_module, keys=None) -> frozenset[str]:
    """Return currently-held keys as upper-case names ("LEFT", "SPACE", ...).

    Kept here so every adapter/display shares one definition of "held keys".
    `keys` optionally restricts polling to a set of pygame key codes (faster).
    """
    pressed = pygame_module.key.get_pressed()
    if keys is None:
        # Scan the common game keys; cheap and avoids enumerating all 512 codes.
        keys = _COMMON_KEYS(pygame_module)
    out = set()
    for code, name in keys.items():
        if pressed[code]:
            out.add(name)
    return frozenset(out)


_COMMON_CACHE: dict = {}


def _COMMON_KEYS(pygame_module) -> dict:
    """{pygame_keycode: NAME} for arrows, space, enter, and letter keys."""
    if _COMMON_CACHE:
        return _COMMON_CACHE
    names = ["UP", "DOWN", "LEFT", "RIGHT", "SPACE", "RETURN", "TAB", "LSHIFT"]
    names += [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    names += [str(d) for d in range(10)]
    for n in names:
        # pygame constants: single letters are lowercase (K_a), everything else
        # is uppercase (K_UP, K_SPACE, K_RETURN, K_0). Getting this wrong makes
        # arrows/space undetectable -> every arrow/space action becomes NOOP.
        attr = "K_" + (n.lower() if (len(n) == 1 and n.isalpha()) else n)
        code = getattr(pygame_module, attr, None)
        if code is not None:
            _COMMON_CACHE[code] = n
    return _COMMON_CACHE
