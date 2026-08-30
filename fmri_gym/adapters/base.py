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
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass
class KeySpec:
    """A keyboard->action mapping for one game.

    :ivar combos: maps a frozenset of pressed key NAMES (pygame key names
        without the "K_" prefix, upper-case: "LEFT", "SPACE", "Z", ...) to the
        action to send. The most specific fully-held combo wins (see resolve()).
    :ivar noop: the action to send when no combo matches.
    """

    combos: dict[frozenset[str], Any]
    noop: Any

    def resolve(self, held: frozenset[str]) -> Any:
        """Map currently-held keys to an action.

        Among combos whose keys are all present in ``held``, pick the longest
        (most specific). Return ``noop`` if none match.

        :param held: frozenset of currently pressed key NAMES.
        :return: the action for the most specific matching combo, or ``noop``.
        """
        best, best_len = self.noop, -1
        for keys, action in self.combos.items():
            if keys <= held and len(keys) > best_len:
                best, best_len = action, len(keys)
        return best


@dataclass
class FrameState:
    """Everything an adapter exposes about the env at one frame.

    :ivar blob: opaque bytes that :meth:`EnvAdapter.restore` can turn back into
        this exact state (e.g. pickled ALE ``clone_state``, retro
        ``em.get_state()``). ``None`` if the backend has no in-memory savestate
        -- then reconstruction relies on seed + action replay instead.
    :ivar variables: named, analysis-friendly scalars/arrays surfaced uniformly
        via ``info``, so the loop never calls ``getRAM()``/``get_ram()`` itself.
        Keys are backend-defined but SHOULD include ``"ram"`` when available.
    """

    blob: bytes | None = None
    variables: dict[str, Any] = field(default_factory=dict)


class EnvAdapter:
    """Base class. Subclasses override the four hooks below."""

    #: short id used in filenames / manifest, e.g. "ale", "retro", "gym"
    name: str = "base"

    def make(self, spec: dict) -> gym.Env:
        """Return a gym-compatible env for one game block.

        ``spec`` is the game phase dict from the curriculum (already validated
        for the keys this adapter cares about). Must render RGB frames via
        ``env.render()`` with ``render_mode="rgb_array"``.

        :param spec: game-phase config dict from the curriculum.
        :return: a Gymnasium-compatible environment.
        :raises NotImplementedError: always in the base class.
        """
        raise NotImplementedError

    def keymap(self, env: gym.Env) -> KeySpec:
        """Return the keyboard->action mapping for this env.

        :param env: the live environment instance from :meth:`make`.
        :return: a :class:`KeySpec` for held-key / combo resolution.
        :raises NotImplementedError: always in the base class.
        """
        raise NotImplementedError

    def reset(self, env: gym.Env, seed: int | None, spec: dict) -> tuple[Any, dict]:
        """Reset the env for a new episode.

        Adapters may use ``spec`` for per-episode setup (e.g. retro load_state).

        :param env: the live environment instance.
        :param seed: RNG seed for this episode, or ``None``.
        :param spec: game-phase config dict (may carry load-state hints).
        :return: ``(obs, info)`` from ``env.reset``.
        """
        return env.reset(seed=seed)

    def step(self, env: gym.Env, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Advance one frame.

        Default is the Gymnasium contract; adapters with non-standard
        signatures (e.g. VGDL's ``step(a, with_img=)``) override this.

        :param env: the live environment instance.
        :param action: action to apply (type depends on the env).
        :return: ``(obs, reward, terminated, truncated, info)``.
        """
        return env.step(action)

    def capture(
        self, env: gym.Env, obs: Any, info: dict, want_blob: bool = True
    ) -> FrameState:
        """Return the :class:`FrameState` to log for the current frame.

        Called once per step. ``obs``/``info`` are the latest :meth:`step`
        outputs so adapters can fold observation-derived state in without
        re-querying. When ``want_blob`` is ``False`` the caller does not need
        the (often expensive) savestate this frame, so adapters SHOULD skip
        computing ``blob`` and leave it ``None`` -- the cheap analysis
        variables should still be filled.

        :param env: the live environment instance.
        :param obs: observation from the latest step/reset.
        :param info: info dict from the latest step/reset.
        :param want_blob: if ``False``, skip expensive savestate capture.
        :return: a :class:`FrameState` (default empty in the base class).
        """
        return FrameState()

    def restore(self, env: gym.Env, blob: bytes) -> None:
        """Inverse of :attr:`FrameState.blob`: restore a captured state.

        :param env: the live environment instance.
        :param blob: opaque bytes previously returned by :meth:`capture`.
        :raises NotImplementedError: if the backend has no in-memory savestate.
        """
        raise NotImplementedError(f"{self.name} adapter has no in-memory savestate")

    def render(self, env: gym.Env) -> np.ndarray:
        """Return the current RGB frame ``(H, W, 3)`` uint8 for display.

        Default assumes the Gymnasium contract (``env.render()`` with the env
        made using ``render_mode="rgb_array"``). Adapters for non-standard envs
        override this (e.g. old-gym's ``env.render(mode="rgb_array")``).

        :param env: the live environment instance.
        :return: RGB frame as a numpy array.
        """
        return env.render()

    def close(self, env: gym.Env) -> None:
        """Close the env if it exposes ``close()``.

        Not every env exposes ``close()`` (e.g. overcooked's OvercookedEnv).

        :param env: the live environment instance.
        """
        closer = getattr(env, "close", None)
        if callable(closer):
            closer()
