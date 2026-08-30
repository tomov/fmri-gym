"""KeySpec (+ its concrete flavors), FrameState, and the EnvAdapter base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass
class KeySpec(ABC):
    """A keyboard->action mapping for one game.

    Subclasses differ only in how they turn the matching combos into one
    action: :class:`SingleKeySpec` picks one, :class:`MultiKeySpec` ORs button
    vectors, :class:`HeldKeysSpec` joins the held key names.

    :ivar combos: maps a frozenset of pressed key NAMES (pygame key names
        without the "K_" prefix, upper-case: "LEFT", "SPACE", "Z", ...) to the
        action to send. A combo overrides its parts (see :meth:`maximal`).
    :ivar noop: the action to send when no combo matches.
    """

    combos: dict[frozenset[str], Any]
    noop: Any

    def maximal(self, held: frozenset[str]) -> list[frozenset[str]]:
        """Return the most specific combos fully held in ``held``.

        A combo matches when all of its keys are held; a match is dropped when
        another match is a strict superset of it, so e.g. an ``UP+SPACE`` combo
        shadows the plain ``UP`` and ``SPACE`` ones.

        :param held: frozenset of currently pressed key NAMES.
        :return: the matching combo key-sets, most specific ones only.
        """
        matched = [keys for keys in self.combos if keys <= held]
        return [keys for keys in matched
                if not any(keys < other for other in matched)]

    @abstractmethod
    def resolve(self, held: frozenset[str]) -> Any:
        """Map currently-held keys to an action.

        :param held: frozenset of currently pressed key NAMES.
        :return: the action to send to the env this frame.
        """


@dataclass
class SingleKeySpec(KeySpec):
    """One action at a time: the most specific fully-held combo wins.

    The right choice for a ``Discrete`` action space, where the env can only be
    told about one action per step.
    """

    def resolve(self, held: frozenset[str]) -> Any:
        """Return the most specific matching combo's action, else ``noop``.

        :param held: frozenset of currently pressed key NAMES.
        :return: the action for the most specific matching combo, or ``noop``.
        """
        matches = self.maximal(held)
        if not matches:
            return self.noop
        return self.combos[max(matches, key=len)]


@dataclass
class MultiKeySpec(KeySpec):
    """Several buttons at once: OR the button vectors of all matching combos.

    The right choice for a ``MultiBinary`` action space, where held keys should
    combine (drive forward while turning, run while shooting, ...).

    :ivar button_map: optional ``action -> per-button 0/1 row`` table. When set,
        combo values (and ``noop``) are indices into it, so a curriculum keymap
        can stay written in the env's ``Discrete`` action indices; when ``None``
        the combo values are already 0/1 button vectors.
    """

    button_map: list[list[int]] | None = None

    def expand(self, action: Any) -> list[int]:
        """Return ``action`` as a per-button 0/1 vector.

        :param action: a :attr:`button_map` index, or a 0/1 vector already.
        :return: the button vector for ``action``.
        """
        if self.button_map is None:
            return [int(v) for v in action]
        return list(self.button_map[int(action)])

    def resolve(self, held: frozenset[str]) -> list[int]:
        """Return the OR of the button vectors of all matching combos.

        :param held: frozenset of currently pressed key NAMES.
        :return: a 0/1 button vector (``noop``'s when nothing matches).
        """
        vec = self.expand(self.noop)
        for keys in self.maximal(held):
            for i, pressed in enumerate(self.expand(self.combos[keys])):
                if pressed:
                    vec[i] = 1
        return vec


@dataclass
class HeldKeysSpec(KeySpec):
    """Pass the held keys through: the env itself interprets the key set.

    For backends with no action space to index into, where ``step()`` turns the
    keys into engine input (browser games press/release them for real;
    supertuxkart assembles a steer/accelerate/brake action struct). ``combos``
    is a whitelist of meaningful keys; the action is a "+"-joined, sorted key
    string, which logs cleanly to npz ("" == nothing held).
    """

    def resolve(self, held: frozenset[str]) -> str:
        """Return the held keys that this game knows about, "+"-joined.

        :param held: frozenset of currently pressed key NAMES.
        :return: sorted key names joined with "+", or ``""`` if none are held.
        """
        known = {key for keys in self.combos for key in keys}
        return "+".join(sorted(held & known))


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
    """The seam that makes the fMRI loop engine-agnostic.

    The experiment loop (session.py) NEVER touches ``env.unwrapped``, a specific
    emulator, or any engine-specific API. Everything engine-specific lives behind
    an EnvAdapter. To support a new backend you write one small adapter subclass;
    the rest of the framework is unchanged.

    An adapter is responsible for four things:

        make(spec)         -> create the gym.Env for one game block
        keymap(env)        -> how held keyboard keys become an env action
        capture(env)       -> the per-frame state we log (opaque blob + named vars)
        restore(env, blob) -> put the env back into a captured state (if supported)

    State is returned in a STANDARD shape (a :class:`FrameState`) so the logger
    and any downstream analysis code are identical across ALE / stable-retro /
    plain gym.
    """

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
        :return: a concrete :class:`KeySpec` -- :class:`SingleKeySpec` for a
            ``Discrete`` space, :class:`MultiKeySpec` when held keys should
            combine, :class:`HeldKeysSpec` when ``step`` takes the key set.
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
