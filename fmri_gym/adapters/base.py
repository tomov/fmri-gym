"""Keymap, FrameState, Sound, and the EnvAdapter base class."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Keymap:
    """The phase's ``keys``: what the subject presses -> what ``step`` gets.

    ``keys`` maps a key NAME (:mod:`fmri_gym.keys`), or several joined with
    ``"+"``, to the action to send, written as the env takes it. There is no
    default map and nothing is merged in: which key does what differs from
    site to site, so the file states all of it.

    - ``MultiBinary(n)``: a value is the index (0..n-1) of the button that key
      holds down, and every held key sets its button, so keys combine as on a
      controller. Nothing held is every button up, so there is no ``noop``.
    - ``Discrete`` or ``Box``: a value is an action of that space (an index, a
      list); the most specific combo whose keys are all held wins, and
      ``noop`` is sent when none is. No action of these spaces means "do
      nothing" everywhere (0 is FrozenLake's LEFT, MiniHack's "move N"), so a
      real-time phase has to name it.

    :param key_spec: the phase's ``keys`` (``validate_config`` has already
        checked that its names are real keys).
    :param env: the env the actions go to; its ``action_space`` is kept.
    :param noop: the phase's ``noop``, or ``None`` if it gives none.
    :raises TypeError: an action space other than those three.
    :raises ValueError: a value that is not an action of the space, or a
        ``noop`` for a ``MultiBinary`` space.
    """

    def __init__(self, key_spec: dict[str, Any], env: gym.Env, noop: Any | None) -> None:
        self.action_space = env.action_space
        self.combos = {frozenset(combo.split("+")): action for combo, action in key_spec.items()}
        if isinstance(self.action_space, spaces.MultiBinary):
            n = self.action_space.n
            bad = {k: v for k, v in key_spec.items()
                   if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v < n}
            if bad:
                raise ValueError(f"keys: {bad}: the env's action is MultiBinary({n}), so each "
                                 f"value is the index (0..{n - 1}) of the button that key holds "
                                 "down")
            if noop is not None:
                raise ValueError(f'"noop": {noop!r}: the env\'s action is MultiBinary({n}), whose '
                                 "no-key action is every button up; remove it")
            self.noop: Any = [0] * n
            return
        if not isinstance(self.action_space, (spaces.Discrete, spaces.Box)):
            raise TypeError(f"keys: the env's action space is {self.action_space}; a keymap "
                            "drives MultiBinary, Discrete or Box only")
        bad = {k: v for k, v in key_spec.items() if not self._is_action(v)}
        if bad:
            raise ValueError(f"keys: {bad} are not actions of the env's {self.action_space}")
        if noop is not None and not self._is_action(noop):
            raise ValueError(f'"noop": {noop!r} is not an action of the env\'s '
                             f"{self.action_space}")
        self.noop = noop

    def _is_action(self, value: Any) -> bool:
        """Whether a config value (an int, or a list for ``Box``) is an action of the space."""
        if isinstance(self.action_space, spaces.Box):
            value = np.asarray(value, dtype=self.action_space.dtype)
        return self.action_space.contains(value)

    def resolve(self, held: frozenset[str]) -> Any:
        """The action for the keys held this frame.

        :param held: the pressed keys' NAMES.
        :return: the button vector, or the most specific matched combo's
            action, or ``noop``.
        """
        if isinstance(self.action_space, spaces.MultiBinary):
            vec = [0] * self.action_space.n
            for keys, button in self.combos.items():
                if keys <= held:
                    vec[button] = 1
            return vec
        matched = [keys for keys in self.combos if keys <= held]
        return self.combos[max(matched, key=len)] if matched else self.noop

    def turn_actions(self) -> dict[str, Any]:
        """``{key: action}`` for the single-key entries: turn-based play steps on one press."""
        return {next(iter(keys)): self.resolve(keys) for keys in self.combos if len(keys) == 1}


@dataclass
class FrameState:
    """Everything an adapter exposes about itself at one frame.

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


@dataclass
class Sound:
    """One chunk of PCM an adapter wants played right now.

    The audio counterpart of the RGB frame :meth:`EnvAdapter.render` returns:
    the samples, plus the one thing an array cannot carry -- the rate they have
    to be played at, which is engine-specific (44100 Hz for Doom, 31440 for the
    ALE).

    :ivar pcm: samples shaped ``(n_samples, n_channels)``; the dtype is the
        sample format (``int16`` for most engines).
    :ivar sample_rate: samples per second at which ``pcm`` must be played.
    """

    pcm: np.ndarray
    sample_rate: float


class EnvAdapter:
    """The seam that makes the fMRI loop engine-agnostic.

    An EnvAdapter WRAPS one game environment for one game block: it builds the
    underlying engine env in ``__init__`` and keeps it (and any per-block state)
    private, exposing only the small interface the experiment loop needs. The
    loop (run.py) never sees the raw env, ``env.unwrapped``, or any
    engine-specific API -- it just calls the methods below on the wrapper.

    A fresh EnvAdapter is constructed per game block (see
    :func:`fmri_gym.adapters.get_adapter`), so per-block state lives naturally
    on ``self`` with no risk of leaking between blocks.

    Subclasses override :meth:`_make` (build the engine env) plus whichever of
    the hooks below they need; state is returned in a STANDARD shape (a
    :class:`FrameState`) so the logger and any downstream analysis code are
    identical across ALE / stable-retro / plain gym.

    :ivar spec: the game-phase config dict this env was built from.
    :ivar env: the underlying ``gymnasium.Env``.
    :ivar keymap: the phase's ``keys`` as a :class:`Keymap`.
    """

    env: gym.Env

    #: short id used in filenames / manifest, e.g. "ale", "retro", "gym"
    name: str = "base"

    def __init__(self, spec: dict) -> None:
        """Build the underlying env for one game block.

        :param spec: game-phase config dict from the curriculum (already
            validated for the keys this backend cares about).
        :raises ValueError: a real-time phase whose keymap has no ``noop``.
        """
        self.spec = spec
        self.env = self._make(spec)
        self.keymap = Keymap(spec["keys"], self.env, spec.get("noop"))
        if self.keymap.noop is None and not spec.get("turn_based", False):
            raise ValueError(f'"noop": missing; the env\'s action space is '
                             f"{self.keymap.action_space}, so state the action sent on a frame "
                             'with no key held, or set "turn_based": true to step only on key '
                             "presses")

    def _make(self, spec: dict) -> gym.Env:
        """Create and return the underlying env for one game block.

        A ``gymnasium.Env`` whose ``render()`` gives an RGB frame
        (``render_mode="rgb_array"``). A game whose own env speaks another API
        (old ``gym``, a bare engine) gets a thin Gymnasium env under ``vendor/``
        first; the adapter never papers over that itself. May also initialise
        per-block state on ``self``.

        :param spec: game-phase config dict from the curriculum.
        :return: the underlying environment, stored as ``self.env``.
        :raises NotImplementedError: always in the base class.
        """
        raise NotImplementedError

    def reset(self, seed: int | None) -> tuple[Any, dict]:
        """Reset the env for a new episode.

        Subclasses may use ``self.spec`` for per-episode setup (e.g. retro
        load_state).

        :param seed: RNG seed for this episode, or ``None``.
        :return: ``(obs, info)`` from ``env.reset``.
        """
        return self.env.reset(seed=seed)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Advance one frame.

        The Gymnasium contract; a subclass overrides it only to shape the
        action first (e.g. a list from the config into the space's dtype).

        :param action: action to apply (type depends on the env).
        :return: ``(obs, reward, terminated, truncated, info)``.
        """
        return self.env.step(action)

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> FrameState:
        """Return the :class:`FrameState` to log for the current frame.

        Called once per step. ``obs``/``info`` are the latest :meth:`step`
        outputs so subclasses can fold observation-derived state in without
        re-querying. When ``want_blob`` is ``False`` the caller does not need
        the (often expensive) savestate this frame, so subclasses SHOULD skip
        computing ``blob`` and leave it ``None`` -- the cheap analysis
        variables should still be filled.

        :param obs: observation from the latest step/reset.
        :param info: info dict from the latest step/reset.
        :param want_blob: if ``False``, skip expensive savestate capture.
        :return: a :class:`FrameState` (default empty in the base class).
        """
        return FrameState()

    def restore(self, blob: bytes) -> None:
        """Inverse of :attr:`FrameState.blob`: restore a captured state.

        :param blob: opaque bytes previously returned by :meth:`capture`.
        :raises NotImplementedError: if the backend has no in-memory savestate.
        """
        raise NotImplementedError(f"{self.name} env has no in-memory savestate")

    def render(self) -> np.ndarray:
        """Return the current RGB frame ``(H, W, 3)`` uint8 for display.

        The Gymnasium contract (``env.render()`` with the env made using
        ``render_mode="rgb_array"``). A subclass overrides it when the picture
        is not the env's render (a terminal drawn to pixels, a pixel
        observation).

        :return: RGB frame as a numpy array.
        """
        return self.env.render()

    def sound(self) -> Sound | None:
        """Return the sound to play for the current frame, or ``None``.

        The audio counterpart of :meth:`render`: the loop calls it once per
        frame and hands the result to the session's audio output, exactly as it
        hands :meth:`render` to the display. Default is ``None`` -- a silent
        backend, which is most of them.

        Return the PCM the engine produced during the last step, at its native
        rate: no resampling, no copying, no timing -- the session places it
        against the flip. It should last about one frame period (``1 / fps``);
        a block where it does not stops with the fps that would fit. Playback
        does not store it: to log the sound as well, return it from
        :meth:`capture` too.

        :return: a :class:`Sound`, or ``None`` if there is nothing to play.
        """
        return None

    def native_fps(self) -> float | None:
        """Steps per second at which the engine plays at its own real speed, or ``None``.

        An engine with a clock of its own (an emulator core's frame rate, a
        tic rate) advances a fixed amount of game time per step, so the block's
        ``fps`` decides how fast the game is, and people and models should meet
        the same game. The session does not enforce it -- ``fps`` has to suit
        the display, and a slowed-down block can be a choice -- it reports
        ``fps / native_fps`` in the manifest and says so when they differ.
        Default is ``None``: no clock of its own (a grid world, a turn-based
        puzzle), where ``fps`` is only how often the screen is redrawn.

        Read it from the live env where the engine tells; divide by any frame
        skip the env was built with.

        :return: steps per second at real speed, or ``None``.
        """
        return None

    def close(self) -> None:
        """Close the underlying env."""
        self.env.close()
