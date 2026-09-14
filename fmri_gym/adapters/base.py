"""FrameState and the EnvAdapter base class."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .keyspec import KeySpec


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


class EnvAdapter:
    """The seam that makes the fMRI loop engine-agnostic.

    An EnvAdapter WRAPS one game environment for one game block: it builds the
    underlying engine env in ``__init__`` and keeps it (and any per-block state)
    private, exposing only the small interface the experiment loop needs. The
    loop (session.py) never sees the raw env, ``env.unwrapped``, or any
    engine-specific API -- it just calls the methods below on the wrapper.

    A fresh EnvAdapter is constructed per game block (see
    :func:`fmri_gym.adapters.get_adapter`), so per-block state lives naturally
    on ``self`` with no risk of leaking between blocks.

    Subclasses override :meth:`_make` (build the engine env) and
    :meth:`_keyspec` (default keyboard map), plus whichever of the hooks
    below they need; state is returned in a STANDARD shape (a
    :class:`FrameState`) so the logger and any downstream analysis code are
    identical across ALE / stable-retro / plain gym.

    :ivar spec: the game-phase config dict this env was built from.
    :ivar env: the underlying engine environment (kept private to the wrapper).
    :ivar keyspec: keyboard->action mapping, with curriculum ``keys`` overrides
        already applied.
    :ivar has_audio: whether native PCM playback is available and enabled.
        Audio-capable backends play by default; ``spec["audio"] = False`` mutes them.
    """

    #: short id used in filenames / manifest, e.g. "ale", "retro", "gym"
    name: str = "base"

    def __init__(self, spec: dict) -> None:
        """Build the underlying env for one game block.

        :param spec: game-phase config dict from the curriculum (already
            validated for the keys this backend cares about).
        """
        self.has_audio = False
        self.spec = spec
        self.env = self._make(spec)
        self.has_audio = self.has_audio and bool(spec.get("audio", True))
        self.keyspec = self._keyspec()
        if spec.get("keys"):
            self.keyspec.apply_overrides(spec["keys"])

    def _make(self, spec: dict) -> Any:
        """Create and return the underlying engine env for one game block.

        Must produce an env that renders RGB frames (``render_mode="rgb_array"``
        for Gymnasium envs). May also initialise per-block state on ``self``.

        :param spec: game-phase config dict from the curriculum.
        :return: the underlying environment, stored as ``self.env``.
        :raises NotImplementedError: always in the base class.
        """
        raise NotImplementedError

    def _keyspec(self) -> KeySpec:
        """Return the default keyboard->action mapping for this env.

        Called once from :meth:`__init__`; the result is stored as
        :attr:`keyspec` after curriculum ``keys`` overrides are applied.

        :return: a concrete :class:`KeySpec` -- :class:`SingleKeySpec` for a
            ``Discrete`` space, :class:`MultiKeySpec` when held keys should
            combine, :class:`PassthroughKeySpec` when ``step`` takes the key set.
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

        Default is the Gymnasium contract; subclasses with non-standard
        signatures (e.g. VGDL's ``step(a, with_img=)``) override this.

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

        Default assumes the Gymnasium contract (``env.render()`` with the env
        made using ``render_mode="rgb_array"``). Subclasses for non-standard
        envs override this (e.g. old-gym's ``env.render(mode="rgb_array")``).

        :return: RGB frame as a numpy array.
        """
        return self.env.render()

    def close(self) -> None:
        """Close the underlying env if it exposes ``close()``.

        Not every env exposes ``close()`` (e.g. overcooked's OvercookedEnv).
        """
        closer = getattr(self.env, "close", None)
        if callable(closer):
            closer()

    def get_audio_buffer(self) -> np.ndarray | None:
        """Return native PCM for the latest reset/step, when ``has_audio`` is set.

        :return: non-overlapping ``(samples, channels)`` PCM array, or None if
            no new samples exist. The playback queue copies retained buffers.
        """
        return None

    def get_audio_sampling_rate(self) -> float:
        """Return the native audio rate for an adapter opting in to playback.

        :return: samples per second; dtype and channels come from the buffer.
        :raises NotImplementedError: if an audio-capable adapter omits this hook.
        """
        raise NotImplementedError(f"{self.name} adapter has no audio sampling rate")
