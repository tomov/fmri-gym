"""FrameState, Sound, and the EnvAdapter base class."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

from .keymap import make_keymap


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
    :ivar keymap: the phase's ``keys`` as a :class:`~.keymap.Keymap`.
    """

    env: gym.Env

    #: short id used in filenames / manifest, e.g. "ale", "retro", "gym"
    name: str = "base"

    def __init__(self, spec: dict) -> None:
        """Build the underlying env for one game block.

        :param spec: game-phase config dict from the curriculum (already
            validated for the keys this backend cares about).
        :raises ValueError: ``keys`` that do not fit the env's action space
            (see :func:`~.keymap.make_keymap`).
        """
        self.spec = spec
        self.env = self._make(spec)
        self.keymap = make_keymap(spec, self.env.action_space)
        #: the last step's ``info`` and reward, for :meth:`outcome`
        self.last_info: dict = {}
        self.last_reward: float = 0.0

    def _make(self, spec: dict) -> gym.Env:
        """Create and return the underlying env for one game block.

        A ``gymnasium.Env`` whose ``render()`` gives an RGB frame
        (``render_mode="rgb_array"``). A game whose own env speaks another API
        (old ``gym``, a bare engine) gets a thin Gymnasium env under ``gym/``
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

        The Gymnasium contract; the action comes from the :class:`~.keymap.Keymap`
        already in the shape the space takes.

        :param action: action to apply (type depends on the env).
        :return: ``(obs, reward, terminated, truncated, info)``.
        """
        out = self.env.step(action)
        self.last_reward, self.last_info = float(out[1]), out[4]
        return out

    def autoplay(self, info: dict) -> Any | None:
        """The action to step with while the env is ignoring input, or ``None``.

        Some games have stretches the player cannot act in: crafter discards
        the action while the player is asleep and forces ``sleep`` until they
        wake. A real-time block steps through those on its own clock and
        nobody notices. A ``turn_based`` block does not: it advances only on a
        press, so it would ask the subject for presses the env throws away, and
        the number of them is a property of the game rather than of the
        decision. Returning an action here says "step with this and do not
        wait", and the loop keeps doing so, at the phase's fps and logging
        every frame, until this returns ``None`` again. The subject still sees
        what happened, because every one of those steps is a drawn frame.

        It is the action and not a flag so that the choice of what a
        do-nothing step is stays with the backend that knows its space: these
        spaces have no action that means "do nothing" everywhere. The default
        is ``None``, which is every game that has no such state and every
        real-time phase.

        :param info: the info dict from the step just taken.
        :return: the action to step with, or ``None`` to wait for the subject.
        """
        return None

    def hud(self, score: float, time_remaining: float) -> list[str] | None:
        """The status lines drawn above the frame every step, or ``None`` for none.

        The session hands over what it knows -- the episode's running score
        (its cumulative reward) and the seconds left in the block -- and the
        backend decides what the subject sees of it, in a strip above the frame
        that never covers game pixels (:meth:`~fmri_gym.display.Display
        .draw_frame`). Whatever goes here has to be state the env already
        reports, so a model reads the same numbers off the log. The default is
        the two the session knows, time left first and the score last; a game
        with a HUD of its own may return ``None`` to draw nothing, and one
        with more to say may extend the list (lines are laid out left to right
        along the strip, the first flush left and the last flush right).

        :param score: the episode's cumulative reward so far.
        :param time_remaining: seconds until the block ends.
        :return: the lines, or ``None``.
        """
        return [f"{max(0, int(time_remaining))} s", f"Score: {score:g}"]

    def overlay(self) -> tuple[list[str], float] | None:
        """The lines drawn *on* the frame every step, or ``None`` for none.

        :meth:`hud` is the session's strip above the frame, the same for every
        game; this is the backend's own, drawn inside the picture at a place
        only the game knows is empty -- a selection the game itself does not
        render, a cue the subject has to read where they are already looking.
        It costs game pixels, so the default is ``None`` and a backend that
        returns lines should keep them short.

        The float is :attr:`overlay_y`, or anywhere else down the frame this
        particular line belongs.

        :return: ``(lines, y_frac)``, or ``None``.
        """
        return None

    #: Where down the frame a line drawn *on* it goes, as a fraction of the
    #: frame's height: what :meth:`overlay` returns beside its lines, and where
    #: the session puts a line of its own when it has one to say inside the
    #: picture (:mod:`fmri_gym.rewind`'s "You died" over the held frame). It is
    #: the backend's call because "where the subject is already looking" is not
    #: always the middle, which is the default: a game that draws its own HUD
    #: into the bottom rows of its frame leaves the played part above centre.
    overlay_y: float = 0.5

    def outcome(self, terminated: bool, truncated: bool) -> tuple[str, str]:
        """How the episode stands after its last step, as a name and a line for the subject.

        Gymnasium says only that an episode ended (``terminated``) or ran out
        of steps (``truncated``), not how it went; a game says "you won" or
        "you lost", and the subject expects to hear which. This is where a
        backend adds that reading, from the flags and whatever its env left in
        :attr:`last_info` / :attr:`last_reward`. The names are a fixed
        vocabulary the session logs per episode and a phase's
        ``advancing_outcomes`` picks from:

        - ``"playing"`` -- neither flag: the block's clock cut the episode off.
        - ``"terminated"`` -- the env ended it and this backend cannot say more.
        - ``"truncated"`` -- the env's own step limit.
        - ``"won"`` / ``"lost"`` -- terminated, with the game's verdict.

        The default is the flags alone. Overrides return the same names with
        their own message; the loop never interprets the message.

        :param terminated: the last step's ``terminated``.
        :param truncated: the last step's ``truncated``.
        :return: ``(outcome, message)``.
        """
        if terminated:
            return "terminated", "Game over"
        if truncated:
            return "truncated", "Timeout"
        return "playing", "Still playing"

    def capture(self, observation: Any, info: dict, want_blob: bool = True) -> FrameState:
        """Return the :class:`FrameState` to log for the current frame.

        Called once per step. ``observation``/``info`` are the latest :meth:`step`
        outputs so subclasses can fold observation-derived state in without
        re-querying. When ``want_blob`` is ``False`` the caller does not need
        the (often expensive) savestate this frame, so subclasses SHOULD skip
        computing ``blob`` and leave it ``None`` -- the cheap analysis
        variables should still be filled.

        :param observation: observation from the latest step/reset.
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

    def block_extra(self) -> dict | None:
        """What the block's log says once, in its ``block_end`` record's ``extra``.

        :meth:`capture` writes a row per frame; this is what the block needs
        said once, and what a per-frame column cannot say about itself: the
        sampling rate that makes logged audio playable, the names behind the
        integers in a column. Called after the last step, so it may read the
        env. Default is ``None`` -- nothing to add.

        :return: a mapping of name to array-like, or ``None``.
        """
        return None

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
