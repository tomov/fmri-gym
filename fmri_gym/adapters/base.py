"""FrameState, HudSettings and the EnvAdapter base class."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import gymnasium as gym
import numpy as np

from ..outputs.sound import Sound
from .keymap import make_keymap


@dataclass
class FrameState:
    """Everything an adapter exposes about itself at one frame.

    :ivar blob: opaque bytes that :meth:`EnvAdapter.restore` can turn back into
        this exact state (e.g. pickled ALE ``clone_state``, retro
        ``em.get_state()``). ``None`` if the backend has no in-memory savestate
        -- then reconstruction relies on seed + action replay instead.
    :ivar variables: named, analysis-friendly scalars/arrays surfaced uniformly
        via ``info``, so nothing outside the adapter calls ``getRAM()``/``get_ram()``.
        Keys are backend-defined but SHOULD include ``"ram"`` when available.
    """

    blob: bytes | None = None
    variables: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HudSettings:
    """A game phase's ``hud`` field: which of the strip's corner lines are shown."""

    #: the score, in the strip's right corner
    show_score: bool = True
    #: the seconds left in the block, in the strip's left corner
    show_time_remaining: bool = True

    @classmethod
    def from_dict(cls, spec: dict) -> HudSettings:
        """Read a phase's ``hud`` dict (already validated by :func:`hud_problems`)."""
        return cls(**spec)


def hud_problems(spec: object) -> list[str]:
    """What is wrong with a phase's ``hud`` dict, for the curriculum's validation.

    :param spec: the phase's ``"hud"`` value.
    :return: human-readable problems, empty when the settings are usable.
    """
    names = [f.name for f in fields(HudSettings)]
    if not isinstance(spec, dict):
        return [f"hud: expected a dict with any of {names}, got {spec!r}"]
    out = [f"hud.{name}: expected true or false, got {value!r}"
           for name, value in spec.items() if name in names and not isinstance(value, bool)]
    unknown = sorted(set(spec) - set(names))
    if unknown:
        out.append(f"hud: unknown fields {unknown}, expected any of {names}")
    return out


class EnvAdapter:
    """One game environment behind an engine-agnostic interface.

    An EnvAdapter WRAPS one game environment for one game block: it builds the
    underlying engine env in ``__init__`` and keeps it (and any per-block state)
    private, exposing only the methods below; the raw env, ``env.unwrapped``
    and any engine-specific API stay inside it. One adapter serves one block,
    so per-block state lives on ``self`` with no risk of leaking between blocks.

    Subclasses override :meth:`_make` (build the engine env) plus whichever of
    the hooks below they need; state is returned in a STANDARD shape (a
    :class:`FrameState`), the same across ALE / stable-retro / plain gym.

    :ivar spec: the game-phase config dict this env was built from.
    :ivar env: the underlying ``gymnasium.Env``.
    :ivar keymap: the phase's ``controls`` as a :class:`~.keymap.Keymap`.
    :ivar hud_settings: the phase's ``hud`` as :class:`HudSettings`.
    """

    env: gym.Env

    #: the backend's short id, e.g. "ale", "retro", "gym"
    name: str = "base"

    def __init__(self, spec: dict) -> None:
        """Build the underlying env for one game block.

        :param spec: the game phase's dict (already validated for the keys this
            backend cares about).
        :raises ValueError: ``controls`` that do not fit the env's action space
            (see :func:`~.keymap.make_keymap`).
        """
        self.spec = spec
        self.hud_settings = HudSettings.from_dict(spec.get("hud", {}))
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

        :param spec: the game phase's dict.
        :return: the underlying environment, stored as ``self.env``.
        :raises NotImplementedError: always in the base class.
        """
        raise NotImplementedError

    def reset(self, seed: int) -> tuple[Any, dict]:
        """Reset the env for a new episode.

        Subclasses may use ``self.spec`` for per-episode setup (e.g. retro
        load_state).

        :param seed: RNG seed for this episode.
        :return: ``(obs, info)`` from ``env.reset``.
        :raises ValueError: ``seed`` is ``None`` (an episode is replayed from its seed).
        """
        if seed is None:
            raise ValueError("an episode needs a seed")
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
        nobody notices. A ``is_turn_based`` block does not: it advances only on a
        press, so it would ask the subject for presses the env throws away, and
        the number of them is a property of the game rather than of the
        decision. Returning an action here says "step with this and do not
        wait", and the block keeps doing so, at its fps and logging every
        frame, until this returns ``None`` again. The subject still sees
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

        Given the episode's running score (its cumulative reward) and the
        seconds left in the block, the backend decides what the subject sees of it, in a strip above the frame
        that never covers game pixels (:meth:`~fmri_gym.outputs.display.Display
        .draw_frame`). Whatever goes here has to be state the env already
        reports, so a model reads the same numbers off the log. The default is
        those two, time left first and the score last, each shown as the
        phase's ``hud`` settings (:attr:`hud_settings`) say; an override reads
        them the same way. A game with a HUD of its own may return ``None`` to
        draw nothing, and one with more to say may extend the list (lines are
        laid out left to right along the strip, the first flush left and the
        last flush right).

        :param score: the episode's cumulative reward so far.
        :param time_remaining: seconds until the block ends.
        :return: the lines, or ``None`` when none is shown.
        """
        # A hidden line is "" rather than dropped: a lone line sits flush left.
        settings = self.hud_settings
        lines = [f"{max(0, int(time_remaining))} s" if settings.show_time_remaining else "",
                 f"Score: {score:g}" if settings.show_score else ""]
        return lines if any(lines) else None

    def overlay(self) -> tuple[list[str], float] | None:
        """The lines drawn *on* the frame every step, or ``None`` for none.

        :meth:`hud` is the strip above the frame, the same for every game; this is the backend's own, drawn inside the picture at a place
        only the game knows is empty -- a selection the game itself does not
        render, a cue the subject has to read where they are already looking.
        It costs game pixels, so the default is ``None`` and a backend that
        returns lines should keep them short.

        The float is where down the frame to centre them, as a fraction of its
        height (0.5 = the middle). It is the backend's call because "where they
        are already looking" is not always the middle: a game that draws its
        own HUD into the bottom rows of its frame leaves the played part above
        centre.

        :return: ``(lines, y_frac)``, or ``None``.
        """
        return None

    def outcome(self, terminated: bool, truncated: bool) -> tuple[str, str]:
        """How the episode stands after its last step, as a name and a line for the subject.

        Gymnasium says only that an episode ended (``terminated``) or ran out
        of steps (``truncated``), not how it went; a game says "you won" or
        "you lost", and the subject expects to hear which. This is where a
        backend adds that reading, from the flags and whatever its env left in
        :attr:`last_info` / :attr:`last_reward`. The names are a fixed
        vocabulary, logged per episode:

        - ``"playing"`` -- neither flag: the block's clock cut the episode off.
        - ``"terminated"`` -- the env ended it and this backend cannot say more.
        - ``"truncated"`` -- the env's own step limit.
        - ``"won"`` / ``"lost"`` -- terminated, with the game's verdict.

        The default is the flags alone. Overrides return the same names with
        their own message, which is for the subject alone.

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
        integers in a column, the state the block ended in (``"state"``: bytes
        :meth:`restore` takes, which the log stores as a frame's ``state`` is).
        Called after the last step, so it may read the env. Default is
        ``None`` -- nothing to add.

        :return: a mapping of name to array-like (or bytes), or ``None``.
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

        The audio counterpart of :meth:`render`: what the speakers play with
        the frame :meth:`render` shows. Default is ``None`` -- a silent
        backend, which is most of them.

        Return the PCM the engine produced during the last step, at its native
        rate: no resampling, no copying, no timing -- placing it against the
        flip is the speakers' job. It should last about one frame period (``1 / fps``);
        a block where it does not stops with the fps that would fit. Playback
        does not store it: to log the sound as well, return it from
        :meth:`capture` too.

        :return: a :class:`~fmri_gym.outputs.sound.Sound`, or ``None`` if there is nothing to play.
        """
        return None

    def native_fps(self) -> float | None:
        """Steps per second at which the engine plays at its own real speed, or ``None``.

        An engine with a clock of its own (an emulator core's frame rate, a
        tic rate) advances a fixed amount of game time per step, so the block's
        ``fps`` decides how fast the game is, and people and models should meet
        the same game. Nothing enforces it -- ``fps`` has to suit the display,
        and a slowed-down block can be a choice.
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
