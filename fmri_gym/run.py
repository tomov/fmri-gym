"""One run: the engine-agnostic experiment loop that plays one curriculum.

A run is what one curriculum file describes and one ``fmri-play`` plays (the
BIDS ``run-NNN`` of its folder). A session -- ``ses-NNN``, several runs -- is
a shell script of those commands and never a Python loop, so nothing here
knows about more than the curriculum it was given.

Everything here is independent of which game engine is used: trigger wait,
clock anchoring, the curriculum of phases (fixation / message / game / survey),
inter-block intervals, timing/pacing, and logging. All engine-specific access
goes through an EnvAdapter, so this file never imports ale_py / stable_retro
and never touches env.unwrapped.

Recording-device concerns (waiting for or sending the scanner start, trigger
codes for MEG/EEG) go through :mod:`fmri_gym.triggers`; with no ``triggers``
config the loop behaves as the fMRI default and sends nothing.

A run is the scanner's and a curriculum the subject's, so a run can stop
part-way through one: given a duration, the run stops whatever phase is on
screen :data:`END_FIXATION_S` before its end and closes on a fixation cross,
for the haemodynamic response to the last thing played. A later run started
with ``--resume`` carries on from there (:func:`~fmri_gym.bids.last_position`).

Three clocks run through this file, and a name says which one it holds:
``*_perf`` is a ``time.perf_counter()`` stamp (what the display and the
deadlines use), ``*_run_time`` is seconds since the scanner trigger (what the
log records), ``*_epoch`` is ``time.time()``.
"""

from __future__ import annotations

import math
import sys
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

import pygame

from . import bids, pad
from .adapters import get_adapter
from .audio import Audio
from .bids import BlockProgress, RunProgress
from .config import RunConfig
from .curriculum import BlockMode, GamePhase, fold_cli_options
from .display import Display, check_monitor
from .keys import get_events, held_key_names, key_name
from .menu import Menu
from .logging import Logger
from .triggers import Triggers

if TYPE_CHECKING:
    from .adapters.base import EnvAdapter

EXPERIMENTER_KEY = " "
#: How far ``fps`` may sit from the engine's own rate and still count as real
#: speed: what the audio output absorbs by resampling.
_SAME_SPEED = 2e-3
#: Seconds between the redraws a ``live_hud`` phase asks for while it waits.
#: The HUD's coarsest field is a whole second, so this only has to be fast
#: enough that the number is never seen a second behind.
HUD_REPAINT_S = 0.2
#: Seconds of fixation that close a run given a duration.
END_FIXATION_S = 10.0
#: Game-phase fields a curriculum may change between runs and still carry a
#: block on; any other change makes it a different game. ``seed`` and ``audio``
#: are the run's own (folded in from the run label and the run config).
RESUMABLE_CHANGES = frozenset({"duration", "max_duration", "n_episodes", "advancing_outcomes",
                               "outcome_duration", "instructions", "keys", "menu", "seed",
                               "audio", "rewind_on", "rewind_frames"})


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """How one episode ended: what its ``episode_end`` record holds."""

    #: the episode's index within its block
    episode_id: int
    #: one of the adapter's names (:meth:`~.adapters.base.EnvAdapter.outcome`:
    #: ``"won"``, ``"lost"``, ``"terminated"``, ``"truncated"``, or
    #: ``"playing"`` when the block's clock cut it off), or the subject's own:
    #: ``"quit"`` for ESC / window close, ``"reset"`` / ``"forfeit"`` for the
    #: choice in the menu
    outcome: str
    #: the last step's flags
    terminated: bool
    truncated: bool
    #: the episode's cumulative reward
    score: float
    #: stalls that dropped the frame debt (the ``pacing_reset`` lines)
    n_pacing_resets: int

    def record(self) -> dict:
        """The ``episode_end`` line's fields."""
        return asdict(self)


class Clock:
    """Anchored at the scanner trigger; gives run + wall-clock time."""

    def __init__(self) -> None:
        """Create an untriggered clock (``trigger_*`` are ``None`` until :meth:`trigger`)."""
        #: ``perf_counter`` at the scanner trigger: the zero of run time.
        self.trigger_perf: float | None = None
        #: ``time.time()`` at the scanner trigger.
        self.trigger_epoch: float | None = None

    def trigger(self) -> None:
        """Anchor the clock at the current time (call on scanner trigger)."""
        self.trigger_perf = time.perf_counter()
        self.trigger_epoch = time.time()

    def run_time(self) -> float:
        """Seconds since the scanner trigger (``perf_counter`` based).

        :return: seconds since this run's trigger.
        """
        return time.perf_counter() - self.trigger_perf

    def from_perf(self, perf: float) -> float:
        """Convert a ``perf_counter`` stamp (e.g. a flip time) to run time.

        :param perf: a ``time.perf_counter()`` value.
        :return: seconds since the scanner trigger.
        """
        return perf - self.trigger_perf

    def wall_time(self) -> float:
        """Current wall-clock epoch time.

        :return: ``time.time()`` seconds since the Unix epoch.
        """
        return time.time()


def _check_quit() -> bool:
    """Drain pygame events and report whether the user requested quit.

    :return: ``True`` if the window was closed or ESC was pressed.
    """
    for event in get_events():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return True
    return False


def _poll_keys_until(
    display: Display,
    deadline_perf: float,
    clock: Clock,
    logger: Logger,
    key_to_action: dict | None = None,
    menu: Menu | None = None,
    latch_keys: bool = False,
    repaint: Callable[[], float] | None = None,
) -> tuple[object | None, bool]:
    """Poll the keyboard until ``deadline_perf``, logging every press/release.

    Replaces a plain sleep between frames: events are time-stamped on arrival
    -- to about a millisecond, or to one refresh when the display is
    vsync-locked and :meth:`Display.idle` re-presents the frame instead of
    sleeping. In turn-based play (``key_to_action`` given) the wait ends at
    the first mapped keydown so the step happens then, not at the tick.
    ``latch_keys`` is the real-time counterpart: the keydown is remembered but
    the wait still runs to the tick, so the frame period is what it says it
    is. The wait also ends the moment the block's menu is armed (its key held
    long enough), which the caller reads off ``menu.pending``.

    :param display: the display, idled between polls.
    :param deadline_perf: ``perf_counter`` at which to stop waiting.
    :param clock: the run's clock, for the log.
    :param logger: the run's logger.
    :param key_to_action: turn-based or latched: map of single key NAMES to
        env actions.
    :param menu: the block's :class:`~.menu.Menu`, if the phase has one.
    :param latch_keys: keep waiting after a mapped keydown and return the last
        one seen, instead of ending the wait at the first.
    :param repaint: called every :data:`HUD_REPAINT_S` while waiting, for a
        wait long enough that the screen would otherwise go stale (see
        ``live_hud``). It redraws; it must not step, log or trigger.
    :return: ``(action_or_None, user_quit)``; ``action`` is set only when
        ``key_to_action`` is given, ``user_quit`` on window close / ESC.
    """
    latched_action = None
    next_repaint_perf = time.perf_counter() + HUD_REPAINT_S
    while True:
        if menu is not None and menu.armed():
            return None, False
        for event in get_events():
            if event.type == pygame.QUIT:
                return None, True
            if event.type not in (pygame.KEYDOWN, pygame.KEYUP):
                continue
            if event.key == pygame.K_ESCAPE:
                return None, True
            key = key_name(event.key)
            if key is None:
                continue
            is_keydown = event.type == pygame.KEYDOWN
            logger.log(type="input_event", run_time=clock.run_time(), name=key, down=is_keydown)
            if is_keydown and key_to_action and key in key_to_action:
                if not latch_keys:
                    return key_to_action[key], False
                # TODO: find out whether every real-time game wants this, not
                # only the slow ones that asked for it -- a press dropped
                # between two frames is a lost action at any fps, and if it is
                # general, latching stops being a phase flag and becomes what
                # the loop does.
                latched_action = key_to_action[key]
        if time.perf_counter() >= deadline_perf:
            return latched_action, False
        if repaint is not None and time.perf_counter() >= next_repaint_perf:
            repaint()
            next_repaint_perf = time.perf_counter() + HUD_REPAINT_S
        display.idle(deadline_perf)


def _wait_for_char(
    display: Display, char: str, dummy_trigger: bool = False, deadline_perf: float = math.inf
) -> None:
    """Block until ``char`` is typed (or briefly sleep in dummy mode).

    :param display: the display, idled between polls (see :meth:`Display.idle`).
    :param char: the unicode character that unblocks the wait, or ``"any"``
        to accept every key (a self-paced "press any key" screen).
    :param dummy_trigger: if ``True``, sleep briefly and return without waiting.
    :param deadline_perf: ``perf_counter`` at which to stop waiting regardless.
    :raises KeyboardInterrupt: on window close or ESC.
    """
    if dummy_trigger:
        time.sleep(0.05)
        return
    while time.perf_counter() < deadline_perf:
        for event in get_events():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt
                if char == "any" or event.unicode == char:
                    return
        display.idle(time.perf_counter() + 0.005, poll=0.005)


def _join_multiline_text(text: str | list | tuple) -> str:
    """Normalize message ``text`` to a single string.

    Accepts a plain string or a list/tuple of lines (joined with ``\\n``), so
    curriculum JSON can keep long instructions readable without ``\\n`` escapes.
    """
    if isinstance(text, (list, tuple)):
        return "\n".join("" if line is None else str(line) for line in text)
    return str(text)


def _wait_for_duration(display: Display, seconds: float, deadline_perf: float = math.inf) -> None:
    """Block for ``seconds``.

    :param display: the display, idled between polls (see :meth:`Display.idle`).
    :param seconds: how long to wait.
    :param deadline_perf: ``perf_counter`` at which to stop waiting regardless.
    :raises KeyboardInterrupt: on window close or ESC.
    """
    end_perf = min(time.perf_counter() + seconds, deadline_perf)
    while time.perf_counter() < end_perf:
        if _check_quit():
            raise KeyboardInterrupt
        display.idle(end_perf, poll=0.005)


def _wait_for_keys(display: Display, key_names: list[str], deadline_perf: float) -> str | None:
    """Block until one of the keys ``key_names`` is pressed, or ``deadline_perf``.

    :param display: the display, idled between polls (see :meth:`Display.idle`).
    :param key_names: key NAMES (:mod:`fmri_gym.keys`) that end the wait.
    :param deadline_perf: ``perf_counter`` at which to stop waiting.
    :return: the key that ended the wait, or ``None`` at the deadline.
    :raises KeyboardInterrupt: on window close or ESC.
    """
    while time.perf_counter() < deadline_perf:
        for event in get_events():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_ESCAPE:
                raise KeyboardInterrupt
            if key_name(event.key) in key_names:
                return key_name(event.key)
        display.idle(min(deadline_perf, time.perf_counter() + 0.005), poll=0.005)
    return None


def _final_outcome(
    adapter: EnvAdapter, outcome: str, terminated: bool, truncated: bool
) -> tuple[str, str]:
    """Settle how the episode ended: its outcome's name and the line for the subject.

    The subject's choices (quit, reset, forfeit) are known here and have
    their own lines; otherwise the adapter reads the env's flags
    (:meth:`~.adapters.base.EnvAdapter.outcome`), which is where "won" and
    "lost" come from. A quit has no line: the run is ending.

    :param adapter: the block's adapter.
    :param outcome: ``"quit"``, ``"reset"``, ``"forfeit"``, or ``""`` for
        the env's own end or the block's clock.
    :param terminated: the last step's ``terminated``.
    :param truncated: the last step's ``truncated``.
    :return: ``(outcome, outcome_message)``; the message is ``""`` on a quit.
    """
    if outcome == "quit":
        return outcome, ""
    if outcome:
        outcome_message = {"reset": "Starting over", "forfeit": "Forfeited"}[outcome]
    else:
        outcome, outcome_message = adapter.outcome(terminated, truncated)
        if outcome == "playing":        # the block's clock, not the game, ended it
            outcome_message = "Block ended"
    return outcome, outcome_message


def _run_progress(
    data_root: str, subject: str, curriculum_path: str, phases: list[dict]
) -> RunProgress | None:
    """Where ``--resume`` starts: the subject's progress in this curriculum, checked against it.

    :param data_root: the BIDS tree's root.
    :param subject: ``sub-<label>``.
    :param curriculum_path: the curriculum file; its name is the task label.
    :param phases: the curriculum's phases.
    :return: :func:`~fmri_gym.bids.last_position`, or ``None`` (said on stderr)
        when no earlier run of this curriculum got past its opening phases.
    :raises ValueError: if that run played the curriculum to its end, or its
        last game phase differs in this curriculum by more than
        :data:`RESUMABLE_CHANGES`.
    """
    progress = bids.last_position(data_root, subject, curriculum_path)
    if progress is None:
        print("resume: no earlier run of this curriculum got past its opening phases; starting "
              "at the beginning", file=sys.stderr)
        return None
    run_label, last_game = progress.run_label, progress.last_game
    if progress.phase_index >= len(phases):
        raise ValueError(f"--resume: {run_label} played this curriculum to its end; start it "
                         "over without --resume")
    if last_game is not None:
        played_phase = last_game.phase
        current_phase = (phases[last_game.phase_index]
                         if last_game.phase_index < len(phases) else {})
        changed_fields = sorted(
            field for field in played_phase.keys() | current_phase.keys()
            if played_phase.get(field) != current_phase.get(field)
            and field not in RESUMABLE_CHANGES and not field.startswith("_"))
        if changed_fields:
            raise ValueError(f"--resume: phase {last_game.phase_index}, the last game "
                             f"{run_label} played, is not the same game in this curriculum "
                             f"({', '.join(changed_fields)} changed). Put it back, or start "
                             "over without --resume")
    print(f"resume: carrying on from {run_label} at phase {progress.phase_index}"
          + (" mid-block" if progress.block_progress is not None else ""), file=sys.stderr)
    return progress


class Run:
    """Plays one curriculum for one subject, dispatching phases to handlers."""

    def __init__(
        self,
        subject: str,
        phases: list[dict],
        display: Display,
        run_folder: str,
        audio: Audio | None = None,
        triggers: Triggers | None = None,
        dummy_trigger: bool = False,
        duration: float | None = None,
        resume: RunProgress | None = None,
    ) -> None:
        """Set up clock, logger, and phase dispatch for one subject.

        :param subject: subject identifier used in log paths / manifest.
        :param phases: the curriculum's phases, in order (dicts: ``type``, timings, …).
        :param display: shared pygame display used by all phases.
        :param run_folder: the run's folder, for its manifest and block logs.
        :param audio: shared audio output used by all phases; one is created if
            omitted, and stays silent unless an adapter returns sound.
        :param triggers: shared trigger output (start sync + codes; see
            :mod:`fmri_gym.triggers`), built by the caller like the display
            and the audio. ``None`` = the fMRI default: wait for ``=``, send
            no trigger codes.
        :param dummy_trigger: if ``True``, skip real experimenter/scanner waits.
        :param duration: the run's seconds from the trigger, the closing
            fixation included; ``None`` plays the curriculum to its end.
        :param resume: where to start in the curriculum, and what the block
            there has left (:func:`~fmri_gym.bids.last_position`); ``None``
            starts at its beginning.
        """
        self.subject = subject
        self.phases = phases
        self.display = display
        self.audio = audio or Audio()
        self.dummy_trigger = dummy_trigger
        self.clock = Clock()
        self.logger = Logger(run_folder, subject, phases, self.clock)
        self.logger.set_extra("display", display.describe())
        self.logger.set_extra("dummy_trigger", dummy_trigger)
        self.logger.set_extra("audio", self.audio.describe())
        self.run_folder = run_folder
        self.triggers = triggers or Triggers.from_dict(None)
        self.sync_settings = self.triggers.sync_settings
        self.duration = duration
        self.resume = resume
        #: ``perf_counter`` at which the phases stop for the closing fixation.
        self._phases_end_perf = math.inf
        self.logger.set_extra("duration", duration)
        self.logger.set_extra("resumed_from", None if resume is None else resume.record())

    @classmethod
    def from_config_and_curriculum(cls, config: RunConfig, curriculum: dict[str, Any]) -> "Run":
        """Everything one run needs, from its run config and its curriculum.

        Works out where the run writes and what seeds it plays, then opens the
        trigger line, the audio output and the window -- in that order, so a
        bad section, an unopenable port or an unusable output stops the run
        before anything is on screen. Each says on stderr what it got.

        :param config: the command line, as the run reads it.
        :param curriculum: the curriculum, already checked
            (:func:`~fmri_gym.curriculum.validate_curriculum`).
        :return: the run, ready to :meth:`play`.
        :raises ValueError: on a BIDS name or number, a monitor or a window
            size this machine or this subject refuses; a duration no longer
            than the closing fixation; a progress to resume from that this
            curriculum cannot carry on (see :func:`_run_progress`).
        """
        phases = curriculum["curriculum"]
        if config.duration is not None and config.duration <= END_FIXATION_S:
            raise ValueError(f"--duration {config.duration:g}: a run closes on "
                             f"{END_FIXATION_S:g} s of fixation, so it must be longer than that")
        output = bids.run_output(config.data_root, config.subject, config.curriculum_path,
                                 config.ses, config.run)
        if output.attempt > 1:
            print(f"{output.label} already has data: this is attempt {output.attempt} at it, "
                  "and writes beside the others", file=sys.stderr)
        print(f"output: {output.folder}", file=sys.stderr)
        resume = None
        if config.resume:
            resume = _run_progress(config.data_root, config.subject, config.curriculum_path,
                                   phases)
        seeds = bids.fold_seeds(phases, output.label)
        check_monitor(config.monitor)
        fold_cli_options(phases, has_audio=config.has_audio)

        triggers = Triggers.from_dict(curriculum.get("triggers"))
        print(f"triggers: {triggers.status()}", file=sys.stderr)
        if config.dummy_trigger:
            print("triggers: --dummy-trigger: the experimenter and scanner waits are skipped; "
                  "this is a test run, not a session", file=sys.stderr)
        audio = Audio(enabled=config.has_audio)
        print(f"audio: {audio.status()}", file=sys.stderr)
        pad_status = pad.init(config.has_pad)
        print(f"pad: {pad_status}", file=sys.stderr)
        display = Display(size=config.size, fullscreen=config.is_fullscreen,
                          vsync=config.has_vsync, monitor=config.monitor)
        run = cls(config.subject, phases, display, output.folder, audio=audio,
                  triggers=triggers, dummy_trigger=config.dummy_trigger,
                  duration=config.duration, resume=resume)
        run.logger.set_extra("run", {"label": output.label, "attempt": output.attempt})
        run.logger.set_extra("seeds", seeds)
        run.logger.set_extra("pad", pad_status)   # which controller answered, and as what
        run.logger.set_extra("versions", {  # the banner fmri_gym hides said these
            "pygame": pygame.version.ver, "sdl": ".".join(map(str, pygame.get_sdl_version()))})
        return run

    def close(self) -> None:
        """Close the window, the audio output, the trigger line and the logger."""
        self.display.close()
        self.audio.close()
        self.triggers.close()
        self.logger.close()

    def _trigger(self) -> None:
        """Wait for experimenter ready, sync with the scanner, start the clock.

        Draws the readiness screen -- with the trigger status on it, so the
        experimenter sees what this run will do before pressing SPACE -- then
        either waits for the trigger key, sends the start code (``sync.mode``),
        or neither; then calls :meth:`Clock.trigger`, records the trigger time
        on the logger and sends ``task_start``.
        """
        self.display.draw_text(
            "Please keep your head as still as possible.\n\n"
            "(experimenter: press SPACE when ready)\n\n"
            f"triggers: {self.triggers.status()}\n"
            f"audio: {self.audio.status()}")
        _wait_for_char(self.display, EXPERIMENTER_KEY, dummy_trigger=self.dummy_trigger)
        if self.sync_settings.mode == "wait":
            self.display.draw_text(f"Waiting for the scanner...\n\n(its trigger key: "
                                   f"{self.sync_settings.key!r})")
            _wait_for_char(self.display, self.sync_settings.key, dummy_trigger=self.dummy_trigger)
        elif self.sync_settings.mode == "send":
            self.display.draw_text("Starting the recording...")
            self.triggers.lifecycle("scanner_start")
            _wait_for_duration(self.display, self.sync_settings.delay)

        self.clock.trigger()
        self.logger.set_trigger_time()
        self.triggers.lifecycle("task_start")

    def _fixation(self, phase: dict, phase_index: int) -> None:
        """Show a fixation cross for ``phase["duration"]`` seconds.

        :param phase: fixation-phase config (``duration``, default 2.0).
        :param phase_index: the phase's index in the curriculum (for the manifest).
        """
        duration = phase.get("duration", 2.0)

        onset = self.clock.from_perf(self.display.draw_fixation())
        _wait_for_duration(self.display, duration, deadline_perf=self._phases_end_perf)

        self.logger.log_phase({"index": phase_index, "type": "fixation",
                               "onset": onset, "offset": self.clock.run_time()})

    def _message(self, phase: dict, phase_index: int) -> None:
        """Show on-screen text until a key press or timed duration.

        :param phase: message-phase config (see :meth:`_show_message`).
        :param phase_index: the phase's index in the curriculum (for the manifest).
        """
        self.logger.log_phase({"index": phase_index, "type": "message",
                               **self._show_message(phase)})

    def _show_message(self, phase: dict) -> dict:
        """Draw a message and wait until a key press or its duration.

        :param phase: ``text`` as a string or list of lines; optional
            ``duration`` / ``key`` / ``keys_to_dismiss`` / ``align``. ``key`` is
            the character that dismisses the screen (default SPACE), or
            ``"any"``; ``keys_to_dismiss`` is a list of key NAMES instead, which
            with a ``duration`` ends the screen at whichever comes first.
        :return: the ``text`` shown, its ``onset`` and ``offset``.
        :raises KeyboardInterrupt: on window close or ESC.
        """
        text = _join_multiline_text(phase.get("text", ""))
        duration = phase.get("duration")

        onset = self.clock.from_perf(
            self.display.draw_text(text, align=phase.get("align", "center")))
        if "keys_to_dismiss" in phase:
            deadline_perf = time.perf_counter() + (math.inf if duration is None else duration)
            if self.dummy_trigger:      # a test run skips the waits for a key
                deadline_perf = min(deadline_perf, time.perf_counter() + 0.05)
            _wait_for_keys(self.display, phase["keys_to_dismiss"],
                           min(deadline_perf, self._phases_end_perf))
        elif duration is None:
            _wait_for_char(self.display, phase.get("key", " "),
                           dummy_trigger=self.dummy_trigger, deadline_perf=self._phases_end_perf)
        else:
            _wait_for_duration(self.display, duration, deadline_perf=self._phases_end_perf)
        return {"text": text, "onset": onset, "offset": self.clock.run_time()}

    def _survey(self, phase: dict, phase_index: int) -> None:
        """Run a Likert-style survey and log each confirmed response.

        :param phase: survey-phase config (``questions``, optional ``n_points``).
        :param phase_index: the phase's index in the curriculum (for the manifest).
        :raises KeyboardInterrupt: on window close or ESC.
        """
        questions = phase.get("questions", [])
        n_points = phase.get("n_points", 7)
        onset = self.clock.run_time()

        responses = []
        try:
            self._ask(questions, n_points, responses)
        finally:  # a quit mid-survey keeps the answers already confirmed
            self.logger.log_phase({"index": phase_index, "type": "survey",
                                   "onset": onset, "offset": self.clock.run_time(),
                                   "responses": responses})

    def _ask(self, questions: list[str], n_points: int, responses: list[dict]) -> None:
        """Append each confirmed Likert answer to ``responses``.

        :raises KeyboardInterrupt: on window close or ESC.
        """
        for question in questions:
            rating = (n_points + 1) // 2
            confirmed = False
            while not confirmed:
                scale = "  ".join((f"[{point}]" if point == rating else f" {point} ")
                                  for point in range(1, n_points + 1))
                self.display.draw_text(
                    f"{question}\n\nDisagree      Agree\n{scale}\n\n"
                    "(LEFT/RIGHT to rate, ENTER to confirm)")
                for event in get_events():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            raise KeyboardInterrupt
                        elif event.key == pygame.K_LEFT:
                            rating = max(1, rating - 1)
                        elif event.key == pygame.K_RIGHT:
                            rating = min(n_points, rating + 1)
                        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                            confirmed = True
                time.sleep(0.005)
            responses.append({"question": question, "value": rating,
                              "run_time": self.clock.run_time()})

    def _episode(
        self,
        adapter: EnvAdapter,
        game_phase: GamePhase,
        *,
        seed: int,
        episode_id: int,
        block_end_perf: float,
        menu: Menu | None,
        initial_state: bytes | None = None,
        rewind_savestates: deque | None = None,
    ) -> EpisodeResult:
        """Run one episode, logging every frame.

        :param adapter: wrapped env for reset/step/render/sound/capture.
        :param game_phase: the phase's settings; read here are
            ``is_turn_based`` (advance only on mapped keydowns),
            ``has_latched_keys`` (real-time: let a fresh keydown win over held
            keys), ``has_live_hud`` (turn-based: redraw the HUD through the
            wait), ``frame_period``, ``state_stride``, ``has_audio`` and
            ``show_outcome_duration``.
        :param seed: RNG seed for this episode's ``reset``.
        :param episode_id: index of this episode within the game block.
        :param block_end_perf: ``perf_counter`` deadline for the game block.
        :param menu: the block's :class:`~.menu.Menu`, or ``None``.
        :param initial_state: a savestate to restore over the reset, so the
            episode starts where an earlier run's stopped; logged on
            ``episode_start``, since it, not the seed, is what the episode
            starts from.
        :param rewind_savestates: if given, emptied, then filled with the
            savestate after every frame (it keeps the last ``maxlen``), for a
            block that rewinds (``rewind_on``).
        :raises ValueError: ``rewind_savestates`` with a backend whose
            ``capture`` returns no savestate.
        :return: how the episode ended, as its ``episode_end`` record says.
        """
        terminated = truncated = False
        outcome = ""
        score = 0.0                     # the episode's cumulative reward
        n_pacing_resets = 0
        ep_frame = 0
        frame_period = game_phase.frame_period
        key_to_action = (adapter.keymap.turn_actions()
                         if game_phase.is_turn_based or game_phase.has_latched_keys else None)

        def repaint_hud() -> float:
            """Present the frame again, for its HUD. Never steps, logs or triggers."""
            return self._show_frame(adapter, False, score, block_end_perf)[0]

        # A turn-based wait is unbounded -- it ends at a press -- and the HUD
        # is drawn only by _show_frame, so without this the block clock a
        # subject reads is the one their last press left behind, while the
        # clock the loop ends on runs on regardless. Off unless the phase
        # asks: a game whose HUD says nothing about time has nothing to gain,
        # and a real-time wait is one frame long and repaints anyway.
        repaint = repaint_hud if game_phase.is_turn_based and game_phase.has_live_hud else None
        # Set while the env is in a state it takes no action in, to the action
        # to step it with; see EnvAdapter.autoplay. Only a turn-based block
        # cares: a real-time one steps through such a stretch anyway.
        autoplay_action: Any = None

        ## Reset environment and show initial state
        if rewind_savestates is not None:
            rewind_savestates.clear()
        observation, info = adapter.reset(seed)
        if initial_state is not None:
            adapter.restore(initial_state)
        self.display.call_on_flip(self.triggers.episode_start)
        # Paced from the flip, so a slow reset does not become a burst of
        # catch-up frames. The reset frame's sound is not played: it is not a
        # step's, and it would start the episode's sound off its flips.
        reset_flip_perf, _, _ = self._show_frame(adapter, False, score, block_end_perf)
        self.logger.log(type="episode_start", episode_id=episode_id, seed=seed,
                        initial_state=initial_state,
                        flip_time=self.clock.from_perf(reset_flip_perf))
        next_tick_perf = reset_flip_perf + frame_period

        ## Loop over frames within episode
        while not (terminated or truncated) and time.perf_counter() < block_end_perf:
            # Wait for the frame tick (turn-based: for a mapped keydown, up to
            # the block end), polling keys as we go so presses are stamped on
            # arrival; a vsync-locked display re-presents the frame meanwhile.
            # TODO(#43): anchor the wait to the last step (step time + frame
            # period), not to the flip.
            # An autoplayed stretch is paced like a real-time frame even in a
            # turn-based block: the env is not waiting on the subject, so the
            # wait is one tick and the keys polled through it are logged but
            # not acted on.
            waits_for_press = game_phase.is_turn_based and autoplay_action is None
            deadline_perf = block_end_perf if waits_for_press else next_tick_perf
            action, user_quit = _poll_keys_until(
                self.display, deadline_perf, self.clock, self.logger,
                key_to_action if waits_for_press else None, menu,
                latch_keys=game_phase.has_latched_keys and not game_phase.is_turn_based,
                repaint=repaint if waits_for_press else None)
            if user_quit:
                outcome = "quit"
                break
            if menu is not None and menu.pending:
                menu_choice = menu.run(self.display, self.clock.run_time, self.logger)
                if menu_choice != "resume":
                    outcome = menu_choice
                    break
                resume_flip_perf, _, _ = self._show_frame(adapter, False, score, block_end_perf)
                next_tick_perf = resume_flip_perf + frame_period
                continue
            next_tick_perf += frame_period
            if waits_for_press and action is None:
                continue                        # block ended without a press
            if autoplay_action is not None:
                action = autoplay_action        # the env is not taking ours
            elif not game_phase.is_turn_based and action is None:
                # No latched press this frame (or the phase never asked for
                # one): the action is whatever is held down at the tick.
                action = adapter.keymap.resolve(held_key_names())

            observation, reward, terminated, truncated, info = adapter.step(action)
            autoplay_action = adapter.autoplay(info) if game_phase.is_turn_based else None
            step_run_time = self.clock.run_time()
            score += float(reward)
            # A savestate at each episode's first frame (the replay anchor), then
            # every stride; frames between are the anchor plus the logged actions.
            anchor_frame = ep_frame % game_phase.state_stride == 0
            frame_state = adapter.capture(observation, info,
                                          want_blob=anchor_frame or rewind_savestates is not None)
            if rewind_savestates is not None:
                if frame_state.blob is None:
                    raise ValueError(f"rewind_on: the {adapter.name} adapter's capture "
                                     "returns no savestate to go back to")
                rewind_savestates.append(frame_state.blob)
            # The frame trigger goes out on the flip that shows this frame.
            self.display.call_on_flip(self.triggers.frame)
            flip_perf, frame_rgb, sound = self._show_frame(adapter, game_phase.has_audio, score,
                                                           block_end_perf)
            # More than a frame behind (a stall): drop the debt, or it is repaid
            # as a burst of one-refresh frames. The frame of slack is what a
            # vsync-locked flip normally lands after its tick.
            # TODO(#43): why frames fall behind at all is not established.
            if game_phase.is_turn_based:
                # Anchor the next tick to this flip. A turn-based wait leaves
                # the tick far behind -- the deadline it waits on is the
                # block's, not the tick -- so an autoplayed stretch starting
                # from the old value would find every tick already past and
                # run flat out instead of at fps.
                next_tick_perf = flip_perf + frame_period
            elif next_tick_perf + frame_period < flip_perf:
                self.logger.log(type="pacing_reset", flip_time=self.clock.from_perf(flip_perf),
                                late=flip_perf - (next_tick_perf - frame_period))
                n_pacing_resets += 1
                next_tick_perf = flip_perf + frame_period

            frame_fields = {
                "episode_id": episode_id, "ep_frame": ep_frame,
                "action": action, "reward": reward,
                "terminated": bool(terminated), "truncated": bool(truncated),
                "run_time": step_run_time, "flip_time": self.clock.from_perf(flip_perf),
                "wall_time": self.clock.wall_time(), "variables": frame_state.variables,
            }
            # env_action: an adapter's own reading of the UI keys, when it differs
            # from the action sent (e.g. Rush Hour select+move -> Discrete).
            if isinstance(info, dict) and "env_action" in info:
                frame_fields["env_action"] = info["env_action"]
            if self.triggers.enabled:
                frame_fields["trigger"] = self.triggers.last_frame
            if game_phase.has_audio:
                # The chunk this frame's sound was queued as (-1: none); when it
                # reached the DAC is in the block_end record's audio onsets.
                frame_fields["audio_chunk"] = self.audio.last_chunk
            self.logger.log_frame(frame_fields, frame_rgb=frame_rgb,
                                  savestate=frame_state.blob if anchor_frame else None)
            self.logger.log_audio(sound)
            ep_frame += 1

        ## Final outcome
        outcome, outcome_message = _final_outcome(adapter, outcome, terminated, truncated)
        episode_result = EpisodeResult(episode_id, outcome, bool(terminated), bool(truncated),
                                       score, n_pacing_resets)
        self.logger.log(type="episode_end", **episode_result.record())
        if outcome_message and game_phase.show_outcome_duration > 0 and not self._phases_over():
            self.audio.stop()               # the episode's last sounds, still queued
            self.display.draw_text(f"Final score: {score:g}\n{outcome_message}")
            _wait_for_duration(self.display, game_phase.show_outcome_duration,
                               deadline_perf=self._phases_end_perf)
        return episode_result

    def _instructions(self, phase: dict, resumed: bool) -> dict | None:
        """Show a game phase's ``instructions``, if any, as a message headed by its title.

        :param phase: the game phase (``instructions``: ``title``, ``text``,
            ``align``, ``duration``, ``keys_to_dismiss``).
        :param resumed: whether the block carries on from an earlier run: the
            heading says "Resuming <title>" rather than "Starting <title>".
        :return: for the game's manifest entry, the ``text`` shown with its
            ``onset`` and ``offset``; ``None`` without instructions.
        :raises KeyboardInterrupt: on window close or ESC.
        """
        instructions_spec = phase.get("instructions")
        if instructions_spec is None:
            return None
        heading = f"{'Resuming' if resumed else 'Starting'} {instructions_spec['title']}"
        text = _join_multiline_text(instructions_spec.get("text", ""))
        return self._show_message({**instructions_spec, "text": f"{heading}\n\n{text}"})

    def _show_frame(
        self, adapter: EnvAdapter, play_sound: bool, score: float, block_end_perf: float
    ) -> tuple[float, Any, Any]:
        """Flip the adapter's frame with its HUD, then queue its sound against that flip.

        :param adapter: the env whose ``render`` / ``hud`` / ``overlay`` /
            ``sound`` to present.
        :param play_sound: pass the sound to the speakers.
        :param score: the episode's running score, for the HUD.
        :param block_end_perf: ``perf_counter`` the block ends at, for the HUD.
        :return: ``perf_counter`` of the flip, the rendered RGB frame, and the
            sound queued this frame (``None`` if ``play_sound`` is ``False``).
        """
        frame_rgb = adapter.render()
        hud_lines = adapter.hud(score, block_end_perf - time.perf_counter())
        flip_perf = self.display.draw_frame(frame_rgb, hud_lines, adapter.overlay())
        sound = adapter.sound() if play_sound else None
        if play_sound:
            self.audio.play(sound, flip_perf)
        return flip_perf, frame_rgb, sound

    def _game(self, phase: dict, phase_index: int,
              block_progress: BlockProgress | None = None) -> None:
        """Run a game block (one or more episodes) and save frame-level data.

        Creates the env via the phase's backend adapter, plays until duration /
        episode count / quit, then writes its log and a manifest phase entry.

        :param phase: the game phase's dict: the fields of
            :class:`~fmri_gym.curriculum.GamePhase`, ``keys``,
            ``instructions``, ``menu``, and the backend's own.
        :param phase_index: the phase's index in the curriculum (for the manifest).
        :param block_progress: what an earlier run that stopped this block
            left of it; ``None`` plays the block from its beginning.
        :raises KeyboardInterrupt: if the subject quits mid-block.
        """
        ## Settings
        game_phase = GamePhase.from_dict(phase, phase_index)
        # A block an earlier run stopped goes on with the time and episodes it has left.
        block_time_elapsed = (0.0 if block_progress is None
                              else block_progress.block_time_elapsed)
        num_completed_episodes = (0 if block_progress is None
                                  else block_progress.num_completed_episodes)
        # The savestate the next episode is restored to over its reset, if any.
        initial_state = (None if block_progress is None
                         else block_progress.episode_initial_state)
        # The savestates a rewinding block keeps (GamePhase.rewind_on): the
        # oldest is rewind_frames before the ending.
        rewind_savestates = (deque(maxlen=game_phase.rewind_frames + 1)
                             if game_phase.rewind_on else None)
        instructions = self._instructions(phase, resumed=block_progress is not None)

        # Some backends (nle, browser games) take several seconds to start;
        # show a Loading screen so the previous fixation "+" doesn't freeze.
        self.display.draw_text(f"Loading {phase.get('text') or game_phase.game} …")
        adapter = get_adapter(game_phase.backend, phase)
        speed_fields = ({} if game_phase.is_turn_based
                        else self._speed(adapter, game_phase.fps, phase_index, game_phase.game))
        block_folder_name = self.logger.open_block(phase_index, game_phase.backend,
                                                   game_phase.game, phase, game_phase.seed)

        ## Init loop over episodes
        vsync_locked = self.display.vsync and self.display.refresh_rate
        flip_period = 1 / self.display.refresh_rate if vsync_locked else None
        self.audio.start(frame_period=None if game_phase.is_turn_based else game_phase.frame_period,
                         flip_period=flip_period)
        onset = self.clock.run_time()
        block_start_perf = time.perf_counter()
        own_end_perf = block_start_perf + game_phase.duration - block_time_elapsed
        block_end_perf = min(own_end_perf, self._phases_end_perf)
        # The hold-a-key pause menu (reset / forfeit / resume), if the phase has one.
        menu = Menu(phase["menu"], block_start_perf) if "menu" in phase else None
        episodes: list[EpisodeResult] = []
        outcome = ""

        ## Loop over episodes within game block
        while outcome not in ("quit", "forfeit") and time.perf_counter() < block_end_perf:
            ## Run one episode
            # Seeded by episodes played, so a restart from the menu replays
            # the very instance the subject gave up on (a start-over).
            episode_result = self._episode(
                adapter, game_phase, seed=game_phase.seed + num_completed_episodes,
                episode_id=len(episodes), block_end_perf=block_end_perf, menu=menu,
                initial_state=initial_state, rewind_savestates=rewind_savestates)
            episodes.append(episode_result)
            outcome = episode_result.outcome
            # A rewound ending goes back to the oldest savestate kept,
            # rewind_frames before it; an ending sooner than that after the
            # start goes back to where the episode started, which is what
            # initial_state still holds.
            rewound = outcome in game_phase.rewind_on
            if not rewound:
                initial_state = None
            elif len(rewind_savestates) == rewind_savestates.maxlen:
                initial_state = rewind_savestates[0]
            # An episode's last sounds are still queued when it ends; drop them
            # so they do not play over the next episode or the next fixation.
            self.audio.stop()
            # An episode the subject restarted from the menu was not played; nor,
            # in a blocking block, one whose outcome does not advance; nor one
            # a clock or ESC stopped, which a later run may carry on; nor a
            # death that is rewound.
            outcome_advances = (game_phase.advance_on is None
                                or outcome in game_phase.advance_on)
            num_completed_episodes += (outcome not in ("reset", "playing", "quit")
                                       and not rewound and outcome_advances)
            if (game_phase.mode is BlockMode.EPISODE
                    and num_completed_episodes >= game_phase.n_episodes):
                break
        user_quit = (outcome == "quit")
        # Stopped before the block's own end (its clock, its episodes, a
        # forfeit) -- by the run's clock or ESC -- so a later run can carry it on.
        block_did_not_finish = user_quit or (
            self._phases_over() and time.perf_counter() < own_end_perf and outcome != "forfeit"
            and not (game_phase.mode is BlockMode.EPISODE
                     and num_completed_episodes >= game_phase.n_episodes))

        self.triggers.block_end()

        # Logging
        block_extra = adapter.block_extra() # run before closing the adapter to get the last state
        audio_log = self.audio.block_log()
        if audio_log is not None:
            audio_log = audio_log.in_run_time(self.clock.from_perf)
        summary = {
            "n_episodes": len(episodes), "n_frames": self.logger.n_frames,
            "outcomes": dict(Counter(episode.outcome for episode in episodes)),
            "n_pacing_resets": sum(episode.n_pacing_resets for episode in episodes),
            "total_reward": sum(episode.score for episode in episodes),
            "block_time_elapsed": block_time_elapsed + time.perf_counter() - block_start_perf,
            "num_completed_episodes": num_completed_episodes,
            "block_did_not_finish": block_did_not_finish,
        }
        self.logger.close_block(**summary, audio=audio_log, extra=block_extra)
        self.logger.log_phase({
            "index": phase_index, "type": "game", "backend": game_phase.backend,
            "game": game_phase.game, "mode": game_phase.mode.value,
            "onset": onset, "offset": self.clock.run_time(), **summary,
            **({"advancing_outcomes": list(game_phase.advance_on)}
               if game_phase.advance_on is not None else {}),
            "data_dir": block_folder_name, **speed_fields,
            **({"menu": menu.describe()} if menu else {}),
            **({"instructions": instructions} if instructions else {}),
        })

        # Close adapter and environment
        adapter.close()
        self.display.ensure() # Some gym envs call pygame.display.quit() on close() which tears down our shared window

        if user_quit:
            raise KeyboardInterrupt

    def _speed(self, adapter: EnvAdapter, fps: float, phase_index: int, game: str) -> dict:
        """The block's speed against the engine's own clock, said aloud when it is not 1.

        :param adapter: the block's adapter (see :meth:`EnvAdapter.native_fps`).
        :param fps: the block's steps per second.
        :param phase_index: the block's phase index.
        :param game: its game id.
        :return: manifest fields -- ``native_fps`` and ``speed`` -- or ``{}`` for
            an engine with no clock of its own.
        """
        native_fps = adapter.native_fps()
        if native_fps is None:
            return {}
        speed = fps / native_fps
        if abs(speed - 1) > _SAME_SPEED:
            print(f"phase {phase_index} ({game}): fps {fps:g} against the engine's own "
                  f"{native_fps:g} -- the game plays at {speed:.2f}x its real speed",
                  file=sys.stderr)
        return {"native_fps": native_fps, "speed": speed}

    def play(self) -> bool:
        """Play the curriculum: trigger wait, then each phase in order.

        A resumed run skips the phases an earlier run played, except the
        ``include_on_resume`` opening ones, and hands the block it carries on
        what that run left of it. A run with a duration stops its phases
        :data:`END_FIXATION_S` before the end and closes on a fixation cross.
        Always writes the run's manifest in ``finally``, including after an
        interrupt (partial data).

        :return: ``True`` if the run played to its end -- the curriculum's or
            its duration's -- and ``False`` if it was quit, so a caller can
            report a run that stopped early.
        """
        user_quit = False
        start_index = 0 if self.resume is None else self.resume.phase_index
        block_progress = None if self.resume is None else self.resume.block_progress
        handlers = {"fixation": self._fixation, "message": self._message,
                    "game": self._game, "survey": self._survey}
        if any(phase["type"].startswith("check_") for phase in self.phases):
            from . import checks  # rig-check phases: measured with this run's display and lines
            handlers.update({kind: partial(checks.run_check, self)
                             for kind in checks.CHECK_TYPES})
        try:
            self._trigger()
            if self.duration is not None:
                self._phases_end_perf = self.clock.trigger_perf + self.duration - END_FIXATION_S

            reached_end = True
            for phase_index, phase in enumerate(self.phases):
                if phase_index < start_index and not phase.get("include_on_resume", False):
                    continue                    # an earlier run played it
                if self._phases_over():
                    reached_end = False
                    break
                handler = handlers.get(phase["type"])
                if handler is None:
                    raise ValueError(f"unknown phase type: {phase['type']!r}")
                if phase_index == start_index and block_progress is not None:
                    # the block an earlier run stopped
                    self._game(phase, phase_index, block_progress)
                    continue
                handler(phase, phase_index)

            if self.duration is not None:
                self._end_fixation()
            if reached_end:
                self.display.draw_text("Done. Thank you!")
                time.sleep(2.0)
        except KeyboardInterrupt:
            user_quit = True
            print("Interrupted -- saving partial data.", file=sys.stderr)
        finally:
            if self.clock.trigger_perf is not None:
                self.triggers.lifecycle("task_stop")
            self.logger.set_extra("triggers", self.triggers.describe(self.clock))
            manifest_path = self.logger.save_manifest()
            print(f"Saved run to: {self.run_folder}")
            print(f"Manifest: {manifest_path}")
        return not user_quit

    def _phases_over(self) -> bool:
        """Whether the run has reached its closing fixation: no phase plays on past it."""
        return time.perf_counter() >= self._phases_end_perf

    def _end_fixation(self) -> None:
        """Close the run on :data:`END_FIXATION_S` seconds of fixation."""
        onset = self.clock.from_perf(self.display.draw_fixation())
        _wait_for_duration(self.display, END_FIXATION_S)
        self.logger.log_phase({"index": None, "type": "fixation", "end_of_run": True,
                               "onset": onset, "offset": self.clock.run_time()})
