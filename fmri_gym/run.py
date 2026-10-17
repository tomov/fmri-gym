"""One run: the engine-agnostic experiment loop that plays one curriculum.

A run is what one config file describes and one ``fmri-play`` plays (the
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

A run is the scanner's unit and a curriculum the subject's, so a run can
stop part-way through one and the next run carry on from there
(:mod:`fmri_gym.progress`). A run given a duration stops whatever phase is on
screen :data:`END_FIXATION_S` before its end and closes on a fixation cross,
so the haemodynamic response to the last thing played is still recorded; a
game block it stops keeps its savestate for the next run to restore.
"""

from __future__ import annotations

import copy
import math
import sys
import time
from collections import Counter
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import pygame

from . import bids, pad
from .adapters import get_adapter
from .audio import Audio
from .config import fold_cli_options
from .display import Display, check_monitor
from .keys import get_events, held_key_names, key_name
from .menu import Menu
from .logging import Logger, read_state
from .progress import Progress, fresh, start_of
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
#: Seconds of fixation that close a run given a duration: the haemodynamic
#: response to the last thing played takes about this long to come back down.
END_FIXATION_S = 10.0


class Clock:
    """Anchored at the scanner trigger; gives run + wall-clock time."""

    def __init__(self) -> None:
        """Create an untriggered clock (``t0_*`` are ``None`` until :meth:`trigger`)."""
        self.t0_perf = None
        self.t0_epoch = None

    def trigger(self) -> None:
        """Anchor the clock at the current time (call on scanner trigger)."""
        self.t0_perf = time.perf_counter()
        self.t0_epoch = time.time()

    def run_time(self) -> float:
        """Seconds since the scanner trigger (``perf_counter`` based).

        :return: seconds since this run's trigger.
        """
        return time.perf_counter() - self.t0_perf

    def from_perf(self, t_perf: float) -> float:
        """Convert a ``perf_counter`` stamp (e.g. a flip time) to run time.

        :param t_perf: a ``time.perf_counter()`` value.
        :return: seconds since the scanner trigger.
        """
        return t_perf - self.t0_perf

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
    deadline: float,
    clock: Clock,
    logger: Logger,
    key_to_action: dict | None = None,
    menu: Menu | None = None,
    latch: bool = False,
    repaint: Callable[[], float] | None = None,
) -> tuple[object | None, bool]:
    """Poll the keyboard until ``deadline``, logging every press/release.

    Replaces a plain sleep between frames: events are time-stamped on arrival
    -- to about a millisecond, or to one refresh when the display is
    vsync-locked and :meth:`Display.idle` re-presents the frame instead of
    sleeping. In turn-based play (``key_to_action`` given) the wait ends at
    the first mapped keydown so the step happens then, not at the tick.
    ``latch`` is the real-time counterpart: the keydown is remembered but the
    wait still runs to the tick, so the frame period is what it says it is.
    The wait also ends the moment the block's menu is armed (its key held long
    enough), which the caller reads off ``menu.pending``.

    :param display: the display, idled between polls.
    :param deadline: ``perf_counter`` at which to stop waiting.
    :param clock: the run's clock, for the log.
    :param logger: the run's logger.
    :param key_to_action: turn-based or latched: map of single key NAMES to
        env actions.
    :param menu: the block's :class:`~.menu.Menu`, if the phase has one.
    :param latch: keep waiting after a mapped keydown and return the last one
        seen, instead of ending the wait at the first.
    :param repaint: called every :data:`HUD_REPAINT_S` while waiting, for a
        wait long enough that the screen would otherwise go stale (see
        ``live_hud``). It redraws; it must not step, log or trigger.
    :return: ``(action_or_None, user_quit)``; ``action`` is set only when
        ``key_to_action`` is given, ``user_quit`` on window close / ESC.
    """
    latched_action = None
    next_repaint = time.perf_counter() + HUD_REPAINT_S
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
            name = key_name(event.key)
            if name is None:
                continue
            down = event.type == pygame.KEYDOWN
            logger.log(type="input_event", run_time=clock.run_time(), name=name, down=down)
            if down and key_to_action and name in key_to_action:
                if not latch:
                    return key_to_action[name], False
                # TODO: find out whether every real-time game wants this, not
                # only the slow ones that asked for it -- a press dropped
                # between two frames is a lost action at any fps, and if it is
                # general, latching stops being a phase flag and becomes what
                # the loop does.
                latched_action = key_to_action[name]
        if time.perf_counter() >= deadline:
            return latched_action, False
        if repaint is not None and time.perf_counter() >= next_repaint:
            repaint()
            next_repaint = time.perf_counter() + HUD_REPAINT_S
        display.idle(deadline)


def _wait_for_char(
    display: Display, char: str, dummy_trigger: bool = False, until: float = math.inf
) -> None:
    """Block until ``char`` is typed (or briefly sleep in dummy mode).

    :param display: the display, idled between polls (see :meth:`Display.idle`).
    :param char: the unicode character that unblocks the wait, or ``"any"``
        to accept every key (a self-paced "press any key" screen).
    :param dummy_trigger: if ``True``, sleep briefly and return without waiting.
    :param until: ``perf_counter`` at which to stop waiting all the same.
    :raises KeyboardInterrupt: on window close or ESC.
    """
    if dummy_trigger:
        time.sleep(0.05)
        return
    while time.perf_counter() < until:
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


def _wait_for_duration(display: Display, duration: float, until: float = math.inf) -> None:
    """Block for ``duration`` seconds.

    :param display: the display, idled between polls (see :meth:`Display.idle`).
    :param duration: seconds to wait.
    :param until: ``perf_counter`` at which to stop waiting all the same.
    :raises KeyboardInterrupt: on window close or ESC.
    """
    end = min(time.perf_counter() + duration, until)
    while time.perf_counter() < end:
        if _check_quit():
            raise KeyboardInterrupt
        display.idle(end, poll=0.005)


def _wait_for_keys(display: Display, names: list[str], deadline: float) -> str | None:
    """Block until one of the keys ``names`` is pressed, or ``deadline``.

    :param display: the display, idled between polls (see :meth:`Display.idle`).
    :param names: key NAMES (:mod:`fmri_gym.keys`) that end the wait.
    :param deadline: ``perf_counter`` at which to stop waiting.
    :return: the name of the key that ended the wait, or ``None`` at the deadline.
    :raises KeyboardInterrupt: on window close or ESC.
    """
    while time.perf_counter() < deadline:
        for event in get_events():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_ESCAPE:
                raise KeyboardInterrupt
            if key_name(event.key) in names:
                return key_name(event.key)
        display.idle(min(deadline, time.perf_counter() + 0.005), poll=0.005)
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
    :return: ``(outcome, message)``; the message is ``""`` on a quit.
    """
    if outcome == "quit":
        return outcome, ""
    if outcome:
        message = {"reset": "Starting over", "forfeit": "Forfeited"}[outcome]
    else:
        outcome, message = adapter.outcome(terminated, truncated)
        if outcome == "playing":        # the block's clock, not the game, ended it
            message = "Block ended"
    return outcome, message


def _block_progress(
    last: dict, elapsed: float, completed: int, base_seed: int, state: bytes | None
) -> dict:
    """How far a game block got before the run stopped it, for the next run to carry on.

    :param last: the block's last ``episode_end`` record.
    :param elapsed: the block's seconds of play, earlier runs' included.
    :param completed: episodes counted toward the block so far.
    :param base_seed: the block's base seed.
    :param state: the savestate the block ended in (``block_extra()["state"]``),
        or ``None`` for a backend that keeps none.
    :return: the ``block`` of :mod:`fmri_gym.progress`'s record: with
        ``episode`` set when the last episode was still going.
    """
    going = last["outcome"] in ("playing", "quit")
    episode = None
    if going:
        episode = {"id": last["episode_id"], "seed": base_seed + completed,
                   "ep_frame": last["ep_frames"], "score": last["score"], "state": state}
    return {"elapsed": elapsed, "completed": completed, "base_seed": base_seed,
            "next_episode": last["episode_id"] + (0 if going else 1), "episode": episode}


class Run:
    """Plays one curriculum for one subject, dispatching phases to handlers."""

    def __init__(
        self,
        subject: str,
        curriculum: list[dict],
        display: Display,
        outdir: str,
        audio: Audio | None = None,
        triggers: Triggers | None = None,
        dummy_trigger: bool = False,
        duration: float | None = None,
        progress: Progress | None = None,
    ) -> None:
        """Set up clock, logger, and phase dispatch for one subject.

        :param subject: subject identifier used in log paths / manifest.
        :param curriculum: ordered list of phase dicts (``type``, timings, …).
        :param display: shared pygame display used by all phases.
        :param outdir: the run's folder, for its manifest and block logs.
        :param audio: shared audio output used by all phases; one is created if
            omitted, and stays silent unless an adapter returns sound.
        :param triggers: shared trigger output (start sync + codes; see
            :mod:`fmri_gym.triggers`), built by the caller like the display
            and the audio. ``None`` = the fMRI default: wait for ``=``, send
            no trigger codes.
        :param dummy_trigger: if ``True``, skip real experimenter/scanner waits.
        :param duration: the run's seconds from the trigger, the closing
            fixation included; ``None`` plays the curriculum to its end.
        :param progress: where the run begins in its curriculum, kept current
            as it plays; ``None`` is the curriculum's beginning.
        :raises ValueError: a duration no longer than the closing fixation.
        """
        if duration is not None and duration <= END_FIXATION_S:
            raise ValueError(f"--duration {duration:g}: a run closes on {END_FIXATION_S:g} s "
                             "of fixation, so it must be longer than that")
        self.subject = subject
        self.curriculum = curriculum
        self.display = display
        self.audio = audio or Audio()
        self.dummy_trigger = dummy_trigger
        self.clock = Clock()
        self.logger = Logger(outdir, subject, curriculum, self.clock)
        self.logger.set_extra("display", display.describe())
        self.logger.set_extra("dummy_trigger", dummy_trigger)
        self.logger.set_extra("audio", self.audio.describe())
        self.outdir = outdir
        self.triggers = triggers or Triggers.from_config(None)
        self.sync = self.triggers.sync
        self.duration = duration
        #: ``perf_counter`` at which the phases stop for the closing fixation.
        self._run_end = math.inf
        self.progress = progress or Progress(copy.deepcopy(curriculum), fresh("not asked to"))
        self.logger.set_extra("duration", duration)
        self.logger.set_extra("resume", self.progress.start.note)
        self._save_progress()

    @classmethod
    def from_config(cls, config: dict, args: Any) -> "Run":
        """Everything one run needs, from its config and the command line.

        Works out where the run writes and what seeds it plays, then opens the
        trigger line, the audio output and the window -- in that order, so a
        bad section, an unopenable port or an unusable output stops the run
        before anything is on screen. Each says on stderr what it got.

        :param config: the run's config, already checked (``validate_config``).
        :param args: ``fmri_play``'s parsed flags.
        :return: the run, ready to :meth:`play`.
        :raises ValueError: on a BIDS name or number, a monitor or a window
            size this machine or this subject refuses.
        """
        curriculum = config["curriculum"]
        as_written = copy.deepcopy(curriculum)  # before the run folds its seeds and options in
        width, height = (int(x) for x in args.size.lower().split("x"))
        out = bids.run_output(args.data_root, args.subject, args.curriculum,
                              args.ses, args.run)
        if out.attempt > 1:
            print(f"{out.label} already has data: this is attempt {out.attempt} at it, and "
                  "writes beside the others", file=sys.stderr)
        print(f"output: {out.folder}", file=sys.stderr)
        start = start_of(as_written, args.data_root, args.subject,
                         bids.task_label(args.curriculum), args.resume)
        seeds = bids.fold_seeds(curriculum, out.label)
        if start.block is not None:  # the block keeps the episodes it began with
            curriculum[start.index]["seed"] = start.block["base_seed"]
            seeds["phases"][start.index] = "resumed"
            print(f"seeds: phase {start.index} = {start.block['base_seed']} (resumed)",
                  file=sys.stderr)
        check_monitor(args.monitor)
        fold_cli_options(curriculum, args)

        triggers = Triggers.from_config(config.get("triggers"))
        print(f"triggers: {triggers.status()}", file=sys.stderr)
        if args.dummy_trigger:
            print("triggers: --dummy-trigger: the experimenter and scanner waits are skipped; "
                  "this is a test run, not a session", file=sys.stderr)
        audio = Audio(enabled=not args.no_audio)
        print(f"audio: {audio.status()}", file=sys.stderr)
        pad_status = pad.init(not getattr(args, "no_pad", False))
        print(f"pad: {pad_status}", file=sys.stderr)
        display = Display(size=(width, height), fullscreen=args.fullscreen,
                          vsync=not args.no_vsync, monitor=args.monitor)
        run = cls(args.subject, curriculum, display, out.folder, audio=audio,
                  triggers=triggers, dummy_trigger=args.dummy_trigger,
                  duration=args.duration, progress=Progress(as_written, start))
        run.logger.set_extra("run", {"label": out.label, "attempt": out.attempt})
        run.logger.set_extra("seeds", seeds)
        run.logger.set_extra("pad", pad_status)   # which controller answered, and as what
        run.logger.set_extra("versions", {  # the banner fmri_gym hides said these
            "pygame": pygame.version.ver, "sdl": ".".join(map(str, pygame.get_sdl_version()))})
        return run

    def _time_up(self) -> bool:
        """Whether the run has reached its closing fixation."""
        return time.perf_counter() >= self._run_end

    def _save_progress(self) -> None:
        """Write where the next run of this curriculum starts into the manifest."""
        self.logger.set_extra("progress", self.progress.describe())

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
        if self.sync.mode == "wait":
            self.display.draw_text(f"Waiting for the scanner...\n\n(its trigger key: "
                                   f"{self.sync.key!r})")
            _wait_for_char(self.display, self.sync.key, dummy_trigger=self.dummy_trigger)
        elif self.sync.mode == "send":
            self.display.draw_text("Starting the recording...")
            self.triggers.lifecycle("scanner_start")
            _wait_for_duration(self.display, self.sync.delay)

        self.clock.trigger()
        self.logger.set_trigger_time()
        self.triggers.lifecycle("task_start")

    def _fixation(self, phase: dict, index: int) -> None:
        """Show a fixation cross for ``phase["duration"]`` seconds.

        :param phase: fixation-phase config (``duration``, default 2.0).
        :param index: phase index in the curriculum (for the manifest).
        """
        duration = phase.get("duration", 2.0)

        onset = self.clock.from_perf(self.display.draw_fixation())
        _wait_for_duration(self.display, duration, until=self._run_end)

        self.logger.log_phase({"index": index, "type": "fixation",
                               "onset": onset, "offset": self.clock.run_time()})

    def _message(self, phase: dict, index: int) -> None:
        """Show on-screen text until a key press or timed duration.

        :param phase: message-phase config (``text`` as a string or list of
            lines; optional ``duration`` / ``key`` / ``align``). ``key`` is the
            character that dismisses the screen (default SPACE), or ``"any"``.
        :param index: phase index in the curriculum (for the manifest).
        """
        text = _join_multiline_text(phase.get("text", ""))
        duration = phase.get("duration")

        onset = self.clock.from_perf(
            self.display.draw_text(text, align=phase.get("align", "center")))
        if duration is None:
            _wait_for_char(self.display, phase.get("key", " "),
                           dummy_trigger=self.dummy_trigger, until=self._run_end)
        else:
            _wait_for_duration(self.display, duration, until=self._run_end)

        self.logger.log_phase({"index": index, "type": "message", "text": text,
                               "onset": onset, "offset": self.clock.run_time()})

    def _survey(self, phase: dict, index: int) -> None:
        """Run a Likert-style survey and log each confirmed response.

        :param phase: survey-phase config (``questions``, optional ``n_points``).
        :param index: phase index in the curriculum (for the manifest).
        :raises KeyboardInterrupt: on window close or ESC.
        """
        questions = phase.get("questions", [])
        n_points = phase.get("n_points", 7)
        onset = self.clock.run_time()

        responses = []
        try:
            self._ask(questions, n_points, responses)
        finally:  # a quit mid-survey keeps the answers already confirmed
            self.logger.log_phase({"index": index, "type": "survey",
                                   "onset": onset, "offset": self.clock.run_time(),
                                   "responses": responses})

    def _ask(self, questions: list[str], n_points: int, responses: list[dict]) -> None:
        """Append each confirmed Likert answer to ``responses``, until the run's closing fixation.

        :raises KeyboardInterrupt: on window close or ESC.
        """
        for q in questions:
            value = (n_points + 1) // 2
            confirmed = False
            while not confirmed:
                if self._time_up():
                    return
                scale = "  ".join((f"[{i}]" if i == value else f" {i} ")
                                  for i in range(1, n_points + 1))
                self.display.draw_text(
                    f"{q}\n\nDisagree      Agree\n{scale}\n\n"
                    "(LEFT/RIGHT to rate, ENTER to confirm)")
                for event in get_events():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            raise KeyboardInterrupt
                        elif event.key == pygame.K_LEFT:
                            value = max(1, value - 1)
                        elif event.key == pygame.K_RIGHT:
                            value = min(n_points, value + 1)
                        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                            confirmed = True
                time.sleep(0.005)
            responses.append({"question": q, "value": value,
                              "run_time": self.clock.run_time()})

    def _episode(
        self,
        adapter: EnvAdapter,
        *,
        seed: int,
        episode_id: int,
        turn_based: bool,
        latched: bool,
        live_hud: bool,
        dt: float,
        state_stride: int,
        block_end: float,
        play_sound: bool,
        menu: Menu | None,
        outcome_duration: float,
        carry: dict | None = None,
    ) -> dict:
        """Run one episode, logging every frame.

        :param adapter: wrapped env for reset/step/render/sound/capture.
        :param seed: RNG seed for this episode's ``reset``.
        :param episode_id: index of this episode within the game block.
        :param turn_based: if True, advance only on mapped keydowns.
        :param latched: real-time only: let a fresh keydown win over held keys.
        :param live_hud: turn-based only: redraw the HUD while waiting for the
            press, so its clock is not the one the last press left behind.
        :param dt: target seconds per frame (``1 / fps``).
        :param state_stride: record a full savestate every this many frames.
        :param block_end: ``perf_counter`` deadline for the game block.
        :param play_sound: pass the adapter's sound to the speakers (the
            phase's ``audio``); muting never changes what is logged.
        :param menu: the block's :class:`~.menu.Menu`, or ``None``.
        :param outcome_duration: seconds the final score and outcome are shown.
        :param carry: an episode an earlier run stopped in (``ep_frame``,
            ``score``, and its savestate, ``state``, as the log encodes one),
            restored over the reset so it carries on where it stopped.
        :return: the episode's ``episode_end`` record, as recorded:
            ``episode_id``, ``outcome`` (one of the adapter's,
            :meth:`~.adapters.base.EnvAdapter.outcome`: ``"won"``, ``"lost"``,
            ``"terminated"``, ``"truncated"``, or ``"playing"`` when the
            block's clock cut it off; or ``"quit"`` for ESC / window close,
            ``"reset"`` / ``"forfeit"`` for the subject's choice in the menu),
            ``terminated``, ``truncated``, ``score``, ``ep_frames`` (the
            episode's frames, the carried ones included) and ``n_pacing_resets``.
        """
        terminated = truncated = False
        outcome = ""
        score = 0.0                     # the episode's cumulative reward
        n_pacing_resets = 0
        ep_frame = 0
        key_to_action = (adapter.keymap.turn_actions()
                         if turn_based or latched else None)

        def redraw() -> float:
            """Present the frame again, for its HUD. Never steps, logs or triggers."""
            return self._show(adapter, False, score, block_end)[0]

        # A turn-based wait is unbounded -- it ends at a press -- and the HUD
        # is drawn only by _show, so without this the block clock a subject
        # reads is the one their last press left behind, while the clock the
        # loop ends on runs on regardless. Off unless the phase asks: a game
        # whose HUD says nothing about time has nothing to gain, and a
        # real-time wait is one frame long and repaints anyway.
        repaint = redraw if turn_based and live_hud else None
        # Set while the env is in a state it takes no action in, to the action
        # to step it with; see EnvAdapter.autoplay. Only a turn-based block
        # cares: a real-time one steps through such a stretch anyway.
        auto: Any = None

        ## Reset environment and show initial state
        observation, info = adapter.reset(seed)
        if carry is not None:
            adapter.restore(read_state(carry["state"]))
            score, ep_frame = carry["score"], carry["ep_frame"]
        self.display.call_on_flip(self.triggers.episode_start)
        # Paced from the flip, so a slow reset does not become a burst of
        # catch-up frames. The reset frame's sound is not played: it is not a
        # step's, and it would start the episode's sound off its flips.
        flip_t0, _, _ = self._show(adapter, False, score, block_end)
        self.logger.log(type="episode_start", episode_id=episode_id, seed=seed,
                        resumed=carry is not None, flip_time=self.clock.from_perf(flip_t0))
        next_t = flip_t0 + dt

        ## Loop over frames within episode
        while not (terminated or truncated) and time.perf_counter() < block_end:
            # Wait for the frame tick (turn-based: for a mapped keydown, up to
            # the block end), polling keys as we go so presses are stamped on
            # arrival; a vsync-locked display re-presents the frame meanwhile.
            # TODO(#43): anchor the wait to the last step (t_step + dt), not to the flip.
            # An autoplayed stretch is paced like a real-time frame even in a
            # turn-based block: the env is not waiting on the subject, so the
            # wait is one tick and the keys polled through it are logged but
            # not acted on.
            waits_for_press = turn_based and auto is None
            deadline = block_end if waits_for_press else next_t
            action, user_quit = _poll_keys_until(
                self.display, deadline, self.clock, self.logger,
                key_to_action if waits_for_press else None, menu,
                latch=latched and not turn_based,
                repaint=repaint if waits_for_press else None)
            if user_quit:
                outcome = "quit"
                break
            if menu is not None and menu.pending:
                choice = menu.run(self.display, self.clock.run_time, self.logger)
                if choice != "resume":
                    outcome = choice
                    break
                flip_t0, _, _ = self._show(adapter, False, score, block_end)
                next_t = flip_t0 + dt
                continue
            next_t += dt
            if waits_for_press and action is None:
                continue                        # block ended without a press
            if auto is not None:
                action = auto                   # the env is not taking ours
            elif not turn_based and action is None:
                # No latched press this frame (or the phase never asked for
                # one): the action is whatever is held down at the tick.
                action = adapter.keymap.resolve(held_key_names())

            observation, reward, terminated, truncated, info = adapter.step(action)
            auto = adapter.autoplay(info) if turn_based else None
            t_step = self.clock.run_time()
            score += float(reward)
            # A savestate at each episode's first frame (the replay anchor), then
            # every stride; frames between are the anchor plus the logged actions.
            save_state = ep_frame % state_stride == 0
            fs = adapter.capture(observation, info, want_blob=save_state)
            # The frame trigger goes out on the flip that shows this frame.
            self.display.call_on_flip(self.triggers.frame)
            flip_t, frame, sound = self._show(adapter, play_sound, score, block_end)
            # More than a frame behind (a stall): drop the debt, or it is repaid
            # as a burst of one-refresh frames. The frame of slack is what a
            # vsync-locked flip normally lands after its tick.
            # TODO(#43): why frames fall behind at all is not established.
            if turn_based:
                # Anchor the next tick to this flip. A turn-based wait leaves
                # next_t far behind -- the deadline it waits on is the block's,
                # not the tick -- so an autoplayed stretch starting from the
                # old value would find every tick already past and run flat
                # out instead of at fps.
                next_t = flip_t + dt
            elif next_t + dt < flip_t:
                self.logger.log(type="pacing_reset", flip_time=self.clock.from_perf(flip_t),
                                late=flip_t - (next_t - dt))
                n_pacing_resets += 1
                next_t = flip_t + dt

            fields = {
                "episode_id": episode_id, "ep_frame": ep_frame,
                "action": action, "reward": reward,
                "terminated": bool(terminated), "truncated": bool(truncated),
                "run_time": t_step, "flip_time": self.clock.from_perf(flip_t),
                "wall_time": self.clock.wall_time(), "variables": fs.variables,
            }
            # env_action: an adapter's own reading of the UI keys, when it differs
            # from the action sent (e.g. Rush Hour select+move -> Discrete).
            if isinstance(info, dict) and "env_action" in info:
                fields["env_action"] = info["env_action"]
            if self.triggers.enabled:
                fields["trigger"] = self.triggers.last_frame
            if play_sound:
                # The chunk this frame's sound was queued as (-1: none); when it
                # reached the DAC is in the block_end record's audio onsets.
                fields["audio_chunk"] = self.audio.last_chunk
            self.logger.log_frame(fields, frame=frame, state=fs.blob)
            self.logger.log_audio(sound)
            ep_frame += 1

        ## Final outcome
        outcome, message = _final_outcome(adapter, outcome, terminated, truncated)
        end = {"episode_id": episode_id, "outcome": outcome, "terminated": bool(terminated),
               "truncated": bool(truncated), "score": score, "ep_frames": ep_frame,
               "n_pacing_resets": n_pacing_resets}
        self.logger.log(type="episode_end", **end)
        if message and outcome_duration > 0 and not self._time_up():
            self.audio.stop()               # the episode's last sounds, still queued
            self.display.draw_text(f"Final score: {score:g}\n{message}")
            _wait_for_duration(self.display, outcome_duration, until=self._run_end)
        return end

    def _instructions(self, phase: dict, resumed: bool) -> dict | None:
        """Show a game phase's ``instructions``, if it has any, until a duration or a key.

        :param phase: the game phase (``instructions``: ``title``, ``text``,
            ``align``, ``duration``, ``keys_to_dismiss``).
        :param resumed: whether the block carries on from an earlier run,
            which the heading says ("Resuming ..." rather than "Starting ...").
        :return: for the manifest, the heading, onset, offset and the key that
            dismissed the screen (``None`` if none did); ``None`` without instructions.
        :raises KeyboardInterrupt: on window close or ESC.
        """
        spec = phase.get("instructions")
        if spec is None:
            return None
        heading = f"{'Resuming' if resumed else 'Starting'} {spec['title']}"
        text = _join_multiline_text(spec.get("text", ""))
        onset = self.clock.from_perf(self.display.draw_text(
            f"{heading}\n\n{text}", align=spec.get("align", "center")))
        keys = spec.get("keys_to_dismiss", [])
        deadline = time.perf_counter() + spec.get("duration", math.inf)
        if keys and self.dummy_trigger:  # a test run skips the waits for a key
            deadline = min(deadline, time.perf_counter() + 0.05)
        key = _wait_for_keys(self.display, keys, min(deadline, self._run_end))
        return {"heading": heading, "onset": onset, "offset": self.clock.run_time(),
                "dismissed_by": key}

    def _log_resume(self, carry: dict) -> None:
        """Log the ``resume`` line: how far the block had got, and the savestate it goes on from.

        :param carry: the block's progress from the earlier run
            (:func:`_block_progress`, with ``state`` as the log encodes one).
        """
        episode = carry["episode"]
        self.logger.log(
            type="resume", from_run=self.progress.start.note["from_run"],
            elapsed=carry["elapsed"], completed=carry["completed"],
            next_episode=carry["next_episode"],
            episode=None if episode is None else {k: v for k, v in episode.items()
                                                   if k != "state"},
            state=None if episode is None else read_state(episode["state"]))

    def _show(
        self, adapter: EnvAdapter, play_sound: bool, score: float, block_end: float
    ) -> tuple[float, Any, Any]:
        """Flip the adapter's frame with its HUD, then queue its sound against that flip.

        :param adapter: the env whose ``render`` / ``hud`` / ``overlay`` /
            ``sound`` to present.
        :param play_sound: pass the sound to the speakers.
        :param score: the episode's running score, for the HUD.
        :param block_end: ``perf_counter`` the block ends at, for the HUD.
        :return: ``perf_counter`` of the flip, the rendered frame, and the
            sound queued this frame (``None`` if ``play_sound`` is ``False``).
        """
        frame = adapter.render()
        hud = adapter.hud(score, block_end - time.perf_counter())
        flip_t = self.display.draw_frame(frame, hud, adapter.overlay())
        sound = adapter.sound() if play_sound else None
        if play_sound:
            self.audio.play(sound, flip_t)
        return flip_t, frame, sound

    def _game(self, phase: dict, index: int) -> None:
        """Run a game block (one or more episodes) and save frame-level data.

        Creates the env via the phase's backend adapter, plays until duration /
        episode count / quit, then writes its log and a manifest phase entry.

        :param phase: game-phase config (``backend``, ``game``, ``mode``,
            ``duration`` / ``n_episodes`` and ``advancing_outcomes``, ``fps``,
            ``seed``, ``state_stride``, ``turn_based``, ``latched_keys``,
            ``live_hud``,
            ``keys``, ``outcome_duration``, …).
        :param index: phase index in the curriculum (for the manifest).
        :raises KeyboardInterrupt: if the subject quits mid-block.
        """
        ## Config
        backend = phase.get("backend", "gym")
        mode = phase.get("mode", "duration")
        duration = phase.get("duration", 30.0)
        n_episodes = phase.get("n_episodes", 1)
        # A blocking block: only an episode whose outcome is listed (usually
        # ["won"]) counts toward n_episodes; any other replays the same instance
        # (the seed follows the count), so the subject stays on a level until
        # they clear it, max_duration runs out, or they forfeit from the menu.
        # None: every episode counts, whatever its outcome.
        advancing = phase.get("advancing_outcomes")
        outcome_duration = float(phase.get("outcome_duration", 2.0))
        base_seed = phase.get("seed", 1000 + index)
        # 1 = a savestate every frame (default); larger trades savestate density
        # for disk -- retro states are ~1 MB/frame.
        state_stride = max(1, int(phase.get("state_stride", 1)))
        cap = duration if mode == "duration" else phase.get("max_duration", 300.0)
        # Turn-based games (grid worlds: FrozenLake, CliffWalking, Taxi, ...) must
        # advance ONE step per deliberate key PRESS, not once per frame. In a
        # real-time loop they'd auto-step every frame with the noop action (which
        # for e.g. FrozenLake is action 0 = LEFT), so the agent "moves on its own"
        # and a single held key fires many times. turn_based fixes both.
        turn_based = bool(phase.get("turn_based", False))
        # Real-time blocks poll HELD keys, so a press that starts and ends
        # between two frames is never seen -- at a grid world's few frames per
        # second that loses most taps. `latched_keys` lets a fresh keydown win
        # instead, falling back to the held-key poll (so holding a key still
        # repeats). Off by default: backends whose actions are key COMBINATIONS
        # must keep polling, and this is exactly what they do today.
        latched = bool(phase.get("latched_keys", False))
        # A turn-based block draws a frame only when the subject presses, so
        # the seconds its HUD shows stop between presses while the clock the
        # block actually ends on does not. `live_hud` redraws the standing
        # frame a few times a second through the wait, which changes nothing
        # but the pixels: no step, no log row, no trigger, no sound. Off by
        # default, so a block that has run this way keeps running this way.
        live_hud = bool(phase.get("live_hud", False))
        if live_hud and not turn_based:
            raise ValueError(f'game phase {index}: "live_hud" is for turn_based '
                             "blocks, whose wait is unbounded; a real-time block "
                             "already redraws every frame")
        play_sound = phase.get("audio", True)
        if not isinstance(play_sound, bool):
            raise ValueError(f'game phase {index}: "audio" must be true or false, '
                             f"got {play_sound!r}")
        # Every game phase states its own fps (validate_config refuses one that
        # does not): what the block plays at is the config's business, not a
        # default that changes with the engine underneath it.
        fps = phase["fps"]
        dt = 1.0 / fps
        # A block an earlier run stopped: its clock, count and episode go on.
        carry = self.progress.start.block if index == self.progress.start.index else None
        instructions = self._instructions(phase, resumed=carry is not None)

        # Some backends (nle, browser games) take several seconds to start;
        # show a Loading screen so the previous fixation "+" doesn't freeze.
        self.display.draw_text(
            f"Loading {phase.get('text') or phase.get('game', 'game')} …")
        adapter = get_adapter(backend, phase)
        speed = {} if turn_based else self._speed(adapter, fps, index, phase["game"])
        data_dir = self.logger.open_block(index, backend, phase["game"], phase, base_seed)
        if carry is not None:
            self._log_resume(carry)

        ## Init loop over episodes
        locked = self.display.vsync and self.display.refresh_rate
        flip_period = 1 / self.display.refresh_rate if locked else None
        self.audio.start(frame_period=None if turn_based else dt, flip_period=flip_period)
        onset = self.clock.run_time()
        block_start = time.perf_counter()
        played_before = carry["elapsed"] if carry else 0.0
        own_end = block_start + cap - played_before
        block_end = min(own_end, self._run_end)
        # The hold-a-key pause menu (reset / forfeit / resume), if the phase has one.
        menu = Menu(phase["menu"], block_start) if "menu" in phase else None
        episodes: list[dict] = []       # each episode's episode_end record
        completed = carry["completed"] if carry else 0
        first_id = carry["next_episode"] if carry else 0
        resumed_episode = carry["episode"] if carry else None
        outcome = ""

        ## Loop over episodes within game block
        while outcome not in ("quit", "forfeit") and time.perf_counter() < block_end:
            ## Run one episode
            # Seeded by episodes played, so a restart from the menu replays
            # the very instance the subject gave up on (a start-over).
            end = self._episode(
                adapter, seed=base_seed + completed, episode_id=first_id + len(episodes),
                turn_based=turn_based, latched=latched, live_hud=live_hud,
                dt=dt, state_stride=state_stride,
                block_end=block_end, play_sound=play_sound, menu=menu,
                outcome_duration=outcome_duration, carry=resumed_episode)
            resumed_episode = None
            episodes.append(end)
            outcome = end["outcome"]
            # An episode's last sounds are still queued when it ends; drop them
            # so they do not play over the next episode or the next fixation.
            self.audio.stop()
            # An episode the subject restarted from the menu was not played; nor,
            # in a blocking block, one whose outcome does not advance; nor one
            # still going when a clock or ESC stopped it, which a later run may
            # carry on.
            completed += (outcome not in ("reset", "playing", "quit")
                          and (advancing is None or outcome in advancing))
            if mode == "episode" and completed >= n_episodes:
                break
        stopped = time.perf_counter()
        user_quit = outcome == "quit"

        self.triggers.block_end()
        audio = self.audio.block_log()
        if audio:
            audio["onsets"] = [[chunk, self.clock.from_perf(t)] for chunk, t in audio["onsets"]]
        extra = adapter.block_extra()
        adapter.close()
        # Some gym envs (classic-control) call pygame.display.quit() on close(),
        # which tears down our shared window; rebuild it if so.
        self.display.ensure()
        summary = {
            "n_episodes": len(episodes), "n_frames": self.logger.n_frames,
            # How the episodes ended, by outcome name (see EnvAdapter.outcome).
            "outcomes": dict(Counter(e["outcome"] for e in episodes)),
            "n_pacing_resets": sum(e["n_pacing_resets"] for e in episodes),
            "total_reward": sum(e["score"] for e in episodes),
        }
        self.logger.close_block(**summary, audio=audio, extra=extra)
        self.logger.log_phase({
            "index": index, "type": "game", "backend": backend,
            "game": phase["game"], "mode": mode,
            "onset": onset, "offset": self.clock.run_time(), **summary,
            **({"advancing_outcomes": advancing} if advancing is not None else {}),
            "data_dir": data_dir, **speed, "resumed": carry is not None,
            **({"menu": menu.describe()} if menu else {}),
            **({"instructions": instructions} if instructions else {}),
        })
        finished = outcome == "forfeit" or not user_quit and (
            (mode == "episode" and completed >= n_episodes) or stopped >= own_end)
        if finished:
            self.progress.passed(index)
        elif episodes:  # one stopped before it played anything stays where it was
            self.progress.cut(index, _block_progress(
                episodes[-1], played_before + stopped - block_start, completed, base_seed,
                (extra or {}).get("state")))
        self._save_progress()
        if user_quit:
            raise KeyboardInterrupt

    def _phases(self) -> bool:
        """Play the run's phases (see :meth:`~.progress.Start.schedule`) until the last, or time.

        Each phase played to its end moves the progress past it; one the run's
        clock stopped is left where it was, to be played again. A game block
        records its own progress, since it can stop part-way and carry on.
        The opening phases a resumed run plays again are not progress at all.

        :return: ``True`` if the last phase was played, ``False`` if the run's
            clock stopped first.
        :raises KeyboardInterrupt: on window close or ESC.
        """
        handlers = {"fixation": self._fixation, "message": self._message,
                    "game": self._game, "survey": self._survey}
        if any(p["type"].startswith("check_") for p in self.curriculum):
            from . import checks  # rig-check phases: measured with this run's display and lines
            handlers.update({kind: partial(checks.run_check, self)
                             for kind in checks.CHECK_TYPES})
        start = self.progress.start
        for index in start.schedule(self.curriculum):
            if self._time_up():
                return False
            phase = self.curriculum[index]
            handler = handlers.get(phase["type"])
            if handler is None:
                raise ValueError(f"unknown phase type: {phase['type']!r}")
            handler(phase, index)
            if phase["type"] == "game" or index < start.index:
                continue
            if self._time_up():
                return False
            self.progress.passed(index)
            self._save_progress()
        return True

    def _end_fixation(self) -> None:
        """Close the run on :data:`END_FIXATION_S` of fixation, after whatever was playing."""
        onset = self.clock.from_perf(self.display.draw_fixation())
        _wait_for_duration(self.display, END_FIXATION_S)
        self.logger.log_phase({"index": None, "type": "fixation", "end_of_run": True,
                               "onset": onset, "offset": self.clock.run_time()})

    def _speed(self, adapter: EnvAdapter, fps: float, index: int, game: str) -> dict:
        """The block's speed against the engine's own clock, said aloud when it is not 1.

        :param adapter: the block's adapter (see :meth:`EnvAdapter.native_fps`).
        :param fps: the block's steps per second.
        :param index: the block's phase index.
        :param game: its game id.
        :return: manifest fields -- ``native_fps`` and ``speed`` -- or ``{}`` for
            an engine with no clock of its own.
        """
        native = adapter.native_fps()
        if native is None:
            return {}
        speed = fps / native
        if abs(speed - 1) > _SAME_SPEED:
            print(f"phase {index} ({game}): fps {fps:g} against the engine's own {native:g} -- "
                  f"the game plays at {speed:.2f}x its real speed", file=sys.stderr)
        return {"native_fps": native, "speed": speed}

    def play(self) -> bool:
        """Play the curriculum from where the run begins: trigger wait, then each phase in order.

        With a duration, the phases stop :data:`END_FIXATION_S` before its end
        and the run closes on a fixation cross. Always writes the run's
        manifest in ``finally``, including after an interrupt (partial data).

        :return: ``True`` if the run played to its end -- the curriculum's or
            its duration's -- and ``False`` if it was quit, so a caller can
            report a run that stopped early.
        """
        completed = False
        try:
            self._trigger()
            if self.duration is not None:
                self._run_end = self.clock.t0_perf + self.duration - END_FIXATION_S
            reached_end = self._phases()
            if self.duration is not None:
                self._end_fixation()
            if reached_end:
                self.display.draw_text("Done. Thank you!")
                time.sleep(2.0)
            completed = True
        except KeyboardInterrupt:
            print("Interrupted -- saving partial data.", file=sys.stderr)
        finally:
            if self.clock.t0_perf is not None:
                self.triggers.lifecycle("task_stop")
            self.logger.set_extra("triggers", self.triggers.describe(self.clock))
            manifest_path = self.logger.save_manifest()
            print(f"Saved run to: {self.outdir}")
            print(f"Manifest: {manifest_path}")
        return completed
