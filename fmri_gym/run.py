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
"""

from __future__ import annotations

import random
import sys
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable

import pygame

from . import bids
from .adapters import get_adapter
from .audio import Audio
from .config import fold_cli_options, seconds_range
from .display import Display, check_monitor
from .keys import held_key_names, key_name
from .logging import Logger
from .triggers import Triggers

if TYPE_CHECKING:
    from .adapters.base import EnvAdapter

TRIGGER_KEY = "="
EXPERIMENTER_KEY = " "
#: How far ``fps`` may sit from the engine's own rate and still count as real
#: speed: what the audio output absorbs by resampling.
_SAME_SPEED = 2e-3


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

    def run_time(self) -> float | None:
        """Seconds since the scanner trigger (``perf_counter`` based).

        :return: seconds since this run's trigger, or ``None`` before it: a
            curriculum may put its instructions above its ``trigger`` phase,
            and a phase the scan has not started for has no run time.
        """
        return None if self.t0_perf is None else time.perf_counter() - self.t0_perf

    def from_perf(self, t_perf: float) -> float | None:
        """Convert a ``perf_counter`` stamp (e.g. a flip time) to run time.

        :param t_perf: a ``time.perf_counter()`` value.
        :return: seconds since the scanner trigger, or ``None`` before it
            (see :meth:`run_time`).
        """
        return None if self.t0_perf is None else t_perf - self.t0_perf

    def wall_time(self) -> float:
        """Current wall-clock epoch time.

        :return: ``time.time()`` seconds since the Unix epoch.
        """
        return time.time()


def _check_quit() -> bool:
    """Drain pygame events and report whether the user requested quit.

    :return: ``True`` if the window was closed or ESC was pressed.
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return True
    return False


def _poll_keys_until(
    display: Display,
    deadline: float,
    key_log: list,
    clock: Clock,
    key_to_action: dict | None = None,
    latch: bool = False,
) -> tuple[object | None, bool]:
    """Poll the keyboard until ``deadline``, logging every press/release.

    Replaces a plain sleep between frames: events are time-stamped on arrival
    -- to about a millisecond, or to one refresh when the display is
    vsync-locked and :meth:`Display.idle` re-presents the frame instead of
    sleeping. In turn-based play (``key_to_action`` given) the wait ends at
    the first mapped keydown so the step happens then, not at the tick.
    ``latch`` is the real-time counterpart: the keydown is remembered but the
    wait still runs to the tick, so the frame period is what it says it is.

    :param display: the display, idled between polls.
    :param deadline: ``perf_counter`` at which to stop waiting.
    :param key_log: list receiving ``(run_time, key_name, is_down)``.
    :param clock: the run's clock for the timestamps.
    :param key_to_action: turn-based or latched: map of single key NAMES to
        env actions.
    :param latch: keep waiting after a mapped keydown and return the last one
        seen, instead of ending the wait at the first.
    :return: ``(action_or_None, user_quit)``; ``action`` is set only when
        ``key_to_action`` is given, ``user_quit`` on window close / ESC.
    """
    latched_action = None
    while True:
        for event in pygame.event.get():
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
            key_log.append((clock.run_time(), name, down))
            if down and key_to_action and name in key_to_action:
                if not latch:
                    return key_to_action[name], False
                latched_action = key_to_action[name]
        if time.perf_counter() >= deadline:
            return latched_action, False
        display.idle(deadline)


def _wait_for_char(display: Display, char: str, dummy_trigger: bool = False) -> None:
    """Block until ``char`` is typed (or briefly sleep in dummy mode).

    :param display: the display, idled between polls (see :meth:`Display.idle`).
    :param char: the unicode character that unblocks the wait, or ``"any"``
        to accept every key (a self-paced "press any key" screen).
    :param dummy_trigger: if ``True``, sleep briefly and return without waiting.
    :raises KeyboardInterrupt: on window close or ESC.
    """
    if dummy_trigger:
        time.sleep(0.05)
        return
    while True:
        for event in pygame.event.get():
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


def _warn_self_paced(curriculum: list[dict]) -> None:
    """Say on stderr which phases let the subject decide how long the run is.

    A phase that waits for a key -- an untimed ``message``, a ``survey`` -- adds
    an unknown number of seconds to every onset after it and to the run's own
    length, so the scan cannot be stopped at a planned volume count. Above the
    curriculum's ``trigger`` phase that is exactly what is wanted and nothing is
    said; below it, it is usually an oversight, and this is a warning rather than
    a refusal because a behavioural run is entitled to be self-paced.

    :param curriculum: the run's phases.
    """
    at = [i for i, p in enumerate(curriculum) if p["type"] == "trigger"]
    start = at[0] + 1 if at else 0
    loose = [f"phase {i} ({p['type']})" for i, p in enumerate(curriculum)
             if i >= start and (p["type"] == "survey"
                                or (p["type"] == "message" and p.get("duration") is None))]
    if loose:
        print(f"curriculum: {', '.join(loose)} wait for a key press inside the run, so this "
              "run's length and every onset after them depend on the subject: put a trigger "
              "phase below them, or give the message a duration", file=sys.stderr)


def _wait_for_duration(display: Display, duration: float) -> None:
    """Block for ``duration`` seconds.

    :param display: the display, idled between polls (see :meth:`Display.idle`).
    :param duration: seconds to wait.
    :raises KeyboardInterrupt: on window close or ESC.
    """
    end = time.perf_counter() + duration
    while time.perf_counter() < end:
        if _check_quit():
            raise KeyboardInterrupt
        display.idle(end, poll=0.005)


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
        label: str | None = None,
    ) -> None:
        """Set up clock, logger, and phase dispatch for one subject.

        :param subject: subject identifier used in log paths / manifest.
        :param curriculum: ordered list of phase dicts (``type``, timings, …).
        :param display: shared pygame display used by all phases.
        :param outdir: the run's folder, for its manifest and game npz files.
        :param audio: shared audio output used by all phases; one is created if
            omitted, and stays silent unless an adapter returns sound.
        :param triggers: shared trigger output (start sync + codes; see
            :mod:`fmri_gym.triggers`), built by the caller like the display
            and the audio. ``None`` = the fMRI default: wait for ``=``, send
            no trigger codes.
        :param dummy_trigger: if ``True``, skip real experimenter/scanner waits.
        :param label: the run's BIDS label (:func:`fmri_gym.bids.run_label`),
            which names its events file; ``from_config`` passes the one it
            worked out.
        """
        self.subject = subject
        self.curriculum = curriculum
        self.display = display
        self.audio = audio or Audio()
        self.dummy_trigger = dummy_trigger
        self.clock = Clock()
        self.logger = Logger(outdir, subject, curriculum, self.clock, label=label)
        self.logger.set_extra("display", display.describe())
        self.logger.set_extra("dummy_trigger", dummy_trigger)
        self.logger.set_extra("audio", self.audio.describe())
        self.outdir = outdir
        self.triggers = triggers or Triggers.from_config(None)
        self.sync = self.triggers.sync

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
        width, height = (int(x) for x in args.size.lower().split("x"))
        out = bids.run_output(args.data_root, args.subject, args.curriculum,
                              args.ses, args.run)
        if out.attempt > 1:
            print(f"{out.label} already has data: this is attempt {out.attempt} at it, and "
                  "writes beside the others", file=sys.stderr)
        print(f"output: {out.folder}", file=sys.stderr)
        seeds = bids.fold_seeds(curriculum, out.label)
        check_monitor(args.monitor)
        fold_cli_options(curriculum, args)
        _warn_self_paced(curriculum)

        triggers = Triggers.from_config(config.get("triggers"))
        print(f"triggers: {triggers.status()}", file=sys.stderr)
        if args.dummy_trigger:
            print("triggers: --dummy-trigger: the experimenter and scanner waits are skipped; "
                  "this is a test run, not a session", file=sys.stderr)
        audio = Audio(enabled=not args.no_audio)
        print(f"audio: {audio.status()}", file=sys.stderr)
        display = Display(size=(width, height), fullscreen=args.fullscreen,
                          vsync=not args.no_vsync, monitor=args.monitor)
        run = cls(args.subject, curriculum, display, out.folder, audio=audio,
                  triggers=triggers, dummy_trigger=args.dummy_trigger, label=out.label)
        run.logger.set_extra("run", {"label": out.label, "attempt": out.attempt})
        run.logger.set_extra("seeds", seeds)
        run.logger.set_extra("versions", {  # the banner fmri_gym hides said these
            "pygame": pygame.version.ver, "sdl": ".".join(map(str, pygame.get_sdl_version()))})
        return run

    def close(self) -> None:
        """Close the window, the audio output and the trigger line."""
        self.display.close()
        self.audio.close()
        self.triggers.close()

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
            self.display.draw_text("Waiting for scanner...")
            _wait_for_char(self.display, self.sync.key, dummy_trigger=self.dummy_trigger)
        elif self.sync.mode == "send":
            self.display.draw_text("Starting the recording...")
            self.triggers.lifecycle("scanner_start")
            _wait_for_duration(self.display, self.sync.delay)

        self.clock.trigger()
        self.logger.set_trigger_time()
        self.triggers.lifecycle("task_start")

    def _trigger_phase(self, phase: dict, index: int) -> None:
        """Start the scan here, in the middle of the curriculum.

        A ``trigger`` phase is where t=0 goes when the phases above it must not
        be inside the run. The instruction screen is the case it exists for: it
        waits for a key, so with the trigger before it the run's length -- its
        number of volumes -- and the onset of everything in it depend on how
        long the subject took to read. Put the trigger under the instructions
        and the scan starts at a fixed distance from the first fixation instead.

        :param phase: the trigger-phase config (it has no fields).
        :param index: phase index in the curriculum (for the manifest).
        """
        self._trigger()
        self.logger.log_phase({"index": index, "type": "trigger",
                               "onset": 0.0, "offset": self.clock.run_time()})

    def _fixation(self, phase: dict, index: int) -> None:
        """Show a fixation cross for ``phase["duration"]`` seconds.

        :param phase: fixation-phase config (``duration``, default 2.0).
        :param index: phase index in the curriculum (for the manifest).
        """
        duration = phase.get("duration", 2.0)

        onset = self.clock.from_perf(self.display.draw_fixation())
        _wait_for_duration(self.display, duration)

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
                           dummy_trigger=self.dummy_trigger)
        else:
            _wait_for_duration(self.display, duration)

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
        """Append each confirmed Likert answer to ``responses``.

        :raises KeyboardInterrupt: on window close or ESC.
        """
        for q in questions:
            value = (n_points + 1) // 2
            confirmed = False
            while not confirmed:
                scale = "  ".join((f"[{i}]" if i == value else f" {i} ")
                                  for i in range(1, n_points + 1))
                self.display.draw_text(
                    f"{q}\n\nDisagree      Agree\n{scale}\n\n"
                    "(LEFT/RIGHT to rate, ENTER to confirm)")
                for event in pygame.event.get():
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
        frames: dict,
        *,
        seed: int,
        episode_id: int,
        turn_based: bool,
        latched: bool,
        dt: float,
        state_stride: int,
        block_end: float,
        play_sound: bool,
        world: Any = None,
    ) -> tuple[bool, str]:
        """Run one episode, appending frame data to ``frames``.

        :param adapter: wrapped env for reset/step/render/sound/capture, and
            optionally an ``overlay()`` returning status lines for the display
            margin, or an ``on_frame_overlay()`` returning lines to draw over
            the frame itself.
        :param frames: mutable frame-log dict; lists are appended in place.
        :param seed: RNG seed for this episode's ``reset``.
        :param world: this episode's ``(obs, info)``, when the interval before it
            already built it (:meth:`_interval`); otherwise it is built here, and
            the subject waits on the blank screen while it is.
        :param episode_id: index of this episode within the game block.
        :param turn_based: if True, advance only on mapped keydowns.
        :param latched: real-time only: let a fresh keydown win over held keys.
        :param dt: target seconds per frame (``1 / fps``).
        :param state_stride: save a full state blob every this many frames.
        :param block_end: ``perf_counter`` deadline for the game block.
        :param play_sound: pass the adapter's sound to the speakers (the
            phase's ``audio``); muting never changes what is logged.
        :return: ``(user_quit, ended)``: whether the user quit (ESC/window
            close), and what ended the episode -- ``terminated`` (the env said
            so), ``truncated`` (its own time limit), ``block_end`` (the block's
            deadline came first) or ``quit``. The caller needs the difference:
            only a real ending is followed by an inter-episode interval, and
            ``block_end`` means the block is over.
        """
        frames["episode_seeds"].append(seed)
        terminated = truncated = False
        ep_frame = 0
        key_to_action = (adapter.keyspec.key_to_action_map()
                         if turn_based or latched else None)
        key_log = frames["key_events"]

        ## Reset environment and show initial state
        obs, info = adapter.reset(seed) if world is None else world
        self.display.call_on_flip(self.triggers.episode_start)
        # Paced from the flip, so a slow reset does not become a burst of
        # catch-up frames. The reset frame's sound is not played: it is not a
        # step's, and it would start the episode's sound off its flips.
        reset_flip = self._show(adapter, play_sound=False)
        # The episode became visible here. Its own first STEPPED frame is one
        # frame later, so a regressor built from `flip_time` alone would start
        # the episode after the subject has already seen the world.
        frames["episode_onset"].append(self.clock.from_perf(reset_flip))
        next_t = reset_flip + dt

        ## Loop over frames within episode
        while not (terminated or truncated) and time.perf_counter() < block_end:
            # Wait for the frame tick (turn-based: for a mapped keydown, up to
            # the block end), polling keys as we go so presses are stamped on
            # arrival; a vsync-locked display re-presents the frame meanwhile.
            # TODO(#43): anchor the wait to the last step (t_step + dt), not to the flip.
            deadline = block_end if turn_based else next_t
            action, user_quit = _poll_keys_until(
                self.display, deadline, key_log, self.clock, key_to_action,
                latch=latched and not turn_based)
            if user_quit:
                return True, "quit"
            next_t += dt
            if turn_based and action is None:
                continue                        # block ended without a press
            if not turn_based and action is None:
                # No latched press this frame (or the phase never asked for
                # one): the action is whatever is held down at the tick.
                action = adapter.keyspec.resolve(held_key_names())

            obs, reward, terminated, truncated, info = adapter.step(action)
            t_step = self.clock.run_time()
            # Anchor a full savestate at episode start and every stride.
            save_blob = (ep_frame % state_stride == 0)
            ep_frame += 1
            fs = adapter.capture(obs, info, want_blob=save_blob)
            # The frame trigger goes out on the flip that shows this frame.
            self.display.call_on_flip(self.triggers.frame)
            flip_t = self._show(adapter, play_sound)
            # More than a frame behind (a stall): drop the debt, or it is repaid
            # as a burst of one-refresh frames. The frame of slack is what a
            # vsync-locked flip normally lands after its tick.
            # TODO(#43): why frames fall behind at all is not established.
            if not turn_based and next_t + dt < flip_t:
                late = flip_t - (next_t - dt)
                frames["pacing_reset"].append((self.clock.from_perf(flip_t), late))
                next_t = flip_t + dt

            # Prefer env_action when an adapter translates UI meta-keys into a
            # different logged action (e.g. Rush Hour select+move -> Discrete).
            if isinstance(info, dict) and "env_action" in info:
                frames["env_action"].append(info["env_action"])
            frames["action"].append(action)
            frames["reward"].append(reward)
            frames["terminated"].append(bool(terminated))
            frames["truncated"].append(bool(truncated))
            frames["episode_id"].append(episode_id)
            frames["run_time"].append(t_step)
            frames["flip_time"].append(self.clock.from_perf(flip_t))
            frames["audio_chunk"].append(self.audio.last_chunk)
            frames["wall_time"].append(self.clock.wall_time())
            frames["state_blob"].append(fs.blob)
            if self.triggers.enabled:
                frames["trigger"].append(self.triggers.last_frame)
            for k, v in fs.variables.items():
                frames["variables"][k].append(v)
        if terminated:
            return False, "terminated"
        return False, "truncated" if truncated else "block_end"

    def _wait(self, deadline: float, key_log: list) -> bool:
        """Hold what is on screen until ``deadline``, logging presses.

        :param deadline: ``perf_counter`` to wait until.
        :param key_log: the block's ``key_events`` list, appended to in place.
        :return: ``True`` if the user quit (ESC / window close).
        """
        return _poll_keys_until(self.display, deadline, key_log, self.clock)[1]

    def _interval(self, key_log: list, *, hold: float, iti: float, cue: float,
                  deadline: float, prepare: Callable[[], Any] | None = None,
                  ) -> tuple[dict, bool, Any]:
        """The interval between two episodes: hold the last frame, then fixation.

        Three things a scanner block needs that consecutive flips do not give
        it. The hold leaves the ending frame up, so the subject reads what
        happened instead of the next world appearing on the frame they died on.
        The blank that follows is the inter-episode interval, and ``iti`` was
        drawn per interval by the caller: at a fixed length every episode onset
        would sit at the same phase of every TR, and the response to one
        episode could not be separated from the response to the next. The cue
        turns the marker red for the last ``cue`` seconds, so the next episode
        does not begin unannounced. (crafter-for-brain-scan v0.33 precedent:
        ``--death-hold-s`` 1.2, ``ITI_RANGE = (4.0, 12.0)``, a red fixation
        ``--cue-s`` before each round.)

        The next episode's world is built here too, in the blank, because
        building one is slow (crafter generates terrain in about 2.4 s) and a
        subject who waited for it after the cue would be waiting through a red
        cross that had already promised the episode. In the blank instead, the
        wait is spent rather than added: ``iti`` and ``cue`` are the lengths they
        say, and the episode starts on the flip the cue counted down to.

        Nothing is stepped here, so no frame is logged; presses still are, and
        ESC still ends the run.

        :param key_log: the block's ``key_events`` list, appended to in place.
        :param hold: seconds the episode's last frame stays up.
        :param iti: seconds of fixation after it, already drawn.
        :param cue: seconds of that fixation the marker is red for.
        :param deadline: ``perf_counter`` the block ends at. The interval is cut
            there rather than running the block past the length its config
            promised, and says in its record that it was cut.
        :param prepare: builds the next episode's world, called inside the blank
            (see below). Its return value is handed back to the caller.
        :return: the interval's manifest record, whether the user quit, and what
            ``prepare`` built (``None`` if it was not called).
        """
        rec: dict = {"hold": hold, "drawn": iti}
        if hold > 0:
            # No flip of its own: the frame is already up, and this is only not
            # replacing it yet. So the hold began at that frame's flip.
            rec["hold_onset"] = self.clock.from_perf(self.display.last_flip)
            if self._wait(min(time.perf_counter() + hold, deadline), key_log):
                return rec, True, None
        end = time.perf_counter() + iti
        rec["clamped"] = end > deadline
        end = min(end, deadline)
        rec["onset"] = self.clock.from_perf(self.display.draw_fixation())
        # A clamped interval is the block's last: its end is the block's, so the
        # loop stops there and no episode follows it. Nothing to announce with a
        # cue, and nothing to build a world for.
        world = None
        if prepare is not None and not rec["clamped"]:
            # Blocking, and nothing is polled while it runs: a press inside it is
            # stamped when polling resumes, which is why it goes in the blank and
            # not in a phase the subject is doing anything in.
            started = time.perf_counter()
            world = prepare()
            rec["built"] = time.perf_counter() - started
            if time.perf_counter() > end:
                print(f"iti: the next world took {rec['built']:.2f} s to build, more than "
                      f"the {iti:.2f} s blank it had to fit in; the blank ran that long "
                      "instead. Raise the low end of `iti` above it.", file=sys.stderr)
        red_at = end - cue if cue > 0 and not rec["clamped"] else None
        if self._wait(end if red_at is None else red_at, key_log):
            return rec, True, world
        if red_at is not None:
            rec["cue_onset"] = self.clock.from_perf(self.display.draw_fixation(red=True))
            if self._wait(end, key_log):
                return rec, True, world
        rec["offset"] = self.clock.run_time()
        return rec, False, world

    def _show(self, adapter: EnvAdapter, play_sound: bool) -> float:
        """Flip the adapter's frame, then queue its sound against that flip.

        An adapter may also offer ``overlay()`` (status lines for the letterbox
        margin) and ``on_frame_overlay()`` (lines drawn over the frame itself);
        both are optional and draw nothing when absent or returning ``None``.

        :param adapter: the env whose ``render`` / ``sound`` to present.
        :param play_sound: pass the sound to the speakers.
        :return: ``perf_counter`` of the flip.
        """
        overlay = getattr(adapter, "overlay", lambda: None)
        on_frame = getattr(adapter, "on_frame_overlay", lambda: None)
        flip_t = self.display.draw_frame(adapter.render(), overlay(), on_frame())
        if play_sound:
            self.audio.play(adapter.sound(), flip_t)
        return flip_t

    def _game(self, phase: dict, index: int) -> None:
        """Run a game block (one or more episodes) and save frame-level data.

        Creates the env via the phase's backend adapter, plays until duration /
        episode count / quit, then writes an npz and a manifest phase entry.

        :param phase: game-phase config (``backend``, ``game``, ``mode``,
            ``duration`` / ``n_episodes``, ``fps``, ``seed``, ``state_stride``,
            ``turn_based``, ``latched_keys``, optional ``keys`` overrides, …).
        :param index: phase index in the curriculum (for the manifest).
        :raises KeyboardInterrupt: if the subject quits mid-block.
        """
        ## Config
        backend = phase.get("backend", "gym")
        mode = phase.get("mode", "duration")
        duration = phase.get("duration", 30.0)
        n_episodes = phase.get("n_episodes", 1)
        base_seed = phase.get("seed", 1000 + index)
        # Save a full savestate every `state_stride` frames (and always at each
        # episode's first frame, the replay anchor). 1 = every frame (default);
        # larger values trade savestate density for disk -- important for retro,
        # whose states are ~1 MB/frame. Between anchors, frames are still
        # reconstructable by restoring the last anchor and replaying actions.
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
        play_sound = phase.get("audio", True)
        if not isinstance(play_sound, bool):
            raise ValueError(f'game phase {index}: "audio" must be true or false, '
                             f"got {play_sound!r}")
        # What goes between two episodes (see _interval). All three default to
        # 0, which is the block every version before them played: one episode's
        # last frame and the next one's first are consecutive flips.
        iti = seconds_range(phase.get("iti", 0), "iti")
        iti_cue = float(phase.get("iti_cue", 0))
        end_hold = float(phase.get("end_hold", 0))
        has_interval = end_hold > 0 or iti[1] > 0
        # One stream per block, seeded from the block's own seed, so a
        # re-acquisition of a run draws the intervals of the run it replaces.
        # The rig keeps one stream over a whole sitting (core.py:1275); a run
        # here is one process, so a block is as far as a stream can reach.
        rng = random.Random(base_seed ^ 0x1717)
        # Every game phase states its own fps (validate_config refuses one that
        # does not): what the block plays at is the config's business, not a
        # default that changes with the engine underneath it.
        fps = phase["fps"]
        dt = 1.0 / fps

        # Some backends (nle, browser games) take several seconds to start;
        # show a Loading screen so the previous fixation "+" doesn't freeze.
        self.display.draw_text(
            f"Loading {phase.get('text') or phase.get('game', 'game')} …")
        adapter = get_adapter(backend, phase)
        speed = {} if turn_based else self._speed(adapter, fps, index, phase["game"])

        ## Frame logging
        frames = defaultdict(list)
        frames["variables"] = defaultdict(list)  # varname -> list, filled lazily

        ## Init loop over episodes
        locked = self.display.vsync and self.display.refresh_rate
        flip_period = 1 / self.display.refresh_rate if locked else None
        self.audio.start(frame_period=None if turn_based else dt, flip_period=flip_period)
        onset = self.clock.run_time()
        block_end = time.perf_counter() + cap
        episode_id = 0
        user_quit = False
        episodes: list[dict] = []
        intervals: list[dict] = []
        # The world the interval before this episode built, if there was one.
        world = None

        ## Loop over episodes within game block
        while not user_quit and time.perf_counter() < block_end:
            ## Run one episode
            first = len(frames["action"])
            user_quit, ended = self._episode(
                adapter, frames, world=world,
                seed=base_seed + episode_id, episode_id=episode_id,
                turn_based=turn_based, latched=latched, dt=dt,
                state_stride=state_stride,
                block_end=block_end, play_sound=play_sound)
            world = None                     # that world is played; a new one is built below
            # An episode's last sounds are still queued when it ends; drop them
            # so they do not play over the next episode or the next fixation.
            self.audio.stop()
            episodes.append({
                "id": episode_id, "seed": base_seed + episode_id, "ended": ended,
                "onset": frames["episode_onset"][episode_id],
                "offset": self.clock.run_time(),
                "n_frames": len(frames["action"]) - first})
            episode_id += 1
            if mode == "episode" and episode_id >= n_episodes:
                break
            ## The interval before the next one, if a next one is coming
            if user_quit or ended == "block_end" or not has_interval:
                continue
            # episode_id is the next episode's now, so this is the seed it will
            # be played with and log; the interval builds that world in its blank.
            next_seed = base_seed + episode_id
            rec, user_quit, world = self._interval(
                frames["key_events"], hold=end_hold, iti=rng.uniform(*iti),
                cue=iti_cue, deadline=block_end,
                prepare=lambda: adapter.reset(next_seed))
            intervals.append({"after_episode": episode_id - 1, **rec})

        self.triggers.block_end()
        extra = getattr(adapter, "block_extra", lambda: None)()
        audio_log = self.audio.block_log(frames["audio_chunk"])
        if audio_log:
            frames["audio_onset"] = self.clock.from_perf(audio_log.pop("audio_onset"))
            extra = {**(extra if extra is not None else {}), **audio_log}
        adapter.close()
        # Some gym envs (classic-control) call pygame.display.quit() on close(),
        # which tears down our shared window; rebuild it if so.
        self.display.ensure()
        path = self.logger.save_game_block(index, backend, phase["game"],
                                           frames, extra=extra)
        self.logger.log_phase({
            "index": index, "type": "game", "backend": backend,
            "game": phase["game"], "mode": mode,
            "onset": onset, "offset": self.clock.run_time(),
            "n_episodes": episode_id, "n_frames": len(frames["action"]),
            "n_pacing_resets": len(frames["pacing_reset"]),
            "total_reward": sum(float(r) for r in frames["reward"]),
            # Where each episode and each interval sat in the run. Here and not
            # in the npz, which is one row per frame and has no row for an
            # interval: what the block was made of is a property of the block.
            "episodes": episodes, "intervals": intervals,
            "data_file": path.split("/")[-1], **speed,
        })
        if user_quit:
            raise KeyboardInterrupt

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
        """Play the full curriculum: trigger wait, then each phase in order.

        Always writes the run's manifest in ``finally``, including after an
        interrupt (partial data).

        :return: ``True`` if the curriculum played to its end, ``False`` if it
            was quit, so a caller can report a run that stopped early.
        """
        completed = False
        handlers = {"fixation": self._fixation, "message": self._message,
                    "game": self._game, "trigger": self._trigger_phase,
                    "survey": self._survey}
        try:
            # A curriculum that says where its scan starts starts it there; one
            # that does not starts it above the first phase, as every
            # curriculum did before there was a way to say otherwise.
            if not any(p["type"] == "trigger" for p in self.curriculum):
                self._trigger()

            for index, phase in enumerate(self.curriculum):
                handler = handlers.get(phase["type"])
                if handler is None:
                    raise ValueError(f"unknown phase type: {phase['type']!r}")
                handler(phase, index)

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
            # After the manifest, and it is derived from it: a raising events
            # writer must not be able to cost the run its record.
            events_path = self.logger.save_events()
            print(f"Saved run to: {self.outdir}")
            print(f"Manifest: {manifest_path}")
            if events_path:
                print(f"Events: {events_path}")
        return completed
