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

import sys
import time
from collections import defaultdict
from functools import partial
from typing import TYPE_CHECKING, Any

import pygame

from . import bids
from .adapters import get_adapter
from .audio import Audio
from .config import fold_cli_options
from .display import Display, check_monitor
from .keys import held_key_names, key_name
from .logging import Logger
from .triggers import Triggers

if TYPE_CHECKING:
    from .adapters.base import EnvAdapter

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
) -> tuple[object | None, bool]:
    """Poll the keyboard until ``deadline``, logging every press/release.

    Replaces a plain sleep between frames: events are time-stamped on arrival
    -- to about a millisecond, or to one refresh when the display is
    vsync-locked and :meth:`Display.idle` re-presents the frame instead of
    sleeping. In turn-based play (``key_to_action`` given) the wait ends at
    the first mapped keydown so the step happens then, not at the tick.

    :param display: the display, idled between polls.
    :param deadline: ``perf_counter`` at which to stop waiting.
    :param key_log: list receiving ``(run_time, key_name, is_down)``.
    :param clock: the run's clock for the timestamps.
    :param key_to_action: turn-based: map of single key NAMES to env actions.
    :return: ``(action_or_None, user_quit)``; ``action`` is set only in
        turn-based play, ``user_quit`` on window close / ESC.
    """
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
                return key_to_action[name], False
        if time.perf_counter() >= deadline:
            return None, False
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
        """
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
                  triggers=triggers, dummy_trigger=args.dummy_trigger)
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
        dt: float,
        state_stride: int,
        block_end: float,
        play_sound: bool,
    ) -> bool:
        """Run one episode, appending frame data to ``frames``.

        :param adapter: wrapped env for reset/step/render/sound/capture.
        :param frames: mutable frame-log dict; lists are appended in place.
        :param seed: RNG seed for this episode's ``reset``.
        :param episode_id: index of this episode within the game block.
        :param turn_based: if True, advance only on mapped keydowns.
        :param dt: target seconds per frame (``1 / fps``).
        :param state_stride: save a full state blob every this many frames.
        :param block_end: ``perf_counter`` deadline for the game block.
        :param play_sound: pass the adapter's sound to the speakers (the
            phase's ``audio``); muting never changes what is logged.
        :return: ``True`` if the user quit (ESC/window close), else ``False``.
        """
        frames["episode_seeds"].append(seed)
        terminated = truncated = False
        ep_frame = 0
        key_to_action = adapter.keyspec.key_to_action_map() if turn_based else None
        key_log = frames["key_events"]

        ## Reset environment and show initial state
        obs, info = adapter.reset(seed)
        self.display.call_on_flip(self.triggers.episode_start)
        # Paced from the flip, so a slow reset does not become a burst of
        # catch-up frames. The reset frame's sound is not played: it is not a
        # step's, and it would start the episode's sound off its flips.
        next_t = self._show(adapter, play_sound=False) + dt

        ## Loop over frames within episode
        while not (terminated or truncated) and time.perf_counter() < block_end:
            # Wait for the frame tick (turn-based: for a mapped keydown, up to
            # the block end), polling keys as we go so presses are stamped on
            # arrival; a vsync-locked display re-presents the frame meanwhile.
            # TODO(#43): anchor the wait to the last step (t_step + dt), not to the flip.
            deadline = block_end if turn_based else next_t
            action, user_quit = _poll_keys_until(
                self.display, deadline, key_log, self.clock, key_to_action)
            if user_quit:
                return True
            next_t += dt
            if turn_based and action is None:
                continue                        # block ended without a press
            if not turn_based:
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
        return False

    def _show(self, adapter: EnvAdapter, play_sound: bool) -> float:
        """Flip the adapter's frame, then queue its sound against that flip.

        :param adapter: the env whose ``render`` / ``sound`` to present.
        :param play_sound: pass the sound to the speakers.
        :return: ``perf_counter`` of the flip.
        """
        flip_t = self.display.draw_frame(adapter.render())
        if play_sound:
            self.audio.play(adapter.sound(), flip_t)
        return flip_t

    def _game(self, phase: dict, index: int) -> None:
        """Run a game block (one or more episodes) and save frame-level data.

        Creates the env via the phase's backend adapter, plays until duration /
        episode count / quit, then writes an npz and a manifest phase entry.

        :param phase: game-phase config (``backend``, ``game``, ``mode``,
            ``duration`` / ``n_episodes``, ``fps``, ``seed``, ``state_stride``,
            ``turn_based``, optional ``keys`` overrides, …).
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
        play_sound = phase.get("audio", True)
        if not isinstance(play_sound, bool):
            raise ValueError(f'game phase {index}: "audio" must be true or false, '
                             f"got {play_sound!r}")
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

        ## Loop over episodes within game block
        while not user_quit and time.perf_counter() < block_end:
            ## Run one episode
            user_quit = self._episode(
                adapter, frames,
                seed=base_seed + episode_id, episode_id=episode_id,
                turn_based=turn_based, dt=dt, state_stride=state_stride,
                block_end=block_end, play_sound=play_sound)
            # An episode's last sounds are still queued when it ends; drop them
            # so they do not play over the next episode or the next fixation.
            self.audio.stop()
            episode_id += 1
            if mode == "episode" and episode_id >= n_episodes:
                break

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
                    "game": self._game, "survey": self._survey}
        if any(p["type"].startswith("check_") for p in self.curriculum):
            from . import checks  # rig-check phases: measured with this run's display and lines
            handlers.update({kind: partial(checks.run_check, self)
                             for kind in checks.CHECK_TYPES})
        try:
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
            print(f"Saved run to: {self.outdir}")
            print(f"Manifest: {manifest_path}")
        return completed
