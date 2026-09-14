"""The engine-agnostic experiment loop.

Everything here is independent of which game engine is used: trigger wait,
clock anchoring, the curriculum of phases (fixation / message / game / survey),
inter-block intervals, timing/pacing, and logging. All engine-specific access
goes through an EnvAdapter, so this file never imports ale_py / stable_retro
and never touches env.unwrapped.

Recording-device concerns (waiting for or sending the scanner start, marker
codes for MEG/EEG) go through :mod:`fmri_gym.triggers`; with no ``triggers``
config the loop behaves as the fMRI default and sends nothing.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from dataclasses import asdict
from typing import TYPE_CHECKING, Union

import pygame

from .adapters import get_adapter
from .display import Display
from .audio import SoundDeviceGameBlockStream
from .keys import held_key_names, key_name
from .logging import Logger
from .triggers import Markers, SyncSettings, TriggerError

if TYPE_CHECKING:
    from .adapters.base import EnvAdapter

TRIGGER_KEY = "="
EXPERIMENTER_KEY = " "


class Clock:
    """Anchored at the scanner trigger; gives session + wall-clock time."""

    def __init__(self) -> None:
        """Create an untriggered clock (``t0_*`` are ``None`` until :meth:`trigger`)."""
        self.t0_perf = None
        self.t0_epoch = None

    def trigger(self) -> None:
        """Anchor the clock at the current time (call on scanner trigger)."""
        self.t0_perf = time.perf_counter()
        self.t0_epoch = time.time()

    def session_time(self) -> float:
        """Seconds since the scanner trigger (``perf_counter`` based).

        :return: elapsed session time in seconds.
        """
        return time.perf_counter() - self.t0_perf

    def from_perf(self, t_perf: float) -> float:
        """Convert a ``perf_counter`` stamp (e.g. a flip time) to session time.

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
    :param key_log: list receiving ``(session_time, key_name, is_down)``.
    :param clock: the session clock for the timestamps.
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
            key_log.append((clock.session_time(), name, down))
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


def _join_multiline_text(text: Union[str, list, tuple]) -> str:
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


class Session:
    """Runs a curriculum for one subject, dispatching phases to handlers."""

    def __init__(
        self,
        subject: str,
        curriculum: list[dict],
        display: Display,
        outdir: str,
        dummy_trigger: bool = False,
        triggers: dict | None = None,
    ) -> None:
        """Set up clock, logger, and phase dispatch for one subject.

        :param subject: subject identifier used in log paths / manifest.
        :param curriculum: ordered list of phase dicts (``type``, timings, …).
        :param display: shared pygame display used by all phases.
        :param outdir: directory for the session manifest and game npz files.
        :param dummy_trigger: if ``True``, skip real experimenter/scanner waits.
        :param triggers: optional ``triggers`` config section (``sync`` and
            ``markers``; see :mod:`fmri_gym.triggers`). ``None`` = fMRI
            default: wait for ``=``, send no markers.
        :raises TriggerError: if a marker backend cannot be opened, or
            ``sync.mode`` is ``send`` with no backend to send on.
        :raises ValueError: on an invalid ``triggers`` section.
        """
        self.subject = subject
        self.curriculum = curriculum
        self.display = display
        self.dummy_trigger = dummy_trigger
        self.clock = Clock()
        self.logger = Logger(outdir, subject, curriculum, self.clock)
        self.logger.set_extra("display", display.describe())
        self.outdir = outdir
        triggers = triggers or {}
        self.sync = SyncSettings.from_dict(triggers.get("sync"))
        self.markers = Markers.from_config(triggers.get("markers"), self.clock)
        if self.sync.mode == "send" and not self.markers.enabled:
            raise TriggerError('triggers: sync.mode "send" sends scanner_start on the marker '
                               'line, so triggers.markers.backend must not be "null"')

    def _trigger(self) -> None:
        """Wait for experimenter ready, sync with the scanner, start the clock.

        Draws the readiness screen, then either waits for the trigger key,
        sends the start code (``sync.mode``), or neither; then calls
        :meth:`Clock.trigger`, records the trigger time on the logger and
        sends ``task_start``.
        """
        self.display.draw_text(
            "Please keep your head as still as possible.\n\n"
            "(experimenter: press SPACE when ready)")
        _wait_for_char(self.display, EXPERIMENTER_KEY, dummy_trigger=self.dummy_trigger)
        if self.sync.mode == "wait":
            self.display.draw_text("Waiting for scanner...")
            _wait_for_char(self.display, self.sync.key, dummy_trigger=self.dummy_trigger)
        elif self.sync.mode == "send":
            self.display.draw_text("Starting the recording...")
            self.markers.lifecycle("scanner_start")
            _wait_for_duration(self.display, self.sync.delay)

        self.clock.trigger()
        self.logger.set_trigger_time()
        self.markers.lifecycle("task_start")

    def _fixation(self, phase: dict, index: int) -> None:
        """Show a fixation cross for ``phase["duration"]`` seconds.

        :param phase: fixation-phase config (``duration``, default 2.0).
        :param index: phase index in the curriculum (for the manifest).
        """
        duration = phase.get("duration", 2.0)

        onset = self.clock.from_perf(self.display.draw_fixation())
        _wait_for_duration(self.display, duration)

        self.logger.log_phase({"index": index, "type": "fixation",
                               "onset": onset, "offset": self.clock.session_time()})

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
                               "onset": onset, "offset": self.clock.session_time()})

    def _survey(self, phase: dict, index: int) -> None:
        """Run a Likert-style survey and log each confirmed response.

        :param phase: survey-phase config (``questions``, optional ``n_points``).
        :param index: phase index in the curriculum (for the manifest).
        :raises KeyboardInterrupt: on window close or ESC.
        """
        questions = phase.get("questions", [])
        n_points = phase.get("n_points", 7)
        onset = self.clock.session_time()

        responses = []
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
                              "session_time": self.clock.session_time()})

        self.logger.log_phase({"index": index, "type": "survey",
                               "onset": onset, "offset": self.clock.session_time(),
                               "responses": responses})

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
    ) -> bool:
        """Run one episode, appending frame data to ``frames``.

        :param adapter: wrapped env for reset/step/render/capture.
        :param frames: mutable frame-log dict; lists are appended in place.
        :param seed: RNG seed for this episode's ``reset``.
        :param episode_id: index of this episode within the game block.
        :param turn_based: if True, advance only on mapped keydowns.
        :param dt: target seconds per frame (``1 / fps``).
        :param state_stride: save a full state blob every this many frames.
        :param block_end: ``perf_counter`` deadline for the game block.
        :return: ``True`` if the user quit (ESC/window close), else ``False``.
        """
        frames["episode_seeds"].append(seed)
        terminated = truncated = False
        ep_frame = 0
        next_t = time.perf_counter()
        key_to_action = adapter.keyspec.key_to_action_map() if turn_based else None
        key_log = frames["key_events"]

        ## Reset environment and show initial state
        obs, info = adapter.reset(seed)
        if adapter.has_audio:
            first_audio_buffer = adapter.get_audio_buffer()
            self.audio_stream = SoundDeviceGameBlockStream(
                adapter.get_audio_sampling_rate(),
                first_audio_buffer.shape[0],
                first_audio_buffer.shape[1],
                dtype=first_audio_buffer.dtype,
            )
            self.audio_stream.play()
        self.display.call_on_flip(self.markers.episode_start)
        self.display.draw_frame(adapter.render())

        ## Loop over frames within episode
        while not (terminated or truncated) and time.perf_counter() < block_end:
            # Wait for the frame tick (turn-based: for a mapped keydown, up to
            # the block end), polling keys as we go so presses are stamped on
            # arrival; a vsync-locked display re-presents the frame meanwhile.
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
            t_step = self.clock.session_time()
            if adapter.has_audio:
                self.audio_stream.put(adapter.get_audio_buffer())
            # Anchor a full savestate at episode start and every stride.
            save_blob = (ep_frame % state_stride == 0)
            ep_frame += 1
            fs = adapter.capture(obs, info, want_blob=save_blob)
            # The frame marker goes out on the flip that shows this frame.
            self.display.call_on_flip(self.markers.frame)
            flip_t = self.display.draw_frame(adapter.render())

            # Prefer env_action when an adapter translates UI meta-keys into a
            # different logged action (e.g. Rush Hour select+move -> Discrete).
            if isinstance(info, dict) and "env_action" in info:
                frames["env_action"].append(info["env_action"])
            frames["action"].append(action)
            frames["reward"].append(reward)
            frames["terminated"].append(bool(terminated))
            frames["truncated"].append(bool(truncated))
            frames["episode_id"].append(episode_id)
            frames["session_time"].append(t_step)
            frames["flip_time"].append(self.clock.from_perf(flip_t))
            frames["wall_time"].append(self.clock.wall_time())
            frames["state_blob"].append(fs.blob)
            if self.markers.enabled:
                frames["marker"].append(self.markers.last_frame)
            for k, v in fs.variables.items():
                frames["variables"][k].append(v)
        if adapter.has_audio:
            self.audio_stream.stop()
        return False

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
        fps = phase.get("fps", 30)
        base_seed = phase.get("seed", 1000 + index)
        # Save a full savestate every `state_stride` frames (and always at each
        # episode's first frame, the replay anchor). 1 = every frame (default);
        # larger values trade savestate density for disk -- important for retro,
        # whose states are ~1 MB/frame. Between anchors, frames are still
        # reconstructable by restoring the last anchor and replaying actions.
        state_stride = max(1, int(phase.get("state_stride", 1)))
        dt = 1.0 / fps
        cap = duration if mode == "duration" else phase.get("max_duration", 300.0)
        # Turn-based games (grid worlds: FrozenLake, CliffWalking, Taxi, ...) must
        # advance ONE step per deliberate key PRESS, not once per frame. In a
        # real-time loop they'd auto-step every frame with the noop action (which
        # for e.g. FrozenLake is action 0 = LEFT), so the agent "moves on its own"
        # and a single held key fires many times. turn_based fixes both.
        turn_based = bool(phase.get("turn_based", False))

        # Some backends (nle, browser games) take several seconds to start;
        # show a Loading screen so the previous fixation "+" doesn't freeze.
        self.display.draw_text(
            f"Loading {phase.get('text') or phase.get('game', 'game')} …")
        adapter = get_adapter(backend, phase)

        ## Frame logging
        frames = defaultdict(list)
        frames["variables"] = defaultdict(list)  # varname -> list, filled lazily

        ## Init loop over episodes
        onset = self.clock.session_time()
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
                block_end=block_end)
            episode_id += 1
            if mode == "episode" and episode_id >= n_episodes:
                break

        self.markers.block_end()
        extra = getattr(adapter, "block_extra", lambda: None)()
        adapter.close()
        # Some gym envs (classic-control) call pygame.display.quit() on close(),
        # which tears down our shared window; rebuild it if so.
        self.display.ensure()
        path = self.logger.save_game_block(index, backend, phase["game"],
                                           frames, extra=extra)
        self.logger.log_phase({
            "index": index, "type": "game", "backend": backend,
            "game": phase["game"], "mode": mode,
            "onset": onset, "offset": self.clock.session_time(),
            "n_episodes": episode_id, "n_frames": len(frames["action"]),
            "total_reward": sum(float(r) for r in frames["reward"]),
            "data_file": path.split("/")[-1],
        })
        if user_quit:
            raise KeyboardInterrupt

    def run(self) -> bool:
        """Run the full curriculum: trigger wait, then each phase in order.

        Always writes the session manifest in ``finally``, including after an
        interrupt (partial data).

        :return: ``True`` if the curriculum completed, ``False`` if the user
            quit (ESC / window close), so a caller playing several runs in a
            row can stop there.
        """
        handlers = {"fixation": self._fixation, "message": self._message,
                    "game": self._game, "survey": self._survey}
        completed = False
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
                self.markers.lifecycle("task_stop")
            self.logger.set_extra("triggers", {"sync": asdict(self.sync),
                                               "markers": self.markers.describe()})
            self.markers.close()
            manifest_path = self.logger.save_manifest()
            print(f"Saved session to: {self.outdir}")
            print(f"Manifest: {manifest_path}")
        return completed
