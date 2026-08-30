"""The engine-agnostic experiment loop.

Everything here is independent of which game engine is used: trigger wait,
clock anchoring, the curriculum of phases (fixation / message / game / survey),
inter-block intervals, timing/pacing, and logging. All engine-specific access
goes through an EnvAdapter, so this file never imports ale_py / stable_retro
and never touches env.unwrapped.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Union

import pygame

from .display import Display
from .keys import held_key_names, key_name
from .logging import Logger

if TYPE_CHECKING:
    from .adapters.base import EnvAdapter, KeySpec

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


def _get_action(key_to_action: dict) -> tuple[object | None, bool]:
    """Drain events; return the mapped action for a fresh keydown, if any.

    :param key_to_action: map of single key NAMES to env actions.
    :return: ``(action_or_None, user_quit)`` where ``user_quit`` is ``True``
        on window close / ESC.
    """
    action = None
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            return None, True
        if event.type == pygame.KEYDOWN:
            name = key_name(event.key)
            if name in key_to_action:
                action = key_to_action[name]
    return action, False


def _wait_for_char(char: str, dummy_trigger: bool = False) -> None:
    """Block until ``char`` is typed (or briefly sleep in dummy mode).

    :param char: the unicode character that unblocks the wait.
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
                if event.unicode == char:
                    return
        time.sleep(0.005)


def _join_multiline_text(text: Union[str, list, tuple]) -> str:
    """Normalize message ``text`` to a single string.

    Accepts a plain string or a list/tuple of lines (joined with ``\\n``), so
    curriculum JSON can keep long instructions readable without ``\\n`` escapes.
    """
    if isinstance(text, (list, tuple)):
        return "\n".join("" if line is None else str(line) for line in text)
    return str(text)


def _wait_for_duration(duration: float) -> None:
    """Block for ``duration`` seconds.

    :param duration: seconds to wait.
    :raises KeyboardInterrupt: on window close or ESC.
    """
    end = time.perf_counter() + duration
    while time.perf_counter() < end:
        if _check_quit():
            raise KeyboardInterrupt
        time.sleep(0.005)


def _apply_key_overrides(keyspec: KeySpec, overrides: dict | None) -> KeySpec:
    """Merge curriculum-provided key combo overrides into a :class:`KeySpec`.

    :param keyspec: base keymap from the adapter.
    :param overrides: optional ``{"LEFT": action, "LEFT+SPACE": action}`` map
        from the curriculum; ``None`` / empty leaves ``keyspec`` unchanged.
    :return: the (possibly mutated) ``keyspec``.
    """
    if not overrides:
        return keyspec
    combos = dict(keyspec.combos)
    for combo_str, action in overrides.items():
        keys = frozenset(k.strip().upper() for k in combo_str.split("+"))
        combos[keys] = action
    keyspec.combos = combos
    return keyspec


def _get_key_to_action_map(keyspec: KeySpec) -> dict:
    """Map single pressed KEY names to actions (for turn-based play).

    Resolving each key on its own (rather than reading ``combos`` directly)
    keeps the action in whatever shape the keymap flavour produces, e.g. a
    button vector for a :class:`~.adapters.base.MultiKeySpec`.

    :param keyspec: keymap whose length-1 combos become the press map.
    :return: ``{key_name: action}`` for single-key combos only.
    """
    return {next(iter(ks)): keyspec.resolve(ks) for ks in keyspec.combos
            if len(ks) == 1}


class Session:
    """Runs a curriculum for one subject, dispatching phases to handlers."""

    def __init__(
        self,
        subject: str,
        curriculum: list[dict],
        adapters: dict[str, EnvAdapter],
        display: Display,
        outdir: str,
        dummy_trigger: bool = False,
    ) -> None:
        """Set up clock, logger, and phase dispatch for one subject.

        :param subject: subject identifier used in log paths / manifest.
        :param curriculum: ordered list of phase dicts (``type``, timings, …).
        :param adapters: ``{backend_name: EnvAdapter}`` for game phases.
        :param display: shared pygame display used by all phases.
        :param outdir: directory for the session manifest and game npz files.
        :param dummy_trigger: if ``True``, skip real experimenter/scanner waits.
        """
        self.subject = subject
        self.curriculum = curriculum
        self.adapters = adapters              # {backend_name: EnvAdapter}
        self.display = display
        self.dummy_trigger = dummy_trigger
        self.clock = Clock()
        self.logger = Logger(outdir, subject, curriculum, self.clock)
        self.outdir = outdir

    def _trigger(self) -> None:
        """Wait for experimenter ready + scanner trigger, then start the clock.

        Draws readiness / waiting screens, then calls :meth:`Clock.trigger` and
        records the trigger time on the logger.
        """
        self.display.draw_text(
            "Please keep your head as still as possible.\n\n"
            "(experimenter: press SPACE when ready)")
        _wait_for_char(EXPERIMENTER_KEY, dummy_trigger=self.dummy_trigger)
        self.display.draw_text("Waiting for scanner...")
        _wait_for_char(TRIGGER_KEY, dummy_trigger=self.dummy_trigger)
        
        self.clock.trigger()
        self.logger.set_trigger_time()

    def _fixation(self, phase: dict, index: int) -> None:
        """Show a fixation cross for ``phase["duration"]`` seconds.

        :param phase: fixation-phase config (``duration``, default 2.0).
        :param index: phase index in the curriculum (for the manifest).
        """
        duration = phase.get("duration", 2.0)
        onset = self.clock.session_time()

        self.display.draw_fixation()
        _wait_for_duration(duration)

        self.logger.log_phase({"index": index, "type": "fixation",
                               "onset": onset, "offset": self.clock.session_time()})

    def _message(self, phase: dict, index: int) -> None:
        """Show on-screen text until a key press or timed duration.

        :param phase: message-phase config (``text`` as a string or list of
            lines; optional ``duration`` / ``key`` / ``align``).
        :param index: phase index in the curriculum (for the manifest).
        """
        text = _join_multiline_text(phase.get("text", ""))
        duration = phase.get("duration")
        onset = self.clock.session_time()

        self.display.draw_text(text, align=phase.get("align", "center"))
        if duration is None:
            _wait_for_char(phase.get("key", " "), dummy_trigger=self.dummy_trigger)
        else:
            _wait_for_duration(duration)

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
        env: Any,
        phase: dict,
        frames: dict,
        *,
        seed: int,
        episode_id: int,
        turn_based: bool,
        key_to_action: dict,
        keyspec: KeySpec,
        dt: float,
        state_stride: int,
        block_end: float,
    ) -> bool:
        """Run one episode, appending frame data to ``frames``.

        :param adapter: env adapter for reset/step/render/capture.
        :param env: the live environment instance.
        :param phase: game-phase config dict (passed through to ``adapter.reset``).
        :param frames: mutable frame-log dict; lists are appended in place.
        :param seed: RNG seed for this episode's ``reset``.
        :param episode_id: index of this episode within the game block.
        :param turn_based: if True, advance only on mapped keydowns.
        :param key_to_action: single-key name -> action map (turn-based only).
        :param keyspec: keymap used to resolve held keys in real-time mode.
        :param dt: target seconds per frame (``1 / fps``).
        :param state_stride: save a full state blob every this many frames.
        :param block_end: ``perf_counter`` deadline for the game block.
        :return: ``True`` if the user quit (ESC/window close), else ``False``.
        """
        frames["episode_seeds"].append(seed)
        terminated = truncated = False
        ep_frame = 0
        next_t = time.perf_counter()

        ## Reset environment and show initial state
        obs, info = adapter.reset(env, seed, phase)
        self.display.draw_frame(adapter.render(env))

        ## Loop over frames within episode
        while not (terminated or truncated):
            # Wait until it's time for the next frame
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
            next_t += dt

            if turn_based:
                # Advance only on a fresh keydown that maps to an action.
                action, user_quit = _get_action(key_to_action)
                if user_quit:
                    return True
                if action is None:
                    continue                    # no press -> don't step
            else:
                if _check_quit():
                    return True
                action = keyspec.resolve(held_key_names())

            obs, reward, terminated, truncated, info = adapter.step(env, action)
            # Anchor a full savestate at episode start and every stride.
            save_blob = (ep_frame % state_stride == 0)
            ep_frame += 1
            fs = adapter.capture(env, obs, info, want_blob=save_blob)

            frames["action"].append(action)
            frames["reward"].append(reward)
            frames["terminated"].append(bool(terminated))
            frames["truncated"].append(bool(truncated))
            frames["episode_id"].append(episode_id)
            frames["session_time"].append(self.clock.session_time())
            frames["wall_time"].append(self.clock.wall_time())
            frames["state_blob"].append(fs.blob)
            for k, v in fs.variables.items():
                frames["variables"][k].append(v)

            self.display.draw_frame(adapter.render(env))
            if time.perf_counter() >= block_end:
                break
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
        adapter = self.adapters[backend]
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

        ## Key mapping
        # Some backends (nle, browser games) take several seconds to start;
        # show a Loading screen so the previous fixation "+" doesn't freeze.
        self.display.draw_text(
            f"Loading {phase.get('text') or phase.get('game', 'game')} …")
        env = adapter.make(phase)
        keyspec = adapter.keymap(env)
        # Allow the curriculum to override the mapping explicitly.
        keyspec = _apply_key_overrides(keyspec, phase.get("keys"))
        # For turn-based play, map single pressed KEY -> action via key names.
        key_to_action = _get_key_to_action_map(keyspec)

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
                adapter, env, phase, frames,
                seed=base_seed + episode_id, episode_id=episode_id,
                turn_based=turn_based, key_to_action=key_to_action,
                keyspec=keyspec, dt=dt, state_stride=state_stride,
                block_end=block_end)
            episode_id += 1
            if mode == "episode" and episode_id >= n_episodes:
                break

        extra = getattr(adapter, "block_extra", lambda: None)()
        adapter.close(env)
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
        
    def run(self) -> None:
        """Run the full curriculum: trigger wait, then each phase in order.

        Always writes the session manifest in ``finally``, including after an
        interrupt (partial data).
        """
        handlers = {"fixation": self._fixation, "message": self._message,
                    "game": self._game, "survey": self._survey}
        try:
            self._trigger()

            for index, phase in enumerate(self.curriculum):
                handler = handlers.get(phase["type"])
                if handler is None:
                    raise ValueError(f"unknown phase type: {phase['type']!r}")
                handler(phase, index)

            self.display.draw_text("Done. Thank you!")
            time.sleep(2.0)
        except KeyboardInterrupt:
            print("Interrupted -- saving partial data.", file=sys.stderr)
        finally:
            manifest_path = self.logger.save_manifest()
            print(f"Saved session to: {self.outdir}")
            print(f"Manifest: {manifest_path}")
