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

import pygame

from .adapters.base import held_key_names, key_name as _key_name
from .display import Display
from .logging import Logger

TRIGGER_KEY = "="
EXPERIMENTER_KEY = " "


class Clock:
    """Anchored at the scanner trigger; gives session + wall-clock time."""

    def __init__(self):
        self.t0_perf = None
        self.t0_epoch = None

    def trigger(self):
        self.t0_perf = time.perf_counter()
        self.t0_epoch = time.time()

    def session_time(self):
        return time.perf_counter() - self.t0_perf

    def wall_time(self):
        return time.time()


def _check_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return True
    return False


def _wait_for_char(char, dummy_trigger=False):
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


def _wait_for_duration(duration):
    """Block for `duration` seconds (ESC/quit raises KeyboardInterrupt)."""
    end = time.perf_counter() + duration
    while time.perf_counter() < end:
        if _check_quit():
            raise KeyboardInterrupt
        time.sleep(0.005)


def _apply_key_overrides(keyspec, overrides):
    """Merge curriculum-provided {"LEFT": action, "LEFT+SPACE": action} combos."""
    if not overrides:
        return keyspec
    combos = dict(keyspec.combos)
    for combo_str, action in overrides.items():
        keys = frozenset(k.strip().upper() for k in combo_str.split("+"))
        combos[keys] = action
    keyspec.combos = combos
    return keyspec


class Session:
    """Runs a curriculum for one subject, dispatching phases to handlers."""

    def __init__(self, subject, curriculum, adapters, display, outdir,
                 dummy_trigger=False):
        self.subject = subject
        self.curriculum = curriculum
        self.adapters = adapters              # {backend_name: EnvAdapter}
        self.display = display
        self.dummy_trigger = dummy_trigger
        self.clock = Clock()
        self.logger = Logger(outdir, subject, curriculum, self.clock)
        self.outdir = outdir

    # -- phase handlers ------------------------------------------------------

    def _fixation(self, phase, index):
        duration = phase.get("duration", 2.0)
        onset = self.clock.session_time()

        self.display.draw_fixation()
        _wait_for_duration(duration)

        self.logger.log_phase({"index": index, "type": "fixation",
                               "onset": onset, "offset": self.clock.session_time()})

    def _message(self, phase, index):
        text = phase.get("text", "")
        duration = phase.get("duration")
        onset = self.clock.session_time()
        
        self.display.draw_text(text)
        if duration is None:
            _wait_for_char(phase.get("key", " "), dummy_trigger=self.dummy_trigger)
        else:
            _wait_for_duration(duration)

        self.logger.log_phase({"index": index, "type": "message", "text": text,
                               "onset": onset, "offset": self.clock.session_time()})

    def _survey(self, phase, index):
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

    def _game(self, phase, index):
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

        # Some backends (nle, browser games) take several seconds to start;
        # show a Loading screen so the previous fixation "+" doesn't freeze.
        self.display.draw_text(
            f"Loading {phase.get('text') or phase.get('game', 'game')} …")
        env = adapter.make(phase)
        keyspec = adapter.keymap(env)
        # Allow the curriculum to override the mapping explicitly.
        keyspec = _apply_key_overrides(keyspec, phase.get("keys"))
        # For turn-based play, map single pressed KEY -> action via key names.
        key_to_action = {next(iter(ks)): a for ks, a in keyspec.combos.items()
                         if len(ks) == 1}

        frames = {k: [] for k in ("action", "reward", "terminal",
                                  "episode_id", "session_time", "wall_time", "state_blob")}
        frames["episode_seeds"] = []
        frames["variables"] = {}   # varname -> list, filled lazily

        onset = self.clock.session_time()
        block_end = time.perf_counter() + cap
        episode_id = 0
        total_reward = 0.0
        user_quit = False

        while not user_quit and time.perf_counter() < block_end:
            seed = base_seed + episode_id
            obs, info = adapter.reset(env, seed, phase)
            frames["episode_seeds"].append(seed)
            terminated = truncated = False
            ep_frame = 0
            next_t = time.perf_counter()
            self.display.draw_frame(adapter.render(env))   # show initial state
            while not (terminated or truncated):
                now = time.perf_counter()
                if now < next_t:
                    time.sleep(next_t - now)
                next_t += dt

                if turn_based:
                    # Advance only on a fresh keydown that maps to an action.
                    action = None
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (
                                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                            user_quit = True
                            break
                        if event.type == pygame.KEYDOWN:
                            name = _key_name(pygame, event.key)
                            if name in key_to_action:
                                action = key_to_action[name]
                    if user_quit:
                        break
                    if action is None:
                        continue                    # no press -> don't step
                else:
                    if _check_quit():
                        user_quit = True
                        break
                    action = keyspec.resolve(held_key_names(pygame))

                obs, reward, terminated, truncated, info = adapter.step(env, action)
                total_reward += float(reward)
                # Anchor a full savestate at episode start and every stride.
                want_blob = (ep_frame % state_stride == 0)
                ep_frame += 1
                fs = adapter.capture(env, obs, info, want_blob=want_blob)

                frames["action"].append(action)
                frames["reward"].append(reward)
                frames["terminal"].append(bool(terminated or truncated))
                frames["episode_id"].append(episode_id)
                frames["session_time"].append(self.clock.session_time())
                frames["wall_time"].append(self.clock.wall_time())
                frames["state_blob"].append(fs.blob)
                for k, v in fs.variables.items():
                    frames["variables"].setdefault(k, []).append(v)

                self.display.draw_frame(adapter.render(env))
                if time.perf_counter() >= block_end:
                    break
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
            "total_reward": total_reward, "data_file": path.split("/")[-1],
        })
        if user_quit:
            raise KeyboardInterrupt

    # -- top level -----------------------------------------------------------

    def _trigger(self):
        """Wait for experimenter ready + scanner trigger, then start the clock."""
        self.display.draw_text(
            "Please keep your head as still as possible.\n\n"
            "(experimenter: press SPACE when ready)")
        _wait_for_char(EXPERIMENTER_KEY, dummy_trigger=self.dummy_trigger)
        self.display.draw_text("Waiting for scanner...")
        _wait_for_char(TRIGGER_KEY, dummy_trigger=self.dummy_trigger)
        
        self.clock.trigger()
        self.logger.set_trigger_time()

    def run(self):
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
