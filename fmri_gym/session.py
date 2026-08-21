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
    """Anchored at the scanner trigger; gives trigger-relative + epoch time."""

    def __init__(self):
        self.t0_perf = None
        self.t0_epoch = None

    def anchor(self):
        self.t0_perf = time.perf_counter()
        self.t0_epoch = time.time()

    def rel(self):
        return time.perf_counter() - self.t0_perf

    def epoch(self):
        return time.time()


def _check_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            return True
    return False


# ASCII names (the default pygame font has no arrow glyphs -> tofu boxes).
_ARROW_GLYPH = {"UP": "UP", "DOWN": "DOWN", "LEFT": "LEFT", "RIGHT": "RIGHT"}
_KEY_ORDER = ["UP", "DOWN", "LEFT", "RIGHT", "SPACE", "Z", "X", "RETURN", "LSHIFT"]


def _controls_from_keyspec(keyspec, adapter, env):
    """Return ordered (keys, meaning) pairs for the controls screen.

    Uses the adapter's explicit `controls` if provided; otherwise derives them
    from the (already override-applied) combos, labelling each action via
    adapter.describe_action so backends like ALE show real meanings
    (RIGHT->'RIGHT', SPACE->'FIRE', ...).
    """
    if keyspec.controls:
        return list(keyspec.controls)

    def canon(keys):
        return "+".join(_ARROW_GLYPH.get(k, k) for k in
                        sorted(keys, key=lambda x: (_KEY_ORDER.index(x)
                               if x in _KEY_ORDER else 99, x)))

    def order(label):
        first = label.split("+")[0]
        name = {v: k for k, v in _ARROW_GLYPH.items()}.get(first, first)
        return (_KEY_ORDER.index(name) if name in _KEY_ORDER else 99, label)

    # Collapse duplicate actions (a `keys` override may leave the default key
    # for the same action in place): keep the canonically-first key per action.
    best = {}   # meaning -> (order_key, label)
    for keys, action in keyspec.combos.items():
        label = canon(keys)
        try:
            meaning = adapter.describe_action(env, action)
        except Exception:
            meaning = str(action)
        # Hide unhelpful meanings: identical to the key label, or a bare action
        # index (e.g. "3") that adds nothing over the key name.
        if meaning == label or meaning.lstrip("-").isdigit():
            meaning = ""
        key = meaning or label
        cand = (order(label), label)
        if key not in best or cand < best[key]:
            best[key] = cand
    rows = [(label, meaning if meaning != label else "")
            for meaning, (_, label) in best.items()]
    return sorted(rows, key=lambda r: order(r[0]))


def _wait_any_key():
    """Block until any key is pressed (ESC raises to quit)."""
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt
                return
        time.sleep(0.005)


def _wait_for_char(char, dummy=False):
    if dummy:
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


class Session:
    """Runs a curriculum for one subject, dispatching phases to handlers."""

    def __init__(self, subject, curriculum, adapters, display, outdir,
                 dummy_trigger=False):
        self.subject = subject
        self.curriculum = curriculum
        self.adapters = adapters              # {backend_name: EnvAdapter}
        self.display = display
        self.dummy = dummy_trigger
        self.clock = Clock()
        self.logger = Logger(outdir, subject, curriculum, self.clock)
        self.outdir = outdir

    # -- phase handlers ------------------------------------------------------

    def _fixation(self, phase, index):
        duration = phase.get("duration", 2.0)
        onset = self.clock.rel()
        self.display.draw_fixation()
        end = time.perf_counter() + duration
        while time.perf_counter() < end:
            if _check_quit():
                raise KeyboardInterrupt
            time.sleep(0.005)
        self.logger.log_phase({"index": index, "type": "fixation",
                               "onset": onset, "offset": self.clock.rel()})

    def _message(self, phase, index):
        text = phase.get("text", "")
        duration = phase.get("duration")
        onset = self.clock.rel()
        self.display.draw_text(text)
        if duration is None:
            _wait_for_char(phase.get("key", " "), dummy=self.dummy)
        else:
            end = time.perf_counter() + duration
            while time.perf_counter() < end:
                if _check_quit():
                    raise KeyboardInterrupt
                time.sleep(0.005)
        self.logger.log_phase({"index": index, "type": "message", "text": text,
                               "onset": onset, "offset": self.clock.rel()})

    def _controls_screen(self, phase, keyspec, adapter, env):
        """Show the game name + its buttons and what they do, before play.

        Waits for any key press (experimenter/subject), or auto-advances after
        `controls_seconds` if set. Skipped in dummy-trigger runs (headless).
        """
        title = phase.get("text") or phase.get("game", "Game")
        lines = [str(title), ""]
        controls = _controls_from_keyspec(keyspec, adapter, env)
        if controls:
            lines.append("Controls:")
            width = max(len(k) for k, _ in controls)
            for keys, meaning in controls:
                lines.append(f"   {keys:<{width}}   {meaning}" if meaning
                             else f"   {keys}")
        elif keyspec.help:
            lines += ["Controls:", "   " + keyspec.help]
        lines += ["", "(press any key to start — ESC quits)"]
        self.display.draw_text("\n".join(lines))
        # Drop any keypresses queued during the (possibly slow) make()/load, so
        # they don't instantly dismiss this screen. Keep honoring ESC.
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                raise KeyboardInterrupt
        secs = phase.get("controls_seconds")
        if self.dummy:
            time.sleep(0.05)
        elif secs is not None:
            end = time.perf_counter() + secs
            while time.perf_counter() < end:
                if _check_quit():
                    raise KeyboardInterrupt
                time.sleep(0.005)
        else:
            _wait_any_key()

    def _survey(self, phase, index):
        questions = phase.get("questions", [])
        n_points = phase.get("n_points", 7)
        onset = self.clock.rel()
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
                              "t_rel": self.clock.rel()})
        self.logger.log_phase({"index": index, "type": "survey",
                               "onset": onset, "offset": self.clock.rel(),
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
        if phase.get("show_controls", True):
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
                                  "episode_id", "t_rel", "t_epoch", "state_blob")}
        frames["episode_seeds"] = []
        frames["variables"] = {}   # varname -> list, filled lazily

        # Show a per-game controls screen: game name + which buttons do what.
        # Skip with "show_controls": false; auto-advance with "controls_seconds".
        if phase.get("show_controls", True):
            self._controls_screen(phase, keyspec, adapter, env)

        onset = self.clock.rel()
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
                frames["t_rel"].append(self.clock.rel())
                frames["t_epoch"].append(self.clock.epoch())
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
            "onset": onset, "offset": self.clock.rel(),
            "n_episodes": episode_id, "n_frames": len(frames["action"]),
            "total_reward": total_reward, "data_file": path.split("/")[-1],
        })
        if user_quit:
            raise KeyboardInterrupt

    # -- top level -----------------------------------------------------------

    def run(self):
        handlers = {"fixation": self._fixation, "message": self._message,
                    "game": self._game, "survey": self._survey}
        try:
            self.display.draw_text(
                "Please keep your head as still as possible.\n\n"
                "(experimenter: press SPACE when ready)")
            _wait_for_char(EXPERIMENTER_KEY, dummy=self.dummy)
            self.display.draw_text("Waiting for scanner...")
            _wait_for_char(TRIGGER_KEY, dummy=self.dummy)

            self.clock.anchor()
            self.logger.set_trigger_time()

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
