"""AI GameStore adapter (aigamestore_gym) -- the ten browser games, lock-stepped.

``aigamestore_gym`` (``vendor/aigamestore/``) wraps each AI GameStore game -- an
LLM-generated p5.js / three.js page -- in a Gymnasium env by taking over the
page's clock: one ``step`` holds a set of keys and advances the game exactly
``frame_skip`` of its 60 Hz frames, then returns the canvas and the game's own
``getGameState()``. A model deliberating between steps and a subject at the
button box therefore meet the same game, and a block replays from
``episode_seeds`` + ``actions``. The action is ``MultiBinary`` over the game's
keys (``env.keys``), so held keys combine, as on a keyboard.

An episode is one level, named in the phase: ``"game": "game6/level3"`` plays
level 3 of game6 afresh every episode, and the episode ends on a win, a loss
or a level clear (the env's rule; see its docstring). A curriculum therefore
lists one game phase per level it wants played. ``"game": "game6"`` is level 1;
game4 has no levels and takes no ``/level``.

Keys: a phase's ``keys`` values are indices into the game's key list
(``aigamestore_gym.GAME_KEYS``; game1: LEFT, RIGHT, SPACE, Z), the order of the
MultiBinary action. The games paint control hints on the canvas; the env
relabels them from the same map, so a hint names the key the subject presses
where it differs from the game's own.

Timing: lock-stepped, one ``step`` per fmri-gym frame, so ``fps`` must equal
``60 / frame_skip`` (10 by default); ``_make`` refuses a config where they
disagree. Not supported: sound (the games have none) and savestates.

Phase fields: ``game`` ("game1".."game10", with an optional "/level<N>", or an
http(s):// URL to an index.html -- then ``game_keys`` lists the keys that game
listens for),
``frame_skip`` (default 6), ``headed``, ``browser_channel`` ("chrome" by
default; ``null`` for Playwright's bundled Chromium), ``games_dir``.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from .base import EnvAdapter, FrameState


class AIGameStoreAdapter(EnvAdapter):
    name: str = "aigamestore"

    def _make(self, spec: dict) -> gym.Env:
        from aigamestore_gym import GAME_KEYS, AIGameStoreEnv

        # "game6/level3" -> game6, level 3; an episode is one level.
        game, _, level = spec["game"].partition("/level")
        game_keys = spec.get("game_keys", GAME_KEYS.get(game, []))
        # Hints name the pressed key where it differs from the game's own; an
        # index the game has no key for is left to the Keymap to refuse.
        labels = {game_keys[i]: pressed for pressed, i in spec["keys"].items()
                  if isinstance(i, int) and 0 <= i < len(game_keys) and game_keys[i] != pressed}
        env = AIGameStoreEnv(
            game,
            level=int(level) if level else None,
            keys=spec.get("game_keys"),
            frame_skip=int(spec.get("frame_skip", 6)),
            headless=not spec.get("headed", False),
            browser_channel=spec.get("browser_channel", "chrome"),
            key_labels=labels,
            games_dir=spec.get("games_dir"),
        )
        fps = spec["fps"]
        if env.metadata["render_fps"] != fps:
            env.close()
            raise ValueError(f"fps={fps} but the game steps at {env.metadata['render_fps']:g} Hz "
                             f"(60 / frame_skip {env.frame_skip}); set frame_skip so they match")
        return env

    def native_fps(self) -> float:
        return self.env.metadata["render_fps"]

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> FrameState:
        # The scalar fields of the game's own state (score, level, gamePhase,
        # lives, ...), one state_* variable each.
        return FrameState(blob=None, variables={
            f"state_{name}": value for name, value in info["state"].items()})
