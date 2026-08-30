"""VGDL adapter (Video Game Description Language games).

Wraps the VGDLEnv from ccolas/language_and_experience behind the standard
EnvAdapter interface. The VGDL source has been ported from old gym to
gymnasium (see that repo's `dbp` branch), so it runs in the SAME numpy-2 env as
the ale/retro/gym backends -- no separate conda env needed. Point the adapter
at the checkout via the phase "repo" field or the VGDL_REPO env var, and add it
to PYTHONPATH so `src.vgdl...` is importable.

VGDL specifics handled here:
- construction is by file paths, not an env id:
    VGDLEnv(game_file=..., level_file=..., obs_type='objects', block_size=...)
- there is no seed= on reset(); seed via env.game.set_seed(s) before reset();
- step takes an int action + with_img kwarg; reset takes with_img;
- render uses old-gym's render(mode='rgb_array');
- exact savestate via env.get_state()/set_state() (picklable hidden state);
- fixed action set UP/DOWN/LEFT/RIGHT/NO_OP/SPACE -> indices 0..5.

Curriculum phase fields (backend "vgdl"):
    repo   : path to the language_and_experience checkout (or set VGDL_REPO env)
    game   : game name, e.g. "aliens" (dir games/<game>_v0/)
    level  : level index (default 0)
    block_size : render tile size in px (default 25)
The phase's "game" is the VGDL game name; "game_file"/"level_file" are derived.
"""

from __future__ import annotations

import os
import pickle
import sys
from typing import Any

import gymnasium as gym
import numpy as np

from .keyspec import SingleKeySpec
from .base import EnvAdapter, FrameState

# Fixed VGDL action order (verified from get_action_meanings / core.py).
_VGDL_ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "NO_OP", "SPACE"]
_KEYS = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3, "SPACE": 5}


class VGDLAdapter(EnvAdapter):
    name: str = "vgdl"

    def __init__(self, repo: str | None = None, save_pixels: bool = False) -> None:
        self.repo = repo or os.environ.get("VGDL_REPO")
        self.save_pixels = save_pixels
        self._VGDLEnv = None

    def _load_env_class(self) -> None:
        if self._VGDLEnv is not None:
            return
        if not self.repo:
            raise RuntimeError(
                "VGDL adapter needs the language_and_experience repo path; set "
                "phase 'repo' or the VGDL_REPO env var.")
        if self.repo not in sys.path:
            sys.path.insert(0, self.repo)
        from src.vgdl.interfaces.gym.env import VGDLEnv
        self._VGDLEnv = VGDLEnv

    def make(self, spec: dict) -> gym.Env:
        self._load_env_class()
        repo = spec.get("repo", self.repo)
        game = spec["game"]
        level = spec.get("level", 0)
        block_size = spec.get("block_size", 25)
        game_file = os.path.join(repo, "games", f"{game}_v0", f"{game}.txt")
        level_file = os.path.join(
            repo, "games", f"{game}_v0", f"{game}_lvl{level}.txt")
        env = self._VGDLEnv(game_file=game_file, level_file=level_file,
                            obs_type="objects", block_size=block_size)
        # VGDL's own renderer calls pygame.display.set_mode(), which would
        # HIJACK and shrink our shared fMRI window to the game's tiny size
        # (the "upper-left crop" bug). Instead attach an OFFSCREEN renderer that
        # draws to a plain Surface; we read the frame from it and never touch
        # the display. (The display must already be initialised by our Display,
        # which it is by the time a game phase runs, so sprite convert_alpha
        # works.)
        self._attach_offscreen_renderer(env)
        return env

    def _attach_offscreen_renderer(self, env: gym.Env) -> None:
        import pygame
        from src.vgdl.render import PygameRenderer
        r = PygameRenderer(env.game, env.render_block_size)
        r.headless = True
        r.screen = pygame.Surface(r.screen_dims)   # offscreen draw target
        r.screen.fill((255, 255, 255))
        r.background = r.screen.copy()
        env.renderer = r

    def keymap(self, env: gym.Env) -> SingleKeySpec:
        combos = {frozenset([k]): idx for k, idx in _KEYS.items()}
        return SingleKeySpec(combos=combos, noop=_VGDL_ACTIONS.index("NO_OP"))

    def reset(self, env: gym.Env, seed: int | None, spec: dict) -> tuple[Any, dict]:
        if seed is not None:
            env.game.set_seed(int(seed))
        return env.reset(with_img=False)

    def render(self, env: gym.Env) -> np.ndarray:
        import numpy as np
        import pygame
        # Draw to the offscreen surface and read it directly. We deliberately do
        # NOT call env.render()/update_display(), which would push to (and
        # resize) the display surface.
        r = env.renderer
        r.draw_all()
        return np.flipud(np.rot90(
            pygame.surfarray.array3d(r.screen).astype(np.uint8)))

    def capture(
        self, env: gym.Env, obs: Any, info: dict, want_blob: bool = True
    ) -> FrameState:
        variables = {}
        # Symbolic per-cell object grid + collision events, if present in info.
        if isinstance(info, dict):
            if "state" in info:
                variables["symbolic_state"] = info["state"]
            if "events_triggered" in info:
                variables["events"] = info["events_triggered"]
        blob = (pickle.dumps(env.get_state(return_orientation=True))
                if want_blob else None)
        return FrameState(blob=blob, variables=variables)

    def restore(self, env: gym.Env, blob: bytes) -> None:
        env.set_state(pickle.loads(blob))

    def step(
        self, env: gym.Env, action: Any
    ) -> tuple[Any, float, bool, bool, dict]:
        # VGDL's step takes an int action plus a with_img kwarg (default frame
        # off; we render separately via render()).
        return env.step(action, with_img=False)
