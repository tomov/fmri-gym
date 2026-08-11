"""VGDL adapter (Video Game Description Language games).

Wraps the VGDLEnv from ccolas/language_and_experience behind the standard
EnvAdapter interface. VGDL is built on OLD gym (0.26) and needs numpy<2, so it
CANNOT share an env with the gymnasium/numpy-2 backends -- run VGDL curricula in
the `language_and_experience` conda env. The fmri_gym core imports fine there
(it needs only pygame + numpy), which is the whole point of the adapter split.

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

from .base import EnvAdapter, FrameState, KeySpec

# Fixed VGDL action order (verified from get_action_meanings / core.py).
_VGDL_ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "NO_OP", "SPACE"]
_KEYS = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3, "SPACE": 5}


class VGDLAdapter(EnvAdapter):
    name = "vgdl"

    def __init__(self, repo=None, save_pixels=False):
        self.repo = repo or os.environ.get("VGDL_REPO")
        self.save_pixels = save_pixels
        self._VGDLEnv = None

    def _load_env_class(self):
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

    def make(self, spec):
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
        env.render(mode="rgb_array")  # lazily init the renderer before reset
        return env

    def keymap(self, env) -> KeySpec:
        combos = {frozenset([k]): idx for k, idx in _KEYS.items()}
        return KeySpec(combos=combos, noop=_VGDL_ACTIONS.index("NO_OP"),
                       help="Arrow keys move, SPACE acts.")

    def reset(self, env, seed, spec):
        if seed is not None:
            env.game.set_seed(int(seed))
        return env.reset(with_img=False)

    def render(self, env):
        return env.render(mode="rgb_array")

    def capture(self, env, obs, info, want_blob=True) -> FrameState:
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

    def restore(self, env, blob):
        env.set_state(pickle.loads(blob))

    def step(self, env, action):
        # VGDL's step takes an int action plus a with_img kwarg (default frame
        # off; we render separately via render()).
        return env.step(action, with_img=False)
