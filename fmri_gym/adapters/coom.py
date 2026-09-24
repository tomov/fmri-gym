"""COOM adapter -- Doom continual-RL scenarios (TTomilin/COOM), via raw ViZDoom.

COOM ships its own Doom scenario configs + WADs, but its Python package pins
gymnasium==0.28.1 (conflicting with minihack's gymnasium==1.2 pin in this
shared env) and its env factory wraps observations for RL training (84x84,
normalized, frame-stacked -- see COOM.env.builder.wrap_env), not a viewable
frame. So, like the `vizdoom` backend, we never import the COOM package: we
drive `vizdoom.DoomGame` directly and point it at COOM's own scenario assets
(conf.cfg + <task>.wad under <repo>/COOM/env/scenarios/<scenario>/, located
via the phase "repo" field or the COOM_REPO env var, same convention as the
vgdl backend). This sidesteps the version conflict entirely and reaches
COOM's own tasks (pitfall, chainsaw, hide_and_seek, health_gathering,
arms_dealer, parkour, raise_the_roof, run_and_gun, floor_is_lava), which the
`vizdoom` backend's stock scenarios (DeadlyCorridor, DefendCenter, ...) don't
cover. conf.cfg also pins a tiny 160x120 render meant for training; we bump
it to 640x480 for a human to actually see.

Every COOM scenario exposes exactly 4 buttons, always in the order
[TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, <scenario-specific 4th button>]
(JUMP/ATTACK/SPEED/USE) -- this is what lets COOM train one continual-learning
agent across all of them via a single unified action table (COOM's own
`build_multi_discrete_actions`, despite the name, is actually a Discrete(12)
space: 3 turn states x 2 move states x 2 execute states). We rebuild that same
12-action table here from whatever buttons the loaded scenario reports, so
the human keymap (and any curriculum `keys` override) lines up with COOM's
own action indices without hardcoding a per-scenario button table.

`step()` returns ViZDoom's raw (near-zero) reward; COOM's actual reward
shaping lives in Python wrapper classes we deliberately don't use here. Game
variables (health, ammo, position, ...) are logged as an analysis variable
instead, mirroring the `vizdoom` backend's `gamevariables`. No native audio
yet (unlike `vizdoom.py`'s `sound()`) -- ViZDoom exposes it the same way
here, just not wired up.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .keyspec import KeySpec, SingleKeySpec
from .base import EnvAdapter, FrameState

# Physical key for each possible 4th (execute) button.
_EXECUTE_KEY = {"JUMP": "SPACE", "ATTACK": "SPACE", "SPEED": "LSHIFT", "USE": "E"}


def _build_actions() -> list[list[bool]]:
    """COOM's unified action table: turn(3) x move(2) x execute(2) = 12."""
    turns = [[False, False], [False, True], [True, False]]
    moves = [[False], [True]]
    execs = [[False], [True]]
    return [t + m + e for t in turns for m in moves for e in execs]


_ACTIONS = _build_actions()
_NOOP = _ACTIONS.index([False, False, False, False])


class COOMAdapter(EnvAdapter):
    name: str = "coom"

    def _make(self, spec: dict) -> Any:
        """Create a raw ViZDoom game for one COOM scenario.

        :param spec: game-phase config dict; ``game`` is the scenario name
            (e.g. "pitfall"), ``env_kwargs.task`` picks the WAD variant
            (default "default", e.g. "hard" for run_and_gun's "blue"/"red"/...).
        :return: an initialised ``vizdoom.DoomGame``, stored as ``self.env``.
        :raises RuntimeError: if no COOM repo checkout path is configured.
        """
        import vizdoom as vzd  # optional dep, only needed by this backend

        repo = spec.get("repo") or os.environ.get("COOM_REPO")
        if not repo:
            raise RuntimeError(
                "COOM adapter needs the TTomilin/COOM repo path; set "
                "phase 'repo' or the COOM_REPO env var.")
        scenario = spec["game"]
        env_kwargs = spec.get("env_kwargs", {})
        task = env_kwargs.get("task", "default")
        scenario_dir = Path(repo) / "COOM" / "env" / "scenarios" / scenario

        game = vzd.DoomGame()
        game.load_config(str(scenario_dir / "conf.cfg"))
        game.set_doom_scenario_path(str(scenario_dir / f"{task}.wad"))
        game.set_window_visible(False)
        game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        game.set_seed(env_kwargs.get("seed", 0))
        game.init()
        self._last_frame: np.ndarray | None = None
        return game

    def _keyspec(self) -> KeySpec:
        """Derive the keymap from the scenario's own button list.

        :return: arrows to turn/move, plus the scenario's 4th button on its
            mapped key (:data:`_EXECUTE_KEY`), both alone and combined with UP.
        :raises RuntimeError: if the scenario doesn't report COOM's standard
            4-button layout, or its 4th button has no default key mapped.
        """
        game = self.spec["game"]
        button_names = [str(b).split(".")[-1] for b in self.env.get_available_buttons()]
        if len(button_names) != 4:
            raise RuntimeError(f"COOM adapter expects 4 buttons (TURN_LEFT, "
                               f"TURN_RIGHT, MOVE_FORWARD, <execute>); {game!r} "
                               f"reports {button_names!r}.")
        execute_button = button_names[3]
        if execute_button not in _EXECUTE_KEY:
            raise RuntimeError(f"No default key for {game!r}'s {execute_button!r} "
                               f"button; add one to _EXECUTE_KEY or pass 'keys'.")
        execute_key = _EXECUTE_KEY[execute_button]

        def action_for(*, turn_left=False, turn_right=False,
                       move=False, execute=False) -> int:
            return _ACTIONS.index([turn_left, turn_right, move, execute])

        combos = {
            frozenset(["LEFT"]): action_for(turn_left=True),
            frozenset(["RIGHT"]): action_for(turn_right=True),
            frozenset(["UP"]): action_for(move=True),
            frozenset(["LEFT", "UP"]): action_for(turn_left=True, move=True),
            frozenset(["RIGHT", "UP"]): action_for(turn_right=True, move=True),
            frozenset([execute_key]): action_for(execute=True),
            frozenset(["UP", execute_key]): action_for(move=True, execute=True),
        }
        return SingleKeySpec(combos=combos, noop=_NOOP)

    def reset(self, seed: int | None) -> tuple[Any, dict]:
        """Start a new episode, optionally reseeding the scenario's RNG."""
        if seed is not None:
            self.env.set_seed(int(seed))
        self.env.new_episode()
        return self._frame(), {}

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict]:
        """Apply one of the 12 unified actions and advance one Doom tic."""
        reward = self.env.make_action([int(b) for b in _ACTIONS[action]])
        terminated = self.env.is_episode_finished()
        return self._frame(), reward, terminated, False, {}

    def _frame(self) -> np.ndarray:
        """RGB frame from the current state, falling back to the last one.

        ViZDoom's ``get_state()`` returns ``None`` on the tic an episode ends,
        so ``render``/``capture`` need a frame to show even then.
        """
        state = self.env.get_state()
        if state is None:
            if self._last_frame is None:
                h, w = self.env.get_screen_height(), self.env.get_screen_width()
                self._last_frame = np.zeros((h, w, 3), dtype=np.uint8)
            return self._last_frame
        self._last_frame = np.transpose(state.screen_buffer, (1, 2, 0))
        return self._last_frame

    def render(self) -> np.ndarray:
        return self._frame()

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> FrameState:
        state = self.env.get_state()
        variables = {}
        if state is not None:
            variables["game_variables"] = np.asarray(state.game_variables)
        return FrameState(blob=None, variables=variables)
