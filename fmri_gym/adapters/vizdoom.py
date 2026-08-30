"""ViZDoom adapter -- Doom action-shooter scenarios (the COOM engine).

COOM (the DBP "action/shooter" pick) is a continual-RL suite built on ViZDoom,
but COOM itself pins gymnasium 0.28 which conflicts with the other backends
(minihack/nle need 1.2). ViZDoom -- the same Doom engine COOM uses -- ships
Gymnasium environments that work with our gymnasium and cover the same
action-shooter category (DefendCenter, DeadlyCorridor, HealthGathering,
TakeCover, MyWayHome, Deathmatch, and full Doom E1M1..). So this backend uses
ViZDoom directly.

The env's observation is a dict {"screen": (H,W,3) uint8, "gamevariables": ...};
`env.render()` (rgb_array) returns the screen for display, and we log
gamevariables (health, ammo, ...) as an analysis variable.

Actions are the scenario's small Discrete(n) button set, and keymaps (both the
default one below and curriculum `keys` overrides) are ALWAYS Discrete action
indices. Setting `env_kwargs.max_buttons_pressed` to 0 switches the env to a
MultiBinary action space so several buttons can be pressed at once; the keymap
is unchanged, we just OR the buttons of every held key (e.g. forward + turn).
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import gymnasium as gym

from .base import EnvAdapter, FrameState, KeySpec, MultiKeySpec, SingleKeySpec

# Physical key -> preferred Doom button (first available for the scenario wins).
# The gymnasium wrapper's Discrete action i presses the buttons set in
# env.unwrapped.button_map[i]; index 0 is the no-op (all buttons up).
# See e.g. https://github.com/Farama-Foundation/ViZDoom/blob/main/scenarios/deathmatch.cfg
_DEFAULT_KEY_TO_BUTTON_MAP: dict[str, list[str]] = {
    "UP": ["MOVE_FORWARD"],
    "DOWN": ["MOVE_BACKWARD"],
    "LEFT": ["TURN_LEFT"],
    "RIGHT": ["TURN_RIGHT"],
    "Z": ["MOVE_LEFT"],
    "X": ["MOVE_RIGHT"],
    "SPACE": ["ATTACK"],
    "ENTER": ["USE"],
    "N": ["SELECT_PREV_WEAPON"],
    "M": ["SELECT_NEXT_WEAPON"],
}


def _get_button_map(env: gym.Env) -> list[list[int]]:
    """Return ``Discrete action index -> per-button 0/1 row`` for this scenario.

    ViZDoom only builds ``env.unwrapped.button_map`` for a Discrete action
    space; under MultiBinary (``max_buttons_pressed=0``) we rebuild the same
    single-button table, so Discrete indices mean the same thing in both modes.

    :param env: a ViZDoom Gymnasium environment.
    :return: one 0/1 row per Discrete action index.
    """
    button_map = getattr(env.unwrapped, "button_map", None)
    if button_map is not None:
        return [[int(v) for v in row] for row in np.asarray(button_map)]
    n = len(env.unwrapped.game.get_available_buttons())
    return [list(row) for row in itertools.product((0, 1), repeat=n)
            if sum(row) <= 1]


def _get_button_to_action_map(env: gym.Env) -> dict[str, int]:
    """Map each available Doom button name to its Discrete action index.

    Only single-button rows of the button map are included; the first index
    for each button wins.

    :param env: a ViZDoom Gymnasium environment.
    :return: ``{BUTTON_NAME: discrete_action_index}``.
    """
    u = env.unwrapped
    names = [str(b).split(".")[-1] for b in u.game.get_available_buttons()]
    out: dict[str, int] = {}
    for i, row in enumerate(_get_button_map(env)):
        on = [names[j] for j, v in enumerate(row) if v]
        if len(on) == 1 and on[0] not in out:
            out[on[0]] = i
    return out


def _get_default_key_to_action_map(env: gym.Env) -> KeySpec:
    """Build the default keyboard->action :class:`KeySpec` for this scenario.

    For each physical key in :data:`_DEFAULT_KEY_TO_BUTTON_MAP`, picks the
    first preferred Doom button that exists in the env's button map.

    :param env: a ViZDoom Gymnasium environment.
    :return: a :class:`KeySpec` with single-key combos and ``noop=0``, both
        given as Discrete action indices. MultiBinary envs get a
        :class:`MultiKeySpec` that ORs the buttons of every held key.
    """
    btn_idx = _get_button_to_action_map(env)
    combos: dict[frozenset[str], int] = {}
    for key, prefs in _DEFAULT_KEY_TO_BUTTON_MAP.items():
        for b in prefs:
            if b in btn_idx:
                combos[frozenset([key])] = btn_idx[b]
                break
    if isinstance(env.action_space, gym.spaces.MultiBinary):
        return MultiKeySpec(combos=combos, noop=0,
                            button_map=_get_button_map(env))
    return SingleKeySpec(combos=combos, noop=0)


class VizDoomAdapter(EnvAdapter):
    name: str = "vizdoom"

    def make(self, spec: dict) -> gym.Env:
        """Create a ViZDoom Gymnasium environment for one game block.

        :param spec: game-phase config dict from the curriculum.
        :return: a ViZDoom Gymnasium environment.
        """
        from vizdoom import gymnasium_wrapper  # noqa: F401  (registers Vizdoom*-v1)
        return gym.make(spec["game"], render_mode="rgb_array",
                        **spec.get("env_kwargs", {}))

    def keymap(self, env: gym.Env) -> KeySpec:
        return _get_default_key_to_action_map(env)

    def render(self, env: gym.Env) -> np.ndarray:
        return np.asarray(env.render())

    def capture(
        self, env: gym.Env, obs: Any, info: dict, want_blob: bool = True
    ) -> FrameState:
        variables = {}
        if isinstance(obs, dict) and "gamevariables" in obs:
            variables["gamevariables"] = np.asarray(obs["gamevariables"])
        return FrameState(blob=None, variables=variables)
