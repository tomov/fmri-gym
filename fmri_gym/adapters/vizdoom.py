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
gamevariables (health, ammo, ...) as an analysis variable. Actions are a small
Discrete(n) button set per scenario; arrows/space map to the first n actions.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from .base import EnvAdapter, FrameState, KeySpec

# Physical key -> preferred Doom button (first available for the scenario wins).
# The gymnasium wrapper's Discrete action i presses the buttons set in
# env.unwrapped.button_map[i]; index 0 is the no-op (all buttons up).
_KEY_BUTTONS = {
    "UP": ["MOVE_FORWARD"],
    "DOWN": ["MOVE_BACKWARD"],
    "LEFT": ["MOVE_LEFT", "TURN_LEFT"],
    "RIGHT": ["MOVE_RIGHT", "TURN_RIGHT"],
    "Z": ["TURN_LEFT"],
    "X": ["TURN_RIGHT"],
    "SPACE": ["ATTACK", "USE"],
}
_FRIENDLY = {"MOVE_FORWARD": "move forward", "MOVE_BACKWARD": "move back",
             "MOVE_LEFT": "strafe left", "MOVE_RIGHT": "strafe right",
             "TURN_LEFT": "turn left", "TURN_RIGHT": "turn right",
             "ATTACK": "shoot", "USE": "use"}


class VizDoomAdapter(EnvAdapter):
    name = "vizdoom"

    def make(self, spec):
        from vizdoom import gymnasium_wrapper  # noqa: F401  (registers Vizdoom*-v1)
        return gym.make(spec["game"], render_mode="rgb_array",
                        **spec.get("make_kwargs", {}))

    def _button_index(self, env):
        """Return {BUTTON_NAME: discrete action index} from the wrapper's button_map."""
        u = env.unwrapped
        names = [str(b).split(".")[-1] for b in u.game.get_available_buttons()]
        out = {}
        for i, row in enumerate(np.asarray(u.button_map)):
            on = [names[j] for j, v in enumerate(row) if v]
            if len(on) == 1 and on[0] not in out:   # single-button action
                out[on[0]] = i
        self._btn_meaning = {}  # action idx -> friendly label, for describe_action
        for name, idx in out.items():
            self._btn_meaning[idx] = _FRIENDLY.get(name, name.lower())
        return out

    def keymap(self, env) -> KeySpec:
        btn_idx = self._button_index(env)
        combos = {}
        for key, prefs in _KEY_BUTTONS.items():
            for b in prefs:
                if b in btn_idx:
                    combos[frozenset([key])] = btn_idx[b]
                    break
        # controls list (key -> friendly meaning) for the auto screen
        controls = []
        for key in ["UP", "DOWN", "LEFT", "RIGHT", "SPACE", "Z", "X"]:
            ks = frozenset([key])
            if ks in combos:
                controls.append((key, self._btn_meaning.get(combos[ks], "")))
        return KeySpec(combos=combos, noop=0,
                       help="", controls=controls)

    def describe_action(self, env, action) -> str:
        return getattr(self, "_btn_meaning", {}).get(int(action), str(action))

    def render(self, env):
        return np.asarray(env.render())

    def capture(self, env, obs, info, want_blob=True) -> FrameState:
        variables = {}
        if isinstance(obs, dict) and "gamevariables" in obs:
            variables["gamevariables"] = np.asarray(obs["gamevariables"])
        return FrameState(blob=None, variables=variables)
