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

A few scenarios (Deathmatch, the full-game maps) also declare *delta* buttons
-- the mouse axes -- which makes ViZDoom's action space a
``Dict{"binary", "continuous"}``. A scanner button box has no mouse, so
:class:`_KeyboardOnlyAction` hides the delta axes and leaves the keyboard
buttons: the subject turns with TURN_LEFT/TURN_RIGHT, as in every other
scenario.

Sound is opt-in per curriculum: `env_kwargs.audio_buffer_enabled` puts one tic
of stereo PCM in `obs["audio"]` (so a model sees the same observation a subject
hears), which `sound()` hands to the session's speakers and `capture()` logs.
The phase's `"audio": false` mutes the speakers; the PCM is still logged.
Doom produces 1/35 s of sound per step whatever the frame rate, so audio only
runs in real time when `fps * env_kwargs.frame_skip == 35`; below that it plays
with gaps, above it lags further behind every frame.
"""

from __future__ import annotations

import itertools
from typing import Any

import gymnasium as gym
import numpy as np

from .base import EnvAdapter, FrameState, Sound
from .keyspec import KeySpec, MultiKeySpec, SingleKeySpec

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


class _KeyboardOnlyAction(gym.ActionWrapper):
    """Drop a scenario's delta (mouse) axes, keeping its keyboard buttons.

    ViZDoom presents a scenario with both kinds of buttons as a
    ``Dict{"binary": ..., "continuous": Box(n_delta)}`` action space. There is
    no mouse in the scanner, so the axes are held at 0 and the action space the
    adapter (and a model) sees is the plain binary one.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.action_space = env.action_space["binary"]
        self._axes = np.zeros(env.action_space["continuous"].shape,
                              dtype=np.float32)

    def action(self, action: Any) -> dict[str, Any]:
        """Return ``action`` as a Dict action with the delta axes at rest.

        :param action: a binary action (Discrete index or button vector).
        :return: the ``{"binary", "continuous"}`` action ViZDoom expects.
        """
        return {"binary": action, "continuous": self._axes}


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
    n = env.unwrapped.num_binary_buttons
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
    # ViZDoom reorders the scenario's buttons to put the delta (mouse) ones
    # first, and the button map only covers the binary ones that follow them.
    names = [str(b).split(".")[-1]
             for b in u.game.get_available_buttons()][u.num_delta_buttons:]
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

    def _make(self, spec: dict) -> gym.Env:
        """Create a ViZDoom Gymnasium environment for one game block.

        :param spec: game-phase config dict from the curriculum.
        :return: a ViZDoom Gymnasium environment.
        """
        from vizdoom import gymnasium_wrapper  # noqa: F401  (registers Vizdoom*-v1)
        env = gym.make(spec["game"], render_mode="rgb_array",
                       **spec.get("env_kwargs", {}))
        if env.unwrapped.game.is_audio_buffer_enabled():
            # ViZDoom 1.3.0 ships an assert-enabled OpenAL Soft whose EFX
            # (reverb) filter setup aborts with "gain > 0.00001f" the moment the
            # audio buffer is on, segfaulting the process inside game.init().
            # Doom's reverb only colours the buffer, so turn EFX off; the audio
            # buffer itself still carries the real sound.
            env.unwrapped.game.add_game_args("+snd_efx 0")
        if isinstance(env.action_space, gym.spaces.Dict):
            return _KeyboardOnlyAction(env)
        return env

    def _keyspec(self) -> KeySpec:
        return _get_default_key_to_action_map(self.env)

    def render(self) -> np.ndarray:
        return np.asarray(self.env.render())

    def native_fps(self) -> float:
        """Doom's tic rate (35) over the tics one step advances (``env_kwargs.frame_skip``)."""
        u = self.env.unwrapped
        return u.game.get_ticrate() / u.frame_skip

    def sound(self) -> Sound | None:
        """Return the frame's Doom audio, or ``None`` if there is none to play.

        ``audio_buffer`` is ``None`` unless the curriculum turned the buffer on,
        and ``state`` itself is ``None`` on a terminal frame -- the episode is
        over, so there is nothing left to hear.

        :return: one tic of stereo PCM at the scenario's rate, or ``None``.
        """
        state = self.env.unwrapped.state
        if state is None or state.audio_buffer is None:
            return None
        return Sound(state.audio_buffer,
                     self.env.unwrapped.game.get_audio_sampling_rate())

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> FrameState:
        variables = {}
        if isinstance(obs, dict) and "gamevariables" in obs:
            variables["gamevariables"] = np.asarray(obs["gamevariables"]).copy()
        if isinstance(obs, dict) and "audio" in obs:
            # What the subject heard this frame. Taken from obs, not sound(),
            # because obs has a (zeroed) audio buffer on the terminal frame too,
            # keeping this series the same length as actions and rewards.
            variables["audio"] = np.asarray(obs["audio"]).copy()
        return FrameState(blob=None, variables=variables)

    def block_extra(self) -> dict | None:
        """Block-level arrays merged into the npz (the audio sample rate).

        :return: ``{"audio_sampling_rate": Hz}`` when sound is on, else
            ``None``. Without it the logged ``audio`` is not playable back.
        """
        game = self.env.unwrapped.game
        if not game.is_audio_buffer_enabled():
            return None
        return {"audio_sampling_rate": game.get_audio_sampling_rate()}
