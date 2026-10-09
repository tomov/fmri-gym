"""ViZDoom adapter -- Doom action-shooter scenarios (the COOM engine).

COOM (the DBP "action/shooter" pick) is a continual-RL suite built on ViZDoom,
but COOM itself pins gymnasium 0.28 which conflicts with the other backends
(minihack/nle need 1.2). ViZDoom -- the same Doom engine COOM uses -- ships
Gymnasium environments that work with our gymnasium and cover the same
action-shooter category (DefendCenter, DeadlyCorridor, HealthGathering,
TakeCover, MyWayHome, Deathmatch, and the full Doom and Freedoom maps). So this
backend uses ViZDoom directly.

The env's observation is a dict {"screen": (H,W,3) uint8, "gamevariables": ...};
`env.render()` (rgb_array) returns the screen for display, and we log
gamevariables (health, ammo, ...) as an analysis variable.

Actions: with `env_kwargs.max_buttons_pressed` 0 (what every config here
uses) the action space is MultiBinary over the scenario's buttons, so a phase's
`keys` are button indices in the scenario's `available_buttons` order (see its
.cfg; DefendCenter: TURN_LEFT, TURN_RIGHT, ATTACK) and held keys combine
(forward + turn). Without it the space is the wrapper's Discrete(n), whose index
k presses button n-k (index 0 is the no-op), and `keys` are those indices.

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

from typing import Any

import gymnasium as gym
import numpy as np

from .base import EnvAdapter, FrameState, Sound


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

    def warm_up(self) -> None:
        self.reset(0)

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

    def outcome(self, terminated: bool, truncated: bool) -> tuple[str, str]:
        # Doom ends an episode on the player's death or the scenario's tic
        # limit; a scenario with a goal (a vest to reach, a monster to kill) ends
        # there too, but the engine does not say which, so only death is read.
        if terminated and self.env.unwrapped.game.is_player_dead():
            return "lost", "You died"
        return super().outcome(terminated, truncated)

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> FrameState:
        variables = {}
        if isinstance(obs, dict):
            for k, v in obs.items():
                if k == "screen":
                    continue
                variables[k] = np.asarray(v).copy()
        variables.update(info or {})
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
