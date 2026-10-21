"""COOM scenarios as a Gymnasium env, driven through ``vizdoom.DoomGame``.

COOM's own package pins ``gymnasium==0.28.1``, which conflicts with a current
gymnasium, and its env factory wraps the screen for training (84x84, normalized,
frame-stacked) rather than a viewable frame. This env never imports that
package. It reads a TTomilin/COOM checkout's own scenario assets --
``conf.cfg`` and ``<task>.wad`` under ``<repo>/COOM/env/scenarios/<scenario>/``
-- and steps the game itself.

Every scenario exposes exactly 4 buttons, always
``[TURN_LEFT, TURN_RIGHT, MOVE_FORWARD, <execute>]`` (JUMP, ATTACK, SPEED, or
USE). The action space is COOM's 12-action table, turn(3) x move(2) x
execute(2): ``index = turn * 4 + move * 2 + execute``, with turn 0 = none,
1 = right, 2 = left. So 0 = noop, 1 = execute, 2 = forward, 3 = forward +
execute, 4 = right, 6 = right + forward, 8 = left, 10 = left + forward.

One step is one Doom tic. ``conf.cfg`` asks for a 160x120 training view; the
screen here is 640x480. Reward is ViZDoom's raw reward, not COOM's shaping.
``audio_buffer_enabled`` puts one 44.1 kHz stereo buffer for that tic on the
live ``DoomGame`` (``env.game``). The labels/objects/sectors buffers are
turned on here (COOM's own scenarios leave them off) so ``game.get_state()``
also carries every object in the level and the level's own sector geometry,
not just ``game_variables``.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from vizdoom_gym import use_openal_soft


def _actions() -> list[list[bool]]:
    """COOM's unified action table: turn(3) x move(2) x execute(2) = 12."""
    turns = [[False, False], [False, True], [True, False]]
    moves = [[False], [True]]
    execs = [[False], [True]]
    return [t + m + e for t in turns for m in moves for e in execs]


_ACTIONS = _actions()

#: fmri-gym's ``external/coom``; this file is ``gym/coom/coom_gym/env.py``.
_DEFAULT_REPO = Path(__file__).resolve().parents[3] / "external" / "coom"
_CLONE = "git submodule update --init external/coom  (a submodule of fmri-gym)"


def _configure_audio(game: Any, *, enabled: bool, efx: bool) -> None:
    """Turn the audio buffer on before ``init``, or leave it off.

    On macOS, turning it on also puts OpenAL Soft on dyld's path
    (:func:`vizdoom_gym.use_openal_soft`).

    :param game: a ``vizdoom.DoomGame`` that has not been initialised yet.
    :param enabled: whether this step should produce a PCM buffer.
    :param efx: whether Doom's reverb filter is on. Older OpenAL builds abort
        while setting it up, so the caller states the choice.
    :raises TypeError: if either flag is not a bool.
    :raises RuntimeError: audio was requested and OpenAL is missing (libopenal1
        on Linux, OpenAL Soft on macOS).
    """
    import vizdoom as vzd

    if not isinstance(enabled, bool) or not isinstance(efx, bool):
        raise TypeError("audio_buffer_enabled and audio_efx must be booleans")
    game.set_audio_buffer_enabled(enabled)
    if not enabled:
        return
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libopenal.so.1")
        except OSError as error:
            raise RuntimeError("COOM audio requires OpenAL: install libopenal1 "
                               "(Ubuntu: sudo apt install libopenal1)") from error
    if sys.platform == "darwin" and use_openal_soft() is None:
        raise RuntimeError("COOM audio on macOS requires OpenAL Soft: brew install openal-soft")
    game.set_audio_sampling_rate(vzd.SamplingRate.SR_44100)
    game.set_audio_buffer_size(1)
    game.add_game_args(f"+snd_efx {int(efx)}")
    game.set_console_enabled(True)  # expose engine audio/MIDI initialization errors
    print(f"COOM audio: 44100 Hz stereo, 1 tic/buffer; EFX reverb={efx}")


class COOMEnv(gym.Env):
    """One COOM scenario.

    :param scenario: directory name under ``COOM/env/scenarios`` (``pitfall``,
        ``chainsaw``, ...).
    :param repo: the TTomilin/COOM checkout. Defaults to ``external/coom``
        under the fmri-gym root.
    :param task: WAD variant inside the scenario (default ``default``; some
        scenarios ship ``hard``, ``blue``, ``red``, ...).
    :param seed: seed applied before the game is initialised.
    :param audio_buffer_enabled: record one PCM buffer per tic.
    :param audio_efx: Doom's reverb on that buffer.
    :raises RuntimeError: no such scenario at the checkout, the scenario does
        not have COOM's 4-button layout, or audio is on and OpenAL is missing.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["rgb_array"], "render_fps": 35}

    def __init__(
        self,
        scenario: str,
        *,
        repo: str | None = None,
        task: str = "default",
        seed: int = 0,
        audio_buffer_enabled: bool = False,
        audio_efx: bool = False,
    ) -> None:
        import vizdoom as vzd

        scenario_dir = Path(repo or _DEFAULT_REPO) / "COOM" / "env" / "scenarios" / scenario
        if not scenario_dir.is_dir():
            raise RuntimeError(f"no COOM scenario {scenario!r} at {scenario_dir}; {_CLONE}, "
                               "or pass repo=")
        game = vzd.DoomGame()
        game.load_config(str(scenario_dir / "conf.cfg"))
        game.set_doom_scenario_path(str(scenario_dir / f"{task}.wad"))
        game.set_window_visible(False)
        game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        game.set_seed(seed)
        _configure_audio(game, enabled=audio_buffer_enabled, efx=audio_efx)
        game.set_labels_buffer_enabled(True)
        game.set_objects_info_enabled(True)
        game.set_sectors_info_enabled(True)
        game.init()
        buttons = [str(b).split(".")[-1] for b in game.get_available_buttons()]
        if len(buttons) != 4:
            game.close()
            raise RuntimeError(f"COOM's 12-action table needs 4 buttons (TURN_LEFT, TURN_RIGHT, "
                               f"MOVE_FORWARD, <execute>); {scenario!r} reports {buttons!r}")
        self.game = game
        self.action_space = spaces.Discrete(len(_ACTIONS))
        h, w = game.get_screen_height(), game.get_screen_width()
        self.observation_space = spaces.Box(0, 255, (h, w, 3), dtype=np.uint8)
        self._last_frame: np.ndarray | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        """Start a new episode, optionally reseeding the scenario's RNG.

        :param seed: if given, replaces the seed from construction.
        :param options: unused.
        :return: ``(frame, info)``.
        """
        super().reset(seed=seed)
        if seed is not None:
            self.game.set_seed(int(seed))
        self.game.new_episode()
        return self._frame(), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Apply one of the 12 actions and advance one Doom tic.

        :param action: an index into the action table.
        :return: ``(frame, reward, terminated, truncated, info)``.
        """
        reward = self.game.make_action([int(b) for b in _ACTIONS[int(action)]])
        terminated = self.game.is_episode_finished()
        return self._frame(), float(reward), bool(terminated), False, {}

    def render(self) -> np.ndarray:
        """Return the current RGB frame.

        :return: ``(H, W, 3)`` uint8. The last frame, if the episode just ended.
        """
        return self._frame()

    def close(self) -> None:
        """Close the Doom game."""
        self.game.close()

    def _frame(self) -> np.ndarray:
        """RGB frame from the current state, falling back to the last one.

        ``get_state()`` returns ``None`` on the tic an episode ends.
        """
        state = self.game.get_state()
        if state is None:
            if self._last_frame is None:
                h, w = self.game.get_screen_height(), self.game.get_screen_width()
                self._last_frame = np.zeros((h, w, 3), dtype=np.uint8)
            return self._last_frame
        self._last_frame = np.transpose(state.screen_buffer, (1, 2, 0))
        return self._last_frame
