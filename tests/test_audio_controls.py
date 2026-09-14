"""Keep playback preferences separate from recorded observations and cleanup.

Mock engine creation and presentation boundaries so these regressions need no
optional game packages, sound device, or pygame window.
"""

from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pygame

import fmri_play
from fmri_gym.adapters.base import FrameState, Sound
from fmri_gym.adapters.keyspec import SingleKeySpec
from fmri_gym.adapters.retro import RetroAdapter
from fmri_gym.adapters.vizdoom import VizDoomAdapter
from fmri_gym.session import Clock, Session


class AdapterAudioTests(unittest.TestCase):
    def test_retro_retains_native_pcm_and_fractional_rate(self) -> None:
        pcm = np.array([[1, -1], [2, -2]], dtype="int16")
        emulator = Mock()
        emulator.get_audio.return_value = pcm
        emulator.get_audio_rate.return_value = 44100.125
        env = SimpleNamespace(unwrapped=SimpleNamespace(buttons=["A", "RIGHT"], em=emulator))
        engine = SimpleNamespace(make=Mock(return_value=env))
        with patch.dict(sys.modules, {"stable_retro": engine}):
            audible = RetroAdapter({"game": "test-game"})
            muted = RetroAdapter({"game": "test-game", "audio": False})
        for adapter in (audible, muted):
            sound = adapter.sound()
            np.testing.assert_array_equal(sound.pcm, pcm)
            self.assertEqual(sound.pcm.dtype, pcm.dtype)
            self.assertEqual(sound.sample_rate, 44100.125)
        held = frozenset({"RIGHT", "Z"})
        self.assertEqual(audible.keyspec.resolve(held), muted.keyspec.resolve(held))

    def test_vizdoom_playback_mute_preserves_recording_buffer_configuration(self) -> None:
        cases = [({}, True), ({"audio": False}, True),
                 ({"env_kwargs": {"audio_buffer_enabled": False}}, False),
                 ({"audio": False, "env_kwargs": {"audio_buffer_enabled": True}}, True),
                 ({"audio": True, "env_kwargs": {"audio_buffer_enabled": False}}, False)]
        wrapper = SimpleNamespace(gymnasium_wrapper=object())
        keys = SingleKeySpec(combos={}, noop=0)
        for options, expected in cases:
            spec = {"game": "test-game", **deepcopy(options)}
            env = Mock()
            env.unwrapped.game.is_audio_buffer_enabled.return_value = expected
            with (self.subTest(options=options),
                  patch.dict(sys.modules, {"vizdoom": wrapper}),
                  patch("fmri_gym.adapters.vizdoom.gym.make", return_value=env) as make,
                  patch.object(VizDoomAdapter, "_keyspec", return_value=keys)):
                VizDoomAdapter(spec)
                self.assertIs(make.call_args.kwargs["audio_buffer_enabled"], expected)
                self.assertEqual(spec, {"game": "test-game", **options})
                if expected:
                    env.unwrapped.game.add_game_args.assert_called_once_with("+snd_efx 0")
                else:
                    env.unwrapped.game.add_game_args.assert_not_called()

    def test_vizdoom_audio_capture_owns_observation_and_preserves_rate_metadata(self) -> None:
        adapter = VizDoomAdapter.__new__(VizDoomAdapter)
        pcm = np.array([[1, -1], [2, -2]], dtype="int16")
        variables = np.array([100, 50])
        game = Mock()
        game.get_audio_sampling_rate.return_value = 44100
        game.is_audio_buffer_enabled.return_value = True
        env = SimpleNamespace(state=SimpleNamespace(audio_buffer=pcm), game=game)
        adapter.env = SimpleNamespace(unwrapped=env)
        sound = adapter.sound()
        np.testing.assert_array_equal(sound.pcm, pcm)
        self.assertEqual(sound.sample_rate, 44100)
        captured = adapter.capture({"audio": pcm, "gamevariables": variables}, {})
        pcm.fill(0)
        variables.fill(0)
        np.testing.assert_array_equal(captured.variables["audio"], [[1, -1], [2, -2]])
        np.testing.assert_array_equal(captured.variables["gamevariables"], [100, 50])
        self.assertEqual(adapter.block_extra(), {"audio_sampling_rate": 44100})
        env.state.audio_buffer = None
        self.assertIsNone(adapter.sound())
        env.state = None
        self.assertIsNone(adapter.sound())
        game.is_audio_buffer_enabled.return_value = False
        self.assertIsNone(adapter.block_extra())


class AudioControlTests(unittest.TestCase):
    @staticmethod
    def _run_cli(curriculum: list[dict], *args: str) -> list[dict]:
        """Return the effective curriculum passed to the session."""
        with (patch.object(sys, "argv", ["fmri_play.py", *args]),
              patch.object(fmri_play, "build_demo_curriculum", return_value=deepcopy(curriculum)),
              patch.object(fmri_play, "Display") as display,
              patch.object(fmri_play, "Audio") as audio,
              patch.object(fmri_play, "Session") as session):
            fmri_play.main()
            session.return_value.run.assert_called_once_with()
            display.return_value.close.assert_called_once_with()
            audio.return_value.close.assert_called_once_with()
            return session.call_args.args[1]

    def test_cli_mutes_all_games_without_changing_recording_or_other_phases(self) -> None:
        curriculum = [{"type": "game", "backend": "retro", "audio": True},
                      {"type": "game", "backend": "future-backend"},
                      {"type": "game", "audio": False,
                       "env_kwargs": {"audio_buffer_enabled": True}},
                      {"type": "message", "text": "Instructions", "audio": True},
                      {"type": "fixation", "duration": 2}]
        expected = deepcopy(curriculum)
        for phase in expected[:3]:
            phase["audio"] = False
        self.assertEqual(self._run_cli(curriculum, "--no-audio"), expected)

    def test_cli_without_mute_preserves_per_game_preferences(self) -> None:
        curriculum = [{"type": "game", "audio": True},
                      {"type": "game", "audio": False}, {"type": "game"}]
        self.assertEqual(self._run_cli(curriculum), curriculum)


class SessionAudioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = Session.__new__(Session)
        self.session.audio = Mock()
        self.session.display = Mock()
        self.session.logger = Mock()
        self.session.logger.save_game_block.return_value = "data/block.npz"
        self.session.clock = Clock()
        self.session.clock.trigger()
        self.adapter = Mock()
        self.adapter.spec = {}
        self.adapter.reset.return_value = (None, {})
        self.adapter.step.return_value = (None, 1.0, True, False, {})
        self.adapter.capture.return_value = FrameState()
        self.adapter.sound.return_value = Sound(np.ones((2, 1), dtype="int16"), 44100)
        self.adapter.keyspec = SingleKeySpec(combos={}, noop=0)
        self.adapter.block_extra.return_value = None
        for target, value in [("fmri_gym.session.get_adapter", self.adapter),
                              ("fmri_gym.session.held_key_names", set()),
                              ("pygame.event.get", [])]:
            patcher = patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _game(self, **options: object) -> None:
        phase = {"type": "game", "backend": "future-backend", "game": "test-game",
                 "mode": "episode", "n_episodes": 1, "max_duration": 1, "fps": 1000,
                 **options}
        self.session._game(phase, 0)

    def test_each_episode_stops_audio_before_next_reset(self) -> None:
        calls = Mock()
        calls.attach_mock(self.adapter.reset, "reset")
        calls.attach_mock(self.session.audio.stop, "stop")
        self._game(n_episodes=2)
        self.assertEqual([call[0] for call in calls.mock_calls], ["reset", "stop", "reset", "stop"])
        self.assertEqual(self.session.audio.play.call_count, 4)

    def test_escape_stops_audio_for_both_input_modes(self) -> None:
        escape = SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_ESCAPE)
        for turn_based in (False, True):
            self.session.audio.reset_mock()
            with (self.subTest(turn_based=turn_based),
                  patch("pygame.event.get", return_value=[escape]),
                  self.assertRaises(KeyboardInterrupt)):
                self._game(turn_based=turn_based)
            self.session.audio.stop.assert_called_once_with()
            self.adapter.step.assert_not_called()

    def test_engine_failures_stop_audio_and_propagate(self) -> None:
        for stage in ("reset", "render", "sound", "step"):
            self.session.audio.reset_mock()
            with (self.subTest(stage=stage),
                  patch.object(self.adapter, stage, side_effect=RuntimeError(f"{stage} failed")),
                  self.assertRaisesRegex(RuntimeError, f"{stage} failed")):
                self._game()
            self.session.audio.stop.assert_called_once_with()

    def test_muted_future_backend_still_captures_audio_without_reading_sound(self) -> None:
        self.adapter.spec = {"audio": False}
        pcm = np.ones((2, 1), dtype="int16")
        self.adapter.capture.return_value = FrameState(variables={"audio": pcm})
        self._game(audio=False)
        self.adapter.sound.assert_not_called()
        self.assertTrue(all(call.args == (None,) for call in self.session.audio.play.call_args_list))
        frames = self.session.logger.save_game_block.call_args.args[3]
        np.testing.assert_array_equal(frames["variables"]["audio"][0], pcm)

    def test_session_failure_outside_game_stops_audio_and_saves_manifest(self) -> None:
        self.session.outdir = "unused-test-output"
        self.session.curriculum = []
        with (patch.object(self.session, "_trigger", side_effect=RuntimeError("trigger failed")),
              self.assertRaisesRegex(RuntimeError, "trigger failed")):
            self.session.run()
        self.session.audio.stop.assert_called_once_with()
        self.session.logger.save_manifest.assert_called_once_with()

    def test_stop_failure_still_saves_session_manifest(self) -> None:
        self.session.outdir = "unused-test-output"
        self.session.curriculum = []
        self.session.audio.stop.side_effect = RuntimeError("device stop failed")
        with (patch.object(self.session, "_trigger"),
              patch("fmri_gym.session.time.sleep"),
              self.assertRaisesRegex(RuntimeError, "device stop failed")):
            self.session.run()
        self.session.logger.save_manifest.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
