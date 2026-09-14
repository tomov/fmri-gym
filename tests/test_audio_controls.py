"""Keep default playback and mute precedence consistent across game backends.

Engine creation and the CLI display/session boundary are mocked so these checks
need neither optional game packages nor an audio device or pygame window.
"""

from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import fmri_play
from fmri_gym.adapters.base import EnvAdapter
from fmri_gym.adapters.keyspec import SingleKeySpec
from fmri_gym.adapters.retro import RetroAdapter
from fmri_gym.adapters.vizdoom import VizDoomAdapter
from fmri_gym.audio import episode_audio


class _AudioAdapter(EnvAdapter):
    """Represent a future backend that only declares its native capability."""

    def _make(self, spec: dict) -> None:
        self.has_audio = True

    def _keyspec(self) -> SingleKeySpec:
        return SingleKeySpec(combos={}, noop=0)


class AudioControlTests(unittest.TestCase):
    """Exercise user preferences through adapter construction and CLI startup."""

    def test_generic_mute_skips_audio_hooks_and_optional_dependency(self) -> None:
        self.assertTrue(_AudioAdapter({}).has_audio)
        adapter = _AudioAdapter({"audio": False})
        self.assertFalse(adapter.has_audio)
        with (patch.dict(sys.modules, {"sounddevice": None}),
              patch.object(adapter, "get_audio_buffer") as read_audio,
              episode_audio(adapter) as play_audio):
            play_audio()
            read_audio.assert_not_called()
        with patch.object(_AudioAdapter, "_make", return_value=None):
            self.assertFalse(_AudioAdapter({"audio": True}).has_audio)

    def test_retro_defaults_to_audio_and_accepts_phase_mute(self) -> None:
        env = SimpleNamespace(unwrapped=SimpleNamespace(buttons=["A", "RIGHT"]))
        engine = SimpleNamespace(make=Mock(return_value=env))
        with patch.dict(sys.modules, {"stable_retro": engine}):
            audible = RetroAdapter({"game": "test-game"})
            muted = RetroAdapter({"game": "test-game", "audio": False})
        self.assertTrue(audible.has_audio)
        self.assertFalse(muted.has_audio)
        held = frozenset({"RIGHT", "Z"})
        self.assertEqual(audible.keyspec.resolve(held), muted.keyspec.resolve(held))

    def test_vizdoom_mute_precedence_reaches_engine_construction(self) -> None:
        cases = [({}, True), ({"audio": False}, False),
                 ({"env_kwargs": {"audio_buffer_enabled": False}}, False),
                 ({"audio": False, "env_kwargs": {"audio_buffer_enabled": True}}, False),
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
                adapter = VizDoomAdapter(spec)
                self.assertIs(make.call_args.kwargs["audio_buffer_enabled"], expected)
                self.assertIs(adapter.has_audio, expected)
                self.assertEqual(spec, {"game": "test-game", **options})

    @staticmethod
    def _run_cli(curriculum: list[dict], *args: str) -> list[dict]:
        """Return the effective curriculum passed to the session."""
        with (patch.object(sys, "argv", ["fmri_play.py", *args]),
              patch.object(fmri_play, "build_demo_curriculum", return_value=deepcopy(curriculum)),
              patch.object(fmri_play, "Display") as display,
              patch.object(fmri_play, "Session") as session):
            fmri_play.main()
            session.return_value.run.assert_called_once_with()
            display.return_value.close.assert_called_once_with()
            return session.call_args.args[1]

    def test_cli_mutes_every_game_and_preserves_other_phases(self) -> None:
        curriculum = [{"type": "game", "backend": "retro", "audio": True},
                      {"type": "game", "backend": "future-backend"},
                      {"type": "game", "audio": False},
                      {"type": "message", "text": "Instructions", "audio": True},
                      {"type": "fixation", "duration": 2}]
        expected = deepcopy(curriculum)
        for phase in expected[:3]:
            phase["audio"] = False
        self.assertEqual(self._run_cli(curriculum, "--no-audio"), expected)

    def test_cli_without_mute_preserves_per_game_preferences(self) -> None:
        curriculum = [{"type": "game", "audio": True},
                      {"type": "game", "audio": False},
                      {"type": "game"}]
        self.assertEqual(self._run_cli(curriculum), curriculum)


if __name__ == "__main__":
    unittest.main()
