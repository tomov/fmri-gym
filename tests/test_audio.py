"""Exercise audio ownership and episode cleanup without opening a sound device.

The callback receives arbitrary frame counts: game-buffer boundaries cannot be
assumed to match them. Distinct sample values expose lost or repeated audio.
"""

from __future__ import annotations

import sys
import time
import unittest
from collections import defaultdict
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pygame

from fmri_gym.audio import SoundDeviceGameBlockStream, episode_audio
from fmri_gym.session import Session


class _OutputStream:
    """Record device lifetime and expose the real callback for deterministic reads."""

    def __init__(self, fail_start: bool, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.fail_start = fail_start
        self.started = False
        self.closed = False

    def start(self) -> None:
        if self.fail_start:
            raise RuntimeError("device failed to start")
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True
        self.started = False

    def read(self, frames: int) -> np.ndarray:
        """Fill a sentinel buffer so unwritten callback output is visible."""
        output = np.full((frames, self.kwargs["channels"]), 99,
                         dtype=self.kwargs["dtype"])
        self.kwargs["callback"](output, frames, None, None)
        return output


class AudioTests(unittest.TestCase):
    """Cover queued playback and the optional per-episode audio capability."""

    def setUp(self) -> None:
        self.devices: list[_OutputStream] = []
        self.fail_start = False
        sounddevice = SimpleNamespace(OutputStream=self._make_device)
        self.device_patch = patch.dict(sys.modules, {"sounddevice": sounddevice})
        self.device_patch.start()
        self.addCleanup(self.device_patch.stop)

    def _make_device(self, **kwargs: Any) -> _OutputStream:
        device = _OutputStream(self.fail_start, **kwargs)
        self.devices.append(device)
        return device

    def _stream(
        self, channels: int = 1, dtype: str = "float32",
    ) -> tuple[SoundDeviceGameBlockStream, _OutputStream]:
        stream = SoundDeviceGameBlockStream(44100, channels=channels, dtype=dtype)
        self.addCleanup(stream.close)
        return stream, self.devices[-1]

    @staticmethod
    def _samples(*values: float, dtype: str = "float32") -> np.ndarray:
        return np.array(values, dtype=dtype).reshape(-1, 1)

    @staticmethod
    def _adapter(*buffers: np.ndarray | None) -> SimpleNamespace:
        pending = iter(buffers)
        return SimpleNamespace(has_audio=True, get_audio_sampling_rate=lambda: 44100,
                               get_audio_buffer=lambda: next(pending))

    def test_mono_preserves_current_buffer_when_queue_is_empty(self) -> None:
        stream, device = self._stream()
        self.assertEqual(device.kwargs["channels"], 1)
        stream.put(self._samples(1, 2, 3, 4))
        stream.play()
        np.testing.assert_array_equal(device.read(2), self._samples(1, 2))
        np.testing.assert_array_equal(device.read(3), self._samples(3, 4, 0))

    def test_callback_combines_buffers_and_silences_underrun(self) -> None:
        stream, device = self._stream()
        stream.put(self._samples(1, 2))
        stream.put(self._samples(3))
        stream.play()
        np.testing.assert_array_equal(device.read(5), self._samples(1, 2, 3, 0, 0))
        np.testing.assert_array_equal(device.read(2), self._samples(0, 0))

    def test_put_copies_reusable_engine_buffers(self) -> None:
        stream, device = self._stream()
        source = self._samples(1, 2)
        stream.put(source)
        source.fill(9)
        stream.play()
        np.testing.assert_array_equal(device.read(2), self._samples(1, 2))

    def test_none_and_empty_buffers_do_not_interrupt_playback(self) -> None:
        stream, device = self._stream()
        stream.put(None)
        stream.put(self._samples())
        stream.put(self._samples(7))
        stream.play()
        np.testing.assert_array_equal(device.read(2), self._samples(7, 0))

    def test_put_rejects_malformed_buffers(self) -> None:
        stream, _ = self._stream()
        invalid = [np.zeros(2, dtype="float32"), np.zeros((2, 2), dtype="float32"),
                   np.zeros((2, 1, 1), dtype="float32")]
        for block in invalid:
            with self.subTest(shape=block.shape), self.assertRaises(ValueError):
                stream.put(block)
        with self.assertRaises((TypeError, ValueError)):
            stream.put(self._samples(1, dtype="int16"))

    def test_stop_clears_current_and_queued_audio_before_restart(self) -> None:
        stream, device = self._stream()
        stream.put(self._samples(1, 2))
        stream.put(self._samples(3))
        stream.play()
        device.read(1)
        stream.stop()
        self.assertFalse(device.started)
        np.testing.assert_array_equal(device.read(2), self._samples(0, 0))
        stream.put(self._samples(8))
        stream.play()
        np.testing.assert_array_equal(device.read(3), self._samples(8, 0, 0))

    def test_flush_requires_stopped_device_and_discards_pending_audio(self) -> None:
        stream, device = self._stream()
        stream.put(self._samples(1, 2))
        stream.play()
        device.read(1)
        with self.assertRaises(RuntimeError):
            stream.flush()
        np.testing.assert_array_equal(device.read(1), self._samples(2))
        stream.stop()
        stream.put(self._samples(3))
        stream.flush()
        stream.play()
        np.testing.assert_array_equal(device.read(3), self._samples(0, 0, 0))

    def test_unsigned_audio_uses_midpoint_silence(self) -> None:
        stream, device = self._stream(dtype="uint8")
        np.testing.assert_array_equal(device.read(1), self._samples(128, dtype="uint8"))
        stream.put(self._samples(130, dtype="uint8"))
        stream.play()
        np.testing.assert_array_equal(device.read(3),
                                      self._samples(130, 128, 128, dtype="uint8"))

    def test_close_releases_device_and_is_repeatable(self) -> None:
        stream, device = self._stream()
        stream.play()
        stream.close()
        stream.close()
        self.assertTrue(device.closed)
        self.assertFalse(device.started)

    def test_silent_adapter_does_not_need_sounddevice_or_audio_hooks(self) -> None:
        with patch.dict(sys.modules, {"sounddevice": None}):
            for adapter in (SimpleNamespace(has_audio=False), SimpleNamespace()):
                with episode_audio(adapter) as play_audio:
                    play_audio()
        self.assertEqual(self.devices, [])

    def test_episode_plays_reset_buffer_and_closes_normally(self) -> None:
        with episode_audio(self._adapter(self._samples(1), None)) as play_audio:
            device, = self.devices
            self.assertTrue(device.started)
            np.testing.assert_array_equal(device.read(1), self._samples(1))
            play_audio()
            np.testing.assert_array_equal(device.read(1), self._samples(0))
        self.assertTrue(device.closed)

    def test_episode_waits_for_first_nonempty_buffer(self) -> None:
        with episode_audio(self._adapter(None, self._samples(), self._samples(5))) as play_audio:
            self.assertEqual(self.devices, [])
            play_audio()
            self.assertEqual(self.devices, [])
            play_audio()
            device, = self.devices
            np.testing.assert_array_equal(device.read(2), self._samples(5, 0))
        self.assertTrue(device.closed)

    def test_episode_closes_on_interrupt_or_step_failure(self) -> None:
        for error in (KeyboardInterrupt, RuntimeError):
            with (self.subTest(error=error), self.assertRaises(error),
                  episode_audio(self._adapter(self._samples(1)))):
                raise error("episode failed")
            self.assertTrue(self.devices[-1].closed)

    def test_episode_closes_device_when_startup_fails(self) -> None:
        self.fail_start = True
        with (self.assertRaisesRegex(RuntimeError, "device failed to start"),
              episode_audio(self._adapter(self._samples(1)))):
            self.fail("startup failure must propagate")
        self.assertTrue(self.devices[-1].closed)

    def _game_adapter(self) -> Mock:
        adapter = Mock(has_audio=True)
        adapter.get_audio_sampling_rate.return_value = 44100
        adapter.get_audio_buffer.return_value = self._samples(1)
        adapter.keyspec.key_to_action_map.return_value = {}
        return adapter

    @staticmethod
    def _run_episode(adapter: Any, turn_based: bool = False) -> bool:
        session = Session.__new__(Session)
        session.display = Mock()
        frames = defaultdict(list)
        return session._episode(
            adapter, frames, seed=1, episode_id=0, turn_based=turn_based,
            dt=0.01, state_stride=1, block_end=time.perf_counter() + 1,
        )

    def test_session_escape_closes_audio_for_both_input_modes(self) -> None:
        escape = SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_ESCAPE)
        for turn_based in (False, True):
            adapter = self._game_adapter()
            with (self.subTest(turn_based=turn_based),
                  patch("pygame.event.get", return_value=[escape])):
                self.assertTrue(self._run_episode(adapter, turn_based))
                adapter.step.assert_not_called()
                self.assertTrue(self.devices[-1].closed)

    def test_session_render_and_step_failures_close_audio(self) -> None:
        for stage in ("render", "step"):
            adapter = self._game_adapter()
            getattr(adapter, stage).side_effect = RuntimeError(f"{stage} failed")
            with (self.subTest(stage=stage), self.assertRaisesRegex(RuntimeError, stage),
                  patch("pygame.event.get", return_value=[]),
                  patch("fmri_gym.session.held_key_names", return_value=set())):
                self._run_episode(adapter)
            self.assertTrue(self.devices[-1].closed)


if __name__ == "__main__":
    unittest.main()
