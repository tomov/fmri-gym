"""Exercise PCM ownership and device lifetime without opening a sound device.

Host callbacks and game steps need not request matching sample counts. Distinct
sample values expose repeated or discarded tails across those boundaries.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

import numpy as np

from fmri_gym.adapters.base import Sound
from fmri_gym.audio import Audio, SoundDeviceGameBlockStream


class _OutputStream:
    """Record device lifetime and let a test request actual callback output."""

    def __init__(self, fail_start: bool, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.fail_start = fail_start
        self.starts = 0
        self.closes = 0
        self.started = False

    def start(self) -> None:
        self.starts += 1
        if self.fail_start:
            raise RuntimeError("device failed to start")
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closes += 1
        self.started = False

    def read(self, frames: int) -> np.ndarray:
        """Fill sentinel output through the same callback PortAudio invokes."""
        output = np.full((frames, self.kwargs["channels"]), 99,
                         dtype=self.kwargs["dtype"])
        self.kwargs["callback"](output, frames, None, None)
        return output


class _DeviceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.devices: list[_OutputStream] = []
        self.fail_start = False
        device_patch = patch("fmri_gym.audio.sounddevice.OutputStream", self._make_device)
        query_patch = patch("fmri_gym.audio.sounddevice.query_devices",
                            side_effect=lambda *args: {"name": "test output"} if args else [])
        device_patch.start()
        query_patch.start()
        self.addCleanup(device_patch.stop)
        self.addCleanup(query_patch.stop)

    def _make_device(self, **kwargs: Any) -> _OutputStream:
        device = _OutputStream(self.fail_start, **kwargs)
        self.devices.append(device)
        return device

    @staticmethod
    def _samples(*values: float, dtype: str = "float32") -> np.ndarray:
        return np.array(values, dtype=dtype).reshape(-1, 1)


class StreamTests(_DeviceTestCase):
    def _stream(self, dtype: str = "float32") -> tuple[SoundDeviceGameBlockStream, _OutputStream]:
        stream = SoundDeviceGameBlockStream(100, channels=1, dtype=dtype)
        self.addCleanup(stream.close)
        return stream, self.devices[-1]

    def test_initial_silence_then_variable_callbacks_preserve_all_samples(self) -> None:
        stream, device = self._stream()
        stream.put(self._samples(1, 2, 3, 4))
        stream.put(self._samples(5))
        stream.put(self._samples(6, 7, 8))
        stream.play()
        output = np.concatenate([device.read(n) for n in (8, 4, 1, 3, 5)])
        expected = self._samples(*([0] * 10), *range(1, 9), 0, 0, 0)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(device.read(2), self._samples(0, 0))

    def test_put_owns_noncontiguous_reusable_engine_buffers(self) -> None:
        stream, device = self._stream()
        stream.flush()
        source = np.arange(8, dtype="float32").reshape(4, 2)[:, :1]
        stream.put(source)
        source.fill(99)
        stream.play()
        device.read(10)
        np.testing.assert_array_equal(device.read(4), self._samples(0, 2, 4, 6))

    def test_empty_chunks_are_ignored_and_invalid_formats_rejected(self) -> None:
        stream, device = self._stream()
        stream.flush()
        stream.put(self._samples())
        invalid = [np.zeros(2, dtype="float32"), np.zeros((2, 2), dtype="float32"),
                   np.zeros((2, 1, 1), dtype="float32"), self._samples(1, dtype="int16")]
        for block in invalid:
            with self.subTest(shape=block.shape, dtype=block.dtype), self.assertRaises(ValueError):
                stream.put(block)
        stream.put(self._samples(7))
        stream.play()
        device.read(10)
        np.testing.assert_array_equal(device.read(2), self._samples(7, 0))

    def test_queue_overflow_keeps_current_tail_and_newest_pending_chunks(self) -> None:
        stream, device = self._stream()
        stream.flush()
        stream.put(self._samples(1, 2, 3))
        stream.play()
        device.read(10)
        np.testing.assert_array_equal(device.read(1), self._samples(1))
        for value in range(10, 50):
            stream.put(self._samples(value))
        expected = self._samples(2, 3, *range(18, 50), 0)
        np.testing.assert_array_equal(device.read(len(expected)), expected)

    def test_stop_discards_current_and_pending_audio_before_restart(self) -> None:
        stream, device = self._stream()
        stream.flush()
        stream.put(self._samples(1, 2))
        stream.put(self._samples(3))
        stream.play()
        device.read(10)
        device.read(1)
        with self.assertRaises(RuntimeError):
            stream.flush()
        stream.stop()
        self.assertFalse(device.started)
        np.testing.assert_array_equal(device.read(2), self._samples(0, 0))
        stream.put(self._samples(8))
        stream.play()
        np.testing.assert_array_equal(device.read(13), self._samples(*([0] * 10), 8, 0, 0))

    def test_unsigned_audio_uses_midpoint_silence(self) -> None:
        stream, device = self._stream(dtype="uint8")
        np.testing.assert_array_equal(device.read(1), self._samples(128, dtype="uint8"))
        stream.put(self._samples(130, dtype="uint8"))
        stream.play()
        np.testing.assert_array_equal(device.read(12),
                                      self._samples(*([128] * 10), 130, 128, dtype="uint8"))

    def test_close_releases_device_once_and_rejects_reuse(self) -> None:
        stream, device = self._stream()
        stream.play()
        stream.play()
        self.assertEqual(device.starts, 1)
        stream.close()
        stream.close()
        stream.stop()
        self.assertEqual(device.closes, 1)
        self.assertFalse(device.started)
        with self.assertRaises(RuntimeError):
            stream.play()
        with self.assertRaises(RuntimeError):
            stream.put(self._samples(1))


class AudioTests(_DeviceTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.audio = Audio()
        self.addCleanup(self.audio.close)

    def test_none_and_empty_chunks_wait_for_first_audio_without_interrupting(self) -> None:
        self.audio.play(None)
        self.audio.play(Sound(self._samples(), 100))
        self.assertEqual(self.devices, [])
        self.audio.play(Sound(self._samples(1, 2), 100))
        self.audio.play(None)
        self.audio.play(Sound(np.zeros((0, 2), dtype="int16"), 44100))
        device, = self.devices
        self.assertTrue(device.started)
        np.testing.assert_array_equal(device.read(12), self._samples(*([0] * 10), 1, 2))

    def test_chunk_length_does_not_reopen_native_fractional_rate_stream(self) -> None:
        self.audio.play(Sound(self._samples(1, 2, 3), 100.25))
        self.audio.play(Sound(self._samples(4), 100.25))
        device, = self.devices
        self.assertEqual(device.kwargs["samplerate"], 100.25)
        self.assertEqual(device.starts, 1)
        np.testing.assert_array_equal(device.read(14), self._samples(*([0] * 10), 1, 2, 3, 4))

    def test_rate_channels_and_dtype_changes_reopen_and_release_old_device(self) -> None:
        sounds = [Sound(self._samples(1), 100), Sound(self._samples(2), 200),
                  Sound(np.ones((1, 2), dtype="float32"), 200),
                  Sound(np.ones((1, 2), dtype="int16"), 200)]
        for sound in sounds:
            self.audio.play(sound)
        self.assertEqual(len(self.devices), 4)
        self.assertEqual([device.closes for device in self.devices], [1, 1, 1, 0])
        self.assertEqual(self.devices[-1].kwargs["channels"], 2)
        self.assertEqual(self.devices[-1].kwargs["dtype"], np.dtype("int16"))

    def test_stop_reuses_device_without_leaking_previous_episode_samples(self) -> None:
        self.audio.play(Sound(self._samples(1, 2), 100))
        device, = self.devices
        device.read(11)
        self.audio.stop()
        self.audio.play(Sound(self._samples(8), 100))
        self.assertEqual(len(self.devices), 1)
        self.assertEqual(device.starts, 2)
        np.testing.assert_array_equal(device.read(13), self._samples(*([0] * 10), 8, 0, 0))

    def test_startup_and_enqueue_failures_close_device_and_propagate(self) -> None:
        self.fail_start = True
        with self.assertRaisesRegex(RuntimeError, "device failed to start"):
            self.audio.play(Sound(self._samples(1), 100))
        self.assertEqual(self.devices[-1].closes, 1)
        self.assertIsNone(self.audio.stream)
        self.fail_start = False
        self.audio.play(Sound(self._samples(2), 100))
        with (patch.object(self.audio.stream, "put", side_effect=RuntimeError("queue failed")),
              self.assertRaisesRegex(RuntimeError, "queue failed")):
            self.audio.play(Sound(self._samples(3), 100))
        self.assertEqual(self.devices[-1].closes, 1)
        self.assertIsNone(self.audio.stream)

    def test_malformed_pcm_does_not_open_a_device(self) -> None:
        for pcm in (np.zeros(2), np.zeros((2, 0)), np.zeros((2, 1, 1))):
            with self.subTest(shape=pcm.shape), self.assertRaises(ValueError):
                self.audio.play(Sound(pcm, 100))
        self.assertEqual(self.devices, [])


if __name__ == "__main__":
    unittest.main()
