"""Play native game PCM without blocking the experiment or repeating old samples.

The game thread queues PCM with put(). After play(), PortAudio calls _callback()
on its own audio thread to fill one device buffer, while the game keeps running.
The callback must return promptly: it never waits for the next game step.

Game blocks and device buffers can differ in length. Each callback joins queued
blocks as needed, preserves any unconsumed tail, and fills missing samples with
silence. Copy on enqueue: engines may reuse buffers while audio is still playing.
Stop the stream before clearing buffers; close it on every episode exit.
Sounddevice is loaded only when an audio-capable adapter supplies samples.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from queue import Empty, SimpleQueue
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .adapters.base import EnvAdapter


class SoundDeviceGameBlockStream:
    """Queue PCM blocks for one episode using the game's native audio format."""

    def __init__(
        self,
        sample_rate: float,
        block_size: int = 0,
        channels: int = 2,
        dtype: str | np.dtype = "float32",
    ) -> None:
        """Open an inactive output stream.

        :param sample_rate: native samples per second.
        :param block_size: device callback size; zero lets the device choose.
        :param channels: number of PCM channels.
        :param dtype: native PCM dtype supported by sounddevice.
        """
        import sounddevice

        self._blocks: SimpleQueue[np.ndarray] = SimpleQueue()
        self._current: np.ndarray | None = None
        self._offset = 0
        self._playing = False
        self._channels = channels
        self._dtype = np.dtype(dtype)
        self._silence = 128 if self._dtype == np.dtype("uint8") else 0
        self.output_stream = sounddevice.OutputStream(
            samplerate=sample_rate, blocksize=block_size, channels=channels,
            dtype=self._dtype, latency=0.1, callback=self._callback,
        )

    def _callback(self, outdata: np.ndarray, frames: int, time: Any, status: Any) -> None:
        """Consume available samples without waiting for the next game step."""
        outdata.fill(self._silence)
        if not self._playing:
            return
        written = 0
        while written < frames:
            if self._current is None:
                self._current = self._next_block()
            if self._current is None:
                return
            count = min(len(self._current) - self._offset, frames - written)
            outdata[written:written + count] = self._current[self._offset:self._offset + count]
            written += count
            self._offset += count
            if self._offset == len(self._current):
                self._current = None
                self._offset = 0

    def _next_block(self) -> np.ndarray | None:
        """Read the queue without waiting for a producer."""
        try:
            return self._blocks.get_nowait()
        except Empty:
            return None

    def put(self, block: np.ndarray | None) -> None:
        """Copy a PCM block for asynchronous playback; ignore absent samples.

        :param block: ``(samples, channels)`` array in the stream's native dtype,
            or None when a game has no new audio (including terminal states).
        :raises ValueError: if the block has the wrong shape or dtype.
        """
        if block is None:
            return
        block = np.asarray(block)
        if block.ndim != 2 or block.shape[1] != self._channels:
            raise ValueError(f"audio must have shape (samples, {self._channels})")
        if block.dtype != self._dtype:
            raise ValueError(f"audio dtype must stay {self._dtype}, got {block.dtype}")
        if len(block):
            self._blocks.put(block.copy(order="C"))

    def play(self) -> None:
        """Start playback of queued samples."""
        self._playing = True
        self.output_stream.start()

    def stop(self) -> None:
        """Stop playback and discard queued and partially consumed samples."""
        self._playing = False
        self.output_stream.stop()
        self.flush()

    def flush(self) -> None:
        """Discard samples while stopped, so an episode cannot leak into the next.

        :raises RuntimeError: if playback has not been stopped.
        """
        if self._playing:
            raise RuntimeError("stop audio before flushing its buffers")
        self._blocks = SimpleQueue()
        self._current = None
        self._offset = 0

    def close(self) -> None:
        """Release the device and discard pending audio, including on early exit."""
        self._playing = False
        try:
            self.output_stream.close()
        finally:
            self.flush()


@contextmanager
def episode_audio(adapter: EnvAdapter) -> Iterator[Callable[[], None]]:
    """Play an adapter's reset/step audio and release the device on every exit.

    :param adapter: environment opting in with ``has_audio`` and audio hooks.
    :return: context yielding a function to queue audio after each step.
    """
    stream: SoundDeviceGameBlockStream | None = None

    def play_audio() -> None:
        nonlocal stream
        if not getattr(adapter, "has_audio", False):
            return
        block = adapter.get_audio_buffer()
        if block is None:
            return
        block = np.asarray(block)
        if block.ndim != 2 or block.shape[1] == 0:
            raise ValueError("audio must have shape (samples, channels)")
        if not len(block):
            return
        if stream is None:
            stream = SoundDeviceGameBlockStream(
                adapter.get_audio_sampling_rate(), channels=block.shape[1], dtype=block.dtype,
            )
            stream.put(block)
            stream.play()
            return
        stream.put(block)

    try:
        play_audio()
        yield play_audio
    finally:
        if stream is not None:
            stream.close()
