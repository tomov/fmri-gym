"""Streaming PCM output via a PortAudio callback (sounddevice).

Two layers. :class:`Audio` is what the experiment loop holds -- the speakers,
exactly as :class:`~fmri_gym.display.Display` is the monitor -- and it takes
whatever :meth:`~fmri_gym.adapters.base.EnvAdapter.sound` produced, engine
unseen. :class:`SoundDeviceGameBlockStream` below it is the PortAudio plumbing;
another way of getting sound out (a different library, a file writer, a
scanner-safe device) is another class with the same ``play``/``stop``.

In that lower layer, :meth:`~SoundDeviceGameBlockStream.play` starts the stream;
PortAudio then calls :meth:`~SoundDeviceGameBlockStream.callback` on
its own realtime thread (not from the caller). Each callback fills one
host buffer (``outdata``) with the next slice of samples and returns
immediately — it does not play a whole clip in one go. Keep the callback
short: if it blocks or the queue runs dry, the buffer underruns and you
hear silence or clicks.

The caller pushes numpy blocks with :meth:`put` on another thread; the
callback pulls from the queue. A queued block is often longer or shorter
than one host buffer, so the callback's inner loop splices across blocks
until that single slot is full, then returns while the caller keeps
running.
"""

from __future__ import annotations

import glob
import os
import platform
from collections import deque
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .adapters.base import Sound

# conda's bundled alsa-lib only ships raw-hardware PCM definitions, so
# PortAudio enumerates `hw:*` cards and nothing else -- its "default" then
# lands on whatever card is first (often a USB mic dongle with no speakers,
# or one that is busy and fails to open). Pointing alsa-lib at the system
# config adds the distro's `pulse`/`default` plugin PCMs, which route to
# whatever the desktop is playing through. Must happen before sounddevice
# imports, i.e. before PortAudio initialises. All no-ops off Linux, where
# PortAudio talks to CoreAudio / WASAPI and its default device is already the
# one the desktop uses.
_SYSTEM_ALSA_CONF = "/usr/share/alsa/alsa.conf"
_SYSTEM_ALSA_PLUGIN_DIRS = ("/usr/lib/*-linux-gnu/alsa-lib", "/usr/lib*/alsa-lib")
if "ALSA_CONFIG_PATH" not in os.environ and os.path.exists(_SYSTEM_ALSA_CONF):
    # Those plugin PCMs are shared objects (libasound_module_pcm_pulse.so)
    # that conda's alsa-lib would look for in its own libdir, so point it at
    # the distro's -- wherever this distro/arch keeps it.
    # A multiarch box has several of these (x86_64 and i386); only the one
    # built for this interpreter's arch can actually be loaded.
    plugin_dirs = [d for pattern in _SYSTEM_ALSA_PLUGIN_DIRS
                   for d in sorted(glob.glob(pattern))
                   if os.path.isdir(d) and glob.glob(
                       os.path.join(d, "libasound_module_pcm_pulse.so"))]
    plugin_dirs.sort(key=lambda d: platform.machine() not in d)
    if plugin_dirs:
        os.environ["ALSA_CONFIG_PATH"] = _SYSTEM_ALSA_CONF
        os.environ.setdefault("ALSA_PLUGIN_DIR", plugin_dirs[0])

import sounddevice  # Needs the ALSA environment above before PortAudio initializes.

# Stream lifecycle. ``NOT_STARTED`` is unused: we construct already
# ``STOPPED`` and only start the PortAudio stream in :meth:`play`.
NOT_STARTED = 0
PLAYING = 1
STOPPED = 2


def _preferred_output_device() -> int | None:
    """Pick the desktop's mixer PCM, so sound goes where the subject hears it.

    :return: index of the first available ``pulse``/``default`` output device,
        or ``None`` to let PortAudio choose (typically a raw ALSA card).
    """
    try:
        devices = sounddevice.query_devices()
    except sounddevice.PortAudioError:  # no audio at all -- let the stream raise instead
        return None
    for name in ("pulse", "default"):
        for device in devices:
            if device["name"] == name and device["max_output_channels"] > 0:
                return int(device["index"])
    return None


class SoundDeviceGameBlockStream:
    """Queue owned PCM blocks for nonblocking playback at their native rate."""

    def __init__(
        self,
        sample_rate: float,
        block_size: int = 0,
        channels: int = 2,
        dtype: str | np.dtype = sounddevice.default.dtype[1],
    ) -> None:
        """Open an inactive stream with at most 32 pending engine chunks.

        :param sample_rate: native samples per second, including fractional rates.
        :param block_size: device callback size; zero lets the host choose.
        :param channels: number of PCM channels.
        :param dtype: native sample format supported by sounddevice.
        """
        self._channels = channels
        self._dtype = np.dtype(dtype)
        self._silence = 128 if self._dtype == np.dtype("uint8") else 0
        # Append/popleft are thread-safe; a full deque drops its oldest pending
        # chunk instead of allowing a slow output device to accumulate audio.
        self._blocks: deque[np.ndarray] = deque(maxlen=32)
        self._current: np.ndarray | None = None
        self._offset = 0
        self.status = STOPPED
        self._closed = False
        self._prime = np.full((int(0.1 * sample_rate), channels), self._silence, dtype=dtype)
        device = _preferred_output_device()
        print("audio out:", sounddevice.query_devices(device, "output")["name"],
              f"({sample_rate:g} Hz, {channels}ch, {self._dtype.name})")
        self.output_stream = sounddevice.OutputStream(
            samplerate=sample_rate, blocksize=block_size, latency=0.1,
            device=device, channels=channels, callback=self.callback, dtype=self._dtype,
            prime_output_buffers_using_stream_callback=False,
        )

    def callback(self, outdata: np.ndarray, frames: int, time: Any, status: Any) -> None:
        """Fill one device buffer without waiting for the game thread.

        :param outdata: writable device buffer shaped ``(frames, channels)``.
        :param frames: number of sample frames requested.
        :param time: PortAudio timing information (unused).
        :param status: PortAudio status flags (unused).
        """
        outdata.fill(self._silence)
        if self.status != PLAYING:
            return
        written = 0
        while written < frames:
            if self._current is None:
                try:
                    self._current = self._blocks.popleft()
                except IndexError:
                    return
            count = min(len(self._current) - self._offset, frames - written)
            outdata[written:written + count] = self._current[self._offset:self._offset + count]
            written += count
            self._offset += count
            if self._offset == len(self._current):
                self._current = None
                self._offset = 0

    def put(self, block: np.ndarray) -> None:
        """Copy native PCM for asynchronous playback; skip empty chunks.

        :param block: array shaped ``(samples, channels)`` in the stream's dtype.
        :raises ValueError: if shape or dtype does not match the stream.
        :raises RuntimeError: if the device has already been closed.
        """
        if self._closed:
            raise RuntimeError("cannot queue audio on a closed stream")
        block = np.asarray(block)
        if block.ndim != 2 or block.shape[1] != self._channels:
            raise ValueError(f"audio must have shape (samples, {self._channels})")
        if block.dtype != self._dtype:
            raise ValueError(f"audio dtype must stay {self._dtype}, got {block.dtype}")
        if len(block):
            self._blocks.append(block.copy(order="C"))

    def play(self) -> None:
        """Start playback once; close a device whose startup fails.

        :raises RuntimeError: if the device has already been closed.
        """
        if self._closed:
            raise RuntimeError("cannot start a closed audio stream")
        if self.status == PLAYING:
            return
        # Each episode starts with the same slack, ahead of any real PCM.
        self._current = self._prime if len(self._prime) else None
        self._offset = 0
        self.status = PLAYING
        try:
            self.output_stream.start()
        except BaseException:
            self.close()
            raise

    def stop(self) -> None:
        """Stop callbacks and discard queued and partially consumed samples."""
        if self._closed:
            return
        self.status = STOPPED
        try:
            self.output_stream.stop()
        finally:
            self.flush()

    def flush(self) -> None:
        """Drop pending samples while callbacks are stopped.

        :raises RuntimeError: if playback is still running.
        """
        if self.status == PLAYING:
            raise RuntimeError("stop audio before flushing its buffers")
        self._blocks.clear()
        self._current = None
        self._offset = 0

    def close(self) -> None:
        """Release the device once, even when stopping or playback failed."""
        if self._closed:
            return
        self.status = STOPPED
        try:
            self.output_stream.close()
        finally:
            self._closed = True
            self.flush()


class Audio:
    """The session's speakers: plays whatever an adapter's ``sound`` returns.

    Engine-agnostic in the same way :class:`~fmri_gym.display.Display` is: it
    receives PCM chunks and never learns which game made them. The output
    stream is opened on the first chunk, because only the chunk says what
    format to open it in, and is reopened if a later block plays at a different
    rate, channel count, or sample format.
    """

    def __init__(self) -> None:
        """Create a silent output; no device is opened until :meth:`play`."""
        self.stream: SoundDeviceGameBlockStream | None = None
        self.format: tuple | None = None

    def play(self, sound: Sound | None) -> None:
        """Queue one chunk of PCM, starting the output if it is not running.

        :param sound: a :class:`~fmri_gym.adapters.base.Sound`, or ``None`` for
            a frame with nothing to play (which leaves playback alone, rather
            than cutting off what is still queued).
        """
        if sound is None:
            return
        pcm = np.asarray(sound.pcm)
        if pcm.ndim != 2 or pcm.shape[1] == 0:
            raise ValueError("audio must have shape (samples, channels)")
        if not len(pcm):
            return
        # Chunk LENGTH is only a hint to PortAudio -- the callback splices
        # across queued blocks -- so a shorter final chunk must not count as a
        # format change and reopen the device mid-episode.
        sound_format = (sound.sample_rate, pcm.shape[1], pcm.dtype)
        if sound_format != self.format:
            self.close()
            self.stream = SoundDeviceGameBlockStream(
                sound.sample_rate, pcm.shape[0], pcm.shape[1], dtype=pcm.dtype)
            self.format = sound_format
        try:
            self.stream.put(pcm)
            if self.stream.status != PLAYING:
                self.stream.play()
        except BaseException:
            self.close()
            raise

    def stop(self) -> None:
        """Stop playback and drop what is still queued.

        Called at the end of every episode, so a death scream does not carry
        over into the next episode or the fixation cross after the block.
        """
        if self.stream is not None:
            self.stream.stop()

    def close(self) -> None:
        """Release the output device (the audio half of a session teardown)."""
        stream = self.stream
        self.stream = None
        self.format = None
        if stream is not None:
            stream.close()
