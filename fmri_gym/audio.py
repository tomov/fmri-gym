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
import logging
import os
import platform
import queue
import threading
from typing import TYPE_CHECKING

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

import sounddevice  # noqa: E402  (needs the ALSA env above at import time)

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
    except Exception:  # no audio at all -- let the stream raise instead
        return None
    for name in ("pulse", "default"):
        for device in devices:
            if device["name"] == name and device["max_output_channels"] > 0:
                return int(device["index"])
    return None


class SoundDeviceGameBlockStream:
    """Queue PCM blocks from the caller onto a PortAudio output stream."""

    def __init__(
        self,
        sample_rate: float,
        block_size: int = 0,
        channels: int = 2,
        dtype=sounddevice.default.dtype[1],
    ) -> None:
        """Open an output stream; does not start playback until :meth:`play`.

        :param sample_rate: samples per second; must match the input blocks.
        :param block_size: PortAudio block size in frames; ``0`` lets the
            host pick. When the caller knows its chunk length, pass it so
            one queued block often fills one callback.
        :param channels: channel count of queued arrays (stereo is 2).
        :param dtype: numpy / PortAudio sample dtype. Default is the
            device's output dtype (``sounddevice.default.dtype[1]``).
        """
        device = _preferred_output_device()
        # Silence is otherwise indistinguishable from a stream that opened on
        # the wrong card, so say where the sound is going.
        # ``kind`` matters: with ``device=None`` (any non-ALSA host, where
        # PortAudio's own default is already the right one) a bare
        # query_devices() would return the whole device list, not a device.
        print("audio out:", sounddevice.query_devices(device, "output")["name"],
              f"({sample_rate:g} Hz, {channels}ch, {np.dtype(dtype).name})")
        self.blocks: queue.Queue = queue.Queue()
        # ~100 ms of silence, matching ``latency=0.1`` below: queued ahead of
        # the real samples so the first callback has something to play. A
        # producer that makes one game frame of sound per game frame never
        # builds that slack up on its own, so :meth:`play` re-queues it.
        self.silence = np.zeros((int(0.1 * sample_rate), channels), dtype=dtype)
        self.blocks.put(self.silence)
        self.lock = threading.Lock()
        self.output_stream = sounddevice.OutputStream(
            samplerate=sample_rate,
            blocksize=block_size,
            latency=0.1,
            device=device,
            channels=channels,
            callback=self.callback,
            dtype=dtype,
            # Let PortAudio zero the initial buffers instead of calling us
            # before :meth:`play` (and before any real game audio is queued).
            prime_output_buffers_using_stream_callback=False,
        )
        self.current_block_idx = 0
        self.current_block = None
        self.status = STOPPED

    def callback(self, outdata, frames: int, time, status) -> None:
        """PortAudio output callback: copy queued PCM into ``outdata``.

        Runs on PortAudio's thread. Keep it short: fill ``outdata`` and
        return. Blocking here underruns and clicks.

        A queued block and ``frames`` have nothing to do with each other -- a
        block can span many callbacks and a callback can span many blocks -- so
        this splices across the queue until the slot is full. Two rules keep
        that honest for a producer that falls silent between sounds, which is
        most of a block for a game that only speaks on events:

        - a block that has started always finishes, across as many callbacks as
          it takes, whether or not anything else is queued behind it;
        - a slot that runs out of samples is completed with zeros, never with
          samples that have already been played.

        Both are the cases a producer of one chunk per frame never reaches, so
        they went unnoticed until sporadic cues arrived (see
        ``docs/audio_queue_check.py``).

        :param outdata: preallocated output array, shape ``(frames, C)``.
        :param frames: number of sample frames PortAudio wants this call.
        :param time: PortAudio timing info (unused).
        :param status: PortAudio status flags (unused).
        """
        if self.status == STOPPED:
            outdata.fill(0)
            return
        out_idx = 0
        while out_idx < frames:
            if self.current_block is None:
                with self.lock:
                    try:
                        # Never wait: this is PortAudio's realtime thread, and
                        # a block that arrives late is simply the next
                        # callback's, which costs it one slot and nothing else.
                        self.current_block = self.blocks.get_nowait()
                    except queue.Empty:
                        outdata[out_idx:] = 0
                        logging.debug("sound queue empty")
                        return
                self.current_block_idx = 0
            block = self.current_block
            take = min(block.shape[0] - self.current_block_idx, frames - out_idx)
            end = self.current_block_idx + take
            outdata[out_idx:out_idx + take] = block[self.current_block_idx:end]
            out_idx += take
            self.current_block_idx = end
            if end == block.shape[0]:
                self.current_block = None

    def put(self, block: np.ndarray) -> None:
        """Enqueue one PCM chunk from the producer thread.

        :param block: array shaped ``(n_samples, channels)``, same dtype
            as the stream. The queue holds a reference; copy first if the
            underlying buffer will be reused.
        """
        with self.lock:
            self.blocks.put(block)

    def play(self) -> None:
        """Start the PortAudio stream (idempotent if already running).

        Re-primes the queue with silence, since :meth:`stop` empties it and a
        restarted stream would otherwise run with no slack at all.
        """
        if self.blocks.empty():
            self.blocks.put(self.silence)
        self.status = PLAYING
        self.output_stream.start()

    def stop(self) -> None:
        """Stop playback and drop any queued samples.

        Flush after stop so leftover PCM is not played on a later
        :meth:`play`.
        """
        self.status = STOPPED
        self.output_stream.stop()
        self.flush()

    def flush(self) -> None:
        """Drop queued blocks, and the one that was half-played.

        Draining while the callback may still hold ``current_block`` is
        racy; a fresh ``Queue`` is the simple cutoff. The half-played block
        goes with them, or :meth:`stop` would only drop the sounds that had
        not started, and the next episode would open on the tail of the last
        one now that a started block is always finished.
        """
        self.blocks = queue.Queue()
        self.current_block = None
        self.current_block_idx = 0

    def close(self) -> None:
        """Stop playback and release the PortAudio stream."""
        self.stop()
        self.output_stream.close()


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
        pcm = sound.pcm
        # Length is not part of the format: the callback splices across queued
        # blocks, so a shorter final chunk must not count as a format change
        # and reopen the device mid-episode. It is not passed as the blocksize
        # either. Only the first chunk of a block would get a say, and it
        # would then set the callback period for the whole 300 s: a game whose
        # chunks are all one frame long would be asking for its own frame,
        # which is reasonable, but one that speaks in sounds of different
        # lengths would be setting it by whichever sound happened to come
        # first. Host's choice is the same for both and answers sooner.
        sound_format = (sound.sample_rate, pcm.shape[1], pcm.dtype)
        if sound_format != self.format:
            self.close()
            self.stream = SoundDeviceGameBlockStream(
                sound.sample_rate, 0, pcm.shape[1], dtype=pcm.dtype)
            self.format = sound_format
        if self.stream.status != PLAYING:
            self.stream.play()
        self.stream.put(pcm)

    def stop(self) -> None:
        """Stop playback and drop what is still queued.

        Called at the end of every episode, so a death scream does not carry
        over into the next episode or the fixation cross after the block.
        """
        if self.stream is not None:
            self.stream.stop()

    def close(self) -> None:
        """Release the output device (the audio half of a session teardown)."""
        if self.stream is not None:
            self.stream.close()
        self.stream = None
        self.format = None
