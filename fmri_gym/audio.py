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

The caller pushes numpy blocks with :meth:`put` on another thread, each
with the time it should start: its frame's flip plus a constant delay. Sound
cannot start at the flip itself -- the chunk only exists once the frame is
computed, and the device needs time to play it -- so the delay is measured
from the device once, at start-up, and logged. The callback places each
chunk by the DAC time PortAudio reports for its buffer, so the sound stays
that far behind the picture however the game loop or the device clock wander.
"""

from __future__ import annotations

import glob
import os
import platform
import time
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

import sounddevice  # noqa: E402  (needs the ALSA env above at import time)

PLAYING = 1
STOPPED = 2

_MAX_OFFSET = 0.003                     # smoothed onset error tolerated before a resync
_SMOOTHING = 0.1                        # of the onset error, per chunk
_MAX_STRETCH = 2e-3                     # 3.5 cents of pitch, 2 ms of correction per second
_GAIN = 0.1                             # of the smoothed error, corrected per chunk
_BLOCKSIZE = 256                        # samples per callback
_DELAY_STEP = 0.010                     # the audio delay is rounded up to this


def _resample(pcm: np.ndarray, n: int) -> np.ndarray:
    """Linearly resample ``pcm`` to ``n`` samples, keeping its dtype.

    :param pcm: array shaped ``(samples, channels)``.
    :param n: target number of samples.
    :return: a new C-contiguous array shaped ``(n, channels)``.
    """
    x = np.linspace(0, len(pcm) - 1, n)
    xp = np.arange(len(pcm))
    out = np.column_stack([np.interp(x, xp, pcm[:, c]) for c in range(pcm.shape[1])])
    if np.issubdtype(pcm.dtype, np.integer):
        out = np.round(out)
    return np.ascontiguousarray(out.astype(pcm.dtype))


def _device_lead(device: int | str | None, blocksize: int, seconds: float = 0.5) -> float:
    """Play silence and measure how far ahead of the DAC the callback runs.

    PortAudio's own ``latency`` figure is an estimate that can be well under
    the real delay (5.8 ms reported vs 14 ms measured on a PipeWire desktop),
    so the delay is chosen from what the callbacks report.

    :param device: PortAudio output device.
    :param blocksize: samples per callback, as the session will use.
    :param seconds: how long to run.
    :return: the longest ``outputBufferDacTime - currentTime`` seen, in seconds.
    :raises RuntimeError: if the device reports no DAC times, so sound cannot
        be placed against the flips.
    """
    leads: list[float] = []

    def callback(outdata: np.ndarray, frames: int, timing: Any, status: Any) -> None:
        outdata.fill(0)
        leads.append(timing.outputBufferDacTime - timing.currentTime
                     if timing.outputBufferDacTime else float("nan"))

    with sounddevice.OutputStream(device=device, blocksize=blocksize, latency="low",
                                  channels=1, callback=callback):
        time.sleep(seconds)
    settled = leads[len(leads) // 4:]       # skip the start-up transient
    if not settled or np.isnan(settled).any():
        raise RuntimeError("audio: this output reports no DAC timestamps, so sound cannot "
                           "be placed against the flips; make another output the system "
                           "default (python -m sounddevice lists them), or run with "
                           "--no-audio")
    return max(settled)


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
    """Play PCM chunks at given times, at their native rate.

    Each chunk carries its onset on the stream clock. The callback knows when
    the first sample of every buffer reaches the DAC (``outputBufferDacTime``),
    so it knows where in the buffer that onset falls. A chunk that follows the
    previous one plays right after it, gapless, while the smoothed error
    between onsets and gapless starts stays within :attr:`max_offset`. Past
    it, the chunk is placed at its onset instead (a resync), with silence
    before it or its already-past samples cut. Late sound is dropped by time,
    never allowed to push the whole stream later.

    Gapless chunks drift off their onsets when the game loop and the sound card
    disagree about time. The callback measures that error and :meth:`put`
    resamples the next chunks by up to :data:`_MAX_STRETCH` to steer it back to
    zero, so resyncs stay for stalls.
    """

    def __init__(
        self,
        sample_rate: float,
        block_size: int,
        channels: int,
        dtype: str | np.dtype,
        device: int | str | None,
        max_offset: float,
        log: dict[str, Any],
    ) -> None:
        """Open an inactive stream.

        :param sample_rate: native samples per second, including fractional rates.
        :param block_size: samples per callback.
        :param channels: number of PCM channels.
        :param dtype: native sample format supported by sounddevice.
        :param device: PortAudio output device (index or name).
        :param max_offset: seconds the smoothed onset error of gapless sound may
            reach before a resync (:attr:`max_offset`, changeable).
        :param log: where the callback records ``onsets`` (``(chunk, perf_counter
            start)`` pairs), ``resyncs`` and ``trimmed`` samples; owned by
            :class:`Audio`, so a block's log survives the stream being reopened.
        """
        self.rate = sample_rate
        self.max_offset = max_offset
        # (pcm, onset on the stream clock). Append/popleft are thread-safe.
        self._blocks: deque[tuple[np.ndarray, float, int]] = deque()
        self._current: np.ndarray | None = None
        self._offset = 0
        self._buffer = 0                    # callbacks so far
        self._end: tuple[int, int] | None = None   # (buffer, index) just after the last chunk
        self._error = 0.0                   # smoothed onset - gapless start, samples
        self._carry = 0.0                   # fractional samples still to stretch
        self.clock_offset = 0.0             # perf_counter - stream clock
        self.log = log
        self.status = STOPPED
        print("audio out:", sounddevice.query_devices(device, "output")["name"],
              f"({sample_rate:g} Hz, {channels}ch, {np.dtype(dtype).name})")
        self.output_stream = sounddevice.OutputStream(
            samplerate=sample_rate, blocksize=block_size, latency="low",
            device=device, channels=channels, callback=self.callback, dtype=dtype,
            prime_output_buffers_using_stream_callback=False,
        )

    def callback(self, outdata: np.ndarray, frames: int, timing: Any, status: Any) -> None:
        """Fill one device buffer with the chunks due in it, without waiting.

        :param outdata: writable device buffer shaped ``(frames, channels)``.
        :param frames: number of sample frames requested.
        :param timing: PortAudio times; ``outputBufferDacTime`` is when
            ``outdata[0]`` plays.
        :param status: PortAudio status flags (unused).
        """
        outdata.fill(0)
        self._buffer += 1
        if self.status != PLAYING:
            return
        t0 = timing.outputBufferDacTime
        pos = 0
        while pos < frames:
            if self._current is None:
                pos = self._take_due(t0, pos, frames)
                if self._current is None:
                    return
            count = min(len(self._current) - self._offset, frames - pos)
            outdata[pos:pos + count] = self._current[self._offset:self._offset + count]
            pos += count
            self._offset += count
            if self._offset == len(self._current):
                self._current = None
                self._end = (self._buffer + 1, 0) if pos == frames else (self._buffer, pos)

    def _take_due(self, t0: float, pos: int, frames: int) -> int:
        """Make the next chunk due in this buffer current; return where it starts.

        :param t0: DAC time of this buffer's first sample.
        :param pos: first free sample of the buffer.
        :param frames: buffer length.
        :return: the buffer index the current chunk starts at (``pos`` if none).
        """
        follows = self._end == (self._buffer, pos)
        while self._blocks:
            pcm, onset, chunk = self._blocks[0]
            want = round((onset - t0) * self.rate)
            error = self._gapless_error(want, pos, follows)
            if want >= frames and error is None:
                return pos                      # due in a later buffer
            self._blocks.popleft()
            if error is None and self._end is not None:   # sound was playing: cut
                self.log["resyncs"] += 1
            start = want if error is None else pos
            self._error = 0.0 if error is None else error
            first = t0 + start / self.rate + self.clock_offset
            if start < pos:                     # late: its first samples are past
                self.log["trimmed"] += min(pos - start, len(pcm))
                pcm, start = pcm[pos - start:], pos
            if len(pcm):
                self.log["onsets"].append((chunk, first))
                self._current, self._offset = pcm, 0
                return start
        return pos

    def _gapless_error(self, want: int, pos: int, follows: bool) -> float | None:
        """The smoothed onset error if a chunk plays gapless, else ``None``.

        A chunk that follows on is taken as soon as the previous one ends, even
        when its onset falls in a later buffer, so a late onset never opens a
        gap unless it is a resync.

        :param want: buffer index of the chunk's onset.
        :param pos: buffer index just after the previous chunk.
        :param follows: whether the previous chunk ended exactly at ``pos``.
        :return: the error in samples, including this chunk, or ``None`` to
            place the chunk at its onset.
        """
        if not follows:
            return None
        error = self._error + _SMOOTHING * (want - pos - self._error)
        return error if abs(error) <= self.max_offset * self.rate else None

    def put(self, block: np.ndarray, onset: float, chunk: int) -> None:
        """Queue a copy of one chunk (engines reuse their buffers).

        Chunks are stretched towards the callback's last error, positive when
        gapless sound starts before its onsets, so a lengthened chunk pushes
        the next ones later.

        :param block: array shaped ``(samples, channels)`` in the stream's dtype.
        :param onset: ``perf_counter`` time its first sample should play.
        :param chunk: its number in the block, for :attr:`log`.
        """
        n = len(block)
        self._carry += max(-_MAX_STRETCH * n, min(_MAX_STRETCH * n, _GAIN * self._error))
        extra = round(self._carry)
        self._carry -= extra
        pcm = _resample(block, n + extra) if extra else block.copy(order="C")
        self._blocks.append((pcm, onset - self.clock_offset, chunk))

    def play(self) -> None:
        """Start the stream and map ``perf_counter`` onto its clock."""
        self._current, self._offset, self._end = None, 0, None
        self._error = self._carry = 0.0
        self.status = PLAYING
        self.output_stream.start()
        before = time.perf_counter()
        stream_time = self.output_stream.time
        self.clock_offset = (before + time.perf_counter()) / 2 - stream_time

    def stop(self) -> None:
        """Stop callbacks and discard queued and partially consumed samples."""
        self.status = STOPPED
        self.output_stream.stop()
        self._blocks.clear()
        self._current, self._offset, self._end = None, 0, None

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

    def __init__(self, enabled: bool = True) -> None:
        """Open the system's output, measure its delay and choose the audio delay.

        No stream is opened until :meth:`play`. The output is the desktop mixer
        (:func:`_preferred_output_device`), i.e. wherever the system plays sound:
        pick the rig's output there, as for any other program.

        :param enabled: ``False`` (``--no-audio``) never touches a sound device,
            so a machine without one can run; a block that then has sound to
            play raises.
        :raises RuntimeError: if there is no usable output, or it reports no
            DAC timestamps.
        """
        self.enabled = enabled
        self._max_offset = _MAX_OFFSET
        self.stream: SoundDeviceGameBlockStream | None = None
        self.format: tuple | None = None
        self._period: float | None = None   # seconds per step, while being checked
        #: number of the chunk queued by the latest :meth:`play` (-1: none), for the log.
        self.last_chunk = -1
        self._queued = 0                    # chunks queued in this block
        self._log: dict[str, Any] = {"onsets": [], "resyncs": 0, "trimmed": 0}
        self._sound_s = 0.0                 # sound produced by the checked steps
        self._steps = 0
        if enabled:
            self._open_output()

    def _open_output(self) -> None:
        """Resolve the output, measure its delay and set :attr:`delay` from it.

        :raises RuntimeError: if there is no usable output, or it reports no
            DAC timestamps.
        """
        self.device = _preferred_output_device()
        try:
            info = sounddevice.query_devices(self.device, "output")
        except (ValueError, sounddevice.PortAudioError) as e:
            raise RuntimeError(f"audio: no usable output device ({e}); "
                               "run with --no-audio for a silent session") from e
        self.device_name = info["name"]
        self.samplerate = info["default_samplerate"]
        self.hostapi = sounddevice.query_hostapis(info["hostapi"])["name"]
        self.lead = _device_lead(self.device, _BLOCKSIZE)
        needed = self.lead + _BLOCKSIZE / self.samplerate
        #: seconds from a frame's flip to its sound's first sample at the DAC.
        self.delay = float((np.ceil(needed / _DELAY_STEP) + 1) * _DELAY_STEP)

    def status(self) -> str:
        """One line for the experimenter screen and the console.

        :return: e.g. ``"out: default (ALSA) | sound at flip + 40 ms (device
            delay 16.4 ms)"``.
        """
        if not self.enabled:
            return ("off (--no-audio): playback muted in every game block, no output "
                    "opened; logged game audio is unchanged")
        return (f"out: {self.device_name} ({self.hostapi}) | sound at flip + "
                f"{self.delay * 1000:.0f} ms (device delay {self.lead * 1000:.1f} ms)")

    def describe(self) -> dict[str, Any]:
        """The output and the delay it got, for the manifest.

        :return: a JSON-serializable dict.
        """
        if not self.enabled:
            return {"enabled": False}
        return {"enabled": True, "device": self.device_name, "hostapi": self.hostapi,
                "device_delay_ms": self.lead * 1000, "delay_ms": self.delay * 1000}

    def start(self, *, frame_period: float | None, flip_period: float | None) -> None:
        """Begin a game block: fit placement to the display, check the frame rate.

        Sound can follow the flips only if each step makes about one frame
        period of it; resampling covers :data:`_MAX_STRETCH`, not a game running
        at the wrong speed. The check runs over the first second of sound.

        :param frame_period: ``1 / fps``, or ``None`` for a turn-based block,
            whose steps are not paced.
        :param flip_period: the refresh period of a vsync-locked display, or
            ``None``. Frames land on refreshes, so their onsets jitter by up to
            one; half of it widens :data:`_MAX_OFFSET` to match.
        """
        self._max_offset = _MAX_OFFSET + (flip_period or 0.0) / 2
        self._period, self._sound_s, self._steps = frame_period, 0.0, 0
        self._queued = 0
        self._log = {"onsets": [], "resyncs": 0, "trimmed": 0}
        if self.stream is not None:
            self.stream.max_offset = self._max_offset
            self.stream.log = self._log

    def block_log(self, chunks: list[int]) -> dict[str, Any]:
        """The block's sound timing for its npz, or ``{}`` if it played none.

        :param chunks: per frame, :attr:`last_chunk` after that frame's flip.
        :return: ``audio_onset`` (``perf_counter`` each frame's sound started
            playing; NaN if it had none, or it never played), ``audio_delay_ms``,
            ``audio_resyncs`` and ``audio_trimmed_samples``.
        """
        if max(chunks, default=-1) < 0:
            return {}
        started = dict(self._log["onsets"])
        return {"audio_onset": np.array([started.get(c, np.nan) for c in chunks]),
                "audio_delay_ms": self.delay * 1000,
                "audio_resyncs": self._log["resyncs"],
                "audio_trimmed_samples": self._log["trimmed"]}

    def _check_rate(self, sound: Sound) -> None:
        """Add one step's sound to the block's rate check; raise once it is off.

        The first chunk after the stream (re)starts is skipped: the first step
        of an episode can carry what the engine made during its reset.

        :param sound: the step's sound.
        :raises ValueError: if the sound per step and the frame period differ
            by more than :data:`_MAX_STRETCH`.
        """
        if self._period is None or self.stream.status != PLAYING:
            return
        self._sound_s += len(sound.pcm) / sound.sample_rate
        self._steps += 1
        if self._sound_s < 1.0:
            return
        per_step = self._sound_s / self._steps
        period, self._period = self._period, None
        if abs(per_step / period - 1) > _MAX_STRETCH:
            raise ValueError(
                f"audio: the game makes {per_step * 1000:.3f} ms of sound per step but "
                f"steps every {period * 1000:.3f} ms ({per_step / period - 1:+.2%}); "
                f"sound can follow the flips only within {_MAX_STRETCH:.1%}. "
                f"Set this block's fps to {1 / per_step:.4f} (ViZDoom: fps * frame_skip "
                "must be 35), or set \"audio\": false")

    def play(self, sound: Sound | None, flip_t: float) -> None:
        """Queue one chunk to start :attr:`delay` after its frame's flip.

        :param sound: a :class:`~fmri_gym.adapters.base.Sound`, or ``None`` for
            a frame with nothing to play (which leaves playback alone, rather
            than cutting off what is still queued).
        :param flip_t: ``perf_counter`` of the flip that showed the frame.
        :raises ValueError: if the PCM is not shaped ``(samples, channels)``,
            or its amount per step does not match the block's frame rate
            (:meth:`start`).
        :raises RuntimeError: if the output is off (``--no-audio``).
        """
        self.last_chunk = -1
        if sound is None or not len(sound.pcm):
            return
        if not self.enabled:
            raise RuntimeError("audio: a game block has sound to play but the output is off "
                               '(--no-audio); set "audio": false on that phase')
        pcm = sound.pcm
        sound_format = (sound.sample_rate, pcm.shape[1:], pcm.dtype)
        if sound_format != self.format:
            if pcm.ndim != 2:
                raise ValueError(f"audio must be shaped (samples, channels), got {pcm.shape}")
            self.close()
            self.stream = SoundDeviceGameBlockStream(
                sound.sample_rate, _BLOCKSIZE, pcm.shape[1], pcm.dtype,
                self.device, self._max_offset, self._log)
            self.format = sound_format
        self._check_rate(sound)
        if self.stream.status != PLAYING:
            self.stream.play()
        self.last_chunk = self._queued
        self._queued += 1
        self.stream.put(pcm, flip_t + self.delay, self.last_chunk)

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
