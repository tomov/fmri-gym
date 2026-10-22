"""What a run showed and played, taken from inside it, to hold against what it logged.

The log is the product of a session, and the claim it makes is that it holds
what the subject saw and heard. :class:`Tap` checks that claim from the
other end: it wraps the four places in the run where a picture or a sound
leaves the program -- :meth:`Display.draw_frame`, :meth:`Audio.play`, the
stream's :meth:`~fmri_gym.audio.SoundDeviceGameBlockStream.put` and the
PortAudio callback that fills the device's buffers -- and keeps a fingerprint
of every frame drawn, the PCM of every chunk queued, and every buffer the
device was handed. Nothing in the run changes: each wrapper calls the real
method and only looks.

What it can say afterwards (:meth:`Tap.report`, written as JSON beside the
run): for every logged frame, the frame that was on screen when it was
logged; for every chunk, the PCM handed to the speakers; and for every chunk
that started playing, whether the samples the device was given from its
onset on are the samples queued (cut short only where the next chunk or an
episode's ``stop`` cut it). The stream may stretch a chunk by a sample or two
to follow the display's clock (:data:`~fmri_gym.audio._MAX_STRETCH`), so the
PCM queued and the PCM played are compared after that stretch, and how many
chunks were stretched is reported, not hidden.

:class:`FakeOutputStream` is a sound card for a machine without one (a CI
runner): it calls the callback on its own thread, a buffer at a time on the
``perf_counter`` clock, as PortAudio would, so the whole audio path runs and
is tapped there too.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import types
from typing import Any

import numpy as np

#: Seconds the fake card's callback runs ahead of its DAC: a desktop-ish figure.
FAKE_LEAD = 0.010


def fingerprint(array: Any) -> str:
    """A short digest of an array's shape, dtype and bytes."""
    a = np.ascontiguousarray(array)
    h = hashlib.blake2b(digest_size=12)
    h.update(f"{a.shape}{a.dtype}".encode())
    h.update(a.tobytes())
    return h.hexdigest()


class FakeOutputStream:
    """A sound card on the ``perf_counter`` clock, for :mod:`sounddevice`'s ``OutputStream``.

    Buffer ``k`` is asked for ``FAKE_LEAD`` seconds before it plays, and plays
    ``blocksize / samplerate`` seconds after buffer ``k - 1``.
    """

    def __init__(self, *, samplerate: float = 48000, blocksize: int = 256, channels: int = 1,
                 callback: Any = None, dtype: Any = "float32", **_: Any) -> None:
        self.samplerate = float(samplerate)
        self.blocksize = blocksize or 256
        self.channels = channels
        self.dtype = np.dtype(dtype)
        self._callback = callback
        self._running = False
        self._thread: threading.Thread | None = None

    @property
    def time(self) -> float:
        return time.perf_counter()

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        period = self.blocksize / self.samplerate
        due = time.perf_counter()
        while self._running:
            wait = due - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            out = np.zeros((self.blocksize, self.channels), dtype=self.dtype)
            now = time.perf_counter()
            timing = types.SimpleNamespace(outputBufferDacTime=due + FAKE_LEAD,
                                           currentTime=now)
            self._callback(out, self.blocksize, timing, None)
            due += period

    def stop(self) -> None:
        self._running = False
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join()

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> FakeOutputStream:
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def use_fake_sound_card() -> None:
    """Point :mod:`fmri_gym.audio` at :class:`FakeOutputStream` (before the run opens audio)."""
    from fmri_gym import audio

    stub = {"name": "fake", "index": 0, "hostapi": 0, "max_output_channels": 2,
            "default_samplerate": 48000.0}

    def query_devices(device: Any = None, kind: Any = None) -> Any:
        return stub if (device is not None or kind is not None) else [stub]

    audio.sounddevice.OutputStream = FakeOutputStream
    audio.sounddevice.query_devices = query_devices
    hostapi = {"name": "fake", "default_output_device": 0}
    audio.sounddevice.query_hostapis = lambda index=None: hostapi if index is not None else [hostapi]


class _StreamTap:
    """One opened stream's buffers, and where in them each chunk started."""

    def __init__(self) -> None:
        self.buffers: dict[int, np.ndarray] = {}    # callback number -> what it wrote
        self.starts: list[tuple[int, int, int, np.ndarray]] = []  # (buffer, index, chunk, pcm)
        self.stops: list[int] = []                  # callback number at each stop()
        self.stretched = 0                          # chunks queued longer or shorter
        self.frames = 0                             # samples per callback


class Tap:
    """Fingerprints of what a run drew and played, by the frame and chunk the log gives them."""

    def __init__(self) -> None:
        self.draws: list[str] = []          # every draw_frame, in order
        #: frame number -> (the draw on screen when it was logged, the frame handed to the log)
        self.logged: dict[int, tuple[int, str]] = {}
        #: per block (in order): chunk number -> fingerprint of the PCM handed to Audio.play
        self.queued: list[dict[int, str]] = []
        self.streams: list[_StreamTap] = []
        self._lock = threading.Lock()

    def install(self) -> None:
        """Wrap the display, the logger and the audio output, for the rest of the process."""
        from fmri_gym import audio, display, logging

        tap = self
        draw_frame = display.Display.draw_frame
        log_frame = logging.Logger.log_frame
        open_block = logging.Logger.open_block
        play = audio.Audio.play
        init = audio.SoundDeviceGameBlockStream.__init__
        callback = audio.SoundDeviceGameBlockStream.callback
        put = audio.SoundDeviceGameBlockStream.put
        take_due = audio.SoundDeviceGameBlockStream._take_due
        stop = audio.SoundDeviceGameBlockStream.stop

        def draw_frame_(self, rgb, *args, **kwargs):
            tap.draws.append(fingerprint(rgb))
            return draw_frame(self, rgb, *args, **kwargs)

        def log_frame_(self, fields, frame=None, state=None):
            if frame is not None:
                tap.logged[self.n_frames] = (len(tap.draws) - 1, fingerprint(frame))
            return log_frame(self, fields, frame, state)

        def open_block_(self, *args, **kwargs):
            tap.queued.append({})
            return open_block(self, *args, **kwargs)

        def play_(self, sound, flip_t):
            play(self, sound, flip_t)
            if self.last_chunk >= 0:
                tap.queued[-1][self.last_chunk] = fingerprint(sound.pcm)

        def init_(self, *args, **kwargs):
            self._tap = _StreamTap()
            tap.streams.append(self._tap)
            init(self, *args, **kwargs)

        def callback_(self, outdata, frames, timing, status):
            callback(self, outdata, frames, timing, status)
            self._tap.frames = frames
            self._tap.buffers[self._buffer] = outdata.copy()

        def put_(self, block, onset, chunk):
            put(self, block, onset, chunk)
            if len(self._blocks[-1][0]) != len(block):
                self._tap.stretched += 1

        def take_due_(self, t0, pos, frames):
            n = len(self.log["onsets"])
            start = take_due(self, t0, pos, frames)
            if len(self.log["onsets"]) > n:
                self._tap.starts.append((self._buffer, start, self.log["onsets"][-1][0],
                                         self._current.copy()))
            return start

        def stop_(self):
            self._tap.stops.append(self._buffer)
            stop(self)

        display.Display.draw_frame = draw_frame_
        logging.Logger.log_frame = log_frame_
        logging.Logger.open_block = open_block_
        audio.Audio.play = play_
        audio.SoundDeviceGameBlockStream.__init__ = init_
        audio.SoundDeviceGameBlockStream.callback = callback_
        audio.SoundDeviceGameBlockStream.put = put_
        audio.SoundDeviceGameBlockStream._take_due = take_due_
        audio.SoundDeviceGameBlockStream.stop = stop_

    def played(self) -> dict[str, int]:
        """Each started chunk against the samples its buffers were given, from its onset on.

        A chunk is compared up to where the next chunk started, the stream
        stopped, or the recorded buffers end, whichever is first.

        :return: chunks and samples compared, and the chunks whose samples differ.
        """
        out = {"chunks": 0, "samples": 0, "mismatched_chunks": 0, "stretched_chunks": 0,
               "first_mismatch": None}
        for s in self.streams:
            out["stretched_chunks"] += s.stretched
            if not s.buffers:
                continue
            f = s.frames
            starts = [b * f + i for b, i, _, _ in s.starts]   # absolute sample positions
            for k, (b, i, chunk, pcm) in enumerate(s.starts):
                end = starts[k] + len(pcm)
                if k + 1 < len(starts):
                    end = min(end, starts[k + 1])
                stop = next((x for x in s.stops if x >= b), None)
                if stop is not None:
                    end = min(end, (stop + 1) * f)
                got = _samples(s.buffers, f, starts[k], end)
                if got is None:
                    continue
                n = len(got)
                out["chunks"] += 1
                out["samples"] += n
                if not np.array_equal(got, pcm[:n]):
                    out["mismatched_chunks"] += 1
                    out["first_mismatch"] = out["first_mismatch"] or f"chunk {chunk}"
        return out

    def report(self) -> dict:
        """Everything a test compares with the log, JSON-able."""
        return {"draws": self.draws,
                "logged": {str(k): list(v) for k, v in self.logged.items()},
                "queued": [{str(k): v for k, v in q.items()} for q in self.queued],
                "played": self.played()}

    def save(self, path: str) -> None:
        """Write :meth:`report` to ``path``."""
        with open(path, "w") as f:
            json.dump(self.report(), f)


def _samples(buffers: dict[int, np.ndarray], frames: int, a: int, b: int) -> np.ndarray | None:
    """Samples ``a`` to ``b`` of the stream, or ``None`` if a buffer between is missing."""
    if b <= a:
        return None
    first, last = a // frames, (b - 1) // frames
    parts = []
    for k in range(first, last + 1):
        if k not in buffers:
            return None
        parts.append(buffers[k])
    joined = np.concatenate(parts)
    return joined[a - first * frames:b - first * frames]


def compare(block: str, report: dict, block_number: int = 0) -> list[str]:
    """What one logged block holds against what the run showed and queued.

    :param block: the block's folder.
    :param report: a :meth:`Tap.report`, read back.
    :param block_number: which of the run's game blocks this is, in order.
    :return: one line per disagreement; empty when the log is what was shown and queued.
    """
    from fmri_gym.logging import AUDIO_FILENAME, read_audio, read_events, read_frames

    problems = []
    pixels, index = read_frames(block)
    for row, frame in enumerate(index):
        drawn = report["logged"].get(str(int(frame)))
        if drawn is None:
            problems.append(f"frame {frame} is in frames.h5 but was never logged by the run")
            continue
        draw, handed = drawn
        stored = fingerprint(pixels[row])
        if stored != handed:
            problems.append(f"frame {frame}: frames.h5 differs from the frame handed to the log")
        if draw < 0 or report["draws"][draw] != stored:
            problems.append(f"frame {frame}: frames.h5 differs from the frame on screen")
    queued = {int(k): v for k, v in report["queued"][block_number].items()}
    chunks = [e["audio_chunk"] for e in read_events(block)
              if e["type"] == "frame" and e.get("audio_chunk", -1) >= 0]
    if not chunks and not queued:
        return problems
    if not os.path.exists(os.path.join(block, AUDIO_FILENAME)):
        return problems + [f"{len(queued)} chunks were queued but audio.h5 is missing"]
    samples, _rate, offsets = read_audio(block)
    ends = np.append(offsets[1:], len(samples))
    if sorted(chunks) != sorted(queued):
        problems.append(f"the frames name chunks {_span(chunks)}, the speakers got "
                        f"{_span(sorted(queued))}")
    if len(offsets) != len(queued):
        problems.append(f"audio.h5 holds {len(offsets)} chunks, the speakers got {len(queued)}")
    for c in chunks:
        if c < len(offsets) and fingerprint(samples[offsets[c]:ends[c]]) != queued.get(c):
            problems.append(f"chunk {c}: audio.h5 differs from the PCM handed to the speakers")
            break
    return problems


def _span(numbers: list[int]) -> str:
    return f"{numbers[0]}..{numbers[-1]} ({len(numbers)})" if numbers else "none"
