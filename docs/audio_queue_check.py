"""Check what SoundDeviceGameBlockStream actually puts on the wire.

The audio path has no test a person can run by listening in a container, and
its one interesting piece -- the callback that splices queued chunks into
PortAudio's slots -- is exactly the piece that only misbehaves under a producer
that goes quiet. Doom queues a chunk every frame and never exposes it; crafter
queues one only when something happened, and did. Run this from the repo root
after touching fmri_gym/audio.py:

    python docs/audio_queue_check.py

There is no PortAudio here and none is needed. A stub ``sounddevice`` records
how the stream was opened, and this file plays the part PortAudio's realtime
thread plays: ask ``callback`` for its frames, over and over, while a producer
queues chunks on the game's clock in between.

Ground truth without a microphone: every queued chunk carries a per-sample
fingerprint (chunk number x 100000 + offset + 1), so each output sample names
the chunk and the offset it came from, and 0 means the callback wrote silence.
Whole, once, in order and unbroken is then a property of the output that can be
read off it rather than argued about. Each chunk is queued with the onset it
should play at, so where its first sample landed in the output is a second
property to read off: the placement error below. The device modelled is an
ideal one whose buffer k plays the moment buffer k-1 is done, which is the only
part a stub can honestly stand in for; a real card's own delay is what
``Audio.lead`` measures at start-up.

The stream also reports where it put each chunk (``log["onsets"]``), and a
recording is analysed against that report rather than against the samples, so
the report agreeing with the samples is checked too. A sparse producer resyncs
on nearly every cue and that is the design: a cue that does not follow the
previous one is placed at its own onset, which the stream counts as a resync.
Crafter's six cues below therefore show five, and Doom's continuous tics none.

Last run 2026-09-24 (onset-scheduled stream): 5 cases, 0 defects.
"""

import importlib.util
import sys
import types

import numpy as np

RATE = 44100
DTYPE = np.int32
FINGERPRINT = 100000

_opened: list[dict] = []


class _FakeStream:
    """Stands in for sounddevice.OutputStream; records the open, plays nothing."""

    def __init__(self, **kwargs):
        _opened.append(kwargs)

    @property
    def time(self):
        """The stream clock. 0, so this file's timeline IS the stream's."""
        return 0.0

    def start(self):
        pass

    def stop(self):
        pass

    def close(self):
        pass


_sd = types.ModuleType("sounddevice")
_sd.default = types.SimpleNamespace(dtype=["float32", "float32"])
_sd.OutputStream = _FakeStream
_sd.query_devices = lambda device=None, kind=None: (
    {"name": "stub", "index": 0, "max_output_channels": 2}
    if (device is not None or kind is not None)
    else [{"name": "stub", "index": 0, "max_output_channels": 2}])
sys.modules["sounddevice"] = _sd

# Loaded by path rather than as fmri_gym.audio: the package __init__ imports
# pygame, and the point of this check is that it needs no hardware at all.
_spec = importlib.util.spec_from_file_location(
    "fmri_gym_audio_under_test", "fmri_gym/audio.py")
_audio = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audio)

MAX_OFFSET = _audio._MAX_OFFSET
# How far ahead of the DAC a callback runs. A real card's figure is measured at
# start-up (`_device_lead`); 10 ms is a desktop-ish value to model one with.
LEAD = 0.010


def delay_for(block_size: int) -> float:
    """The flip-to-sound delay :class:`Audio` would choose for this device.

    Onsets are the reason the checks below pass at all: sound is queued for
    ``flip + delay``, always far enough ahead of the DAC that the callback
    which must place it has not run yet.

    :param block_size: samples per callback.
    :return: seconds from a frame's flip to its sound's first sample.
    """
    needed = LEAD + block_size / RATE
    return float((np.ceil(needed / _audio._DELAY_STEP) + 1) * _audio._DELAY_STEP)


def chunk(number: int, n: int) -> np.ndarray:
    """``n`` samples that each name which chunk and which offset they are."""
    return (number * FINGERPRINT + np.arange(n) + 1).astype(DTYPE).reshape(n, 1)


def open_stream(block_size: int):
    """Open a stub-backed stream and a fresh log for it.

    :param block_size: samples per callback, 0 to let the host API pick.
    :return: ``(stream, log)``.
    """
    _opened.clear()
    log = {"onsets": [], "resyncs": 0, "trimmed": 0}
    stream = _audio.SoundDeviceGameBlockStream(
        RATE, block_size, 1, DTYPE, device=None, max_offset=MAX_OFFSET, log=log)
    stream.play()
    # play() maps perf_counter onto the stream clock; here they are the same
    # clock, so onsets below are plain seconds from the start of the block.
    stream.clock_offset = 0.0
    return stream, log


def run(block_size: int, puts: list, seconds: float, host_default: int = 1024):
    """Drive the callback over a timeline of producer puts.

    :param block_size: what :meth:`Audio.play` passes PortAudio.
    :param puts: ``(flip_s, label, n_samples)`` per queued chunk, time-ordered.
    :param seconds: how long a stream to drive.
    :param host_default: slot size to use when ``block_size`` is 0, i.e. the
        size the host API would have picked for itself.
    :return: ``(frames_per_callback, output samples, log, delay)``.
    """
    stream, log = open_stream(block_size)
    frames = _opened[0]["blocksize"] or host_default
    delay = delay_for(frames)
    out, t, i = [], 0.0, 0
    while t < seconds:
        # The producer queues a chunk when the frame that made it was shown,
        # for that flip plus the device's delay.
        while i < len(puts) and puts[i][0] <= t:
            stream.put(chunk(i + 1, puts[i][2]), puts[i][0] + delay, i + 1)
            i += 1
        slot = np.zeros((frames, 1), dtype=DTYPE)
        # An ideal device, running LEAD ahead of its DAC: this buffer's first
        # sample plays when every sample already handed over has played.
        stream.callback(slot, frames,
                        types.SimpleNamespace(outputBufferDacTime=t + LEAD), None)
        out.append(slot)
        t += frames / RATE
    return frames, np.concatenate(out).ravel(), log, delay


def report(title: str, block_size: int, puts: list, seconds: float) -> int:
    """Play one timeline; print the defects and a summary. Returns defects."""
    frames, y, log, delay = run(block_size, puts, seconds)
    print(f"\n{title}")
    print(f"  opened blocksize={block_size}, "
          f"{frames} frames a callback ({frames / RATE * 1000:.0f} ms), "
          f"sound at flip + {delay * 1000:.0f} ms")
    number = np.where(y > 0, (y - 1) // FINGERPRINT, 0)
    offset = np.where(y > 0, (y - 1) % FINGERPRINT, -1)
    said = dict(log["onsets"])
    defects, errors = 0, []
    for k, (t_flip, label, n) in enumerate(puts, start=1):
        t_put = t_flip + delay
        at = np.flatnonzero(number == k)
        if at.size == 0:
            print(f"    {label} ({n} smp @ {t_flip:.2f}s): NEVER PLAYED")
            defects += 1
            continue
        heard = offset[at]
        problems = []
        if at.size != n or np.unique(heard).size != n:
            problems.append(f"{at.size} samples out of {n}")
        if np.any(np.diff(heard) <= 0):
            problems.append("restarts mid-cue")
        pieces = int(np.count_nonzero(np.diff(at) != 1)) + 1
        if pieces > 1:
            problems.append(f"split across {pieces} pieces")
        played = LEAD + at[0] / RATE     # output sample 0 reaches the DAC at LEAD
        errors.append((played - t_put) * 1000)
        # What the stream logged is what analysis will use, so it has to be
        # where the samples actually went, to well within a callback.
        if abs(said.get(k, float("nan")) - played) > frames / RATE:
            problems.append(f"logged onset {said.get(k)} but played at {played:.4f}s")
        if problems:
            defects += 1
            print(f"    {label} ({n} smp @ {t_flip:.2f}s): " + "; ".join(problems))
    print(f"    {len(puts) - defects}/{len(puts)} whole, once, in order, unbroken"
          f"   |   placement error {min(errors):+.1f} to {max(errors):+.1f} ms"
          f"   |   {log['resyncs']} resyncs, {log['trimmed']} samples trimmed")
    return defects


def episode_boundary() -> int:
    """stop() must drop the sound that was half-played, not just the queue.

    The cue has to be genuinely in flight when stop() lands, or this checks
    nothing, so drain a few buffers first and say so out loud if it never
    started.
    """
    print("\nepisode boundary: stop() mid-cue, then the next episode's first cue")
    stream, _ = open_stream(0)
    stream.put(chunk(1, 30000), LEAD, 1)
    started, t = 0, 0.0
    for _ in range(8):
        slot = np.zeros((1024, 1), DTYPE)
        stream.callback(slot, 1024,
                        types.SimpleNamespace(outputBufferDacTime=t + LEAD), None)
        started += int(np.count_nonzero(slot > 0))
        t += 1024 / RATE
    if not (0 < started < 30000):
        print(f"    SETUP FAILED: {started}/30000 samples played, the cue was"
              " not mid-flight when stopped, so nothing was tested")
        return 1
    stream.stop()
    stream.play()
    stream.clock_offset = 0.0
    stream.put(chunk(2, 2000), t + LEAD, 2)
    out = np.zeros((4096, 1), DTYPE)
    stream.callback(out, 4096,
                    types.SimpleNamespace(outputBufferDacTime=t + LEAD), None)
    leaked = int(np.count_nonzero((out > 0) & (out < 2 * FINGERPRINT)))
    print(f"    cue interrupted {started}/30000 samples in; {leaked} of its"
          f" samples leaked into the next episode:"
          f" {'clean' if leaked == 0 else 'LEAK'}")
    return int(leaked != 0)


# A plausible crafter minute at 2.5 fps: an unlock, some chopping, a refusal.
CRAFTER = [(0.4, "score", 17640), (1.2, "hit", 2646), (1.6, "hit", 2646),
           (3.6, "blocked", 4410), (4.0, "hit", 2646), (4.4, "score", 17640)]

# Doom with the audio buffer on: one equal chunk every tic, 35 a second.
DOOM = [(i / 35, f"tic{i}", 1260) for i in range(40)]

if __name__ == "__main__":
    bad = 0
    # Both games, each at the host's choice of slot and at a slot the size of
    # one of its own chunks: the splicing must not care which it gets.
    bad += report("crafter, host picks the slot", 0, CRAFTER, 8.0)
    bad += report("crafter, slot the length of a score cue", 17640, CRAFTER, 8.0)
    bad += report("doom, host picks the slot", 0, DOOM, 2.0)
    bad += report("doom, slot the length of one tic", 1260, DOOM, 2.0)
    bad += episode_boundary()
    print(f"\ndefects: {bad}")
    sys.exit(1 if bad else 0)
