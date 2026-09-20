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
read off it rather than argued about. The onsets printed are queue-side only:
they say when a sample entered the stream, not when a speaker moved, since the
device's own buffering is the part no stub can stand in for.

Last run 2026-09-20: 4 cases, 0 defects.
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


def chunk(number: int, n: int) -> np.ndarray:
    """``n`` samples that each name which chunk and which offset they are."""
    return (number * FINGERPRINT + np.arange(n) + 1).astype(DTYPE).reshape(n, 1)


def run(block_size: int, puts: list, seconds: float, host_default: int = 1024):
    """Drive the callback over a timeline of producer puts.

    :param block_size: what :meth:`Audio.play` passes PortAudio.
    :param puts: ``(time_s, label, n_samples)`` per queued chunk, time-ordered.
    :param seconds: how long a stream to drive.
    :param host_default: slot size to use when ``block_size`` is 0, i.e. the
        size the host API would have picked for itself.
    :return: ``(frames_per_callback, output samples)``.
    """
    _opened.clear()
    stream = _audio.SoundDeviceGameBlockStream(RATE, block_size, 1, dtype=DTYPE)
    frames = _opened[0]["blocksize"] or host_default
    stream.play()
    out, t, i = [], 0.0, 0
    while t < seconds:
        while i < len(puts) and puts[i][0] <= t:
            stream.put(chunk(i + 1, puts[i][2]))
            i += 1
        slot = np.zeros((frames, 1), dtype=DTYPE)
        stream.callback(slot, frames, None, None)
        out.append(slot)
        t += frames / RATE
    return frames, np.concatenate(out).ravel()


def report(title: str, block_size: int, puts: list, seconds: float) -> int:
    """Play one timeline; print the defects and a summary. Returns defects."""
    frames, y = run(block_size, puts, seconds)
    print(f"\n{title}")
    print(f"  opened blocksize={block_size}, "
          f"{frames} frames a callback ({frames / RATE * 1000:.0f} ms)")
    number = np.where(y > 0, (y - 1) // FINGERPRINT, 0)
    offset = np.where(y > 0, (y - 1) % FINGERPRINT, -1)
    defects, onsets = 0, []
    for k, (t_put, label, n) in enumerate(puts, start=1):
        at = np.flatnonzero(number == k)
        if at.size == 0:
            print(f"    {label} ({n} smp @ {t_put:.2f}s): NEVER PLAYED")
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
        onsets.append((at[0] / RATE - t_put) * 1000)
        if problems:
            defects += 1
            print(f"    {label} ({n} smp @ {t_put:.2f}s): "
                  + "; ".join(problems))
    print(f"    {len(puts) - defects}/{len(puts)} whole, once, in order, unbroken"
          f"   |   queued to output in {min(onsets):.0f}-{max(onsets):.0f} ms")
    return defects


def episode_boundary() -> int:
    """stop() must drop the sound that was half-played, not just the queue.

    The cue has to be genuinely in flight when stop() lands, or this checks
    nothing: the stream opens with ~100 ms of priming silence queued ahead of
    it, and a couple of callbacks go into that instead. So drain first, and say
    so out loud if the cue never started.
    """
    print("\nepisode boundary: stop() mid-cue, then the next episode's first cue")
    stream = _audio.SoundDeviceGameBlockStream(RATE, 0, 1, dtype=DTYPE)
    stream.play()
    stream.put(chunk(1, 30000))
    started = 0
    for _ in range(8):  # 8 x 1024 frames, past the 4410 samples of silence
        slot = np.zeros((1024, 1), DTYPE)
        stream.callback(slot, 1024, None, None)
        started += int(np.count_nonzero(slot > 0))
    if not (0 < started < 30000):
        print(f"    SETUP FAILED: {started}/30000 samples played, the cue was"
              " not mid-flight when stopped, so nothing was tested")
        return 1
    stream.stop()
    stream.play()
    stream.put(chunk(2, 2000))
    out = np.zeros((4096, 1), DTYPE)
    stream.callback(out, 4096, None, None)
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
