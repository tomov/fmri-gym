"""One run's data, on disk as it happens.

::

    <outdir>/manifest.json                            the run: subject, curriculum, one entry per phase, rig
    <outdir>/block-NN_<backend>_<game>/events.jsonl   one JSON object per line, in order
    <outdir>/block-NN_<backend>_<game>/frames.h5      the rendered frames, when the phase keeps them
    <outdir>/block-NN_<backend>_<game>/audio.h5       the queued PCM, when the block played any

There is one :class:`Logger` per run and it writes all three. The manifest is
rewritten (atomically) whenever a phase entry or a run-level field lands, so
it is current at every phase boundary. A block's lines and frames go through
a queue to a writer process that owns the file handles, flushes on a timer
and finishes on its own if the run dies: a crash mid-block loses at most the
last flush interval, and a partial last line is skipped on read. One block is
open at a time (a run plays its curriculum in order).

Every line in ``events.jsonl`` has a ``type``. The logger writes the first
and last itself -- ``block_start`` (``format``, ``version``, ``subject``,
``block_index``, ``backend``, ``game``, ``phase``: the config, ``base_seed``)
and ``block_end`` (whatever summary :meth:`Logger.close_block` is given) --
and stamps each ``frame`` with its index in the block, also ``frame``. The
other lines are the run's, verbatim (:mod:`fmri_gym.run`).

``frames.h5`` holds ``frames`` ``(N, H, W, C)``, one gzip'd chunk per frame
so any one reads alone, and ``frame_index`` ``(N,)``, the ``frame`` of each
row: every ``frame_stride``-th frame of the block (``0``: no frames at all).
SWMR mode makes a flushed frame durable the way a flushed line is.

``audio.h5`` holds ``samples``, every :meth:`Logger.log_audio` chunk
concatenated, and ``chunk_len`` ``(n_chunks,)``, so a chunk's offset into
``samples`` is a running sum of the ones before it. ``onset_chunk``/
``onset_time`` and the ``delay_ms``/``resyncs``/``trimmed_samples`` attrs land
at :meth:`Logger.close_block`, from :meth:`~.audio.Audio.block_log`.
"""

from __future__ import annotations

import base64
import json
import multiprocessing as mp
import os
import queue as queue_mod
import time
import zlib
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .run import Clock

FORMAT = "fmri-gym-log"
VERSION = 1
EVENTS_FILENAME = "events.jsonl"
FRAMES_FILENAME = "frames.h5"
AUDIO_FILENAME = "audio.h5"
#: Seconds between flushes of a block's files to the OS (not every frame).
FLUSH_INTERVAL = 1.0


class Logger:
    """Writes one run's ``manifest.json`` and its per-block logs."""

    def __init__(self, outdir: str, subject: str, curriculum: list[dict], clock: Clock) -> None:
        """Create the run's output directory, an empty manifest and the writer process.

        :param outdir: the run's folder.
        :param subject: subject identifier stored in the manifest and each block.
        :param curriculum: full curriculum list stored in the manifest.
        :param clock: the run's clock (used for trigger epoch / perf times).
        """
        self.outdir = outdir
        self.clock = clock
        os.makedirs(outdir, exist_ok=True)
        self.manifest: dict = {
            "subject": subject, "curriculum": curriculum, "start_epoch": None, "phases": [],
        }
        self._block: str | None = None
        #: frames recorded in the open block so far.
        self.n_frames = 0
        self._frame_stride = 1
        # Unbounded queue: a full one would silently stall put(), the exact
        # invisible data loss this exists to avoid.
        self._queue: mp.Queue = mp.Queue()
        self._process = mp.Process(target=_writer_main, args=(self._queue,), daemon=True)
        self._process.start()

    # -- the manifest --------------------------------------------------------

    def set_trigger_time(self) -> None:
        """Record the scanner-trigger wall and perf times on the manifest."""
        self.manifest["start_epoch"] = self.clock.t0_epoch
        self.manifest["trigger_perf"] = self.clock.t0_perf
        self.save_manifest()

    def log_phase(self, entry: dict) -> None:
        """Append one phase summary entry to the manifest and write it out.

        :param entry: phase dict (``index``, ``type``, onset/offset, …).
        """
        self.manifest["phases"].append(entry)
        self.save_manifest()

    def set_extra(self, key: str, value: Any) -> None:
        """Store a run-level entry in the manifest (e.g. trigger settings) and write it out.

        :param key: top-level manifest key.
        :param value: JSON-serializable value.
        """
        self.manifest[key] = value
        self.save_manifest()

    def save_manifest(self) -> str:
        """Write ``manifest.json`` atomically to the run's output directory.

        :return: absolute path of the written manifest file.
        """
        path = os.path.join(self.outdir, "manifest.json")
        _atomic_write(path, json.dumps(self.manifest, indent=2, default=_json_default).encode())
        return path

    # -- one game block ------------------------------------------------------

    def open_block(self, index: int, backend: str, game: str, phase: dict, base_seed: int) -> str:
        """Start a block's log; its ``block_start`` line is queued at once.

        :param index: the phase's index in the curriculum (in the folder name).
        :param backend: adapter name (in the folder name).
        :param game: game id / path (its last path component in the folder name).
        :param phase: the game-phase config; its ``frame_stride`` (1 by
            default) says which rendered frames this block keeps.
        :param base_seed: the phase's base seed.
        :return: the block's folder name, for the manifest.
        """
        safe_game = game.split("/")[-1].replace(":", "_")
        name = f"block-{index:02d}_{backend}_{safe_game}"
        self._block = os.path.join(self.outdir, name)
        self.n_frames = 0
        self._frame_stride = int(phase.get("frame_stride", 1))
        self.log(type="block_start", format=FORMAT, version=VERSION,
                 subject=self.manifest["subject"], block_index=index, backend=backend,
                 game=game, phase=phase, base_seed=base_seed)
        return name

    def log(self, state: bytes | None = None, **record: Any) -> None:
        """Append one line to the open block's ``events.jsonl``.

        :param state: an opaque savestate to store on the line as ``state``:
            zlib'd, then base64 (both done by the writer process, off the
            caller's loop), the way :meth:`log_frame` stores a frame's.
        :param record: the line's fields; give it a ``type``.
        """
        self._queue.put({"op": "line", "block": self._block, "line": record,
                         "state": state})

    def log_frame(self, fields: dict, frame: Any = None, state: bytes | None = None) -> None:
        """Append one ``frame`` line, and the rendered frame when the stride says so.

        :param fields: the line's fields, written verbatim.
        :param frame: the rendered ``(H, W, C)`` array, or ``None``.
        :param state: an opaque savestate, stored as ``state``: zlib'd, then
            base64 (both done by the writer process, off the caller's loop).
        """
        line = {"type": "frame", "frame": self.n_frames, **fields}
        self._queue.put({"op": "line", "block": self._block, "line": line, "state": state})
        if frame is not None and self._frame_stride and self.n_frames % self._frame_stride == 0:
            # Copied before queuing: some adapters (MiniHack) reuse their render buffer.
            self._queue.put({"op": "frame", "block": self._block,
                             "index": self.n_frames, "frame": np.array(frame)})
        self.n_frames += 1

    def log_audio(self, sound: Any) -> None:
        """Append one frame's queued PCM chunk to the block's ``audio.h5``.

        :param sound: the chunk played this frame (``.pcm``, ``.sample_rate``),
            or ``None`` for a frame that queued no sound.
        """
        if sound is None:
            return
        self._queue.put({"op": "audio", "block": self._block,
                         "pcm": np.array(sound.pcm), "sample_rate": sound.sample_rate})

    def close_block(self, **summary: Any) -> None:
        """Append the ``block_end`` line and release the block's files.

        :param summary: the line's fields (episode and frame counts, totals,
            …); ``audio`` (:meth:`~.audio.Audio.block_log`), if given, is
            written to ``audio.h5`` instead of the line.
        """
        audio = summary.pop("audio", None)
        self.log(type="block_end", **summary)
        if audio:
            self._queue.put({"op": "audio_summary", "block": self._block, "audio": audio})
        self._queue.put({"op": "close", "block": self._block})
        self._block = None

    def close(self) -> None:
        """Flush everything and stop the writer process. Safe to call more than once."""
        if not self._process.is_alive():
            return
        self._queue.put({"op": "shutdown"})
        self._process.join()


# --------------------------------------------------------------------------- #
# The writer process
# --------------------------------------------------------------------------- #

#: How long an orphaned writer waits for more queued records before giving up.
_ORPHAN_GRACE = 1.0


def _writer_main(q: mp.Queue) -> None:
    """Consume queued ops, owning every file handle; flush on a timer; exit on
    ``shutdown`` or, having drained the queue, when the parent is gone."""
    lines: dict[str, Any] = {}
    h5: dict[str, Any] = {}
    audio: dict[str, Any] = {}
    parent = os.getppid()
    last_flush = time.monotonic()

    def _flush() -> None:
        for fh in lines.values():
            fh.flush()
        for f in h5.values():
            f.flush()  # SWMR: this is what makes a written frame durable
        for f in audio.values():
            f.flush()

    def _open_h5(block: str, frame: np.ndarray) -> Any:
        """Sized from the first frame; SWMR mode must be set after every dataset exists."""
        import h5py
        f = h5py.File(os.path.join(block, FRAMES_FILENAME), "w", libver="latest")
        f.create_dataset("frames", shape=(0, *frame.shape), maxshape=(None, *frame.shape),
                         dtype=frame.dtype, chunks=(1, *frame.shape),
                         compression="gzip", compression_opts=1)
        f.create_dataset("frame_index", shape=(0,), maxshape=(None,), dtype="int64",
                         chunks=(1024,))
        f.swmr_mode = True
        return f

    def _open_audio_h5(block: str, pcm: np.ndarray, sample_rate: float) -> Any:
        """Sized from the first chunk; SWMR mode must be set after every dataset exists."""
        import h5py
        f = h5py.File(os.path.join(block, AUDIO_FILENAME), "w", libver="latest")
        f.create_dataset("samples", shape=(0, pcm.shape[1]), maxshape=(None, pcm.shape[1]),
                         dtype=pcm.dtype, chunks=(4096, pcm.shape[1]))
        f.create_dataset("chunk_len", shape=(0,), maxshape=(None,), dtype="int64",
                         chunks=(1024,))
        f.attrs["sample_rate"] = sample_rate
        f.swmr_mode = True
        return f

    def _handle(rec: dict) -> bool:
        """Apply one op; True when it was ``shutdown``."""
        op, block = rec["op"], rec.get("block")
        if op == "shutdown":
            return True
        if op == "line":
            if block not in lines:
                os.makedirs(block, exist_ok=True)
                lines[block] = open(os.path.join(block, EVENTS_FILENAME), "a")
            line = rec["line"]
            if rec.get("state") is not None:
                # Savestates are large and mostly redundant (a crafter pickle is
                # 2.3 MB, ~57 KB zlib'd), so they are compressed here, not in
                # the game loop.
                line["state"] = base64.b64encode(zlib.compress(rec["state"], 1)).decode("ascii")
            lines[block].write(json.dumps(line, default=_json_default) + "\n")
        elif op == "frame":
            if block not in h5:
                h5[block] = _open_h5(block, rec["frame"])
            for name, value in (("frames", rec["frame"]), ("frame_index", rec["index"])):
                ds = h5[block][name]
                n = ds.shape[0]
                ds.resize(n + 1, axis=0)
                ds[n] = value
        elif op == "audio":
            pcm = rec["pcm"]
            if block not in audio:
                audio[block] = _open_audio_h5(block, pcm, rec["sample_rate"])
            samples, chunk_len = audio[block]["samples"], audio[block]["chunk_len"]
            n = samples.shape[0]
            samples.resize(n + len(pcm), axis=0)
            samples[n:n + len(pcm)] = pcm
            m = chunk_len.shape[0]
            chunk_len.resize(m + 1, axis=0)
            chunk_len[m] = len(pcm)
        elif op == "audio_summary":
            f = audio.get(block)
            if f is not None:
                a = rec["audio"]
                f.attrs["delay_ms"] = a["delay_ms"]
                f.attrs["resyncs"] = a["resyncs"]
                f.attrs["trimmed_samples"] = a["trimmed_samples"]
                f.create_dataset("onset_chunk", data=[o[0] for o in a["onsets"]], dtype="int64")
                f.create_dataset("onset_time", data=[o[1] for o in a["onsets"]], dtype="float64")
        elif op == "close":
            for handles in (lines, h5, audio):
                fh = handles.pop(block, None)
                if fh is not None:
                    fh.close()
        return False

    def _finish() -> None:
        _flush()
        for handles in (lines, h5, audio):
            for fh in handles.values():
                fh.close()

    while True:
        wait = max(0.0, FLUSH_INTERVAL - (time.monotonic() - last_flush))
        try:
            rec = q.get(timeout=wait)
        except queue_mod.Empty:
            rec = None
        if rec is not None:
            if _handle(rec):
                return _finish()
        elif os.getppid() != parent:
            # Orphaned: take what is still queued, then persist and leave.
            while True:
                try:
                    rec = q.get(timeout=_ORPHAN_GRACE)
                except queue_mod.Empty:
                    break
                if _handle(rec):
                    break
            return _finish()
        if time.monotonic() - last_flush >= FLUSH_INTERVAL:
            _flush()
            last_flush = time.monotonic()


def _atomic_write(path: str, data: bytes) -> None:
    """tmp file + fsync + ``os.replace``, so a crash mid-write never corrupts ``path``."""
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _json_default(o: Any) -> Any:
    """JSON for numpy scalars and arrays; anything else as its ``str``."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# --------------------------------------------------------------------------- #
# Reading a block back
# --------------------------------------------------------------------------- #

def read_events(block: str) -> list[dict]:
    """Every line of a block's ``events.jsonl``, in order.

    :param block: the block's folder.
    :return: the records; a partial last line (a crash mid-write) is dropped, not raised.
    """
    events = []
    with open(os.path.join(block, EVENTS_FILENAME)) as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return events


def read_frames(block: str) -> tuple[np.ndarray, np.ndarray]:
    """A block's recorded frames.

    :param block: the block's folder.
    :return: ``(frames, frame_index)``: ``(N, H, W, C)`` and, parallel, each
        row's ``frame`` in the block (not contiguous if ``frame_stride`` > 1).
    :raises FileNotFoundError: if the block kept no frames.
    """
    import h5py
    with h5py.File(os.path.join(block, FRAMES_FILENAME), "r") as f:
        return f["frames"][:], f["frame_index"][:]


def read_frame(block: str, frame: int) -> np.ndarray | None:
    """One recorded frame.

    :param block: the block's folder.
    :param frame: its ``frame`` index in the block.
    :return: the ``(H, W, C)`` array, or ``None`` if that frame was not kept.
    """
    import h5py
    with h5py.File(os.path.join(block, FRAMES_FILENAME), "r") as f:
        index = f["frame_index"][:]
        pos = int(np.searchsorted(index, frame))
        if pos >= len(index) or index[pos] != frame:
            return None
        return f["frames"][pos]


def read_audio(block: str) -> tuple[np.ndarray, float, np.ndarray]:
    """A block's recorded audio.

    :param block: the block's folder.
    :return: ``(samples, sample_rate, chunk_offset)``: every queued chunk
        concatenated, its sample rate, and each chunk's starting row in
        ``samples`` (parallel to the block's per-frame ``audio_chunk``).
    :raises FileNotFoundError: if the block queued no audio.
    """
    import h5py
    with h5py.File(os.path.join(block, AUDIO_FILENAME), "r") as f:
        chunk_len = f["chunk_len"][:]
        chunk_offset = np.concatenate(([0], np.cumsum(chunk_len)[:-1]))
        return f["samples"][:], f.attrs["sample_rate"], chunk_offset
