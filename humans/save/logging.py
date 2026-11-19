"""One run's data, on disk as it happens.

::

    <outdir>/manifest.json                            the run: what it played, on what, with which code
    <outdir>/events.jsonl                             the run's log: what the run read, showed, sent, printed
    <outdir>/block-NN_<backend>_<game>/events.jsonl   one block's log: what its game did
    <outdir>/block-NN_<backend>_<game>/frames.h5      the rendered frames, when the phase keeps them
    <outdir>/block-NN_<backend>_<game>/audio.h5       the queued PCM, when the block played any

A phase played again in the same run writes ``block-NN_<backend>_<game>_2``, ``_3``...

There is one :class:`Logger` per run. The manifest is written when the logger is created,
with the log format, and rewritten (atomically) whenever a field or a phase entry lands: it
says what the logs are read with. Each line is written once, by what produced it: the run's
log (:meth:`Logger.log_run_event`) holds what the run read, showed, sent and printed, a
block's (:meth:`Logger.log`) what its game did. Lines, frames and PCM go
through a queue to a writer process that owns the file handles, flushes them on a timer (the
run's log to disk) and finishes on its own if the run dies: a crash loses at most the last
flush interval, and a partial last line is skipped on read. One block is open at a time (a
run plays its curriculum in order). A run that logs no run line (an agent's) has no
run's log.

A run line has a ``type``, the ``perf_counter`` time it happened (``perf_time``, given by its
source; the logger converts no time) and the folder name of the block open when it was
written (``block``, ``None`` outside one). A block's ``events.jsonl`` starts with
``block_start`` (``format``, ``version``, ``subject``, ``block_index``, ``backend``,
``game``, ``phase``: the config, ``base_seed``, then what :meth:`Logger.open_block` is
given besides: a run's ``run_time``) and ends with ``block_end``
(whatever summary :meth:`Logger.close_block` is given); each ``frame`` line is stamped with its
index in the block, also ``frame``; the other lines are written as given.

``frames.h5`` holds ``frames`` ``(N, H, W, C)``, one gzip'd chunk per frame
so any one reads alone, and ``frame_index`` ``(N,)``, the ``frame`` of each
row: every ``frame_stride``-th frame of the block (``0``: no frames at all).
SWMR mode makes a flushed frame durable the way a flushed line is.

``audio.h5`` holds ``samples``, every :meth:`Logger.log_audio` chunk
concatenated, and ``chunk_len`` ``(n_chunks,)``, so a chunk's offset into
``samples`` is a running sum of the ones before it. ``onset_chunk``/
``onset_time``, ``stretched_chunk``/``stretched_len`` and the
``delay_ms``/``resyncs``/``trimmed_samples`` attrs land at
:meth:`Logger.close_block`, from the block's :class:`AudioBlockLog`. A chunk in
``stretched_chunk`` was played linearly resampled to its ``stretched_len`` samples, to keep
the sound on the display's clock; every other chunk played as it is stored.
"""

from __future__ import annotations

import base64
import json
import multiprocessing as mp
import os
import queue as queue_mod
import sys
import threading
import time
import zlib
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

FORMAT = "fmri-gym-log"
#: 2: the run's own ``events.jsonl``, beside its blocks'.
VERSION = 2
EVENTS_FILENAME = "events.jsonl"
FRAMES_FILENAME = "frames.h5"
AUDIO_FILENAME = "audio.h5"
#: Seconds between flushes of the files to the OS, and of the run's log to disk.
FLUSH_INTERVAL = 1.0


@dataclass(frozen=True, slots=True)
class AudioBlockLog:
    """One block's sound timing: what ``audio.h5`` keeps besides the samples."""

    #: ``(chunk, time)``: when each queued chunk started playing, in seconds; a chunk that
    #: never played has no entry
    onsets: list[tuple[int, float]]
    #: milliseconds from a frame's flip to its sound's first sample at the DAC
    delay_ms: float
    #: chunks placed at their onset rather than gaplessly, after a stall
    resyncs: int
    #: samples cut from chunks that were already late when placed
    trimmed_samples: int
    #: ``(chunk, samples)``: each chunk resampled to follow the display's clock,
    #: and how many samples it played instead of the ones logged (a linear
    #: resampling of the logged PCM to that length is what played); every other
    #: chunk played as logged
    stretched: list[tuple[int, int]]

    def record(self) -> dict:
        """The fields as ``audio.h5`` takes them."""
        return asdict(self)


class Logger:
    """Writes one run's ``manifest.json``, its own log and its blocks' logs."""

    def __init__(self, outdir: str, subject: str, phases: list[dict]) -> None:
        """Create the run's folder, its manifest (written at once, with the log format) and the
        writer process.

        :param outdir: the run's folder.
        :param subject: subject identifier stored in the manifest and each block.
        :param phases: the curriculum's phases, stored whole in the manifest (``curriculum``).
        """
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.manifest: dict = {
            "subject": subject, "curriculum": phases, "start_epoch": None, "phases": [],
            "log_format": {"format": FORMAT, "version": VERSION},
        }
        self.save_manifest()
        #: the open block's folder name; ``None`` between blocks.
        self._block: str | None = None
        #: how many blocks of each folder name this run has opened.
        self._block_names: Counter[str] = Counter()
        #: frames recorded in the open block so far.
        self.n_frames = 0
        self._frame_stride = 1
        self._console_copies: list[_ConsoleCopy] = []
        # Unbounded queue: a full one would silently stall put(), the exact
        # invisible data loss this exists to avoid.
        self._queue: mp.Queue = mp.Queue()
        self._process = mp.Process(target=_writer_main, args=(self._queue, outdir), daemon=True)
        self._process.start()

    # -- the manifest --------------------------------------------------------

    def log_trigger_time(self, perf_time: float, wall_time: float) -> None:
        """Record the run clock's zero on the manifest (``trigger_perf``, ``start_epoch``).

        :param perf_time: ``perf_counter`` at the trigger.
        :param wall_time: ``time.time()`` at the trigger.
        """
        self.manifest["start_epoch"] = wall_time
        self.manifest["trigger_perf"] = perf_time
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

    # -- the run's log ---------------------------------------------------------

    def log_run_event(self, line_type: str, perf_time: float, **fields: Any) -> None:
        """Append one line to the run's ``events.jsonl``: its ``type``, ``perf_time``, the open
        block's folder name (``block``, ``None`` between blocks), then ``fields``. Safe to call
        from any thread.

        :param line_type: the line's ``type``.
        :param perf_time: ``perf_counter`` when it happened.
        :param fields: the rest of the line, written as given.
        """
        self._queue.put({"op": "run_line", "line": {
            "type": line_type, "perf_time": perf_time, "block": self._block, **fields}})

    def copy_console(self) -> None:
        """Until :meth:`close`, log each line printed to stdout or stderr as well, as a
        ``console`` line (``stream``, ``text``) at the time it was printed."""
        self._console_copies = [_ConsoleCopy(sys.stdout, "stdout", self),
                                _ConsoleCopy(sys.stderr, "stderr", self)]
        sys.stdout, sys.stderr = self._console_copies

    # -- one game block ------------------------------------------------------

    def open_block(self, index: int, backend: str, game: str, phase: dict, base_seed: int,
                   **fields: Any) -> str:
        """Start a block's log; its ``block_start`` line is queued at once.

        :param index: the phase's index in the curriculum (in the folder name).
        :param backend: adapter name (in the folder name).
        :param game: game id / path (its last path component in the folder name).
        :param phase: the game-phase config; its ``frame_stride`` (1 by
            default) says which rendered frames this block keeps.
        :param base_seed: the phase's base seed.
        :param fields: the rest of the ``block_start`` line (a run gives ``run_time``, the
            run clock as the block opens).
        :return: the block's folder name, for the manifest.
        """
        safe_game = game.split("/")[-1].replace(":", "_")
        name = f"block-{index:02d}_{backend}_{safe_game}"
        self._block_names[name] += 1
        if self._block_names[name] > 1:
            name += f"_{self._block_names[name]}"
        self._block = name
        self.n_frames = 0
        self._frame_stride = int(phase.get("frame_stride", 1))
        self.log(type="block_start", format=FORMAT, version=VERSION,
                 subject=self.manifest["subject"], block_index=index, backend=backend,
                 game=game, phase=phase, base_seed=base_seed, **fields)
        return name

    def log(self, **record: Any) -> None:
        """Append one line to the open block's ``events.jsonl``.

        :param record: the line's fields; give it a ``type``.
        :raises RuntimeError: if no block is open.
        """
        if self._block is None:
            raise RuntimeError(f"a {record.get('type')!r} block line, with no block open")
        self._queue.put({"op": "line", "block": self._block, "line": record})

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

    def close_block(self, audio: AudioBlockLog | None = None, **summary: Any) -> None:
        """Append the ``block_end`` line and release the block's files.

        :param audio: the block's sound timing, its onsets in run time; written to
            ``audio.h5`` rather than to the line; ``None`` for a block that played none.
        :param summary: the line's fields (when it ended, episode and frame counts, totals, …).
        """
        self.log(type="block_end", **summary)
        if audio is not None:
            self._queue.put({"op": "audio_summary", "block": self._block,
                             "audio": audio.record()})
        self._queue.put({"op": "close", "block": self._block})
        self._block = None

    def close(self) -> None:
        """Stop copying the console, flush everything and stop the writer process. Safe to call
        more than once."""
        if not self._process.is_alive():
            return
        for copy in self._console_copies:
            copy.stop()
        self._queue.put({"op": "shutdown"})
        self._process.join()


class _ConsoleCopy:
    """A console stream that also logs each whole line written to it."""

    def __init__(self, stream: Any, name: str, logger: Logger) -> None:
        """Wrap ``stream``.

        :param stream: the console stream written to.
        :param name: ``stdout`` or ``stderr``: the stream's name, and the line's ``stream``.
        :param logger: the log that gets each line.
        """
        self.stream = stream
        self.name = name
        self._logger = logger
        self._partial_line = ""
        self._lock = threading.Lock()   # a line printed from another thread stays whole

    def write(self, text: str) -> int:
        """Write ``text`` to the stream, then log each line it completes.

        :param text: what was printed.
        :return: what the stream's own ``write`` returned.
        """
        written = self.stream.write(text)
        now = time.perf_counter()
        with self._lock:
            *lines, self._partial_line = (self._partial_line + text).split("\n")
        for line in lines:
            self._logger.log_run_event("console", now, stream=self.name, text=line)
        return written

    def stop(self) -> None:
        """Log what is left of an unfinished line, and give the console its stream back."""
        if self._partial_line:
            self._logger.log_run_event("console", time.perf_counter(), stream=self.name,
                                       text=self._partial_line)
            self._partial_line = ""
        if getattr(sys, self.name) is self:
            setattr(sys, self.name, self.stream)

    def __getattr__(self, attribute: str) -> Any:
        """Everything else is the stream's."""
        return getattr(self.stream, attribute)


# --------------------------------------------------------------------------- #
# The writer process
# --------------------------------------------------------------------------- #

#: How long an orphaned writer waits for more queued records before giving up.
_ORPHAN_GRACE = 1.0


def _writer_main(q: mp.Queue, outdir: str) -> None:
    """Consume queued ops, owning every file handle, until ``shutdown`` or, the queue drained,
    the parent is gone.

    Files are flushed every :data:`FLUSH_INTERVAL`, the run's log to disk.

    :param q: the logger's queue.
    :param outdir: the run's folder.
    """
    #: the run's log, opened by its first line.
    run_lines: list[Any] = []
    lines: dict[str, Any] = {}
    h5: dict[str, Any] = {}
    audio: dict[str, Any] = {}
    parent = os.getppid()
    last_flush = time.monotonic()

    def _flush() -> None:
        for fh in run_lines:
            fh.flush()
            os.fsync(fh.fileno())
        for fh in lines.values():
            fh.flush()
        for f in h5.values():
            f.flush()  # SWMR: this is what makes a written frame durable
        for f in audio.values():
            f.flush()

    def _open_h5(folder: str, frame: np.ndarray) -> Any:
        """Sized from the first frame; SWMR mode must be set after every dataset exists."""
        import h5py
        f = h5py.File(os.path.join(folder, FRAMES_FILENAME), "w", libver="latest")
        f.create_dataset("frames", shape=(0, *frame.shape), maxshape=(None, *frame.shape),
                         dtype=frame.dtype, chunks=(1, *frame.shape),
                         compression="gzip", compression_opts=1)
        f.create_dataset("frame_index", shape=(0,), maxshape=(None,), dtype="int64",
                         chunks=(1024,))
        f.swmr_mode = True
        return f

    def _open_audio_h5(folder: str, pcm: np.ndarray, sample_rate: float) -> Any:
        """Sized from the first chunk; SWMR mode must be set after every dataset exists."""
        import h5py
        f = h5py.File(os.path.join(folder, AUDIO_FILENAME), "w", libver="latest")
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
        if op == "run_line":
            if not run_lines:
                run_lines.append(open(os.path.join(outdir, EVENTS_FILENAME), "a"))
            run_lines[0].write(json.dumps(rec["line"], default=_json_default) + "\n")
        elif op == "line":
            if block not in lines:
                os.makedirs(os.path.join(outdir, block), exist_ok=True)
                lines[block] = open(os.path.join(outdir, block, EVENTS_FILENAME), "a")
            line = rec["line"]
            if rec.get("state") is not None:
                # Savestates are large and mostly redundant (a crafter pickle is
                # 2.3 MB, ~57 KB zlib'd), so they are compressed here, by
                # _json_default as every logged bytes value is, not in the game loop.
                line["state"] = rec["state"]
            lines[block].write(json.dumps(line, default=_json_default) + "\n")
        elif op == "frame":
            if block not in h5:
                h5[block] = _open_h5(os.path.join(outdir, block), rec["frame"])
            for name, value in (("frames", rec["frame"]), ("frame_index", rec["index"])):
                ds = h5[block][name]
                n = ds.shape[0]
                ds.resize(n + 1, axis=0)
                ds[n] = value
        elif op == "audio":
            pcm = rec["pcm"]
            if block not in audio:
                audio[block] = _open_audio_h5(os.path.join(outdir, block), pcm,
                                              rec["sample_rate"])
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
                f.create_dataset("stretched_chunk", data=[c for c, _ in a["stretched"]],
                                 dtype="int64")
                f.create_dataset("stretched_len", data=[n for _, n in a["stretched"]],
                                 dtype="int64")
        elif op == "close":
            for handles in (lines, h5, audio):
                fh = handles.pop(block, None)
                if fh is not None:
                    fh.close()
        return False

    def _finish() -> None:
        _flush()
        for fh in run_lines:
            fh.close()
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
    """JSON for numpy scalars and arrays, bytes as a savestate is (zlib, base64); else ``str``."""
    if isinstance(o, bytes):
        return base64.b64encode(zlib.compress(o, 1)).decode("ascii")
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# --------------------------------------------------------------------------- #
# Reading the logs back
# --------------------------------------------------------------------------- #

def read_lines(path: str) -> list[dict]:
    """Every line of an ``events.jsonl``, the run's or a block's, in order.

    :param path: the file.
    :return: the records; a partial last line (a crash mid-write) is dropped, not raised.
    """
    lines = []
    with open(path) as f:
        for line in f:
            try:
                lines.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return lines


def read_events(block: str) -> list[dict]:
    """Every line of a block's ``events.jsonl``, in order.

    :param block: the block's folder.
    :return: the records; a partial last line (a crash mid-write) is dropped, not raised.
    """
    return read_lines(os.path.join(block, EVENTS_FILENAME))


def read_state(text: str) -> bytes:
    """A savestate as the log writes it (zlib'd, then base64), as the bytes ``restore`` takes."""
    return zlib.decompress(base64.b64decode(text))


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
