"""One run's data, on disk as it happens.

::

    <outdir>/manifest.json                            the run: subject, curriculum, one entry per phase, rig
    <outdir>/events.jsonl                             every line of the run, in order, each with its ``block``
    <outdir>/block-NN_<backend>_<game>/events.jsonl   one block's lines, in order
    <outdir>/block-NN_<backend>_<game>/frames.h5      the rendered frames, when the phase keeps them
    <outdir>/block-NN_<backend>_<game>/audio.h5       the queued PCM, when the block played any

A phase played again in the same run writes ``block-NN_<backend>_<game>_2``, ``_3``...

There is one :class:`Logger` per run. The manifest is rewritten (atomically)
whenever a phase entry or a run-level field lands, so it is current at every
phase boundary. Everything else is written as it happens: lines go to the run's
``events.jsonl``, a block's frames and PCM to its folder, all through a queue to a
writer process that owns the file handles, flushes them to disk on a timer and
finishes on its own if the run dies, so a crash loses at most the last flush
interval, and a partial last line is skipped on read. A line written outside a
block has a ``block`` of ``None``. Each block's ``events.jsonl`` is cut from the
run's when the logger closes (:func:`write_blocks`) and checked against its frames;
a block read before then is cut first. One block is open at a time (a run plays its
curriculum in order).

Every line in ``events.jsonl`` has a ``type``. The logger writes the first
and last itself -- ``block_start`` (``format``, ``version``, ``subject``,
``block_index``, ``backend``, ``game``, ``phase``: the config, ``base_seed``)
and ``block_end`` (whatever summary :meth:`Logger.close_block` is given) --
and stamps each ``frame`` with its index in the block, also ``frame``. The
other lines are written as given (:meth:`Logger.log`).

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
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from ..sync.clock import Clock

FORMAT = "fmri-gym-log"
VERSION = 1
EVENTS_FILENAME = "events.jsonl"
FRAMES_FILENAME = "frames.h5"
AUDIO_FILENAME = "audio.h5"
#: Seconds between flushes of the run's files to disk (not every frame).
FLUSH_INTERVAL = 1.0


@dataclass(frozen=True, slots=True)
class AudioBlockLog:
    """One block's sound timing: what ``audio.h5`` keeps besides the samples."""

    #: ``(chunk, time)``: when each queued chunk started playing, on the
    #: ``perf_counter`` clock; a chunk that never played has no entry.
    #: :meth:`Logger.close_block` writes them in run time.
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
    """Writes one run's ``manifest.json``, its run-level log, and at the end its blocks."""

    def __init__(self, outdir: str, subject: str, phases: list[dict], clock: Clock) -> None:
        """Create the run's output directory, an empty manifest and the writer process.

        :param outdir: the run's folder.
        :param subject: subject identifier stored in the manifest and each block.
        :param phases: the curriculum's phases, stored whole in the manifest (``curriculum``).
        :param clock: the run's clock (used for trigger epoch / perf times).
        """
        self.outdir = outdir
        self.clock = clock
        os.makedirs(outdir, exist_ok=True)
        self.manifest: dict = {
            "subject": subject, "curriculum": phases, "start_epoch": None, "phases": [],
        }
        self._block: str | None = None
        #: how many blocks of each folder name this run has opened.
        self._block_names: Counter[str] = Counter()
        #: frames recorded in the open block so far.
        self.n_frames = 0
        self._frame_stride = 1
        # Unbounded queue: a full one would silently stall put(), the exact
        # invisible data loss this exists to avoid.
        self._queue: mp.Queue = mp.Queue()
        self._process = mp.Process(target=_writer_main, args=(self._queue, outdir), daemon=True)
        self._process.start()
        self._console_copies: list[_ConsoleCopy] = []

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
        self._block_names[name] += 1
        if self._block_names[name] > 1:
            name += f"_{self._block_names[name]}"
        self._block = name
        self.n_frames = 0
        self._frame_stride = int(phase.get("frame_stride", 1))
        self.log(type="block_start", format=FORMAT, version=VERSION,
                 subject=self.manifest["subject"], block_index=index, backend=backend,
                 game=game, phase=phase, base_seed=base_seed)
        return name

    def log(self, **record: Any) -> None:
        """Append one line to the run's ``events.jsonl``, in the open block if there is one.

        :param record: the line's fields; give it a ``type``.
        """
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

        :param audio: the block's sound timing, its onsets on ``perf_counter``; written to
            ``audio.h5`` in run time rather than to the line; ``None`` for a block that
            played none.
        :param summary: the line's fields (episode and frame counts, totals, …).
        """
        self.log(type="block_end", **summary)
        if audio is not None:
            in_run_time = [(chunk, self.clock.from_perf(t)) for chunk, t in audio.onsets]
            self._queue.put({"op": "audio_summary", "block": self._block,
                             "audio": replace(audio, onsets=in_run_time).record()})
        self._queue.put({"op": "close", "block": self._block})
        self._block = None

    def copy_console(self) -> None:
        """Until :meth:`close`, log each line printed to stdout or stderr as well, as a
        ``console`` line (``stream``, ``text``) at the time it was printed."""
        self._console_copies = [_ConsoleCopy(sys.stdout, "stdout", self),
                                _ConsoleCopy(sys.stderr, "stderr", self)]
        sys.stdout, sys.stderr = self._console_copies

    def close(self) -> None:
        """Stop copying the console, flush everything, stop the writer process, then write and
        check the block folders (:func:`write_blocks`), saying on stderr and in the manifest
        (``block_files``) what is wrong with any. Safe to call more than once."""
        if not self._process.is_alive():
            return
        for copy in self._console_copies:
            copy.stop()
        self._queue.put({"op": "shutdown"})
        self._process.join()
        problems = write_blocks(self.outdir)
        for problem in problems:
            print(f"log: {problem}", file=sys.stderr)
        self.set_extra("block_files", {"problems": problems})


class _ConsoleCopy:
    """A console stream that also logs each whole line written to it."""

    def __init__(self, stream: Any, name: str, logger: Logger) -> None:
        self.stream = stream
        self.name = name
        self._logger = logger
        self._partial_line = ""
        self._lock = threading.Lock()   # a line printed from another thread stays whole

    def write(self, text: str) -> int:
        written = self.stream.write(text)
        now = time.perf_counter()
        with self._lock:
            *lines, self._partial_line = (self._partial_line + text).split("\n")
        for line in lines:
            self._log(now, line)
        return written

    def _log(self, perf_time: float, line: str) -> None:
        clock = self._logger.clock
        run_time = None if clock.t0_perf is None else clock.from_perf(perf_time)
        self._logger.log(type="console", stream=self.name, text=line, perf_time=perf_time,
                         run_time=run_time)

    def stop(self) -> None:
        """Log what is left of an unfinished line, and give the console its stream back."""
        if self._partial_line:
            self._log(time.perf_counter(), self._partial_line)
            self._partial_line = ""
        if getattr(sys, self.name) is self:
            setattr(sys, self.name, self.stream)

    def __getattr__(self, attribute: str) -> Any:
        return getattr(self.stream, attribute)


# --------------------------------------------------------------------------- #
# The writer process
# --------------------------------------------------------------------------- #

#: How long an orphaned writer waits for more queued records before giving up.
_ORPHAN_GRACE = 1.0


def _writer_main(q: mp.Queue, outdir: str) -> None:
    """Consume queued ops, owning every file handle; flush to disk on a timer; exit on
    ``shutdown`` or, having drained the queue, when the parent is gone."""
    events = open(os.path.join(outdir, EVENTS_FILENAME), "a")
    #: the open block's HDF5 files, by path, each in SWMR mode
    h5_files: dict[str, Any] = {}
    parent = os.getppid()
    last_flush = time.monotonic()

    def _flush() -> None:
        events.flush()
        os.fsync(events.fileno())
        for path, f in h5_files.items():
            f.flush()  # SWMR: this is what makes a written frame durable
            _fsync(path)

    def _reopen(path: str, add: Callable[[Any], None]) -> Any:
        """The HDF5 file at ``path`` with ``add`` applied: SWMR mode allows no new object, so
        the file is closed, changed, and put back in SWMR mode."""
        import h5py
        if path in h5_files:
            h5_files.pop(path).close()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        f = h5py.File(path, "a", libver="latest")
        add(f)
        f.swmr_mode = True
        h5_files[path] = f
        return f

    def _block_file(block: str, filename: str, create: Callable[[Any], None]) -> Any:
        """``block``'s ``filename``, made by ``create`` when it is new."""
        path = os.path.join(outdir, block, filename)
        return h5_files[path] if path in h5_files else _reopen(path, create)

    def _handle(rec: dict) -> bool:
        """Apply one op; True when it was ``shutdown``."""
        op, block = rec["op"], rec.get("block")
        if op == "shutdown":
            return True
        if op == "line":
            line = {"block": block, **rec["line"]}
            if rec.get("state") is not None:
                # Savestates are large and mostly redundant (a crafter pickle is
                # 2.3 MB, ~57 KB zlib'd), so they are compressed here, by
                # _json_default as every logged bytes value is, not in the game loop.
                line["state"] = rec["state"]
            events.write(json.dumps(line, default=_json_default) + "\n")
        elif op == "frame":
            f = _block_file(block, FRAMES_FILENAME, lambda f: _create_frames(f, rec["frame"]))
            _append(f["frames"], rec["frame"][None])
            _append(f["frame_index"], [rec["index"]])
        elif op == "audio":
            pcm = rec["pcm"]
            f = _block_file(block, AUDIO_FILENAME,
                            lambda f: _create_audio(f, pcm, rec["sample_rate"]))
            _append(f["samples"], pcm)
            _append(f["chunk_len"], [len(pcm)])
        elif op == "audio_summary":
            path = os.path.join(outdir, block, AUDIO_FILENAME)
            if path in h5_files:
                _reopen(path, lambda f: _write_audio_summary(f, rec["audio"]))
        elif op == "close":
            _flush()
            for path in list(h5_files):
                h5_files.pop(path).close()
        return False

    def _finish() -> None:
        _flush()
        events.close()
        for f in h5_files.values():
            f.close()

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


def _create_frames(group: Any, frame: np.ndarray) -> None:
    """A block's frame datasets in ``group`` (its file), sized from its first frame."""
    group.create_dataset("frames", shape=(0, *frame.shape), maxshape=(None, *frame.shape),
                         dtype=frame.dtype, chunks=(1, *frame.shape),
                         compression="gzip", compression_opts=1)
    group.create_dataset("frame_index", shape=(0,), maxshape=(None,), dtype="int64",
                         chunks=(1024,))


def _create_audio(group: Any, pcm: np.ndarray, sample_rate: float) -> None:
    """A block's audio datasets, sized from its first chunk."""
    group.create_dataset("samples", shape=(0, pcm.shape[1]), maxshape=(None, pcm.shape[1]),
                         dtype=pcm.dtype, chunks=(4096, pcm.shape[1]))
    group.create_dataset("chunk_len", shape=(0,), maxshape=(None,), dtype="int64",
                         chunks=(1024,))
    group.attrs["sample_rate"] = sample_rate


def _write_audio_summary(group: Any, audio: dict) -> None:
    """A block's sound timing (:meth:`AudioBlockLog.record`), beside its samples."""
    group.attrs["delay_ms"] = audio["delay_ms"]
    group.attrs["resyncs"] = audio["resyncs"]
    group.attrs["trimmed_samples"] = audio["trimmed_samples"]
    group.create_dataset("onset_chunk", data=[o[0] for o in audio["onsets"]], dtype="int64")
    group.create_dataset("onset_time", data=[o[1] for o in audio["onsets"]], dtype="float64")
    group.create_dataset("stretched_chunk", data=[c for c, _ in audio["stretched"]],
                         dtype="int64")
    group.create_dataset("stretched_len", data=[n for _, n in audio["stretched"]],
                         dtype="int64")


def _append(dataset: Any, rows: Any) -> None:
    """Add ``rows`` at the end of ``dataset``."""
    n = dataset.shape[0]
    dataset.resize(n + len(rows), axis=0)
    dataset[n:] = rows


def _fsync(path: str) -> None:
    """Force a file's written data to disk."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


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
# The block folders, cut from the run's files
# --------------------------------------------------------------------------- #

def write_blocks(outdir: str) -> list[str]:
    """Write each block's ``events.jsonl`` from its lines of the run's, then check each block's
    folder against those lines.

    :param outdir: the run's folder.
    :return: what is wrong with each block, one line each; empty when every block is whole.
    """
    lines_of_block: dict[str, list[dict]] = {}
    for line in _read_lines(os.path.join(outdir, EVENTS_FILENAME)):
        block = line.pop("block")
        if block is not None:
            lines_of_block.setdefault(block, []).append(line)
    frames_logged = _frames_in_manifest(outdir)
    problems = []
    for block, lines in lines_of_block.items():
        folder = os.path.join(outdir, block)
        os.makedirs(folder, exist_ok=True)
        _atomic_write(os.path.join(folder, EVENTS_FILENAME), "".join(
            json.dumps(line, default=_json_default) + "\n" for line in lines).encode())
        problems += [f"{block}: {problem}"
                     for problem in _block_problems(folder, lines, frames_logged.get(block))]
    return problems


def _frames_in_manifest(outdir: str) -> dict[str, int]:
    """Each block's frame count as its phase entry in the manifest gives it; none without one."""
    path = os.path.join(outdir, "manifest.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        phases = json.load(f)["phases"]
    return {p["data_dir"]: p["n_frames"] for p in phases if "data_dir" in p and "n_frames" in p}


def _block_problems(folder: str, lines: list[dict], frames_in_manifest: int | None) -> list[str]:
    """What a block's folder lacks or contradicts: its lines' order and frame numbers, and its
    ``frames.h5`` and the manifest against those lines."""
    out = []
    if lines[0]["type"] != "block_start":
        out.append(f"opens with a {lines[0]['type']!r} line, not block_start")
    if lines[-1]["type"] != "block_end":
        out.append("has no block_end line: the run stopped inside it")
    frames = [line["frame"] for line in lines if line["type"] == "frame"]
    if frames != list(range(len(frames))):
        out.append(f"its {len(frames)} frame lines are not numbered 0 to {len(frames) - 1}")
    if frames_in_manifest is not None and frames_in_manifest != len(frames):
        out.append(f"the manifest gives {frames_in_manifest} frames, its lines {len(frames)}")
    frames_file = os.path.join(folder, FRAMES_FILENAME)
    if os.path.exists(frames_file):
        import h5py
        with h5py.File(frames_file, "r") as f:
            kept, index = f["frames"].shape[0], f["frame_index"][:]
        if kept != len(index) or not set(index.tolist()) <= set(frames):
            out.append(f"frames.h5 holds {kept} frames indexed {len(index)}, not all of "
                       "them frames of its lines")
    return out


# --------------------------------------------------------------------------- #
# Reading a block back
# --------------------------------------------------------------------------- #

def _written(block: str) -> str:
    """``block``, its folder first written from its run's files when the run stopped
    before it was (:func:`write_blocks`), saying so on stderr."""
    run = os.path.dirname(os.path.normpath(block))
    if os.path.exists(os.path.join(block, EVENTS_FILENAME)) \
            or not os.path.exists(os.path.join(run, EVENTS_FILENAME)):
        return block
    print(f"log: {run}: its block folders were never written; writing them from its log",
          file=sys.stderr)
    for problem in write_blocks(run):
        print(f"log: {problem}", file=sys.stderr)
    return block


def _read_lines(path: str) -> list[dict]:
    """Every line of a ``.jsonl`` file; a partial last line (a crash mid-write) is dropped."""
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
    return _read_lines(os.path.join(_written(block), EVENTS_FILENAME))


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
    with h5py.File(os.path.join(_written(block), FRAMES_FILENAME), "r") as f:
        return f["frames"][:], f["frame_index"][:]


def read_frame(block: str, frame: int) -> np.ndarray | None:
    """One recorded frame.

    :param block: the block's folder.
    :param frame: its ``frame`` index in the block.
    :return: the ``(H, W, C)`` array, or ``None`` if that frame was not kept.
    """
    import h5py
    with h5py.File(os.path.join(_written(block), FRAMES_FILENAME), "r") as f:
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
    with h5py.File(os.path.join(_written(block), AUDIO_FILENAME), "r") as f:
        chunk_len = f["chunk_len"][:]
        chunk_offset = np.concatenate(([0], np.cumsum(chunk_len)[:-1]))
        return f["samples"][:], f.attrs["sample_rate"], chunk_offset
