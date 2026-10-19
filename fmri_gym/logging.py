"""One run's data, on disk as it happens.

::

    <run_folder>/manifest.json                        the run: subject, curriculum, phases, rig
    <run_folder>/block-NN_<backend>_<game>/events.jsonl   one JSON object per line, in order
    <run_folder>/block-NN_<backend>_<game>/frames.h5      the rendered frames the phase keeps
    <run_folder>/block-NN_<backend>_<game>/audio.h5       the queued PCM, when the block played any

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
    from .audio import AudioBlockLog
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

    def __init__(self, run_folder: str, subject: str, phases: list[dict],
                 clock: Clock) -> None:
        """Create the run's folder, an empty manifest and the writer process.

        :param run_folder: the run's folder.
        :param subject: subject identifier stored in the manifest and each block.
        :param phases: the curriculum's phases, stored whole in the manifest (``curriculum``).
        :param clock: the run's clock (used for trigger epoch / perf times).
        """
        self.run_folder = run_folder
        self.clock = clock
        os.makedirs(run_folder, exist_ok=True)
        self.manifest: dict = {
            "subject": subject, "curriculum": phases, "start_epoch": None, "phases": [],
        }
        #: the open block's folder, or ``None`` between blocks.
        self._block_folder: str | None = None
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
        self.manifest["start_epoch"] = self.clock.trigger_epoch
        self.manifest["trigger_perf"] = self.clock.trigger_perf
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
        """Write ``manifest.json`` atomically to the run's folder.

        :return: absolute path of the written manifest file.
        """
        path = os.path.join(self.run_folder, "manifest.json")
        _atomic_write(path, json.dumps(self.manifest, indent=2, default=_json_default).encode())
        return path

    # -- one game block ------------------------------------------------------

    def open_block(self, phase_index: int, backend: str, game: str, phase: dict,
                   base_seed: int) -> str:
        """Start a block's log; its ``block_start`` line is queued at once.

        :param phase_index: the phase's index in the curriculum (in the folder name).
        :param backend: adapter name (in the folder name).
        :param game: game id / path (its last path component in the folder name).
        :param phase: the game-phase config; its ``frame_stride`` (1 by
            default) says which rendered frames this block keeps.
        :param base_seed: the phase's base seed.
        :return: the block's folder name, for the manifest.
        """
        safe_game = game.split("/")[-1].replace(":", "_")
        block_folder_name = f"block-{phase_index:02d}_{backend}_{safe_game}"
        self._block_folder = os.path.join(self.run_folder, block_folder_name)
        self.n_frames = 0
        self._frame_stride = int(phase.get("frame_stride", 1))
        self.log(type="block_start", format=FORMAT, version=VERSION,
                 subject=self.manifest["subject"], block_index=phase_index, backend=backend,
                 game=game, phase=phase, base_seed=base_seed)
        return block_folder_name

    def log(self, **record: Any) -> None:
        """Append one line to the open block's ``events.jsonl``.

        :param record: the line's fields; give it a ``type``.
        """
        self._queue.put({"op": "line", "block": self._block_folder, "line": record})

    def log_frame(self, fields: dict, frame_rgb: Any = None,
                  savestate: bytes | None = None) -> None:
        """Append one ``frame`` line, and the rendered frame when the stride says so.

        :param fields: the line's fields, written verbatim.
        :param frame_rgb: the rendered ``(H, W, C)`` array, or ``None``.
        :param savestate: an opaque savestate, stored as ``state``: zlib'd,
            then base64 (both done by the writer process, off the caller's loop).
        """
        record = {"type": "frame", "frame": self.n_frames, **fields}
        self._queue.put({"op": "line", "block": self._block_folder, "line": record,
                         "state": savestate})
        if (frame_rgb is not None and self._frame_stride
                and self.n_frames % self._frame_stride == 0):
            # Copied before queuing: some adapters (MiniHack) reuse their render buffer.
            self._queue.put({"op": "frame", "block": self._block_folder,
                             "index": self.n_frames, "frame": np.array(frame_rgb)})
        self.n_frames += 1

    def log_audio(self, sound: Any) -> None:
        """Append one frame's queued PCM chunk to the block's ``audio.h5``.

        :param sound: the chunk played this frame (``.pcm``, ``.sample_rate``),
            or ``None`` for a frame that queued no sound.
        """
        if sound is None:
            return
        self._queue.put({"op": "audio", "block": self._block_folder,
                         "pcm": np.array(sound.pcm), "sample_rate": sound.sample_rate})

    def close_block(self, audio: AudioBlockLog | None = None, **summary: Any) -> None:
        """Append the ``block_end`` line and release the block's files.

        :param audio: the block's sound timing (:meth:`~.audio.Audio.block_log`),
            written to ``audio.h5`` rather than the line; ``None`` for a block
            that played none.
        :param summary: the line's fields (episode and frame counts, totals, …).
        """
        self.log(type="block_end", **summary)
        if audio is not None:
            self._queue.put({"op": "audio_summary", "block": self._block_folder,
                             "audio": audio.record()})
        self._queue.put({"op": "close", "block": self._block_folder})
        self._block_folder = None

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


def _writer_main(queue: mp.Queue) -> None:
    """Consume queued ops, owning every file handle; flush on a timer; exit on
    ``shutdown`` or, having drained the queue, when the parent is gone."""
    # One open file per block folder, for each of the block's three files.
    events_files: dict[str, Any] = {}
    frames_files: dict[str, Any] = {}
    audio_files: dict[str, Any] = {}
    parent_pid = os.getppid()
    last_flush_monotonic = time.monotonic()

    def _flush() -> None:
        for file in events_files.values():
            file.flush()
        for file in frames_files.values():
            file.flush()  # SWMR: this is what makes a written frame durable
        for file in audio_files.values():
            file.flush()

    def _open_frames_h5(block_folder: str, frame_rgb: np.ndarray) -> Any:
        """Sized from the first frame; SWMR mode must be set after every dataset exists."""
        import h5py
        file = h5py.File(os.path.join(block_folder, FRAMES_FILENAME), "w", libver="latest")
        file.create_dataset("frames", shape=(0, *frame_rgb.shape),
                            maxshape=(None, *frame_rgb.shape), dtype=frame_rgb.dtype,
                            chunks=(1, *frame_rgb.shape), compression="gzip",
                            compression_opts=1)
        file.create_dataset("frame_index", shape=(0,), maxshape=(None,), dtype="int64",
                            chunks=(1024,))
        file.swmr_mode = True
        return file

    def _open_audio_h5(block_folder: str, pcm: np.ndarray, sample_rate: float) -> Any:
        """Sized from the first chunk; SWMR mode must be set after every dataset exists."""
        import h5py
        file = h5py.File(os.path.join(block_folder, AUDIO_FILENAME), "w", libver="latest")
        file.create_dataset("samples", shape=(0, pcm.shape[1]), maxshape=(None, pcm.shape[1]),
                            dtype=pcm.dtype, chunks=(4096, pcm.shape[1]))
        file.create_dataset("chunk_len", shape=(0,), maxshape=(None,), dtype="int64",
                            chunks=(1024,))
        file.attrs["sample_rate"] = sample_rate
        file.swmr_mode = True
        return file

    def _handle(message: dict) -> bool:
        """Apply one op; True when it was ``shutdown``."""
        op, block_folder = message["op"], message.get("block")
        if op == "shutdown":
            return True
        if op == "line":
            if block_folder not in events_files:
                os.makedirs(block_folder, exist_ok=True)
                events_files[block_folder] = open(os.path.join(block_folder, EVENTS_FILENAME),
                                                  "a")
            record = message["line"]
            if message.get("state") is not None:
                # Savestates are large and mostly redundant (a crafter pickle is
                # 2.3 MB, ~57 KB zlib'd), so they are compressed here, by
                # _json_default as every logged bytes value is, not in the game loop.
                record["state"] = message["state"]
            events_files[block_folder].write(json.dumps(record, default=_json_default) + "\n")
        elif op == "frame":
            if block_folder not in frames_files:
                frames_files[block_folder] = _open_frames_h5(block_folder, message["frame"])
            for name, value in (("frames", message["frame"]), ("frame_index", message["index"])):
                dataset = frames_files[block_folder][name]
                n_rows = dataset.shape[0]
                dataset.resize(n_rows + 1, axis=0)
                dataset[n_rows] = value
        elif op == "audio":
            pcm = message["pcm"]
            if block_folder not in audio_files:
                audio_files[block_folder] = _open_audio_h5(block_folder, pcm,
                                                           message["sample_rate"])
            samples = audio_files[block_folder]["samples"]
            chunk_len = audio_files[block_folder]["chunk_len"]
            n_samples = samples.shape[0]
            samples.resize(n_samples + len(pcm), axis=0)
            samples[n_samples:n_samples + len(pcm)] = pcm
            n_chunks = chunk_len.shape[0]
            chunk_len.resize(n_chunks + 1, axis=0)
            chunk_len[n_chunks] = len(pcm)
        elif op == "audio_summary":
            file = audio_files.get(block_folder)
            if file is not None:
                audio_log = message["audio"]
                file.attrs["delay_ms"] = audio_log["delay_ms"]
                file.attrs["resyncs"] = audio_log["resyncs"]
                file.attrs["trimmed_samples"] = audio_log["trimmed_samples"]
                file.create_dataset("onset_chunk", dtype="int64",
                                    data=[chunk for chunk, _ in audio_log["onsets"]])
                file.create_dataset("onset_time", dtype="float64",
                                    data=[onset for _, onset in audio_log["onsets"]])
        elif op == "close":
            for files in (events_files, frames_files, audio_files):
                file = files.pop(block_folder, None)
                if file is not None:
                    file.close()
        return False

    def _finish() -> None:
        _flush()
        for files in (events_files, frames_files, audio_files):
            for file in files.values():
                file.close()

    while True:
        wait_s = max(0.0, FLUSH_INTERVAL - (time.monotonic() - last_flush_monotonic))
        try:
            message = queue.get(timeout=wait_s)
        except queue_mod.Empty:
            message = None
        if message is not None:
            if _handle(message):
                return _finish()
        elif os.getppid() != parent_pid:
            # Orphaned: take what is still queued, then persist and leave.
            while True:
                try:
                    message = queue.get(timeout=_ORPHAN_GRACE)
                except queue_mod.Empty:
                    break
                if _handle(message):
                    break
            return _finish()
        if time.monotonic() - last_flush_monotonic >= FLUSH_INTERVAL:
            _flush()
            last_flush_monotonic = time.monotonic()


def _atomic_write(path: str, data: bytes) -> None:
    """tmp file + fsync + ``os.replace``, so a crash mid-write never corrupts ``path``."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp_path, path)


def _json_default(value: Any) -> Any:
    """JSON for numpy scalars and arrays, bytes as a savestate is (zlib, base64); else ``str``."""
    if isinstance(value, bytes):
        return base64.b64encode(zlib.compress(value, 1)).decode("ascii")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


# --------------------------------------------------------------------------- #
# Reading a block back
# --------------------------------------------------------------------------- #

def read_events(block_folder: str) -> list[dict]:
    """Every line of a block's ``events.jsonl``, in order.

    :param block_folder: the block's folder.
    :return: the records; a partial last line (a crash mid-write) is dropped, not raised.
    """
    events = []
    with open(os.path.join(block_folder, EVENTS_FILENAME)) as file:
        for line in file:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return events


def read_savestate(text: str) -> bytes:
    """A savestate as the log writes it (zlib'd, then base64), as the bytes ``restore`` takes."""
    return zlib.decompress(base64.b64decode(text))


def read_frames(block_folder: str) -> tuple[np.ndarray, np.ndarray]:
    """A block's recorded frames.

    :param block_folder: the block's folder.
    :return: ``(frames, frame_index)``: ``(N, H, W, C)`` and, parallel, each
        row's ``frame`` in the block (not contiguous if ``frame_stride`` > 1).
    :raises FileNotFoundError: if the block kept no frames.
    """
    import h5py
    with h5py.File(os.path.join(block_folder, FRAMES_FILENAME), "r") as file:
        return file["frames"][:], file["frame_index"][:]


def read_frame(block_folder: str, frame: int) -> np.ndarray | None:
    """One recorded frame.

    :param block_folder: the block's folder.
    :param frame: its ``frame`` index in the block.
    :return: the ``(H, W, C)`` array, or ``None`` if that frame was not kept.
    """
    import h5py
    with h5py.File(os.path.join(block_folder, FRAMES_FILENAME), "r") as file:
        frame_index = file["frame_index"][:]
        row = int(np.searchsorted(frame_index, frame))
        if row >= len(frame_index) or frame_index[row] != frame:
            return None
        return file["frames"][row]


def read_audio(block_folder: str) -> tuple[np.ndarray, float, np.ndarray]:
    """A block's recorded audio.

    :param block_folder: the block's folder.
    :return: ``(samples, sample_rate, chunk_offset)``: every queued chunk
        concatenated, its sample rate, and each chunk's starting row in
        ``samples`` (parallel to the block's per-frame ``audio_chunk``).
    :raises FileNotFoundError: if the block queued no audio.
    """
    import h5py
    with h5py.File(os.path.join(block_folder, AUDIO_FILENAME), "r") as file:
        chunk_len = file["chunk_len"][:]
        chunk_offset = np.concatenate(([0], np.cumsum(chunk_len)[:-1]))
        return file["samples"][:], file.attrs["sample_rate"], chunk_offset
