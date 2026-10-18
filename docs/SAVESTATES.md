# How a run is recorded: states, actions, frames, audio

A rundown of the recording pipeline as of 2026-09-30, with pointers into the
code. Line numbers refer to the tree at that date and will drift.

## How it works generally

Everything goes through one `Logger` per run (`fmri_gym/logging.py:62`). The
game loop never touches a file. It puts small dicts on an unbounded
`multiprocessing.Queue` (`logging.py:85`), and a separate **writer process**
(`_writer_main`, `logging.py:210`) owns every file handle, does the expensive
encoding, and flushes on a timer. The queue is deliberately unbounded so a slow
disk stalls nothing in the game loop, at the cost of memory growth if the
writer falls behind (`logging.py:83-84`).

Per block, the output is one folder with up to three files (`logging.py:5-8`):

- `events.jsonl`: one JSON line per event. Savestates live inside the `frame`
  lines here, not in a separate file.
- `frames.h5`: rendered frames, HDF5 with SWMR (single-writer multiple-reader)
  so flushed frames are durable.
- `audio.h5`: PCM chunks concatenated.

Plus `manifest.json` at the run level, rewritten atomically on every phase
boundary (`logging.py:114-121`, `_atomic_write` at `logging.py:332`).

The per-frame producer side is in `Run._episode` (`fmri_gym/run.py:570-613`):

1. `adapter.step(action)`, then
   `adapter.capture(observation, info, want_blob=anchor_frame)` returns a `FrameState`
   with `blob` (savestate bytes or `None`) and `variables` (RAM, info)
   (`run.py:573-574`, contract at `adapters/base.py:14-29`).
2. `_show_frame` renders, flips, and queues sound (`run.py:640-646`).
3. `logger.log_frame(frame_fields, frame_rgb=frame_rgb, savestate=...)` then
   `logger.log_audio(sound)` (`run.py:612-613`).

## How often we flush

**Every 1 second**, on a timer in the writer process. `FLUSH_INTERVAL = 1.0`
(`logging.py:59`). The writer's loop computes the remaining wait, blocks on
`queue.get(timeout=wait)`, and calls `_flush()` when the interval has elapsed
(`logging.py:308-329`). `_flush` calls `.flush()` on the jsonl file handle and
on each open HDF5 file (`logging.py:219-225`). This is a flush to the OS page
cache, not an `fsync`, so a power loss can still lose more; a process crash
loses at most about 1 s.

There is also an orphan path: if the parent dies, the writer notices via
`os.getppid()`, drains the queue with a 1 s grace (`_ORPHAN_GRACE`,
`logging.py:207`), flushes, closes, and exits (`logging.py:317-326`).
`Logger.close` sends a `shutdown` op and joins (`logging.py:194-199`).

Everything else is buffered Python I/O and HDF5's internal buffering between
flushes. There is no end-of-block compression or "finalize" step: `close_block`
queues a `close` op and the writer just closes the block's handles
(`logging.py:191`, `logging.py:295-299`).

## How each thing is stored

**States and actions (events.jsonl).** Actions, reward, times, and `variables`
go verbatim into each `frame` line (`run.py:597-611`). The savestate blob rides
along in the queue record as raw bytes and the *writer process* does
`zlib.compress(blob, 1)` then base64 and stuffs it into the line's `"state"`
key (`logging.py:261-265`). The line is then `json.dumps` with `_json_default`
converting numpy arrays to Python lists (`logging.py:266`,
`logging.py:342-350`). Only every `state_stride`-th frame of each episode
carries a blob (`run.py:573`, `run.py:677`), always including `ep_frame == 0`
as the replay anchor. The blob itself is adapter-specific: pickled
`clone_state` for ALE (`adapters/ale.py:61`), `em.get_state()` for retro
(`adapters/retro.py:60`), a pickle of the whole env for crafter
(`adapters/crafter.py:609`), `None` for adapters without an in-memory savestate
(vizdoom, baba, rushhour, aigamestore).

**Frames (frames.h5).** Every `frame_stride`-th frame (`logging.py:163`, `0`
means none) is copied with `np.array(frame)` before queuing because some
adapters reuse the render buffer (`logging.py:164-166`). The writer lazily
creates the file on the first frame, sized from its shape, as a resizable
dataset with **one chunk per frame, gzip level 1** (`_open_h5`,
`logging.py:227-237`), plus an `int64` `frame_index` dataset. Each frame
appends by `resize` then item assignment (`logging.py:270-274`). Compression
happens per frame as it arrives, spread across the block.

**Audio (audio.h5).** Each frame's `Sound` (`adapters/base.py:32-49`) is what
the adapter produced, whether or not it was played; muting does not change what
is logged (`run.py:483-484`). `log_audio` copies the PCM and queues it
(`logging.py:169-178`). The writer appends to a `samples (N, channels)`
dataset, uncompressed, chunked at 4096 rows, and appends one entry to
`chunk_len` so offsets are a cumsum (`_open_audio_h5`, `logging.py:239-249`;
append at `logging.py:275-285`). At `close_block`, the audio summary from
`Audio.block_log` lands as attrs and two small datasets (`onset_chunk`,
`onset_time`) (`logging.py:286-294`, produced at `run.py:758-773`). The
per-frame `audio_chunk` field in events.jsonl links a frame to its row in
`chunk_len` (`run.py:608-611`).

## Ballpark overhead, from reading the code

Estimates, not measurements. Split by where the cost lands.

### On the game loop (the part that matters for frame timing)

| Item | Cost per frame | Notes |
|---|---|---|
| `capture` blob | ALE: tens of µs. Retro: ~1 MB `get_state`, likely 1-5 ms. Crafter: 2.3 MB pickle, ~3 ms (documented at `adapters/crafter.py:63`) | Only on stride frames. The biggest loop-side cost. |
| `capture` variables | RAM `.copy()`: µs | Every frame. |
| `log_frame` queue put | Pickles the dict, the raw blob, and a full frame copy through a pipe. 210x160x3 ALE frame ~100 KB; 384x384x3 crafter ~440 KB; retro Genesis ~200 KB | Roughly 0.1-0.5 ms per frame. A retro blob adds ~1 MB through the pipe on stride frames. |
| `log_audio` | `np.array(pcm)` copy of a ~1-2 KB chunk, plus pipe | Negligible. |
| `Audio.play` | `block.copy()` or a linear resample (`audio.py:298`) | Negligible. |

The loop side is dominated by the savestate call and by shipping the frame
through the multiprocessing pipe. Both are well under a 16.7 ms frame budget
on ALE. `state_stride: 15` on the retro configs exists because a ~1 MB state
every frame would be a meaningful fraction of it.

### In the writer process (off the critical path, bounded by throughput)

| Item | Cost | Notes |
|---|---|---|
| zlib level 1 of a savestate | ALE ~0.4 KB: nothing. Retro 1 MB: ~5-10 ms. Crafter 2.3 MB: ~10 ms, down to ~57 KB | Only on stride frames. |
| base64 + `json.dumps` of the line | Small, **except `variables`** | See caveat below. |
| gzip level 1 of a frame in h5py | Roughly 1-3 ms per frame at these sizes (a benchmark on 240x320x3 measured 2.2 ms) | One compression call per frame; no batching. |
| HDF5 `resize` + assign, three datasets per frame | Tens to a few hundred µs each | Fine at 60 fps. |
| audio append | Negligible | Uncompressed, chunked in 4096 rows. |

At 60 fps the writer has ~16 ms per frame to keep up. gzip on a Genesis-sized
frame plus zlib on a state every 15 frames should fit, but not by a huge
margin. Since the queue is unbounded, if it does not fit the symptom is memory
growth rather than dropped frames.

### Disk footprint

- Frames: gzip level 1 on game pixels compresses well. Crafter measured 7.4 KB
  median in daylight but ~200 KB at night because of per-pixel noise
  (`configs/dbp_games/crafter__crafter_L4.json`). ALE frames are flat colour,
  likely 2-5 KB each, so ~10-20 MB per 5-minute block at 60 fps.
- States: ALE ~0.4 KB per frame (README storage note). Retro ~1 MB raw per
  anchor, perhaps 100-300 KB after zlib, so with stride 15 at 60 fps roughly
  4 anchors/s, on the order of 0.5-1 MB/s. Crafter ~57 KB per anchor after
  zlib.
- Audio: uncompressed int16 stereo. 44.1 kHz is ~176 KB/s, about 53 MB per
  5-minute block. ALE's 31.4 kHz is ~125 KB/s.

### Caveat: `variables` as JSON

The `variables` dict is JSON-encoded every frame via `_json_default`'s
`tolist()` (`logging.py:349`). For ALE that is 128 bytes of RAM, about 400
bytes of JSON text. For retro it is the full console RAM
(`adapters/retro.py:55`): a Genesis has 64 KB of work RAM, which as a JSON list
of ints is ~250 KB of text per frame, or ~15 MB/s at 60 fps, all in the writer
process's `json.dumps` and in events.jsonl. If that is what the retro
`get_ram()` returns, it dwarfs the frames, the states, and the audio combined,
and is likely the single largest write cost in the system. The README's
storage note only calls out savestates, not this. Measure before trusting the
number.
