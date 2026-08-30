"""Session logging: a manifest.json + one compressed .npz per game block.

The logger is engine-agnostic: it consumes the standard FrameState objects the
adapter produces, so the on-disk schema is identical across backends. Analysis
code loads the same fields whether the block was Atari, Genesis, or CartPole.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .session import Clock


class Logger:
    def __init__(
        self, outdir: str, subject: str, curriculum: list[dict], clock: Clock
    ) -> None:
        self.outdir = outdir
        self.clock = clock
        os.makedirs(outdir, exist_ok=True)
        self.manifest = {
            "subject": subject,
            "curriculum": curriculum,
            "start_epoch": None,
            "phases": [],
        }

    def set_trigger_time(self) -> None:
        self.manifest["start_epoch"] = self.clock.t0_epoch
        self.manifest["trigger_perf"] = self.clock.t0_perf

    def log_phase(self, entry: dict) -> None:
        self.manifest["phases"].append(entry)

    def save_game_block(
        self,
        block_index: int,
        backend: str,
        game: str,
        frames: dict,
        extra: dict | None = None,
    ) -> str:
        """Write one game block's per-frame arrays.

        `frames` holds parallel lists collected by the session loop. Backend-
        specific variables (ram, obs, screen_index, retro info vars, ...) are
        stacked under their own keys. `extra` is a dict of block-level arrays
        (e.g. an ALE palette) merged in verbatim.
        """
        safe_game = game.split("/")[-1].replace(":", "_")
        path = os.path.join(
            self.outdir, f"block-{block_index:02d}_{backend}_{safe_game}.npz")
        arrays = dict(
            actions=_to_array(frames["action"]),
            rewards=np.asarray(frames["reward"], dtype=np.float32),
            terminated=np.asarray(frames["terminated"], dtype=bool),
            truncated=np.asarray(frames["truncated"], dtype=bool),
            episode_id=np.asarray(frames["episode_id"], dtype=np.int32),
            session_time=np.asarray(frames["session_time"], dtype=np.float64),
            wall_time=np.asarray(frames["wall_time"], dtype=np.float64),
            # Opaque per-frame savestate blobs (object array of bytes|None).
            states=np.array(frames["state_blob"], dtype=object),
            episode_seeds=np.asarray(frames["episode_seeds"], dtype=np.int64),
            backend=backend,
            game=game,
        )
        # Stack every named variable the adapter surfaced (ram, obs, ...).
        for key, series in frames["variables"].items():
            try:
                arrays[key] = np.asarray(series)
            except Exception:
                arrays[key] = np.array(series, dtype=object)
        if extra:
            arrays.update(extra)
        np.savez_compressed(path, **arrays)
        return path

    def save_manifest(self) -> str:
        path = os.path.join(self.outdir, "manifest.json")
        with open(path, "w") as f:
            json.dump(self.manifest, f, indent=2, default=_json_default)
        return path


def _to_array(actions: list) -> np.ndarray:
    """Actions may be ints (Discrete), arrays (Box), or button lists (retro)."""
    try:
        return np.asarray(actions)
    except Exception:
        return np.array(actions, dtype=object)


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
