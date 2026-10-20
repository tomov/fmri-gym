"""A model's internal state over a transcript, one row per turn, in one HDF5 file.

::

    states.h5
      hidden      (n_turns, n_layers, hidden_size) float16   the state at each turn's header
      frame       (n_turns,) int64    the turn's frame in the block (the fMRI alignment key)
      episode_id  (n_turns,) int64
      ep_frame    (n_turns,) int64
      run_time    (n_turns,) float64  the frame's logged run_time
      context_start (n_turns,) int64  the episode turn its window began at
      attrs: model, backend (JSON: how the model was run), layers, transcript, window,
             system_prompt, game_backend, game, subject

Rows land in turn order as each window is read and are flushed then, so a
run that dies keeps every window it finished. ``context_start`` with the
transcript and ``system_prompt`` rebuilds the exact chat a row was read from
(:func:`agents.lrm.prompting.messages`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import h5py
import numpy as np

from .transcript import TranscriptHeader, Turn

#: the per-turn index columns, beside ``hidden``
COLUMNS = {"frame": "int64", "episode_id": "int64", "ep_frame": "int64",
           "run_time": "float64", "context_start": "int64"}


@dataclass(frozen=True)
class StateConfig:
    """How the states in a file were made; stored as the file's attributes.

    :param backend: the backend's :meth:`~agents.lrm.backends.LRMBackend.record`.
    :param layers: hidden-state layers kept, ``0`` the embeddings.
    :param transcript: the transcript file read.
    :param window: turns per window (:func:`agents.lrm.prompting.windows`).
    :param system_prompt: the system message every window began with.
    """

    backend: dict
    layers: tuple[int, ...]
    transcript: str
    window: int
    system_prompt: str

    def record(self, header: TranscriptHeader) -> dict:
        """The file's attributes: this config, and the transcript's game and subject."""
        return {"model": self.backend["model"], "backend": json.dumps(self.backend),
                "layers": json.dumps(list(self.layers)), "transcript": self.transcript,
                "window": self.window, "system_prompt": self.system_prompt,
                "game_backend": header.backend, "game": header.game, "subject": header.subject}


class StateWriter:
    """Appends turns' states to an HDF5 file, flushing after each window.

    :param path: the file; an existing one is replaced.
    :param config: how the states are made.
    :param header: the transcript's header (game and subject, for the attrs).
    """

    def __init__(self, path: str, config: StateConfig, header: TranscriptHeader) -> None:
        self.file = h5py.File(path, "w")
        self.file.attrs.update(config.record(header))
        for name, dtype in COLUMNS.items():
            self.file.create_dataset(name, shape=(0,), maxshape=(None,), dtype=dtype)

    def append(self, turns: list[Turn], hidden: np.ndarray, context_start: int) -> None:
        """Add one window's reported turns and their states, then flush.

        :param turns: the turns, in order.
        :param hidden: ``(len(turns), n_layers, hidden_size)``.
        :param context_start: the episode turn the window began at.
        """
        if len(turns) != len(hidden):
            raise ValueError(f"{len(turns)} turns but {len(hidden)} states")
        if "hidden" not in self.file:
            # The width is the model's, known from its first states.
            shape = hidden.shape[1:]
            self.file.create_dataset("hidden", shape=(0, *shape), maxshape=(None, *shape),
                                     dtype="float16", chunks=(1, *shape))
        n = self.file["hidden"].shape[0]
        values = {"frame": [t.frame for t in turns], "episode_id": [t.episode_id for t in turns],
                  "ep_frame": [t.ep_frame for t in turns], "run_time": [t.run_time for t in turns],
                  "context_start": [context_start] * len(turns)}
        for name, column in {"hidden": hidden, **values}.items():
            dataset = self.file[name]
            dataset.resize(n + len(turns), axis=0)
            dataset[n:] = column
        self.file.flush()

    def close(self) -> None:
        self.file.close()
