"""Reconstruction and analysis on top of a logged block (:mod:`humans.save.logging`).

An episode is replayed from its seed and its actions through a fresh adapter, one for the
backend and phase of the block's own ``block_start`` line, so a block's
log is enough to recreate the game the subject saw, frame for frame, on a deterministic engine.
An episode that carried on from an earlier run starts from the savestate its
``episode_start`` line holds (``initial_state``), restored over the reset.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from .logging import read_events, read_state


class EpisodeReconstructionPlan(NamedTuple):
    """One episode's start and every action it took, from its block's log."""

    episode_id: int
    #: the seed its reset took
    seed: int
    #: the savestate restored over that reset, or ``None`` for the seed alone
    initial_state: bytes | None
    #: each frame's logged ``action``, in order
    actions: list


def reconstruction_plan(block: str, episode_id: int | None = None) -> EpisodeReconstructionPlan:
    """One episode's seed and actions.

    :param block: the block's folder.
    :param episode_id: which episode; ``None`` = the last one started (the
        crash-recovery case).
    :return: what replays the episode.
    :raises ValueError: if the block has no episode.
    """
    events = read_events(block)
    starts = {e["episode_id"]: e for e in events if e["type"] == "episode_start"}
    if not starts:
        raise ValueError(f"no episode_start records in {block}")
    if episode_id is None:
        episode_id = max(starts)
    actions = [e["action"] for e in events
               if e["type"] == "frame" and e["episode_id"] == episode_id]
    state = starts[episode_id].get("initial_state")  # blocks logged before it was a field lack it
    return EpisodeReconstructionPlan(episode_id, starts[episode_id]["seed"],
                                     None if state is None else read_state(state), actions)


def reconstruct_episode(
    block: str, episode_id: int | None = None, *, adapter: Any,
    n_actions: int | None = None,
) -> tuple[Any, EpisodeReconstructionPlan]:
    """Replay one episode through an adapter, leaving it at the episode's last frame.

    :param block: the block's folder.
    :param episode_id: which episode (see :func:`reconstruction_plan`).
    :param adapter: the adapter to drive, fresh, for the ``backend`` and ``phase`` of the
        block's first line.
    :param n_actions: replay only the first this many of the episode's actions
        (``0``: the reset alone), leaving the adapter at the logged frame
        ``ep_frame == n_actions - 1``; ``None`` replays them all.
    :return: ``(adapter, plan)``; the caller owns the adapter's ``close()``.
    """
    plan = reconstruction_plan(block, episode_id)
    adapter.reset(plan.seed)
    if plan.initial_state is not None:
        adapter.restore(plan.initial_state)
    actions = plan.actions if n_actions is None else plan.actions[:n_actions]
    for action in actions:
        obs, _, _, _, info = adapter.step(action)
        adapter.capture(obs, info, want_blob=False)
    return adapter, plan


def frame_arrays(block: str) -> dict[str, np.ndarray]:
    """A block's ``frame`` lines as parallel arrays, one entry per frame.

    :param block: the block's folder.
    :return: one array per field, the adapter's ``variables`` flattened to
        top-level keys; a field a frame lacks is ``None`` there. Heterogeneous
        fields become object arrays.
    """
    frames = [e for e in read_events(block) if e["type"] == "frame"]
    columns: dict[str, list] = {}
    for i, f in enumerate(frames):
        for key, value in {**f, **f.get("variables", {})}.items():
            if key in ("type", "variables"):
                continue
            columns.setdefault(key, [None] * len(frames))[i] = value
    arrays = {}
    for key, values in columns.items():
        try:
            arrays[key] = np.asarray(values)
        except Exception:
            arrays[key] = np.array(values, dtype=object)
    return arrays
