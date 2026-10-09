"""Reconstruction and analysis on top of a logged block (:mod:`fmri_gym.logging`).

An episode is replayed from its seed and its actions through a fresh adapter
built from the block's own ``block_start`` line, so a block's log is enough to
recreate the game the subject saw, frame for frame, on a deterministic engine.
An episode that was handed an earlier block's world starts from that world
instead of from its seed, which is also on the block's own log.

An episode the subject went back through is the one case where its actions in
order are not the game it played: a rollback (:mod:`fmri_gym.rewind`) undoes
the frames between where it went back to and where it was, and everything
after that happened in the restored world. So an episode is a *sequence of
segments* rather than one action list, cut at each ``rewind`` line, each
starting from the world on that line. A block nobody went back through has one
segment and reads as it always did.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .logging import read_events, read_state


def reconstruction_plan(block: str, episode_id: int | None = None) -> dict:
    """One episode's seed and actions, and the worlds its seed does not name.

    :param block: the block's folder.
    :param episode_id: which episode; ``None`` = the last one started (the
        crash-recovery case).
    :return: ``{"episode_id", "seed", "actions", "resumed", "state",
        "rewinds", "segments"}``. ``state`` is the world to start from and
        ``None`` unless ``resumed``. ``actions`` is every action the subject
        sent, in order, which for an episode with a rollback includes the ones
        it undid: ``segments`` is the replayable form, a list of
        ``{"state", "actions"}`` to restore and step in turn (``state``
        ``None`` only for a first segment that started from the seed).
        ``rewinds`` is each rollback's record without its blob.
    :raises ValueError: if the block has no episode, or a resumed one has no
        world on the block's ``resume`` line to start from, or a rollback has
        no world on its own line.
    """
    events = read_events(block)
    starts = {e["episode_id"]: e for e in events if e["type"] == "episode_start"}
    if not starts:
        raise ValueError(f"no episode_start records in {block}")
    if episode_id is None:
        episode_id = max(starts)
    actions = [e["action"] for e in events
               if e["type"] == "frame" and e["episode_id"] == episode_id]
    # A resumed episode's seed opened a world nobody played, so its seed and
    # its actions reproduce a different game; what it was handed is the block's
    # own `resume` line (see "Resuming a world" in the README).
    resumed = bool(starts[episode_id].get("resumed"))
    state = next((read_state(e) for e in events if e["type"] == "resume"), None)
    if resumed and state is None:
        raise ValueError(f"episode {episode_id} of {block} was resumed, but the block "
                         "has no resume line to reconstruct it from")
    # The episode's frames and its rollbacks, in the order they were written:
    # a rollback ends a segment where it happened and starts the next one in
    # the world it restored (see "Going back a few frames" in the README).
    rewinds, segments = [], [{"state": state if resumed else None, "actions": []}]
    for event in events:
        if event.get("episode_id") != episode_id:
            continue
        if event["type"] == "frame":
            segments[-1]["actions"].append(event["action"])
        elif event["type"] == "rewind":
            world = read_state(event)
            if world is None:
                raise ValueError(f"episode {episode_id} of {block} was rewound at frame "
                                 f"{event.get('from_ep_frame')}, but that rewind line has no "
                                 "world to reconstruct the rest of the episode from")
            rewinds.append({k: v for k, v in event.items() if k != "state"})
            segments.append({"state": world, "actions": []})
    return {"episode_id": episode_id, "seed": starts[episode_id]["seed"],
            "actions": actions, "resumed": resumed, "state": state if resumed else None,
            "rewinds": rewinds, "segments": segments}


def reconstruct_episode(
    block: str, episode_id: int | None = None, adapter: Any = None,
) -> tuple[Any, dict]:
    """Replay one episode through an adapter, leaving it at the episode's last frame.

    :param block: the block's folder.
    :param episode_id: which episode (see :func:`reconstruction_plan`).
    :param adapter: the adapter to drive; built from the block's ``phase`` if omitted.
    :return: ``(adapter, plan)``; the caller owns the adapter's ``close()``.
    """
    plan = reconstruction_plan(block, episode_id)
    if adapter is None:
        from .adapters import get_adapter
        start = read_events(block)[0]
        adapter = get_adapter(start["backend"], start["phase"])
    if plan["episode_id"] != 0:
        getattr(adapter, "warm_up", lambda: None)()
    adapter.reset(plan["seed"])
    # One segment unless the subject went back through the episode, and then
    # one per rollback: its actions were played in the world on its own line,
    # and the ones before it were undone. Reset first regardless: the episode
    # state an adapter builds there is what restore then replaces the env
    # underneath, as the run's own loop does it.
    for segment in plan["segments"]:
        if segment["state"] is not None:
            adapter.restore(segment["state"])
        for action in segment["actions"]:
            obs, _, _, _, info = adapter.step(action)
            # The adapter's own per-frame state, for the same reason the run
            # loop asks for it: one that folds observation-derived state in
            # updates it here and nowhere else. No blob -- a reconstruction
            # walks to a frame, it does not record one.
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
