"""Where a subject stands in a curriculum, so that a later run carries on from there.

A curriculum belongs to the subject and a run to the scanner: the subject works
through the levels at their own pace, and a run is wherever the scanner's clock
cut that work. So every run's ``manifest.json`` holds a ``progress`` record,
kept current as the run plays, which says where the next run of the same
curriculum starts; ``fmri-play --resume`` reads the subject's latest one,
across all their sessions. The same curriculum means the same config file
name, which is the run's BIDS task label.

``progress`` is self-contained: the next run needs nothing from the run
folder but its manifest. It holds ``next_phase``, the first phase not played
to its end, and ``game``, the last game phase reached: its ``index``, the
``phase`` as the config file has it, whether it ``finished``, and, for a block
the run cut off, ``block``: the seconds of play it has had, the episodes
counted toward it, its base seed, and the episode it was in, with that
episode's savestate (the ``block_extra()["state"]`` of the block's adapter).

The position is kept by the game phase rather than by its index, because a
curriculum is edited between runs: levels are added, removed or retimed. The
next run finds that phase again by what makes its game -- its backend, game
id and every field the adapter or the env reads -- and refuses one that is
gone or ambiguous. A field the run loop owns (:data:`LOOP_FIELDS`, such as
``duration`` or ``keys``) may change; the run says which did and plays the
new value. Non-game phases after the last game phase reached are counted
from it, and one the run cut off is played again from its beginning.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from typing import Any, NamedTuple

#: Game-phase fields the run loop reads and the adapter does not depend on: a
#: phase that differs only in these plays the same game. Any other field
#: reaches the adapter or the env.
LOOP_FIELDS = frozenset({
    "type", "mode", "duration", "n_episodes", "max_duration", "advancing_outcomes",
    "outcome_duration", "seed", "state_stride", "frame_stride", "turn_based",
    "latched_keys", "live_hud", "audio", "fps", "keys", "menu", "text", "instructions",
    "include_on_resume",
})


class Start(NamedTuple):
    """Where a run begins in its curriculum."""

    #: The first phase played after the opening ones (``include_on_resume``).
    index: int
    #: The ``progress`` record as it stands before the run plays anything.
    record: dict
    #: For the manifest: which run this one continues and what changed since.
    note: dict

    @property
    def block(self) -> dict | None:
        """The progress of the block at :attr:`index`, when the run continues one."""
        game = self.record["game"]
        if game is None or game["finished"] or game["index"] != self.index:
            return None
        return game["block"]

    def schedule(self, curriculum: list[dict]) -> list[int]:
        """The phase indices this run plays, in order.

        :param curriculum: the run's phases.
        :return: the opening phases before :attr:`index`, when the run
            resumes, then every phase from :attr:`index` on.
        """
        opening = [i for i in range(self.index)
                   if curriculum[i].get("include_on_resume", False)]
        return opening + list(range(self.index, len(curriculum)))


def fresh(why: str) -> Start:
    """The start of a curriculum.

    :param why: for the manifest, why the run did not resume.
    """
    return Start(0, {"next_phase": 0, "game": None}, {"resumed": False, "why": why})


def start_of(curriculum: list[dict], root: str, subject: str, task: str,
             wanted: bool) -> Start:
    """Where a run of this curriculum begins: its start, or where the subject left off.

    :param curriculum: the phases as the config file has them.
    :param root: the BIDS tree's root.
    :param subject: ``sub-<label>``.
    :param task: the curriculum's task label.
    :param wanted: whether the run resumes (``--resume``).
    :return: the start; with no earlier run to resume, the curriculum's
        beginning, said on stderr and in the start's note.
    :raises ValueError: see :func:`resume`.
    """
    if not wanted:
        return fresh("not asked to (--resume)")
    previous = latest(root, subject, task)
    if previous is None:
        print(f"resume: {subject} has no earlier run of task {task}: starting at the "
              "beginning", file=sys.stderr)
        return fresh("no earlier run of this curriculum")
    return resume(curriculum, *previous)


def game_of(phase: dict) -> dict:
    """The fields of a game phase that make its game: all but :data:`LOOP_FIELDS` and notes."""
    return {k: v for k, v in phase.items() if k not in LOOP_FIELDS and not k.startswith("_")}


def latest(root: str, subject: str, task: str) -> tuple[str, dict] | None:
    """The subject's last saved progress through a curriculum, from any session.

    :param root: the BIDS tree's root.
    :param subject: ``sub-<label>``.
    :param task: the curriculum's task label (:func:`~fmri_gym.bids.task_label`).
    :return: ``(run_label, progress)`` of the run that saved last, or ``None``.
    """
    pattern = os.path.join(root, subject, "ses-*", "beh",
                           f"{subject}_ses-*_task-{task}_run-*", "manifest.json")
    found = []
    for path in glob.glob(pattern):
        with open(path) as f:
            manifest = json.load(f)
        if "progress" in manifest:
            found.append((manifest["progress"]["saved"], manifest["run"]["label"],
                          manifest["progress"]))
    if not found:
        return None
    _, label, record = max(found, key=lambda t: t[0])
    return label, record


def resume(curriculum: list[dict], run: str, record: dict) -> Start:
    """Where this curriculum carries on from a run's saved progress.

    :param curriculum: the phases as the config file has them.
    :param run: the label of the run that saved ``record``.
    :param record: its ``progress``.
    :return: the start, with its record in this curriculum's indices.
    :raises ValueError: if the last game phase reached is not in this
        curriculum (or is twice), or the curriculum was played to its end.
    """
    game, changed = record["game"], []
    if game is None:
        index = record["next_phase"]
    else:
        found = _find(curriculum, game, run)
        changed = _changes(game["phase"], curriculum[found], found)
        index = found
        if game["finished"]:
            index = _after(curriculum, found, record["next_phase"] - game["index"] - 1)
        game = {**game, "index": found, "phase": curriculum[found]}
    if index >= len(curriculum):
        raise ValueError(f"--resume: {run} played this curriculum to its end; there is "
                         "nothing to resume. Start it over without --resume")
    note = {"resumed": True, "from_run": run, "phase": index, "changed": changed}
    start = Start(index, {"next_phase": index, "game": game}, note)
    block = start.block
    if block is not None and block["episode"] is not None and block["episode"]["state"] is None:
        print(f"resume: phase {index}'s backend keeps no savestate, so episode "
              f"{block['episode']['id']} starts again from its seed", file=sys.stderr)
        note["episode_restarts"] = block["episode"]["id"]
        block["episode"] = None
    print(f"resume: {run} left off at phase {index}"
          + (" mid-block" if block is not None else ""), file=sys.stderr)
    return start


def _find(curriculum: list[dict], game: dict, run: str) -> int:
    """The index of the game phase that plays ``game["phase"]``'s game."""
    want = game_of(game["phase"])
    matches = [i for i, p in enumerate(curriculum)
               if p["type"] == "game" and game_of(p) == want]
    if game["index"] in matches:
        return game["index"]
    if len(matches) == 1:
        return matches[0]
    where = "no phase of this curriculum plays" if not matches else f"phases {matches} all play"
    raise ValueError(f"--resume: {run} was last in game phase {game['index']} ({want}), which "
                     f"{where}. Put that phase back, keep only one, or start over without "
                     "--resume")


def _after(curriculum: list[dict], found: int, passed: int) -> int:
    """The first phase after ``found`` not yet played: past ``passed`` non-game phases at most."""
    index = found + 1
    while passed > 0 and index < len(curriculum) and curriculum[index]["type"] != "game":
        index, passed = index + 1, passed - 1
    return index


def _changes(old: dict, new: dict, index: int) -> list[str]:
    """The loop fields of a game phase that changed since the run that saved it, said aloud."""
    changed = []
    for key in sorted((old.keys() | new.keys()) & LOOP_FIELDS):
        if old.get(key) != new.get(key):
            changed.append(f"{key}: {old.get(key)!r} -> {new.get(key)!r}")
            print(f"resume: phase {index}'s {changed[-1]}; this run plays the new value",
                  file=sys.stderr)
    return changed


class Progress:
    """The ``progress`` record of one run, kept current as it plays.

    :ivar start: where the run began.
    :ivar record: ``next_phase`` and ``game``, in this run's indices (see the
        module docstring).
    """

    def __init__(self, curriculum: list[dict], start: Start) -> None:
        """
        :param curriculum: the phases as the config file has them, before the
            run folds seeds and options into its own copy.
        :param start: where the run begins.
        """
        self._curriculum = curriculum
        self.start = start
        self.record = start.record

    def passed(self, index: int) -> None:
        """Phase ``index`` was played to its end."""
        game = self.record["game"]
        if self._curriculum[index]["type"] == "game":
            game = {"index": index, "phase": self._curriculum[index], "finished": True,
                    "block": None}
        self.record = {"next_phase": index + 1, "game": game}

    def cut(self, index: int, block: dict) -> None:
        """Game phase ``index`` stopped before its end; ``block`` is how far it got."""
        self.record = {"next_phase": index, "game": {
            "index": index, "phase": self._curriculum[index], "finished": False,
            "block": block}}

    def describe(self) -> dict[str, Any]:
        """The record for the manifest, stamped with when it was saved."""
        return {**self.record, "saved": time.time()}
