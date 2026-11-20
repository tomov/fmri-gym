"""A run's BIDS events table, made from its timeline.

``<label>_events.tsv`` and its sidecar ``<label>_events.json``, where ``<label>`` is the run's
BIDS name: one row per screen, game block, episode and press of a control, from the trigger on
(:mod:`.run_timeline`). ``onset`` and ``duration`` are seconds of the run clock, whose zero is
the trigger; a press's onset is when its device sent it if the run's devices were read
from the OS, else when the run read it, which ``read_onset`` always gives. The scanner's
pulses, each frame and each code sent stay in the logs and the readable timeline.
"""

from __future__ import annotations

import json
import os
from typing import NamedTuple

from .run_timeline import Timeline

#: The table's columns, in order, each with its sidecar description.
COLUMNS = {
    "onset": "Seconds from the trigger to the start of the event; for a press, when its device "
             "sent it if the run's devices were read from the OS, else when the run read it.",
    "duration": "Seconds the event lasted.",
    "trial_type": "What the event is.",
    "block": "The game block it belongs to: the folder holding its frames.",
    "episode": "The episode of the block it belongs to, from 0.",
    "frame": "The block's frame (step) a press was given to.",
    "control": "The control pressed.",
    "read_onset": "For a press: seconds from the trigger to when the run read it.",
    "outcome": "How an episode ended.",
    "score": "An episode's score.",
    "text": "A text screen's text, each line break written as \\n.",
}
#: Each ``trial_type``, with its sidecar description.
TRIAL_TYPES = {
    "fixation": "A fixation cross on screen.",
    "text": "A screen of text (instructions, a controls image, loading, an outcome, a pop-up).",
    "block": "A game block, from its start to its end.",
    "episode": "An episode, from its first frame on screen to its end.",
    "press": "A control pressed, until it was let go.",
}
MISSING = "n/a"


class EventRow(NamedTuple):
    """One row of the table; a field left ``None`` is written as ``n/a``."""

    onset: float
    duration: float | None
    trial_type: str
    block: str | None = None
    episode: int | None = None
    frame: int | None = None
    control: str | None = None
    read_onset: float | None = None
    outcome: str | None = None
    score: float | None = None
    text: str | None = None


def event_rows(timeline: Timeline) -> list[EventRow]:
    """The table's rows, in order of onset; nothing before the trigger, no scanner pulse.

    :param timeline: the run's timeline.
    :return: the rows.
    """
    rows = [EventRow(screen.onset, screen.duration, screen.kind, block=screen.block,
                     text=screen.text) for screen in timeline.screens]
    rows += [EventRow(block.onset, block.duration, "block", block=block.name)
             for block in timeline.blocks]
    rows += [EventRow(episode.onset, episode.duration, "episode", block=episode.block,
                      episode=episode.episode, outcome=episode.outcome, score=episode.score)
             for episode in timeline.episodes]
    rows += [EventRow(press.onset, press.duration, "press", block=press.block,
                      episode=press.episode, frame=press.frame, control=press.control,
                      read_onset=press.read_time)
             for press in timeline.presses if not press.is_scanner_pulse]
    return sorted((row for row in rows if row.onset >= 0), key=lambda row: row.onset)


def write_events_table(timeline: Timeline, table_folder: str) -> str:
    """Write a run's events table and its sidecar.

    :param timeline: the run's timeline; its label starts both files' names.
    :param table_folder: the folder the two files go in, made if missing.
    :return: the table's path.
    """
    os.makedirs(table_folder, exist_ok=True)
    path = os.path.join(table_folder, f"{timeline.label}_events.tsv")
    with open(path, "w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for row in event_rows(timeline):
            f.write("\t".join(_cell(value) for value in row) + "\n")
    sidecar = {name: {"Description": description} for name, description in COLUMNS.items()}
    for name in ("onset", "duration", "read_onset"):
        sidecar[name]["Units"] = "s"
    sidecar["trial_type"]["Levels"] = TRIAL_TYPES
    with open(os.path.join(table_folder, f"{timeline.label}_events.json"), "w") as f:
        json.dump(sidecar, f, indent=2)
    return path


def _cell(value: object) -> str:
    """One value as a cell: ``n/a`` for none, a number to the microsecond, text on one line."""
    if value is None:
        return MISSING
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("\t", " ").replace("\n", "\\n")
