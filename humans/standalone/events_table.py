"""A run's BIDS events table, made from its log.

``<label>_events.tsv`` and its sidecar ``<label>_events.json``, where ``<label>`` is the
run's BIDS name. One row per screen, game block, episode and press of a control, from the
trigger on; ``onset`` and ``duration`` are seconds of the run clock, whose
zero is the trigger. A screen lasts until the next screen, the next game frame or the end of
the run; a press until the control is let go. The scanner's pulses and each frame stay in the
log, where they are already lines of their own.
"""

from __future__ import annotations

import json
import os
from typing import NamedTuple

from ..save.logging import read_lines

#: The table's columns, in order, each with its sidecar description.
COLUMNS = {
    "onset": "Seconds from the trigger to the start of the event.",
    "duration": "Seconds the event lasted.",
    "trial_type": "What the event is.",
    "block": "The game block it belongs to: the folder holding its frames.",
    "episode": "The episode of the block it belongs to, from 0.",
    "frame": "The block's frame (step) a press was given to.",
    "control": "The control pressed.",
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
    outcome: str | None = None
    score: float | None = None
    text: str | None = None


def write_events_table(log_path: str, table_folder: str, label: str) -> str:
    """Write a run's events table and its sidecar from its log.

    :param log_path: the run's ``events.jsonl``.
    :param table_folder: the folder the two files go in, made if missing.
    :param label: the run's BIDS name, which the two files start with.
    :return: the table's path.
    """
    rows = sorted(event_rows(read_lines(log_path)), key=lambda row: row.onset)
    os.makedirs(table_folder, exist_ok=True)
    path = os.path.join(table_folder, f"{label}_events.tsv")
    with open(path, "w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for row in rows:
            f.write("\t".join(_cell(value) for value in row) + "\n")
    sidecar = {name: {"Description": description} for name, description in COLUMNS.items()}
    for name in ("onset", "duration"):
        sidecar[name]["Units"] = "s"
    sidecar["trial_type"]["Levels"] = TRIAL_TYPES
    with open(os.path.join(table_folder, f"{label}_events.json"), "w") as f:
        json.dump(sidecar, f, indent=2)
    return path


def event_rows(lines: list[dict]) -> list[EventRow]:
    """The table's rows from a run's log lines, in the log's order; a line before the trigger
    makes none."""
    lines = [line for line in lines if line.get("run_time") is not None
             or line["type"] == "episode_start"]
    if not lines:
        return []
    return (_screen_rows(lines) + _block_rows(lines) + _episode_rows(lines)
            + _press_rows(lines))


def _screen_rows(lines: list[dict]) -> list[EventRow]:
    """Each screen, until what is shown next: a screen, a game frame, or the run's end."""
    shown = [(_shown_at(line), line) for line in lines if _shown_at(line) is not None]
    rows = []
    for position, (onset, line) in enumerate(shown):
        if line["type"] != "screen":
            continue
        later = [at for at, _ in shown[position + 1:] if at > onset]
        offset = later[0] if later else lines[-1]["run_time"]
        rows.append(EventRow(onset, offset - onset, line["kind"], block=line["block"],
                             text=line["text"]))
    return rows


def _shown_at(line: dict) -> float | None:
    """When a line put something new on screen, by the run clock; None if it did not."""
    if line["type"] == "screen":
        return line["run_time"]
    if line["type"] in ("frame", "episode_start"):
        return line["flip_time"]
    return None


def _block_rows(lines: list[dict]) -> list[EventRow]:
    """Each game block, from its start line to its end line, or to the run's last line."""
    starts = {line["block"]: line for line in lines if line["type"] == "block_start"}
    ends = {line["block"]: line for line in lines if line["type"] == "block_end"}
    return [EventRow(start["run_time"], ends.get(block, lines[-1])["run_time"] - start["run_time"],
                     "block", block=block)
            for block, start in starts.items()]


def _episode_rows(lines: list[dict]) -> list[EventRow]:
    """Each episode, from its first frame on screen to its end line."""
    rows, start = [], None
    for line in lines:
        if line["type"] == "episode_start":
            start = line
        elif line["type"] == "episode_end" and start is not None:
            rows.append(EventRow(start["flip_time"], line["run_time"] - start["flip_time"],
                                 "episode", block=line["block"], episode=line["episode_id"],
                                 outcome=line["outcome"], score=line["score"]))
            start = None
    return rows


def _press_rows(lines: list[dict]) -> list[EventRow]:
    """Each press of a control, until it is let go, with the frame of its block it was given
    to: the next frame of its episode that names it among its ``controls_pressed``."""
    rows: list[EventRow] = []
    open_presses: dict[str, int] = {}       # control -> its row, until it is let go
    waiting_for_frame: list[int] = []       # rows of the presses since the last frame
    episode: int | None = None
    for line in lines:
        if line["type"] == "episode_start":
            episode, waiting_for_frame = line["episode_id"], []
        elif line["type"] in ("episode_end", "block_end"):
            episode, waiting_for_frame = None, []
        elif line["type"] == "frame":
            given = list(line["controls_pressed"])
            for row in waiting_for_frame:
                if rows[row].control in given:
                    given.remove(rows[row].control)
                    rows[row] = rows[row]._replace(frame=line["frame"])
            waiting_for_frame = []
        elif line["type"] == "input_event" and line["down"]:
            open_presses[line["name"]] = len(rows)
            if episode is not None:
                waiting_for_frame.append(len(rows))
            rows.append(EventRow(line["run_time"], None, "press", block=line["block"],
                                 episode=episode, control=line["name"]))
        elif line["type"] == "input_event" and line["name"] in open_presses:
            row = open_presses.pop(line["name"])
            rows[row] = rows[row]._replace(duration=line["run_time"] - rows[row].onset)
    return rows


def _cell(value: object) -> str:
    """One value as a cell: ``n/a`` for none, a number to the microsecond, text on one line."""
    if value is None:
        return MISSING
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value).replace("\t", " ").replace("\n", "\\n")
