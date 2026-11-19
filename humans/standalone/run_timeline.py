"""A run's timeline: its manifest and its logs read together, every event on the run clock.

A run's data is one record (:class:`RunRecord`): the manifest says what the logs are read
with -- the trigger's ``perf_counter`` (``trigger_perf``), the rig and its scanner key, the
display, the log format -- the run's ``pygame_events.jsonl`` holds what the run read, showed,
sent and printed, and each block's ``game_events.jsonl`` what its game did. :func:`build_timeline`
joins them once, in seconds from the trigger: screens, blocks, episodes, presses with the step
each was given to, the scanner's pulses, the codes sent, what was printed, and the numbers a run
is checked by.
The events table (:mod:`.events_table`) and the readable timeline (:mod:`.timeline_text`)
are both written from it.
"""

from __future__ import annotations

import glob
import json
import math
import os
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..save.logging import GAME_EVENTS_FILENAME, RUN_EVENTS_FILENAME, read_lines

#: The oldest log format read here: the run's own log beside its blocks'.
OLDEST_LOG_FORMAT = 2
#: A gap between flips longer than this many refreshes is a missed refresh.
MISSED_REFRESH = 1.5
#: A gap between scanner pulses longer than this many of their median is a missed pulse.
MISSED_PULSE = 1.5

#: ``perf_counter`` time to run time.
RunClock = Callable[[float], float]


class LogFormatError(ValueError):
    """A run recorded in a log format older than :data:`OLDEST_LOG_FORMAT`."""


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One run's manifest and logs, as written."""

    #: the run's folder
    folder: str
    manifest: dict[str, Any]
    #: the run's ``pygame_events.jsonl``
    run_lines: list[dict[str, Any]]
    #: each block's ``game_events.jsonl``, by its folder's name
    block_lines: dict[str, list[dict[str, Any]]]

    @classmethod
    def read(cls, folder: str) -> RunRecord:
        """The record in ``folder``.

        :param folder: the run's folder.
        :return: its manifest and every log in it.
        :raises LogFormatError: if it was recorded in an older log format.
        """
        manifest = json.loads(Path(folder, "manifest.json").read_text())
        version = manifest.get("log_format", {}).get("version", 1)
        if version < OLDEST_LOG_FORMAT:
            raise LogFormatError(f"{folder}: log format {version}, recorded before runs kept a "
                                 f"log of their own; these tools read {OLDEST_LOG_FORMAT} on")
        blocks = {Path(path).parent.name: read_lines(path) for path in
                  sorted(glob.glob(os.path.join(folder, "block-*", GAME_EVENTS_FILENAME)))}
        return cls(folder, manifest, read_lines(os.path.join(folder, RUN_EVENTS_FILENAME)), blocks)


@dataclass(frozen=True, slots=True)
class Screen:
    """A text or fixation screen, from the flip that showed it to the next new picture."""

    onset: float
    duration: float
    #: ``text`` or ``fixation``
    kind: str
    text: str | None
    #: the block it was shown in (a pop-up's), ``None`` outside one
    block: str | None


@dataclass(frozen=True, slots=True)
class Block:
    """A game block, from its first line to its last."""

    name: str
    onset: float
    duration: float
    backend: str
    game: str
    n_frames: int
    #: ``False`` when its log has no ``block_end``: the run stopped inside it
    has_ended: bool


@dataclass(frozen=True, slots=True)
class Episode:
    """An episode, from its first frame on screen to its end."""

    block: str
    episode: int
    onset: float
    #: ``None`` when the run stopped inside it
    duration: float | None
    outcome: str | None
    score: float | None


@dataclass(frozen=True, slots=True)
class Press:
    """A control pressed, until it was let go."""

    control: str
    #: when the run read it
    onset: float
    #: until it was let go, by the reads; ``None`` when it never was
    duration: float | None
    block: str | None
    episode: int | None
    #: the block's frame (step) it was given to; ``None``: none
    frame: int | None
    #: a press of the key the scanner types at every volume
    is_scanner_pulse: bool


@dataclass(frozen=True, slots=True)
class Mark:
    """A moment of the run for the readable timeline: a code sent, a printed line, a missed
    refresh, the run's start and end."""

    time: float
    what: str
    detail: str


@dataclass(frozen=True, slots=True)
class Timeline:
    """A run's events on the run clock, and the numbers it is checked by."""

    label: str
    #: seconds from the trigger to the run's last line
    end: float
    screens: list[Screen]
    blocks: list[Block]
    episodes: list[Episode]
    presses: list[Press]
    marks: list[Mark]
    #: the checks' numbers (:func:`_quality`)
    quality: dict[str, Any] = field(default_factory=dict)


def build_timeline(record: RunRecord) -> Timeline:
    """Join a run's logs on the run clock (seconds from the trigger).

    :param record: a run that reached its trigger (its manifest has ``trigger_perf``).
    :return: its timeline.
    """
    trigger_perf = record.manifest["trigger_perf"]

    def to_run_time(perf_time: float) -> float:
        return perf_time - trigger_perf
    end = to_run_time(max(line["perf_time"] for line in record.run_lines))
    blocks, episodes = _blocks(record, end), _episodes(record)
    presses = _presses(record, to_run_time, episodes)
    gaps = _flip_gaps(record, to_run_time)
    marks = _marks(record, to_run_time) + [
        Mark(time, "display", f"no flip for {1000 * gap:.1f} ms") for time, gap in gaps]
    timeline = Timeline(record.manifest["run"]["label"], end, _screens(record, to_run_time, end),
                        blocks, episodes, presses, marks)
    return replace(timeline, quality=_quality(record, timeline, gaps))


def _screens(record: RunRecord, to_run_time: RunClock, end: float) -> list[Screen]:
    """Each text or fixation flip, until the next flip that showed a new picture."""
    shown = [(to_run_time(line["perf_time"]), line) for line in record.run_lines
             if line["type"] == "flip" and line["kind"] != "repeat"]
    screens = []
    for position, (onset, line) in enumerate(shown):
        if line["kind"] not in ("text", "fixation"):
            continue
        offset = shown[position + 1][0] if position + 1 < len(shown) else end
        screens.append(Screen(onset, offset - onset, line["kind"], line.get("text"),
                              line["block"]))
    return screens


def _blocks(record: RunRecord, end: float) -> list[Block]:
    """Each block, from its ``block_start`` to its ``block_end``, or to the run's last line."""
    blocks = []
    for name, lines in record.block_lines.items():
        start = lines[0]
        stop = next((line for line in lines if line["type"] == "block_end"), None)
        offset = end if stop is None else stop["run_time"]
        blocks.append(Block(name, start["run_time"], offset - start["run_time"],
                            start["backend"], start["game"],
                            sum(line["type"] == "frame" for line in lines), stop is not None))
    return blocks


def _episodes(record: RunRecord) -> list[Episode]:
    """Each episode of each block, from its first frame's flip to its ``episode_end``."""
    episodes = []
    for name, lines in record.block_lines.items():
        start = None
        for line in lines:
            if line["type"] == "episode_start":
                start = line
            elif line["type"] == "episode_end" and start is not None:
                episodes.append(Episode(name, line["episode_id"], start["flip_time"],
                                        line["run_time"] - start["flip_time"], line["outcome"],
                                        line["score"]))
                start = None
        if start is not None:
            episodes.append(Episode(name, start["episode_id"], start["flip_time"], None, None,
                                    None))
    return episodes


def _presses(record: RunRecord, to_run_time: RunClock, episodes: list[Episode]) -> list[Press]:
    """Each press, with how long it was held, its episode and the step it was given to."""
    sync = record.manifest["rig"]["triggers"]["sync"]
    scanner_key = sync["key"] if sync["mode"] == "wait" else None
    open_presses: dict[str, int] = {}
    presses: list[Press] = []
    for line in record.run_lines:
        if line["type"] != "input_event":
            continue
        read_time = to_run_time(line["perf_time"])
        if not line["down"]:
            if line["name"] in open_presses:
                index = open_presses.pop(line["name"])
                presses[index] = replace(presses[index],
                                         duration=read_time - presses[index].onset)
            continue
        open_presses[line["name"]] = len(presses)
        presses.append(Press(line["name"], read_time, None, line["block"],
                             _episode_at(episodes, line["block"], read_time), None,
                             line["name"] == scanner_key))
    return _given_frames(presses, record)


def _episode_at(episodes: list[Episode], block: str | None, time: float) -> int | None:
    """The episode of ``block`` playing at ``time``, if any."""
    for episode in episodes:
        end = math.inf if episode.duration is None else episode.onset + episode.duration
        if episode.block == block and episode.onset <= time <= end:
            return episode.episode
    return None


def _given_frames(presses: list[Press], record: RunRecord) -> list[Press]:
    """``presses``, each with the frame of its block it was given to: the next frame of its
    episode whose ``controls_pressed`` names it."""
    out = list(presses)
    for name, lines in record.block_lines.items():
        stream = sorted(
            [(press.onset, 0, index) for index, press in enumerate(out)
             if press.block == name and not press.is_scanner_pulse]
            + [(line["run_time"] if line["type"] == "frame" else line["flip_time"], 1, line)
               for line in lines if line["type"] in ("frame", "episode_start")],
            key=lambda timed: (timed[0], timed[1]))
        waiting: list[int] = []
        for _, is_line, item in stream:
            if not is_line:
                waiting.append(item)
                continue
            if item["type"] == "frame":
                given = list(item["controls_pressed"])
                for index in waiting:
                    if out[index].control in given:
                        given.remove(out[index].control)
                        out[index] = replace(out[index], frame=item["frame"])
            waiting = []
    return out


def _marks(record: RunRecord, to_run_time: RunClock) -> list[Mark]:
    """The run's other moments: start and end, the trigger, codes sent, printed lines."""
    marks = []
    for line in record.run_lines:
        time = to_run_time(line["perf_time"])
        if line["type"] == "run_start":
            marks.append(Mark(time, "run", "start"))
        elif line["type"] == "run_end":
            marks.append(Mark(time, "run", f"end: {line['how']}"))
        elif line["type"] == "scanner_trigger":
            marks.append(Mark(time, "trigger", "the run clock's zero"))
        elif line["type"] == "trigger_sent" and line["name"] != "frame":
            marks.append(Mark(time, "code", f"{line['name']} = {line['value']}"))
        elif line["type"] == "console" and line["text"].strip():
            marks.append(Mark(time, "console", line["text"]))
    return marks


def _flip_gaps(record: RunRecord, to_run_time: RunClock) -> list[tuple[float, float]]:
    """Where the display went more than :data:`MISSED_REFRESH` refreshes without a flip: the
    flip that ended each gap, and the gap; none unless the display was vsync-locked.

    :param record: the run.
    :param to_run_time: the run clock.
    :return: ``(time, seconds)`` of each gap.
    """
    display = record.manifest["display"]
    if not display["vsync"] or not display["refresh_rate"]:
        return []
    flips = [to_run_time(line["perf_time"]) for line in record.run_lines
             if line["type"] == "flip"]
    longest = MISSED_REFRESH / display["refresh_rate"]
    return [(after, after - before) for before, after in zip(flips, flips[1:])
            if after - before > longest]


def _quality(record: RunRecord, timeline: Timeline,
             gaps: list[tuple[float, float]]) -> dict[str, Any]:
    """The numbers a run is checked by: its pulses, its refreshes, its presses."""
    return {"pulses": _pulse_quality(timeline),
            "refreshes": _refresh_quality(record, gaps),
            "presses": _press_quality(timeline)}


def _pulse_quality(timeline: Timeline) -> dict[str, Any]:
    """How many pulses the scanner's key gave, their intervals, and the gaps that miss one."""
    times = [press.onset for press in timeline.presses if press.is_scanner_pulse]
    out: dict[str, Any] = {"count": len(times)}
    if len(times) < 2:
        return out
    intervals = [b - a for a, b in zip(times, times[1:])]
    median = statistics.median(intervals)
    return {**out, "median_interval_s": median, "min_interval_s": min(intervals),
            "max_interval_s": max(intervals),
            "gaps_missing_a_pulse": sum(gap > MISSED_PULSE * median for gap in intervals)}


def _refresh_quality(record: RunRecord, gaps: list[tuple[float, float]]) -> dict[str, Any]:
    """The flips, and the gaps between them longer than :data:`MISSED_REFRESH` refreshes."""
    display = record.manifest["display"]
    out: dict[str, Any] = {"flips": sum(line["type"] == "flip" for line in record.run_lines),
                           "vsync": display["vsync"], "refresh_rate": display["refresh_rate"]}
    if not display["vsync"] or not display["refresh_rate"]:
        return out
    return {**out, "missed": len(gaps),
            "longest_gap_s": max((gap for _, gap in gaps), default=0.0)}


def _press_quality(timeline: Timeline) -> dict[str, Any]:
    """The presses: how many, in episodes, given to a step, given to none."""
    presses = [press for press in timeline.presses if not press.is_scanner_pulse]
    in_episodes = [press for press in presses if press.episode is not None]
    return {"count": len(presses), "in_episodes": len(in_episodes),
            "given_to_a_step": sum(press.frame is not None for press in in_episodes),
            "given_to_none": sum(press.frame is None for press in in_episodes)}
