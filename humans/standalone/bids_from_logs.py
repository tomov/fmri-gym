"""``bids-from-logs``: a behavioural BIDS dataset made from the runs' logs, and each run's
timeline in words, by hand.

Each run that reached its trigger is read as one record -- its manifest with every log of it
(:mod:`.run_timeline`) -- and becomes ``sub-*/ses-*/beh/<label>_events.tsv`` and its sidecar
(:mod:`.events_table`), beside the dataset's ``README`` and ``participants.tsv``; and
``derivatives/timelines/sub-*/ses-*/beh/<label>_timeline.txt``, its timeline for a person to
read, with the numbers it is checked by (:mod:`.timeline_text`). The dataset's
``dataset_description.json`` is the user's to write. The logs are read, never changed. A
conversion that would have to choose -- between two attempts at one run, or for a run in a
log format these tools do not read -- writes nothing and says which runs and why.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

from ..config.rig import RigConfig
from .events_table import write_events_table
from .run_timeline import OLDEST_LOG_FORMAT, RunRecord, build_timeline
from .timeline_text import timeline_text

#: Where in the dataset the runs' timelines go: a BIDS derivative of its own.
TIMELINES_FOLDER = os.path.join("derivatives", "timelines")


class RunAttempt(NamedTuple):
    """One attempt at a run: its folder, its BIDS name, whether it reached the trigger, and the
    format of its logs."""

    folder: str
    label: str
    subject: str
    session: str
    has_started: bool
    log_format: int

    @classmethod
    def from_folder(cls, folder: str) -> RunAttempt:
        """The attempt whose manifest is in ``folder``."""
        manifest = json.loads(Path(folder, "manifest.json").read_text())
        subject, session = Path(folder).parts[-4:-2]
        label = manifest["run"]["label"] if "run" in manifest else Path(folder).name
        return cls(folder, label, subject, session, manifest["start_epoch"] is not None,
                   manifest.get("log_format", {}).get("version", 1))


def main(argv: list[str] | None = None) -> None:
    """``bids-from-logs --rig <rig file> [--subject sub-..] [--out folder]``.

    :param argv: the command line; ``None`` reads the process's.
    :raises SystemExit: when a run is ambiguous or in a log format not read, naming each.
    """
    parser = argparse.ArgumentParser(description="Write a behavioural BIDS dataset, and each "
                                     "run's timeline, from the runs' logs in a rig's data root.")
    parser.add_argument("--rig", required=True, help="the rig file: its data root holds the runs")
    parser.add_argument("--subject", help="convert this subject's runs only")
    parser.add_argument("--out", help="the dataset's folder (default: <data root>/bids)")
    args = parser.parse_args(argv)
    data_root = RigConfig.open(args.rig).data_root
    attempts = find_run_attempts(data_root, args.subject)
    for attempt in attempts:
        if not attempt.has_started:
            print(f"skipped, never reached its trigger: {attempt.folder}", file=sys.stderr)
    runs = [attempt for attempt in attempts if attempt.has_started]
    problems = run_problems(runs)
    if problems:
        sys.exit("bids-from-logs: nothing was written:\n" + "\n".join(problems))
    write_dataset(runs, args.out or os.path.join(data_root, "bids"))


def find_run_attempts(data_root: str, subject: str | None) -> list[RunAttempt]:
    """Every attempt at a run under ``data_root``, of ``subject`` or of every subject.

    :param data_root: the BIDS tree the runs are in.
    :param subject: ``sub-<label>``, or ``None`` for every subject.
    :return: the attempts, in folder order.
    """
    pattern = os.path.join(data_root, subject or "sub-*", "ses-*", "beh", "*", "manifest.json")
    return [RunAttempt.from_folder(os.path.dirname(path)) for path in sorted(glob.glob(pattern))]


def run_problems(runs: list[RunAttempt]) -> list[str]:
    """Why ``runs`` cannot become a dataset as they are.

    :param runs: the attempts that reached their trigger.
    :return: one line per problem; none when they can.
    """
    attempts_of_run: dict[str, list[str]] = defaultdict(list)
    for run in runs:
        attempts_of_run[run.label].append(run.folder)
    problems = [f"{label}: {len(folders)} attempts reached the trigger, and BIDS keeps one "
                f"run: move the others out of the data root ({', '.join(folders)})"
                for label, folders in attempts_of_run.items() if len(folders) > 1]
    problems += [f"{run.folder}: log format {run.log_format}, recorded before runs kept a log "
                 f"of their own; these tools read {OLDEST_LOG_FORMAT} on" for run in runs
                 if run.log_format < OLDEST_LOG_FORMAT]
    return problems


def write_dataset(runs: list[RunAttempt], dataset_folder: str) -> None:
    """Write the dataset -- its README and participants, each run's events -- and each run's
    timeline beside it, as a derivative.

    :param runs: the attempts to write, one per run.
    :param dataset_folder: the dataset's folder, made if missing.
    """
    timelines_folder = os.path.join(dataset_folder, TIMELINES_FOLDER)
    os.makedirs(timelines_folder, exist_ok=True)
    with open(os.path.join(dataset_folder, "README"), "w") as f:
        f.write("The behavioural events of fmri-gym runs, one events table per run, written from "
                "the runs' logs by bids-from-logs; each run's timeline, for a person to read, is "
                f"in {TIMELINES_FOLDER}.\n")
    with open(os.path.join(dataset_folder, "participants.tsv"), "w") as f:
        f.write("participant_id\n" + "".join(f"{s}\n" for s in sorted({r.subject for r in runs})))
    for run in runs:
        record = RunRecord.read(run.folder)
        timeline = build_timeline(record)
        write_events_table(timeline, os.path.join(dataset_folder, run.subject, run.session,
                                                  "beh"))
        text_folder = os.path.join(timelines_folder, run.subject, run.session, "beh")
        os.makedirs(text_folder, exist_ok=True)
        Path(text_folder, f"{run.label}_timeline.txt").write_text(timeline_text(record,
                                                                                timeline))
    print(f"runs written: {len(runs)}, to {dataset_folder}")
