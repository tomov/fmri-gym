"""``bids-from-logs``: a behavioural BIDS dataset made from the runs' logs, by hand.

Each run that reached its trigger becomes ``sub-*/ses-*/beh/<label>_events.tsv`` and its
sidecar (:mod:`.events_table`), beside the dataset's ``dataset_description.json``,
``README`` and ``participants.tsv``; the logs are read, never changed. A conversion that
would have to choose -- between two attempts at one run, or for a run with no run log --
writes nothing and says which runs and why.
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
from ..save.logging import EVENTS_FILENAME
from .events_table import write_events_table

#: The BIDS version the dataset is written in.
BIDS_VERSION = "1.10.0"
#: A team's description of its datasets, in its folder: what ``dataset_description.json``
#: starts from.
DESCRIPTION_FILENAME = "dataset_description.json"


class RunAttempt(NamedTuple):
    """One attempt at a run: its folder, its BIDS name, and whether it reached the trigger."""

    folder: str
    label: str
    subject: str
    session: str
    has_started: bool

    @classmethod
    def from_folder(cls, folder: str) -> RunAttempt:
        """The attempt whose manifest is in ``folder``."""
        with open(os.path.join(folder, "manifest.json")) as f:
            manifest = json.load(f)
        subject, session = Path(folder).parts[-4:-2]
        label = manifest["run"]["label"] if "run" in manifest else Path(folder).name
        return cls(folder, label, subject, session, manifest["start_epoch"] is not None)


def main(argv: list[str] | None = None) -> None:
    """``bids-from-logs --rig <rig file> [--subject sub-..] [--out folder]``.

    :raises SystemExit: when a run is ambiguous or has no run log, naming each.
    """
    parser = argparse.ArgumentParser(description="Write a behavioural BIDS dataset from the "
                                     "runs' logs in a rig's data root.")
    parser.add_argument("--rig", required=True,
                        help="the rig file: its data root holds the runs, its team folder the "
                        f"{DESCRIPTION_FILENAME}")
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
    description = team_description(Path(args.rig).resolve().parent.parent)
    write_dataset(runs, description, args.out or os.path.join(data_root, "bids"))


def find_run_attempts(data_root: str, subject: str | None) -> list[RunAttempt]:
    """Every attempt at a run under ``data_root``, of ``subject`` or of every subject."""
    pattern = os.path.join(data_root, subject or "sub-*", "ses-*", "beh", "*", "manifest.json")
    return [RunAttempt.from_folder(os.path.dirname(path)) for path in sorted(glob.glob(pattern))]


def run_problems(runs: list[RunAttempt]) -> list[str]:
    """Why ``runs`` cannot become a dataset as they are: one line each; none when they can."""
    attempts_of_run: dict[str, list[str]] = defaultdict(list)
    for run in runs:
        attempts_of_run[run.label].append(run.folder)
    problems = [f"{label}: {len(folders)} attempts reached the trigger, and BIDS keeps one "
                f"run: move the others out of the data root ({', '.join(folders)})"
                for label, folders in attempts_of_run.items() if len(folders) > 1]
    problems += [f"{run.folder}: no {EVENTS_FILENAME}: recorded before runs kept one log, its "
                 "events cannot be listed" for run in runs
                 if not os.path.exists(os.path.join(run.folder, EVENTS_FILENAME))]
    return problems


def team_description(team_folder: Path) -> dict:
    """The team's dataset description; asked for on the terminal, and saved, when it has none.

    :raises SystemExit: when it has none and the terminal cannot be asked.
    """
    path = team_folder / DESCRIPTION_FILENAME
    if path.exists():
        return json.loads(path.read_text())
    if not sys.stdin.isatty():
        sys.exit(f"bids-from-logs: no {path}: run this in a terminal to make it, or write it")
    print(f"{path} is missing; it describes every dataset of this team.")
    name = input("Dataset name: ").strip()
    authors = [a.strip() for a in input("Authors, separated by commas: ").split(",") if a.strip()]
    description = {"Name": name, "Authors": authors}
    path.write_text(json.dumps(description, indent=2) + "\n")
    print(f"saved {path}")
    return description


def write_dataset(runs: list[RunAttempt], description: dict, dataset_folder: str) -> None:
    """Write the dataset: its description, README and participants, and each run's events."""
    os.makedirs(dataset_folder, exist_ok=True)
    full_description = {**description, "BIDSVersion": BIDS_VERSION, "DatasetType": "raw"}
    with open(os.path.join(dataset_folder, DESCRIPTION_FILENAME), "w") as f:
        json.dump(full_description, f, indent=2)
    with open(os.path.join(dataset_folder, "README"), "w") as f:
        f.write(f"{description['Name']}\n\nThe behavioural events of fmri-gym runs, one events "
                "table per run, written from the runs' logs by bids-from-logs.\n")
    with open(os.path.join(dataset_folder, "participants.tsv"), "w") as f:
        f.write("participant_id\n" + "".join(f"{s}\n" for s in sorted({r.subject for r in runs})))
    for run in runs:
        table_folder = os.path.join(dataset_folder, run.subject, run.session, "beh")
        write_events_table(os.path.join(run.folder, EVENTS_FILENAME), table_folder, run.label)
    print(f"runs written: {len(runs)}, to {dataset_folder}")
