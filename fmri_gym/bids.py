"""Where a run's data goes, BIDS-style, and the numbers in its name.

    data/sub-01/ses-001/beh/sub-01_ses-001_task-pong_run-001/      manifest.json, block-*/
    data/sub-01/ses-001/beh/sub-01_ses-001_task-pong_run-002/      the same task again
    data/sub-01/ses-001/beh/sub-01_ses-001_task-pong_run-002_02/   ... re-acquired
    data/sub-01/ses-001/beh/sub-01_ses-001_task-crafter_run-001/

**The numbers come from the session design, not from the disk.** ``--ses`` and
``--run`` are required: a run's number is which line of the session script it
is, so it is the same however the session went -- skip a line on a resumed
session and the runs after it keep the numbers they were designed with. A
session script picks its session once, in its first line (``fmri-ses``), and
passes it to every run; it takes the session number as its own argument, so a
stopped session is resumed into the one it started.

Data is never overwritten. A run whose folder is already there is a
re-acquisition and writes to ``..._02`` (then ``_03``); the aborted attempt
stays where it is. The suffix names the folder only: the run's **label** is
the canonical ``sub-01_ses-001_task-pong_run-002``, and that is what keys its
seeds (:func:`phase_seed`), so a re-acquisition replays the episodes of the
run it replaces and a session's stimuli can still be regenerated from its
script alone. No two runs -- of a participant, or of two participants --
replay each other's episodes unless a phase pins its ``"seed"``.

The names follow BIDS, apart from that attempt suffix; the contents do not yet
(a manifest and a folder per block in a folder per run, not ``_beh.tsv`` +
``_events.tsv`` sidecars).

A curriculum can outlast a run: :func:`last_position` reads where the
subject's latest run of a curriculum stopped, from that run's manifest and its
last block's ``block_end`` line, so the next run can carry on from there.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Callable, NamedTuple

from .config.rig import RigConfig

_LABEL = re.compile(r"[A-Za-z0-9]+")


class RunOutput(NamedTuple):
    """Where one run writes, and what names it (see :func:`run_output`)."""

    #: The folder to write into: the label, plus ``_02``, ``_03``... for a re-acquisition.
    folder: str
    #: :func:`run_label`, with no attempt suffix: the seed key, and the run this is of.
    label: str
    #: 1 the first time this run is played, 2 the next time, ...
    attempt: int


class LastGame(NamedTuple):
    """The last game phase a run played: where it sat, and what it was."""

    #: its index in the curriculum
    phase_index: int
    #: the phase as that run's curriculum had it (a phase dict, diffed field by field)
    phase: dict


@dataclass(frozen=True, slots=True)
class BlockProgress:
    """What a game block an earlier run stopped hands the run that carries it on."""

    #: seconds the block has played so far, over every run that played it
    block_time_elapsed: float
    #: episodes counted toward its ``n_episodes`` so far
    num_completed_episodes: int
    #: the savestate of the episode it stopped in, restored over that
    #: episode's reset; ``None`` starts the episode over
    episode_initial_state: bytes | None


@dataclass(frozen=True, slots=True)
class RunProgress:
    """Where a subject's latest run of a curriculum stopped (see :func:`last_position`)."""

    #: the label of the run that stopped there
    run_label: str
    #: the phase the next run starts at
    phase_index: int
    #: the last game phase that run played; ``None`` if it played none
    last_game: LastGame | None
    #: set when phase ``phase_index`` is a game block that run stopped
    #: part-way, with what it has left; ``None`` plays the phase from its
    #: beginning
    block_progress: BlockProgress | None = None

    def record(self) -> dict:
        """The manifest's ``resumed_from`` entry: everything but the savestate's bytes."""
        last_game = self.last_game
        out = {"run": self.run_label,
               "last_game": None if last_game is None else
               {"index": last_game.phase_index, "phase": last_game.phase},
               "index": self.phase_index}
        if self.block_progress is not None:
            out["block_time_elapsed"] = self.block_progress.block_time_elapsed
            out["num_completed_episodes"] = self.block_progress.num_completed_episodes
        return out


def subject_label(subject: str) -> str:
    """The ``sub-<label>`` a subject id must be.

    :param subject: e.g. ``sub-01``.
    :return: it, unchanged.
    :raises ValueError: if it is not ``sub-`` and letters or digits (BIDS allows no other).
    """
    if not (subject.startswith("sub-") and _LABEL.fullmatch(subject[4:])):
        raise ValueError(f"--subject {subject!r}: BIDS needs sub-<letters or digits>, "
                         "such as sub-01 or sub-pilot3")
    return subject


def task_label(curriculum_path: str) -> str:
    """A curriculum's path after ``curricula/`` (else its file name), in camelCase.

    ``curricula/ale/levels/pong.json`` gives ``aleLevelsPong``: BIDS labels are letters and digits.

    :raises ValueError: if nothing is left (a file named ``__.json``).
    """
    path = os.path.splitext(os.path.abspath(curriculum_path))[0]
    _, found, tail = path.rpartition(os.sep + "curricula" + os.sep)
    parts = re.findall(r"[A-Za-z0-9]+", tail if found else os.path.basename(path))
    label = "".join(parts[:1] + [p[0].upper() + p[1:] for p in parts[1:]])
    if not label:
        raise ValueError(f"{curriculum_path}: no letter or digit in its name to make a BIDS "
                         "task label from; rename the file")
    return label


def next_session(root: str, subject: str) -> int:
    """The subject's first ``ses-NNN`` with no folder under ``root``: 1, 2, 3..."""
    return _next_free(os.path.join(root, subject), lambda n: f"ses-{n:03d}")


def has_session(root: str, subject: str, session: int) -> bool:
    """Whether the subject's session already has a run filed under ``root``."""
    return os.path.isdir(_beh(root, subject, session))


def run_label(subject: str, session: int, task: str, run: int) -> str:
    """``sub-01_ses-001_task-pong_run-002``: what names the run, and keys its seeds.

    The same for every attempt at that run: only the folder takes a suffix.
    """
    return f"{subject}_ses-{session:03d}_task-{task}_run-{run:03d}"


def run_output(root: str, subject: str, curriculum_path: str,
               session: int, run: int) -> RunOutput:
    """Where this run writes, what names it, and which attempt at it this is.

    Nothing is ever overwritten: the folder of a run that already has data
    goes to the next free attempt beside it, and the caller is told which.

    :param root: the BIDS tree's root (``--data-root``).
    :param subject: ``sub-<label>``.
    :param curriculum_path: the run's curriculum file; its name is the task label.
    :param session: the BIDS session number (``--ses``), from 1.
    :param run: this task's run number in the session (``--run``), from 1.
    :raises ValueError: on a subject or curriculum name BIDS refuses, or a
        session or run number below 1.
    """
    subject = subject_label(subject)
    for flag, number in (("ses", session), ("run", run)):
        if number < 1:
            raise ValueError(f"--{flag} counts from 1, got {number}")
    label = run_label(subject, session, task_label(curriculum_path), run)
    base = os.path.join(_beh(root, subject, session), label)
    folder, attempt = base, 1
    while os.path.exists(folder):
        attempt += 1
        folder = f"{base}_{attempt:02d}"
    return RunOutput(folder, label, attempt)


def phase_seed(label: str, index: int) -> int:
    """A game phase's base seed when it pins none: its episodes get ``seed, seed + 1, ...``.

    A hash of the run's label and the phase's index: stable across processes
    (unlike ``hash()``), so the editor can show what a launch will use, and
    different for every run, participant and phase. Being random 31-bit numbers,
    two blocks' ranges practically never meet (``1000 + phase`` made them share).

    :param label: :func:`run_label`.
    :param index: the phase's index in the curriculum.
    :return: a seed in ``0 .. 2**31 - 1``.
    """
    digest = hashlib.sha256(f"{label}|phase-{index}".encode()).digest()
    return int.from_bytes(digest[:4], "big") >> 1


def last_position(root: str, subject: str, curriculum_path: str) -> RunProgress | None:
    """Where the subject's latest run of this curriculum (in any session) stopped.

    The latest run is the last one to start that played something besides
    its ``include_on_resume`` opening phases. A game block it stopped
    (``block_did_not_finish`` in its ``block_end`` line) is carried on: its
    index, the seconds it has played, the episodes counted toward it and, if
    an episode was still going, the savestate it stopped in. Otherwise the
    next run picks up after the last game block played, or at the last other
    phase shown, played again from its beginning (unless it was the
    curriculum's last).

    :param root: the BIDS tree's root.
    :param subject: ``sub-<label>``.
    :param curriculum_path: the curriculum; its name is the task label.
    :return: the progress, with ``block_progress`` for a block that is carried
        on; ``None`` if no run of this curriculum got past its opening phases.
    :raises ValueError: if that run stopped a block mid-episode on a backend
        that keeps no savestate (its ``block_end`` has no ``extra.state``).
    """
    from .logging import read_events, read_state
    task = task_label(curriculum_path)
    pattern = os.path.join(root, subject, "ses-*", "beh", f"{subject}_ses-*_task-{task}_run-*",
                           "manifest.json")
    runs = []
    for path in glob.glob(pattern):
        with open(path) as f:
            manifest = json.load(f)
        # index None: phases that are always shown at the start of the run (e.g. general instructions)
        played = [p for p in manifest["phases"] if p["index"] is not None
                  and not manifest["curriculum"][p["index"]].get("include_on_resume", False)]
        if played:
            runs.append((manifest["start_epoch"], os.path.dirname(path), manifest, played))
    if not runs:
        return None
    _, folder, manifest, played = max(runs, key=lambda r: r[0])
    games = [p for p in played if p["type"] == "game"]
    last_game = None if not games else LastGame(
        games[-1]["index"], manifest["curriculum"][games[-1]["index"]])
    run = manifest["run"]["label"]
    last = played[-1]
    if last["type"] != "game":
        final = last["index"] == len(manifest["curriculum"]) - 1
        return RunProgress(run, last["index"] + final, last_game)
    events = read_events(os.path.join(folder, last["data_dir"]))
    end = events[-1]
    if not end["block_did_not_finish"]:
        return RunProgress(run, last["index"] + 1, last_game)
    episode_ends = [e for e in events if e["type"] == "episode_end"]
    state = (end["extra"] or {}).get("state")
    going = bool(episode_ends) and episode_ends[-1]["outcome"] in ("playing", "quit")
    if going and state is None:
        raise ValueError(f"--resume: {run} stopped phase {last['index']} mid-episode, and the "
                         f"{last['backend']} backend keeps no savestate to carry it on. Start "
                         "the curriculum over without --resume, or at a later phase")
    block_progress = BlockProgress(
        end["block_time_elapsed"], end["num_completed_episodes"],
        read_state(state) if going and state is not None else None)
    return RunProgress(run, last["index"], last_game, block_progress)


def _beh(root: str, subject: str, session: int) -> str:
    return os.path.join(root, subject, f"ses-{session:03d}", "beh")


def _next_free(folder: str, name: Callable[[int], str]) -> int:
    n = 1
    while os.path.exists(os.path.join(folder, name(n))):
        n += 1
    return n


def main() -> None:
    """``fmri-ses``: the subject's next free session number, for a session script.

    A script's first line takes the session once -- ``SES=${1:-$(fmri-ses
    --subject sub-01)}`` -- and passes it to every run, so they land in the
    same one; giving the script a number instead resumes that session.
    """
    p = argparse.ArgumentParser(
        description="Print a subject's next free BIDS session number (001, 002, ...).")
    p.add_argument("--subject", required=True, help="BIDS subject: sub-<letters/digits>")
    p.add_argument("--rig", required=True, help="the rig file whose data root holds the BIDS "
                   "tree, <root>/sub-XX/ses-NNN/")
    args = p.parse_args()
    root = RigConfig.open(args.rig).data_root
    print(f"{next_session(root, subject_label(args.subject)):03d}")
