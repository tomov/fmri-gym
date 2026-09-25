"""Where a run's data goes, BIDS-style, and the numbers in its name.

    data/sub-01/ses-001/beh/sub-01_ses-001_task-pong_run-001/      manifest.json, block-*.npz
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

The names follow BIDS, apart from that attempt suffix; the contents partly do.
``<label>_events.tsv`` and its sidecar (:func:`events_rows`) are what a BIDS
analysis reads, and the manifest and ``.npz`` blocks beside them are the record
the events are a view of: a folder per run rather than a ``_beh.tsv``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from typing import Any, Callable, NamedTuple

_LABEL = re.compile(r"[A-Za-z0-9]+")

#: The events file's columns, in order, with what the sidecar says each is.
#: ``onset`` and ``duration`` in seconds from the scanner trigger, which is
#: what BIDS means by them; the rest are this rig's own and are described
#: because nothing else would explain them.
EVENT_COLUMNS: dict[str, str] = {
    "onset": "Seconds from the scanner trigger to the flip that showed this event.",
    "duration": "Seconds the event lasted, from its own onset to the next thing shown.",
    "trial_type": "What the event is: a curriculum phase's type (fixation, message, game, "
                  "survey, trigger), or, inside a game phase, episode / hold / iti / response.",
    "phase": "Index of the curriculum phase this event belongs to.",
    "episode": "Index of the episode within its game phase (an episode, or the interval "
               "after it); n/a outside a game phase.",
    "seed": "The seed an episode was reset with, so its world can be rebuilt.",
    "ended": "What ended an episode: terminated (the game said so), truncated (its own time "
             "limit), block_end (the block's time ran out) or quit.",
    "response": "The value a survey question was answered with.",
}


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


def task_label(config_path: str) -> str:
    """A config's task label: its file name, letters and digits only (``ale__pong`` -> ``alepong``).

    :raises ValueError: if nothing is left (a file named ``__.json``).
    """
    stem = os.path.splitext(os.path.basename(config_path))[0]
    label = "".join(re.findall(r"[A-Za-z0-9]", stem))
    if not label:
        raise ValueError(f"{config_path}: no letter or digit in its name to make a BIDS task "
                         "label from; rename the file")
    return label


def next_session(root: str, subject: str) -> int:
    """The subject's first ``ses-NNN`` with no folder under ``root``: 1, 2, 3..."""
    return _next_free(os.path.join(root, subject), lambda n: f"ses-{n:03d}")


def run_label(subject: str, session: int, task: str, run: int) -> str:
    """``sub-01_ses-001_task-pong_run-002``: what names the run, and keys its seeds.

    The same for every attempt at that run: only the folder takes a suffix.
    """
    return f"{subject}_ses-{session:03d}_task-{task}_run-{run:03d}"


class RunOutput(NamedTuple):
    """Where one run writes, and what names it (see :func:`run_output`)."""

    #: The folder to write into: the label, plus ``_02``, ``_03``... for a re-acquisition.
    folder: str
    #: :func:`run_label`, with no attempt suffix: the seed key, and the run this is of.
    label: str
    #: 1 the first time this run is played, 2 the next time, ...
    attempt: int


def run_output(root: str, subject: str, config_path: str,
               session: int, run: int) -> RunOutput:
    """Where this run writes, what names it, and which attempt at it this is.

    Nothing is ever overwritten: the folder of a run that already has data
    goes to the next free attempt beside it, and the caller is told which.

    :param root: the BIDS tree's root (``--data-root``).
    :param subject: ``sub-<label>``.
    :param config_path: the run's config file; its name is the task label.
    :param session: the BIDS session number (``--ses``), from 1.
    :param run: this task's run number in the session (``--run``), from 1.
    :raises ValueError: on a subject or config name BIDS refuses, or a session
        or run number below 1.
    """
    subject = subject_label(subject)
    for flag, number in (("ses", session), ("run", run)):
        if number < 1:
            raise ValueError(f"--{flag} counts from 1, got {number}")
    label = run_label(subject, session, task_label(config_path), run)
    base = os.path.join(_beh(root, subject, session), label)
    folder, attempt = base, 1
    while os.path.exists(folder):
        attempt += 1
        folder = f"{base}_{attempt:02d}"
    return RunOutput(folder, label, attempt)


def fold_seeds(curriculum: list[dict], label: str) -> dict:
    """Give each game phase that pins no ``"seed"`` the one derived for this run.

    The run plays episode e with ``seed + e`` as for any phase, and the
    manifest's curriculum shows every seed; what was derived is said here.

    :param curriculum: the run's phases, seeded in place.
    :param label: :func:`run_label` -- so every attempt at a run replays the
        same episodes.
    :return: the manifest's ``seeds`` entry: the run label, and phase -> how set.
    """
    how = {}
    for i, phase in enumerate(curriculum):
        if phase["type"] != "game":
            continue
        how[i] = "pinned" if "seed" in phase else "derived"
        phase.setdefault("seed", phase_seed(label, i))
        print(f"seeds: phase {i} = {phase['seed']} ({how[i]})", file=sys.stderr)
    return {"derived_from": label, "phases": how}


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


def events_rows(manifest: dict) -> list[dict]:
    """The run's events, from its manifest: what happened, when, for how long.

    A view, not a second record. Everything here is already in the manifest
    (and the episode onsets in the npz); this is the same run in the one shape a
    BIDS analysis will look for, so nobody has to write the reader.

    Phases whose ``onset`` is ``None`` are dropped: they played above the
    curriculum's ``trigger`` phase, before there was a scan for them to have an
    onset in. A manifest from an older run, or from a block with no intervals,
    simply has fewer rows.

    :param manifest: a run's ``manifest.json`` contents.
    :return: rows keyed by :data:`EVENT_COLUMNS`, in onset order. A row may
        leave any key but ``onset`` and ``trial_type`` out.
    """
    rows = [row for phase in manifest.get("phases", []) for row in _phase_rows(phase)]
    return sorted(rows, key=lambda r: r["onset"])


def _phase_rows(phase: dict) -> list[dict]:
    """One phase's rows: itself, then what it was made of."""
    if phase.get("onset") is None:
        return []
    index = phase.get("index")
    rows = [_event_row(phase["onset"], phase.get("offset"), phase["type"], index)]
    for ep in phase.get("episodes", []):
        rows.append(_event_row(ep.get("onset"), ep.get("offset"), "episode", index,
                               episode=ep.get("id"), seed=ep.get("seed"),
                               ended=ep.get("ended")))
    for iv in phase.get("intervals", []):
        after = iv.get("after_episode")
        # The hold ends where the blank begins, so one bounds the other; a run
        # quit inside either leaves the row with no duration rather than a guess.
        rows.append(_event_row(iv.get("hold_onset"), iv.get("onset"), "hold", index,
                               episode=after))
        rows.append(_event_row(iv.get("onset"), iv.get("offset"), "iti", index,
                               episode=after))
    for answer in phase.get("responses", []):
        # A confirmed answer is an instant, not a span: the seconds before it
        # were the subject reading the question, which the phase's own row covers.
        at = answer.get("run_time")
        rows.append(_event_row(at, at, "response", index, response=answer.get("value")))
    return [r for r in rows if r["onset"] is not None]


def _event_row(onset: float | None, offset: float | None, trial_type: str,
               phase: int | None, **rest: Any) -> dict:
    """One row, with the duration worked out from the two ends it has."""
    duration = None if onset is None or offset is None else offset - onset
    return {"onset": onset, "duration": duration, "trial_type": trial_type,
            "phase": phase, **rest}


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
    p.add_argument("--data-root", default="data",
                   help="where the BIDS tree is: <root>/sub-XX/ses-NNN/ (default: data)")
    args = p.parse_args()
    print(f"{next_session(args.data_root, subject_label(args.subject)):03d}")
