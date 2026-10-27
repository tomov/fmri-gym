"""Session scripts: the ``.sh`` a session is played from, one ``fmri-play`` line per run.

A session is a shell script (:func:`write_session`). Its head sets three
variables once: ``SUBJECT`` and ``RIG``, each fixed in the script or required
from whoever starts it (``SUBJECT=sub-01 RIG=configs/<team>/rigs/<rig>.json sh
<script>``), and ``SES``, the script's own argument or the subject's next free
session. Each line after it plays one run with them, or is any command of
yours. The session editor (:mod:`fmri_gym.gui.session_editor`) writes and reads
them; any shell plays them, so nothing in Python loops over runs.
"""

from __future__ import annotations

import argparse
import re
import shlex
from dataclasses import dataclass

from . import bids

HEADER = ["#!/bin/sh", "# fmri-gym session: one line per run, in order.", "set -e"]
PLAY = "uv run fmri-play"
#: What a script's head asks for the session number when it is given none.
SES_TOOL = "uv run fmri-ses"
#: The head's variables, as a run's line spells them: written as typed, not
#: quoted, since the shell has to expand them.
SUBJECT_VAR, RIG_VAR, SES_VAR = '"$SUBJECT"', '"$RIG"', '"$SES"'
#: How a script that leaves a variable to its launch says how to start it.
USAGE = "run as: SUBJECT=sub-01 RIG=configs/<team>/rigs/<rig>.json sh $0 [session]"
#: The flags of ``fmri-play`` a run's line may carry besides where it is filed, as
#: ``(argparse name, flag, kind)``: a number, or a switch.
RUN_FLAGS = (("duration", "--duration", float), ("end_fixation", "--end-fixation", float),
             ("loop", "--loop", bool), ("resume", "--resume", bool))
_HEAD = re.compile(r"(SUBJECT|RIG|SES)=(.*)")
_PINNED = re.compile(r"\$\{1:-(\d+)\}")


@dataclass(frozen=True, slots=True)
class Session:
    """A session script: its lines, and what its head fixes for all of them."""

    #: the lines, in order: ``{"config": path, "skip": bool}`` plays a run, with
    #: ``"flags"`` (:data:`RUN_FLAGS` by argparse name) when it sets any;
    #: ``{"command": text, "skip": bool}`` is any command, as typed
    steps: list[dict]
    #: the subject every run is filed under; ``None``: given at launch (``SUBJECT=``)
    subject: str | None
    #: the rig file every run plays on; ``None``: given at launch (``RIG=``)
    rig: str | None
    #: the session number; ``None``: the script's argument, else the next free one
    ses: int | None


def write_session(session: Session) -> str:
    """The session script.

    ::

        #!/bin/sh
        # fmri-gym session: one line per run, in order.
        set -e
        SUBJECT=sub-01
        RIG=${RIG:?run as: SUBJECT=sub-01 RIG=configs/<team>/rigs/<rig>.json sh $0 [session]}
        SES=${1:-$(uv run fmri-ses --subject "$SUBJECT" --rig "$RIG")}
        uv run fmri-play --curriculum configs/pong.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG"
        ./scripts/localizer.sh "$SES"
        # uv run fmri-play --curriculum configs/mario.json --subject "$SUBJECT" --ses "$SES" ...

    A subject or rig the session leaves open must be given when it is started,
    or the script stops at once, saying how. ``SES=`` picks the session once for
    all the runs: the script's own argument (``sh ses1.sh 003`` resumes that
    session), else the number fixed here, else the subject's next free one.
    Each run's ``--run`` is :func:`run_number`, its place among the lines that
    play its task -- fixed by the design, so skipping a line renumbers nothing.
    A commented line is a skipped run. ``set -e`` stops the session at the
    first line that fails or is quit.
    """
    lines = [*HEADER, _head_line("SUBJECT", session.subject),
             _head_line("RIG", session.rig), _ses_line(session.ses)]
    for i, step in enumerate(session.steps):
        line = (step["command"] if "command" in step
                else _shell(play_command(step["config"], SUBJECT_VAR, SES_VAR,
                                         run_number(session.steps, i), RIG_VAR,
                                         step.get("flags", {}))))
        lines.append(f"# {line}" if step["skip"] else line)
    return "\n".join(lines) + "\n"


def run_number(steps: list[dict], index: int) -> int:
    """Which run of its task the line at ``index`` is: 1, 2, 3...

    Counted over the lines before it that play the same task, skipped ones
    included: a run's number belongs to the session's design, so resuming a
    session with some lines skipped leaves every other number where it was.

    :param steps: the session's lines (see :class:`Session`).
    :param index: the line to number; it must be a run, not a command.
    :return: its ``--run``.
    """
    task = bids.task_label(steps[index]["config"])
    return 1 + sum(1 for step in steps[:index]
                   if "config" in step and bids.task_label(step["config"]) == task)


def read_session(text: str) -> Session:
    """The session a script holds (see :func:`write_session`).

    A script whose run lines spell out the subject and the rig instead of using
    the head's variables is read as fixing them, when every run says the same.
    An ``fmri-play`` line the editor could not write back as it is -- other
    flags, another subject or rig, or a ``--run`` other than the one its place
    gives it (:func:`run_number`) -- is kept as a command, as typed.
    """
    body = [line for line in text.splitlines() if line.strip() and line not in HEADER]
    head = {m[1]: m[2] for line in body if (m := _HEAD.fullmatch(line))}
    # What each run line must say: the head's variable, or (no head) one literal for all.
    wanted = {"subject": SUBJECT_VAR.strip('"') if "SUBJECT" in head else None,
              "rig": RIG_VAR.strip('"') if "RIG" in head else None}
    fixed = {"subject": _fixed(head.get("SUBJECT")), "rig": _fixed(head.get("RIG"))}
    steps: list[dict] = []
    for line in body:
        if _HEAD.fullmatch(line):
            continue
        skip = line.startswith("#")
        bare = line[1:].lstrip() if skip else line
        steps.append(_read_line(bare, skip, steps, wanted, fixed))
    pinned = _PINNED.fullmatch(head.get("SES", ""))
    return Session(steps=steps, subject=fixed["subject"], rig=fixed["rig"],
                   ses=int(pinned[1]) if pinned else None)


def _read_line(line: str, skip: bool, steps: list[dict], wanted: dict, fixed: dict) -> dict:
    """One line as a step: a run when the editor can write it back as it is, else a command."""
    played = _parse_play_line(line)
    if played is None:
        return {"command": line, "skip": skip}
    config, said, run, flags = played
    if any(wanted[k] not in (None, said[k]) for k in said):
        return {"command": line, "skip": skip}
    step = {"config": config, "skip": skip, **({"flags": flags} if flags else {})}
    if run != run_number([*steps, step], len(steps)):
        return {"command": line, "skip": skip}  # a run number of its own
    for k, value in said.items():
        if wanted[k] is None:  # the first run of a script with no head: its value is the session's
            wanted[k] = fixed[k] = value
    return step


def play_command(config_path: str, subject: str, ses: str, run: int, rig: str,
                 flags: dict | None = None) -> list[str]:
    """The command that plays one run: ``fmri-play``, its config, where it is filed, and
    how it is played.

    :param ses: the ``--ses`` value -- a number, or :data:`SES_VAR` in a script.
    :param run: the ``--run`` value (see :func:`run_number`).
    :param flags: :data:`RUN_FLAGS` by argparse name; one left out is not written.
    :return: the command, word by word.
    """
    words = [*PLAY.split(), "--curriculum", config_path, "--subject", subject,
             "--ses", ses, "--run", str(run), "--rig", rig]
    for name, flag, kind in RUN_FLAGS:
        value = (flags or {}).get(name)
        if kind is bool and value:
            words.append(flag)
        elif kind is not bool and value is not None:
            words += [flag, f"{value:g}"]
    return words


def _head_line(var: str, value: str | None) -> str:
    """``SUBJECT=sub-01``, or one that stops the script when its launch does not give it."""
    return f"{var}={shlex.quote(value)}" if value else f"{var}=${{{var}:?{USAGE}}}"


def _ses_line(ses: int | None) -> str:
    """``SES=``: the script's own argument, else the number pinned here or the next free one."""
    if ses is not None:
        return f"SES=${{1:-{ses:03d}}}"
    ses_tool = [*SES_TOOL.split(), "--subject", SUBJECT_VAR, "--rig", RIG_VAR]
    return f"SES=${{1:-$({_shell(ses_tool)})}}"


def _fixed(raw: str | None) -> str | None:
    """A head variable's value as fixed in the script; ``None`` when left to the launch."""
    if raw is None or raw.startswith("${"):
        return None
    return shlex.split(raw)[0]


def _shell(command: list[str]) -> str:
    """A command as a script line: each word quoted, but the head's variables left for the
    shell."""
    return " ".join(word if word in (SUBJECT_VAR, RIG_VAR, SES_VAR) else shlex.quote(word)
                    for word in command)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # argparse would print usage and exit
        raise ValueError(message)


def _parse_play_line(line: str) -> tuple[str, dict[str, str], int, dict] | None:
    """``(config path, {"subject", "rig"} as written, run, flags)`` if ``line`` is a run's
    line the editor can write; ``flags`` are those of :data:`RUN_FLAGS` it sets.

    :return: ``None`` if it is not one -- another command, flags the editor has no
        field for, or a session of its own -- and the caller then keeps the line as typed.
    """
    if not line.startswith(PLAY + " "):
        return None
    parser = _Parser(add_help=False, allow_abbrev=False)
    for flag in ("--curriculum", "--subject", "--ses", "--rig"):
        parser.add_argument(flag, required=True)
    parser.add_argument("--run", type=int, required=True)
    for name, flag, kind in RUN_FLAGS:
        if kind is bool:
            parser.add_argument(flag, dest=name, action="store_true")
        else:
            parser.add_argument(flag, dest=name, type=kind)
    try:
        args = parser.parse_args(shlex.split(line)[3:])
    except ValueError:
        return None
    if args.ses != SES_VAR.strip('"'):
        return None  # a run pinned to a session of its own: not one the editor can write
    flags = {name: getattr(args, name) for name, _, kind in RUN_FLAGS
             if getattr(args, name) not in (None, False)}
    return args.curriculum, {"subject": args.subject, "rig": args.rig}, args.run, flags
