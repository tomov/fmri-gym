"""A game a subject comes back to: the world one block ends in, kept for the next.

Most blocks in a session are a level, and a level starts where it starts. An
open-ended game is not: crafter's world is the one the subject has been living
in, and cutting it at the block's clock and handing the next block a fresh one
throws away everything they built. A block that names a ``"resume"`` slot in
its config keeps that world instead, and the next block with the same slot
continues it.

**This adds no savestate machinery.** An adapter already hands the run a
savestate every ``state_stride`` frames --
:meth:`~fmri_gym.adapters.base.EnvAdapter.capture` with ``want_blob=True``
returns the bytes, and :meth:`~fmri_gym.adapters.base.EnvAdapter.restore` turns
them back into the env. All that was missing was somewhere to put one blob
where the *next process* can find it, since the runs of a session are separate
``fmri-play`` commands. That is this module: a named slot, one file, written
atomically.

    <data-root>/sub-01/ses-001/resume/crafter_L4.state

Beside ``beh/``, not inside it: the run folders under ``beh/`` are the data,
and this is the handoff between them, which is a copy of a savestate that the
block's own log already holds. Deleting the whole ``resume/`` folder loses
nothing but the continuity, and the session starts every world fresh again.

The file is one JSON header line, a newline, then the raw blob. The header is
written in the same write as the blob and read first, so a file that was
half-written when the machine died fails its own length check and is ignored,
and the block starts a new world rather than restoring a truncated one. Nothing
is ever appended: a slot holds the last world, not a history of them.

A slot is a *thread of play*, not a game. Four crafter levels are all
``CrafterMenu-v0``, so a name derived from the backend and the game id would
put L1's world into L4's block; naming the slot in the config is what keeps
them apart, and what lets two blocks that should be one continuous world say so.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .adapters.base import EnvAdapter

#: What the header says this file is, so a stray ``.state`` is not unpickled.
FORMAT = "fmri-gym-resume"
SCHEMA_VERSION = 1
SUFFIX = ".state"
#: A slot names a file, so it is letters, digits, ``_`` and ``-``.
_SLOT = re.compile(r"[A-Za-z0-9_-]+")


@dataclass(frozen=True)
class Carry:
    """One slot's saved world: where it came from, and the bytes to restore.

    :ivar header: the file's JSON header -- ``slot``, ``backend``, ``game``,
        and which run, block, episode and frame the state was taken at.
    :ivar blob: the savestate itself, as
        :meth:`~fmri_gym.adapters.base.EnvAdapter.restore` wants it.
    :ivar path: the file it was read from.
    """

    header: dict
    blob: bytes
    path: str


def slot_of(phase: dict) -> str | None:
    """The slot this game phase continues, or ``None`` if it starts fresh.

    :param phase: a game-phase config.
    :return: the phase's ``"resume"``, or ``None``.
    """
    slot = phase.get("resume")
    return slot if slot else None


def slot_problems(phase: dict) -> list[str]:
    """What a bad ``"resume"`` field looks like, for ``validate_config``.

    Whether the *backend* can resume is not decided here: it takes a built env
    to know, and this check is the cheap one the editor runs.

    :param phase: a game-phase config.
    :return: human-readable problems, empty when the field is absent or usable.
    """
    if "resume" not in phase:
        return []
    slot = phase["resume"]
    if not isinstance(slot, str) or not _SLOT.fullmatch(slot):
        return [f'resume: expected the name of the thread of play this block continues '
                f'(letters, digits, "_" and "-"; e.g. "crafter_L4"), got {slot!r}. It names '
                "a file, and two blocks that name the same one are one continuous world -- "
                "so levels of the same game must not share it"]
    return []


def supported(adapter: EnvAdapter) -> bool:
    """Whether this backend has the savestate a resume is made of.

    The base :meth:`~fmri_gym.adapters.base.EnvAdapter.restore` raises, and most
    backends do not override it; a block that asks to resume on one of those
    would silently start fresh every time, which is exactly the kind of quiet
    wrong answer a session must not produce.

    :param adapter: the block's adapter.
    :return: ``True`` if the backend overrides ``restore``.
    """
    from .adapters.base import EnvAdapter
    return type(adapter).restore is not EnvAdapter.restore


def path_of(folder: str, slot: str) -> str:
    """The file a slot lives in.

    :param folder: the session's ``resume/`` folder.
    :param slot: the slot name.
    :return: its path.
    """
    return os.path.join(folder, slot + SUFFIX)


def save(folder: str, slot: str, blob: bytes, header: dict) -> str:
    """Write ``blob`` as this slot's world, replacing whatever was there.

    :param folder: the session's ``resume/`` folder, created if absent.
    :param slot: the slot name.
    :param blob: the savestate, from ``capture(..., want_blob=True).blob``.
    :param header: provenance to record beside it (run, block, episode, frame).
    :return: the path written.
    """
    os.makedirs(folder, exist_ok=True)
    head = {"format": FORMAT, "version": SCHEMA_VERSION, "slot": slot,
            "n_bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(),
            **header}
    path = path_of(folder, slot)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(json.dumps(head, default=str).encode() + b"\n")
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    # The rename is the commit: a reader sees the old world or the new one.
    os.replace(tmp, path)
    return path


def load(folder: str, slot: str) -> Carry | None:
    """This slot's world, or ``None`` if there is none to continue.

    ``None`` is the ordinary case at the first block of a session, so it is a
    return value and not an error. A file that is there but unreadable is not
    ordinary, and raises.

    :param folder: the session's ``resume/`` folder.
    :param slot: the slot name.
    :return: the :class:`Carry`, or ``None``.
    :raises ValueError: if the file is not this format, or its blob is not
        the length and digest its header claims (a half-written file).
    """
    path = path_of(folder, slot)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        line = f.readline()
        blob = f.read()
    try:
        header = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: no {FORMAT} header on the first line ({exc})") from exc
    if header.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file ({header.get('format')!r})")
    if header.get("version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: written by schema version {header.get('version')!r}, "
                         f"this is {SCHEMA_VERSION}")
    if len(blob) != header.get("n_bytes") or hashlib.sha256(blob).hexdigest() != header.get(
            "sha256"):
        raise ValueError(f"{path}: {len(blob)} bytes after the header, header says "
                         f"{header.get('n_bytes')}: the file was not finished being written. "
                         "Delete it to start this world fresh")
    return Carry(header=header, blob=blob, path=path)


def describe(carry: Carry | None) -> Any:
    """What the manifest says about a block's resume, in one value.

    :param carry: what the block restored, or ``None``.
    :return: the world's provenance, or ``None``.
    """
    if carry is None:
        return None
    return {k: v for k, v in carry.header.items() if k not in ("sha256", "format", "version")}
