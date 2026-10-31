"""The curriculum: what one run plays. Load, save, check.

A curriculum is one JSON file: a ``curriculum`` list of phases (see README);
a key that starts with ``##`` is a comment, anywhere. Several runs make a session, which is a shell
script (:func:`fmri_gym.gui.write_session`), not a curriculum.

The file says what is played, not who plays it or where: subject and test
switches are the run config (:mod:`fmri_gym.config.run`), and the window,
audio, triggers and output folder the rig's (:mod:`fmri_gym.config.rig`), so
one file serves every participant at every site. Nothing here reads a rig.

There is one shape. A bare list, an unknown top-level key, an unknown phase
type or a phase key starting with ``_`` (the one prefix the loader keeps for
itself) is refused at start-up, with the reason.

A phase may start from a shared one: ``"__include": "common/<game>.json"``
names a file (relative to the curriculum's own) holding one phase object, and
the phase's other keys are laid over it (:func:`overlay`). Curricula of one
game then write the setup once and only what differs per level. Shared
phases live in folders named ``common`` (:data:`COMMON_DIRNAME`), which hold no
curricula. Includes are resolved when the file is loaded, so everything
downstream, the manifest included, sees each phase written out in full.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from .phases import PHASE_CLASSES, CheckPhase, GamePhase, Phase, phase_from_dict, phase_problems

#: The phase types, one class each (:mod:`fmri_gym.config.phases`).
PHASE_TYPES = tuple(PHASE_CLASSES)
SECTIONS = ("curriculum",)
#: A key that starts with this is a comment: nothing reads it.
COMMENT_PREFIX = "##"
#: A phase's key naming the file of the shared phase it starts from.
INCLUDE_KEY = "__include"
#: Folders of this name hold shared phases, which are included, not played.
COMMON_DIRNAME = "common"
#: Game-phase fields a curriculum may change between runs and still carry a
#: block on; any other change makes it a different game. ``seed`` is the
#: run's own (folded in from the run label).
RESUMABLE_CHANGES = frozenset({"duration", "n_episodes", "advance_on", "show_outcome_duration",
                               "instructions", "keys", "menu", "seed", "has_audio",
                               "rewind_on", "rewind_frames"})
#: Exit status of ``fmri_play`` when the run was quit (ESC) before its end; a
#: session script (``set -e``) stops on it instead of starting the next run.
EXIT_QUIT = 3

def check_shape(curriculum: Any, where: str) -> dict:
    """Enforce the one curriculum shape.

    :param curriculum: parsed JSON.
    :param where: file path (or another name for the source), for the message.
    :return: ``curriculum``, unchanged.
    :raises ValueError: if it is not an object with a ``curriculum`` list of
        typed phases, has a top-level key that is neither a section nor a
        comment, or a phase key that starts with ``_``.
    """
    if not isinstance(curriculum, dict) or not isinstance(curriculum.get("curriculum"), list):
        raise ValueError(f'{where}: expected a JSON object with a "curriculum" list')
    unknown = [k for k in curriculum if k not in SECTIONS and not is_comment(k)]
    if unknown:
        raise ValueError(f"{where}: unknown top-level key(s) {unknown}; the sections are "
                         f'{SECTIONS}, and a comment starts with "{COMMENT_PREFIX}"')
    for i, phase in enumerate(curriculum["curriculum"]):
        if not isinstance(phase, dict) or phase.get("type") not in PHASE_TYPES:
            raise ValueError(f'{where}: phase {i} must be an object whose "type" is one of '
                             f"{PHASE_TYPES}")
        underscored = [k for k in phase if k.startswith("_") and k != INCLUDE_KEY]
        if underscored:
            raise ValueError(f"{where}: phase {i} has {underscored}; a comment starts with "
                             f'"{COMMENT_PREFIX}", and "{INCLUDE_KEY}" names a file of '
                             f"{COMMON_DIRNAME}/")
    return curriculum


def is_comment(key: str) -> bool:
    """Whether ``key`` is a comment (it starts with :data:`COMMENT_PREFIX`)."""
    return key.startswith(COMMENT_PREFIX)


def game_changes(then: dict, now: dict) -> list[str]:
    """The keys that make ``now`` another game than ``then``, sorted.

    Comments and :data:`RESUMABLE_CHANGES` do not count: a block played under
    ``then`` carries on under ``now`` when nothing else differs.

    :param then: a game phase as a run played it (its manifest's copy).
    :param now: the phase at the same place in the curriculum today.
    """
    return sorted(k for k in then.keys() | now.keys() if then.get(k) != now.get(k)
                  and k not in RESUMABLE_CHANGES and not is_comment(k))


def load_curriculum(path: str, keep_includes: bool = False) -> dict:
    """Load a curriculum file, each phase's ``__include`` resolved.

    :param path: JSON file path.
    :param keep_includes: a phase that includes a shared one also keeps its
        ``__include``, as a path from the working directory, so that
        :func:`save_curriculum` writes it as an include again.
    :return: the curriculum dict, every phase in full.
    :raises ValueError: if the file does not have the curriculum shape
        (see :func:`check_shape`).
    """
    with open(path) as f:
        curriculum = json.load(f)
    if isinstance(curriculum, dict) and isinstance(curriculum.get("curriculum"), list):
        curriculum["curriculum"] = [_resolve(phase, path, frozenset(), keep_includes)
                                    for phase in curriculum["curriculum"]]
    return check_shape(curriculum, path)


def is_common_file(path: str) -> bool:
    """Whether ``path`` is a shared phase (in a :data:`COMMON_DIRNAME` folder), not a curriculum."""
    return os.path.basename(os.path.dirname(os.path.abspath(path))) == COMMON_DIRNAME


def overlay(base: dict, top: dict) -> dict:
    """``top`` laid over ``base``: an object in both merges key by key, anything else replaces.

    So a phase can change one line of its instructions, or add a key to its
    ``keys``, and keep the rest; a list (the instructions' ``text``) is
    replaced whole, and a key cannot be taken away.
    """
    merged = dict(base)
    for key, value in top.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = overlay(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve(phase: Any, path: str, seen: frozenset[str], keep: bool = False) -> Any:
    """``phase`` laid over the phase its ``__include`` names, which may include another.

    :param keep: the result keeps ``__include``, as a path from the working directory.
    :raises ValueError: if the include is not a path to a JSON object, or comes back round.
    """
    if not isinstance(phase, dict) or INCLUDE_KEY not in phase:
        return phase
    name = phase[INCLUDE_KEY]
    target = os.path.normpath(os.path.join(os.path.dirname(path), str(name)))
    if not isinstance(name, str) or not os.path.isfile(target):
        raise ValueError(f'{path}: "{INCLUDE_KEY}" must name a file next to it, got {name!r}')
    if target in seen:
        raise ValueError(f"{path}: {name} includes itself, through {sorted(seen)}")
    own = overlay(_shared_phase(target, seen | {target}),
                  {k: v for k, v in phase.items() if k != INCLUDE_KEY})
    return {INCLUDE_KEY: os.path.relpath(target), **own} if keep else own


def _shared_phase(path: str, seen: frozenset[str] = frozenset()) -> dict:
    """The shared phase in ``path``, its own includes resolved.

    :raises ValueError: if the file is not one JSON object.
    """
    with open(path) as f:
        base = json.load(f)
    if not isinstance(base, dict):
        raise ValueError(f"{path}: a shared phase is one JSON object")
    return _resolve(base, path, seen | {path})


def shared_overrides(curriculum: dict) -> list[str]:
    """What the phases that keep their ``__include`` change of the shared phase, one line each.

    Laying a phase over its shared one only adds and replaces, so a changed
    value is saved as this phase's own; a removed one cannot be, and the
    phase is saved in full, without its include (:func:`save_curriculum`).
    Either way the other curricula that include the shared phase keep it as
    it is.

    :param curriculum: as :func:`load_curriculum` with ``keep_includes`` gives it, edited.
    """
    lines = []
    for i, phase in enumerate(curriculum["curriculum"]):
        if INCLUDE_KEY not in phase:
            continue
        shared = phase[INCLUDE_KEY]
        own = {k: v for k, v in phase.items() if k != INCLUDE_KEY}
        changed, removed = _departures(_shared_phase(shared), own)
        if changed:
            lines.append(f"phase {i}: {', '.join(changed)} differ from {shared}; saved as "
                         "this phase's own")
        if removed:
            lines.append(f"phase {i}: {', '.join(removed)} removed from what {shared} gives; "
                         f'saved in full, without "{INCLUDE_KEY}"')
    return lines


def _departures(base: dict, phase: dict, at: str = "") -> tuple[list[str], list[str]]:
    """The keys ``base`` sets that ``phase`` gives another value, and those it lacks."""
    changed, removed = [], []
    for key, value in base.items():
        name = at + key
        if key not in phase:
            removed.append(name)
        elif isinstance(value, dict) and isinstance(phase[key], dict):
            more = _departures(value, phase[key], name + ".")
            changed += more[0]
            removed += more[1]
        elif phase[key] != value:
            changed.append(name)
    return changed, removed


def _difference(phase: dict, base: dict) -> dict:
    """What ``phase`` lays over ``base`` to become itself (:func:`overlay`'s inverse)."""
    out = {}
    for key, value in phase.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            inner = _difference(value, base[key])
            if inner:
                out[key] = inner
        elif key not in base or base[key] != value:
            out[key] = value
    return out


def _compact(phase: dict, path: str) -> dict:
    """``phase`` as the file at ``path`` writes it: its include and what it adds, if it can."""
    if INCLUDE_KEY not in phase:
        return phase
    shared = _shared_phase(phase[INCLUDE_KEY])
    own = {k: v for k, v in phase.items() if k != INCLUDE_KEY}
    if _departures(shared, own)[1]:
        return own
    name = os.path.relpath(phase[INCLUDE_KEY], os.path.dirname(os.path.abspath(path)))
    return {INCLUDE_KEY: name, **_difference(own, shared)}


def save_curriculum(curriculum: dict, path: str) -> None:
    """Write ``curriculum`` as indented JSON.

    A phase that keeps its ``__include`` (:func:`load_curriculum`) is written
    as the include and what the phase adds to the shared one, or in full when
    it lacks something the shared phase sets (:func:`shared_overrides`).

    :param curriculum: curriculum dict.
    :param path: destination file path.
    """
    phases = [_compact(phase, path) for phase in curriculum["curriculum"]]
    with open(path, "w") as f:
        json.dump({**curriculum, "curriculum": phases}, f, indent=2)
        f.write("\n")


def new_curriculum() -> dict:
    """A minimal starting curriculum: one message, one fixation, one game."""
    return {
        "curriculum": [
            {"type": "message", "text": "Get ready\n\n(press SPACE to start)"},
            {"type": "fixation", "duration": 2.0},
            {"type": "game", "backend": "ale", "game": "ALE/Pong-v5", "mode": "duration",
             "duration": 60.0, "fps": 30, "keys": {"UP": 2, "DOWN": 3}},
        ],
    }


def validate_curriculum(curriculum: dict) -> list[str]:
    """Problems that would stop ``fmri_play`` before the first phase.

    Cheap checks only (no env is built), for the editor's Check and the load:
    each phase as its class has it (:func:`~fmri_gym.config.phases.phase_problems`),
    and the phases a resumed run plays first.

    :param curriculum: a curriculum dict of the right shape.
    :return: human-readable problems, empty when the curriculum looks runnable.
    """
    problems: list[str] = []
    phases = curriculum["curriculum"]
    if not phases:
        problems.append("curriculum: needs at least one phase")
    for i, phase in enumerate(phases):
        problems.extend(f"phase {i}: {p}" for p in phase_problems(phase))
    problems.extend(_opening_problems(phases))
    return problems


def _opening_problems(phases: list[dict]) -> list[str]:
    """``include_on_resume`` marks the phases a curriculum opens with, and only those.

    A resumed run plays them before carrying on (a fixation to settle in), so
    they are an unbroken run of non-game phases from phase 0.
    """
    out = []
    opening = True
    for i, phase in enumerate(phases):
        flag = phase.get("include_on_resume", False)
        if not isinstance(flag, bool):
            out.append(f"phase {i}: include_on_resume: expected true or false, got {flag!r}")
        elif flag and (phase["type"] == "game" or not opening):
            out.append(f"phase {i}: include_on_resume: only for the non-game phases the "
                       "curriculum opens with, with none without it before them")
        opening = opening and flag is True
    return out


@dataclass(frozen=True, slots=True)
class Curriculum:
    """A curriculum, read once: its file and its phases, each its type's object.

    Not one game's: a curriculum is a sequence of phases, and its games may be
    several (a block design of fixed-duration blocks of different games). The
    run's task label is its file's name.
    """

    #: the file it was read from; its name is the run's task label
    path: str
    phases: tuple[Phase, ...]

    @classmethod
    def load(cls, path: str) -> Curriculum:
        """Read and check a curriculum file (:func:`load_curriculum`, :func:`validate_curriculum`).

        :raises ValueError: naming every problem, before anything is played.
        """
        return cls.from_dict(load_curriculum(path), path)

    @classmethod
    def from_dict(cls, curriculum: dict, path: str) -> Curriculum:
        """Check a curriculum dict of the right shape, and read its phases.

        :raises ValueError: naming every problem.
        """
        problems = validate_curriculum(curriculum)
        if problems:
            raise ValueError(f"{path}: " + "; ".join(problems))
        return cls(path=path, phases=tuple(phase_from_dict(d, i)
                                           for i, d in enumerate(curriculum["curriculum"])))

    @property
    def has_checks(self) -> bool:
        """Whether it is a rig check: any of its phases is a check (:mod:`fmri_gym.checks`)."""
        return any(isinstance(p, CheckPhase) for p in self.phases)

    def with_seeds(self, seed_of: Callable[[int], int]) -> tuple[Curriculum, dict[int, str]]:
        """The curriculum with every game phase seeded: its own ``seed``, else ``seed_of(index)``.

        The seed is written into the phase's ``spec`` too, so the manifest shows
        every seed played.

        :return: the seeded curriculum, and per game phase ``"pinned"`` or ``"derived"``.
        """
        phases, how = [], {}
        for i, phase in enumerate(self.phases):
            if isinstance(phase, GamePhase):
                how[i] = "pinned" if phase.seed is not None else "derived"
                seed = phase.seed if phase.seed is not None else seed_of(i)
                phase = replace(phase, seed=seed, spec={**phase.spec, "seed": seed})
            phases.append(phase)
        return replace(self, phases=tuple(phases)), how

    def record(self) -> list[dict]:
        """Its phases as the file has them (includes resolved): the manifest's ``curriculum``."""
        return [phase.spec for phase in self.phases]
