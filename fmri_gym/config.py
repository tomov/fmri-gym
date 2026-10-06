"""A run's file: load, save, check, and the trigger presets.

A run is one JSON file: a ``curriculum`` of phases (see README) and an
optional ``triggers`` section (:mod:`fmri_gym.triggers`); ``_``-prefixed keys
are notes. Several runs make a session, which is a shell script
(:func:`fmri_gym.gui.write_session`), not a config.

The file says what is played, not who plays it or where: subject, output
folder, window and the test switches are command-line flags, so one file
serves every participant.

There is one shape. A bare list, an unknown top-level key or an unknown phase
type is refused at start-up, with the reason.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from .keys import is_key, keycode, typed_keys
from .menu import menu_problems
from .rig import CONTROLS
from .triggers import TriggerError, TriggerSettings

#: The phases of a run; the ``check_*`` ones make a rig check (:mod:`fmri_gym.checks`).
PHASE_TYPES = ("fixation", "message", "game", "survey", "check_display", "check_frames",
               "check_triggers", "check_controls", "check_photodiode")
SECTIONS = ("curriculum", "triggers")
#: Exit status of ``fmri_play`` when the run was quit (ESC) before its end; a
#: session script (``set -e``) stops on it instead of starting the next run.
EXIT_QUIT = 3

#: Trigger presets the editor offers. Each names ``sync.mode`` and ``backend``
#: outright: a section that leaves them out runs, but is reported as NOT SET.
TRIGGER_PRESETS: dict[str, dict] = {
    "fMRI (wait for '=', no codes)": {
        "sync": {"mode": "wait", "key": "="}, "backend": "null"},
    # MEG and EEG alike: the stimulus PC starts the recording and marks it, over serial.
    "MEG (start from the trigger line, serial codes)": {
        "sync": {"mode": "send", "delay": 0.0}, "backend": "serial", "port": "/dev/ttyUSB0"},
    "EEG (start from the trigger line, serial codes)": {
        "sync": {"mode": "send", "delay": 0.0}, "backend": "serial", "port": "/dev/ttyUSB0"},
    "Behavioural (no scanner)": {
        "sync": {"mode": "none"}, "backend": "null"},
}


#: An episode's possible outcomes: EnvAdapter.outcome's names, and the subject's own endings.
OUTCOMES = frozenset({"won", "lost", "terminated", "truncated", "playing", "forfeit", "reset"})

def check_shape(config: Any, where: str) -> dict:
    """Enforce the one config shape.

    :param config: parsed JSON.
    :param where: file path (or another name for the source), for the message.
    :return: ``config``, unchanged.
    :raises ValueError: if it is not an object with a ``curriculum`` list of
        typed phases, or has a top-level key that is neither a section nor a
        ``_`` note.
    """
    if not isinstance(config, dict) or not isinstance(config.get("curriculum"), list):
        raise ValueError(f'{where}: expected a JSON object with a "curriculum" list')
    unknown = [k for k in config if k not in SECTIONS and not k.startswith("_")]
    if unknown:
        raise ValueError(f"{where}: unknown top-level key(s) {unknown}; the sections are "
                         f'{SECTIONS}, and a note must start with "_"')
    for i, phase in enumerate(config["curriculum"]):
        if not isinstance(phase, dict) or phase.get("type") not in PHASE_TYPES:
            raise ValueError(f'{where}: phase {i} must be an object whose "type" is one of '
                             f"{PHASE_TYPES}")
    return config


def load_config(path: str) -> dict:
    """Load a config file.

    :param path: JSON file path.
    :return: the config dict.
    :raises ValueError: if the file does not have the config shape
        (see :func:`check_shape`).
    """
    with open(path) as f:
        return check_shape(json.load(f), path)


def save_config(config: dict, path: str) -> None:
    """Write ``config`` as indented JSON.

    :param config: config dict.
    :param path: destination file path.
    """
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def new_config() -> dict:
    """A minimal starting config: one message, one fixation, one game."""
    return {
        "triggers": copy.deepcopy(next(iter(TRIGGER_PRESETS.values()))),
        "curriculum": [
            {"type": "message", "text": "Get ready\n\n(press SPACE to start)"},
            {"type": "fixation", "duration": 2.0},
            {"type": "game", "backend": "ale", "game": "ALE/Pong-v5", "mode": "duration",
             "duration": 60.0, "fps": 30,
             "keys": {"": 0, "A": 1, "UP": 2, "DOWN": 3, "UP+A": 4, "DOWN+A": 5}},
            {"type": "fixation", "duration": 2.0},
        ],
    }


def fold_cli_options(curriculum: list[dict], args: Any) -> None:
    """Fold the run's CLI-global backend options into the game phases they concern.

    Each per-block ``EnvAdapter`` then reads everything it needs from its own
    spec, and the manifest's curriculum shows what was actually played.

    :param curriculum: the run's phases, changed in place.
    :param args: ``fmri_play``'s parsed flags (``no_audio``).
    """
    for phase in curriculum:
        if phase.get("type") != "game":
            continue
        if args.no_audio:
            phase["audio"] = False


def validate_config(config: dict) -> list[str]:
    """Problems that would stop ``fmri_play`` before the first phase.

    Cheap checks only (no env is built, no port opened), for the editor's
    Check: the required game fields and the triggers section.

    :param config: a config dict of the right shape.
    :return: human-readable problems, empty when the config looks runnable.
    """
    problems: list[str] = []
    if not config["curriculum"]:
        problems.append("curriculum: needs at least one phase")
    for i, phase in enumerate(config["curriculum"]):
        problems.extend(f"phase {i}: {p}" for p in _phase_problems(phase))
    problems.extend(trigger_problems(config.get("triggers")))
    try:
        typed_keys()
    except ValueError as exc:  # this machine's rig file, which every game is read through
        problems.append(str(exc))
    problems.extend(trigger_key_clashes(config))
    return problems


def trigger_key_clashes(config: dict) -> list[str]:
    """The key the scanner types at every volume, where a game would hear it.

    A button box in a digit or letter mode sends its trigger as a key too
    (Current Designs: ``5``, ``t``). With ``sync.mode`` ``wait`` on that key, a
    game hears it at every volume when it stands for a rig key on this machine
    (the keyboard's, or the rig file's ``keys``).

    :param config: a config of the right shape.
    :return: at most one problem.
    """
    try:
        sync = TriggerSettings.from_dict(config.get("triggers")).sync
    except (TypeError, ValueError, TriggerError):
        return []  # trigger_problems says why
    if sync.mode != "wait" or not is_key(sync.key):
        return []
    try:
        rig_key = typed_keys().get(keycode(sync.key))
    except ValueError:
        return []  # validate_config says why
    if rig_key is None or not any(p["type"] == "game" for p in config["curriculum"]):
        return []
    return [f"triggers.sync.key: {sync.key!r}, the key the scanner types at every volume, "
            f"stands for the rig key {rig_key} on this machine, so a game would hear it: "
            "map that key to no rig key in the rig file, or wait for another"]


def _phase_problems(phase: dict) -> list[str]:
    if phase["type"].startswith("check_"):
        from .checks import phase_problems
        return phase_problems(phase)
    if phase["type"] != "game":
        return []
    out = []
    if not phase.get("game"):
        out.append("game: missing env id")
    if phase.get("mode", "duration") not in ("duration", "episode"):
        out.append(f"mode: expected 'duration' or 'episode', got {phase.get('mode')!r}")
    out.extend(_fps_problems(phase))
    out.extend(_keys_problems(phase))
    # advancing_outcomes gates n_episodes, so it means nothing to a block that
    # ends on the clock; the names are EnvAdapter.outcome's plus the menu's.
    if "advancing_outcomes" in phase:
        names = phase["advancing_outcomes"]
        if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
            out.append(f"advancing_outcomes: expected a list of outcome names, got {names!r}")
        elif set(names) - OUTCOMES:
            out.append(f"advancing_outcomes: unknown {sorted(set(names) - OUTCOMES)}; "
                       f"the names are {sorted(OUTCOMES)}")
        elif phase.get("mode", "duration") != "episode":
            out.append("advancing_outcomes: only in mode 'episode' (a duration block ends on the clock)")
    if "outcome_duration" in phase and not (
            isinstance(phase["outcome_duration"], (int, float))
            and phase["outcome_duration"] >= 0):
        out.append(f"outcome_duration: expected seconds >= 0, got {phase['outcome_duration']!r}")
    if "menu" in phase:
        out.extend(menu_problems(phase["menu"]))
    return out


def _keys_problems(phase: dict) -> list[str]:
    """``keys`` is required, and names rig keys only.

    The rig keys (:data:`~fmri_gym.rig.CONTROLS`) are what a game hears, on
    every rig, so a binding to any other name could never be pressed; nor could
    a combo in a ``turn_based`` phase, which steps on single presses. ``""`` is
    the no-key action. Whether the values fit the env's action space is checked
    when the env is built (:mod:`fmri_gym.adapters.keymap`).
    """
    keys = phase.get("keys")
    if not isinstance(keys, dict) or not keys:
        return ['keys: missing; map each rig key to the env action it sends, e.g. {"": 0, '
                '"LEFT": 3, "RIGHT": 2, "UP+A": 5} ("" is no key held; the rig keys are '
                f'{", ".join(CONTROLS)}; there is no default map)']
    unknown = [combo for combo in keys
               if combo and not set(combo.split("+")) <= set(CONTROLS)]
    if unknown:
        return [f"keys: {', '.join(unknown)}: not rig keys; a game hears only "
                f"{', '.join(CONTROLS)}, joined with + for a combo (\"UP+A\")"]
    combos = [combo for combo in keys if "+" in combo]
    if phase.get("turn_based", False) and combos:
        return [f"keys: {', '.join(combos)}: a turn_based phase steps on one key press, so a "
                "combo is never played; give the action a rig key of its own"]
    return []


def _fps_problems(phase: dict) -> list[str]:
    """``fps`` is required: a block's rate is the config's, not the engine's.

    It was the engine's own rate when the phase left it out, so the same file
    played at a different speed depending on the backend underneath it -- and
    silently changed rate when that backend did. The manifest reports the
    engine's own rate against it after a run.
    """
    fps = phase.get("fps")
    if fps is None:
        return ["fps: missing; state the block's steps per second (the engine's own rate is "
                "not assumed: 60 for console cores and Atari, 35 / frame_skip for Doom)"]
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
        return [f"fps: expected a positive number of steps per second, got {fps!r}"]
    return []


def trigger_problems(section: dict | None) -> list[str]:
    """What :class:`~fmri_gym.triggers.TriggerSettings` refuses, as text.

    The port is the one check added here: the settings accept a missing port
    and the backend refuses it on opening, which the editor's Check never does.

    :param section: the ``triggers`` section, or ``None``.
    :return: at most one problem (the first the settings raise).
    """
    try:
        s = TriggerSettings.from_dict(section)
    except (TypeError, ValueError, TriggerError) as exc:
        return [str(exc)]
    if s.backend in ("serial", "parallel") and not s.port:
        return [f"triggers: backend {s.backend!r} needs a port"]
    return []
