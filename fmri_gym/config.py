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
             "duration": 60.0, "fps": 30, "keys": {"UP": 2, "DOWN": 3}},
            {"type": "fixation", "duration": 2.0},
        ],
    }


def fold_cli_options(curriculum: list[dict], args: Any) -> None:
    """Fold the run's CLI-global backend options into the game phases they concern.

    Each per-block ``EnvAdapter`` then reads everything it needs from its own
    spec, and the manifest's curriculum shows what was actually played.

    :param curriculum: the run's phases, changed in place.
    :param args: ``fmri_play``'s parsed flags (``no_audio``, the repo paths).
    """
    for phase in curriculum:
        if phase.get("type") != "game":
            continue
        if args.no_audio:
            phase["audio"] = False
        if phase.get("backend") == "vgdl" and args.vgdl_repo:
            phase.setdefault("repo", args.vgdl_repo)
        if phase.get("backend") == "coom" and args.coom_repo:
            phase.setdefault("repo", args.coom_repo)


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
    problems.extend(trigger_key_clashes(config))
    return problems


def trigger_key_clashes(config: dict) -> list[str]:
    """Phases whose ``keys`` bind the key the scanner types at every volume.

    A button box in a digit mode sends its trigger as a digit too (Current
    Designs: ``5``). With ``sync.mode`` ``wait`` on that key, a game bound to
    it would act at every volume. Only a phase's own ``keys`` are checked: a
    backend's default map is known once its env is built.

    :param config: a config of the right shape.
    :return: one problem per clashing binding.
    """
    try:
        sync = TriggerSettings.from_dict(config.get("triggers")).sync
    except (TypeError, ValueError, TriggerError):
        return []  # trigger_problems says why
    name = sync.key.upper()
    if sync.mode != "wait" or not (len(name) == 1 and name.isalnum()):
        return []  # "=" is no game key: nothing can clash
    return [f"phase {i}: keys: {combo} uses {sync.key!r}, the key the scanner types at every "
            f"volume (triggers.sync.key): bind another key"
            for i, phase in enumerate(config["curriculum"])
            for combo in phase.get("keys", {}) if name in combo.upper().split("+")]


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
    return out


def _fps_problems(phase: dict) -> list[str]:
    """``fps`` is required: a block's rate is the config's, not the engine's.

    It was the engine's own rate when the phase left it out, so the same file
    played at a different speed depending on the backend underneath it -- and
    silently changed rate when that backend did. The editor's Controls tab
    shows what the engine runs at, to write here.
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
