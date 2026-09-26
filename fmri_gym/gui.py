"""``fmri-edit``: a config editor so a rig can be set up without JSON.

Session manager has two panels. Session design is the session's lines
(:func:`write_session`) -- each a run, a config file, or an external
command; add, repeat, remove, reorder, skip -- as a list or as the script
itself. Run design is the selected run's config (:mod:`fmri_gym.config`), as a
phase list with one form per phase type or as its JSON. Two more tabs edit
that config: Controls (the ``keys`` remap of a game phase,
with the backend's defaults on request), Triggers (the start sync, the
backend and the codes, with fMRI/MEG/EEG presets, a check and a live test). The
last tab, Launch, holds the command-line flags
of this one launch (subject, window, test switches): they are never saved to
the file. File > New / Open / Save / Save As.

A lone config is a session of one run with no script. Save writes the configs
shown here and, for several lines, the session script; Play saves and then
hands this process over to what plays what it showed -- one ``fmri-play`` for
a run, the script itself for a session -- so there is one process per run and
nothing in Python loops over runs. File > Open takes either kind of file, a
config (``.json``) or a session (``.sh``).

The triggers section is always written in full. A file that leaves
``sync.mode`` or ``backend`` out runs, but is reported as NOT SET on the
experimenter screen; the forms show a value for both, so Save states them.

The field tables, the parsers, the session scripts and the ``fmri-edit``
command line live here; the window itself is :mod:`fmri_gym.gui_qt`, imported
by :func:`edit_config` so that the rest is readable without PySide6. Field
labels are the config keys and each field's tip says what it does, so the
dialog and the file read alike.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import sys
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from . import bids
from . import config as cfg
from .triggers import (SYNC_MODES, TRIGGER_BACKENDS, TriggerSettings)

#: Suggestions for a game phase's ``backend`` (free text is accepted too; the
#: adapter registry, not this list, decides what exists).
BACKENDS = ("ale", "retro", "gym", "vgdl", "crafter", "minihack", "nethack", "vizdoom",
            "overcooked", "baba", "rushhour", "supertuxkart", "aigamestore")


# ---------------------------------------------------------------------------
# Field tables (pure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Field:
    """One labelled row of a form.

    ``kind``: ``str`` | ``int`` | ``float`` | ``bool`` | ``choice`` (fixed
    list) | ``combo`` (list plus free text) | ``text`` (multi-line string,
    always written) | ``lines`` (multi-line, one list item per line) | ``list``
    (one line, items separated by commas or spaces; numbers stay numbers). For the
    other kinds a blank entry means "not set", so the engine default applies;
    a ``bool`` is only written when it differs from ``default``.
    """

    key: str
    label: str
    kind: str = "str"
    choices: Sequence[str] = ()
    tip: str = ""
    default: Any = None


#: What the Launch tab starts at, and so what a run's command carries unless the
#: tab (or an opened session script) says otherwise. Fullscreen, as a session in
#: the scanner needs; untick it to pilot at the desk.
DEFAULT_LAUNCH: dict[str, Any] = {
    "subject": "sub-test", "ses": None, "data_root": "data", "size": "1024x768",
    "monitor": 0, "fullscreen": True, "no_vsync": False, "no_audio": False,
    "dummy_trigger": False,
}

#: Keys are the argparse names of ``fmri_play.py``'s flags; labels are the flags.
LAUNCH_FIELDS = [
    Field("subject", "--subject", tip="BIDS subject, sub-<letters/digits>: the first folder of "
                                      "the output (data/sub-01/ses-001/beh/...)."),
    Field("ses", "--ses", "int", tip="Session number. Blank = the subject's next free one, "
                                     "counted from the folders already there (1, 2, 3...); it is "
                                     "settled when Play starts, and a session's script asks for "
                                     "it then too. A number here is what the script takes when "
                                     "it is run without one -- `sh ses1.sh 003` overrides it, "
                                     "which is how a stopped session is resumed."),
    Field("data_root", "--data-root", tip="Where the BIDS tree goes. Blank = data."),
    Field("size", "--size", "combo", (),
          tip="The window's size as <w>x<h>, when not fullscreen: common sizes that fit the "
              "monitor, or type another. Fullscreen always takes the monitor's resolution."),
    Field("fullscreen", "--fullscreen", "bool", default=False,
          tip="Fill the screen. Windowed is handier for piloting at the desk."),
    Field("no_vsync", "--no-vsync", "bool", default=False,
          tip="Do not lock flips to the monitor refresh. Only if the display self-test "
              "(python -m fmri_gym.display) reports it cannot lock."),
    Field("no_audio", "--no-audio", "bool", default=False,
          tip="Mute game audio in every block; the manifest's curriculum then shows "
              "\"audio\": false."),
    Field("dummy_trigger", "--dummy-trigger", "bool", default=False,
          tip="Skip the experimenter and scanner waits. A test run, not a session -- though its "
              "data is saved like any run's and takes the next run number."),
]

SYNC_FIELDS = [
    Field("mode", "mode", "choice", SYNC_MODES,
          tip="After the experimenter's SPACE: wait = wait for 'key' from the trigger box (fMRI); "
              "send = send the start code on the trigger line, which starts the recording "
              "(MEG, EEG); none = start at once."),
    Field("key", "key", tip="The character the trigger box types at each volume (usually '=' "
                            "or '5'); the first one starts the clock."),
    # Saved as codes.scanner_start: shown here, where it is used, not among the other codes.
    Field("scanner_start", "start code", "int",
          tip="The value sent on the trigger line to start the recording. Saved as "
              "codes.scanner_start; it must share no bit with the other codes."),
    Field("delay", "delay (s)", "float",
          tip="Seconds between the start code and the clock anchor (t=0), for acquisitions that "
              "need a moment to come up."),
]

#: The sync rows each mode uses; the others are hidden in the editor and left out of the file.
SYNC_ROWS = {"wait": ("key",), "send": ("scanner_start", "delay"), "none": ()}

TRIGGER_FIELDS = [
    Field("backend", "backend", "choice", TRIGGER_BACKENDS,
          tip="null = no codes sent (fMRI). serial (pyserial), parallel (pyparallel) or lsl "
              "(pylsl). A backend that cannot be opened stops the run at the desk with the reason."),
    Field("port", "port", tip="serial/parallel device, e.g. /dev/ttyUSB0, COM3, /dev/parport0."),
    Field("lsl_stream_name", "lsl_stream_name", tip="lsl only: name of the trigger stream."),
    Field("pulse_ms", "pulse_ms", "float",
          tip="parallel only: how long a code stays on the port between blocks "
              "(inside a block the next frame code replaces it)."),
    Field("on_frame", "on_frame", "bool", default=True,
          tip="Send a code on every displayed game frame (cycling 1..2^frame_bits-1)."),
    Field("frame_every", "frame_every", "int",
          tip="Send the frame code every N frames (1 = all). Thin it for slow trigger links."),
    Field("on_episode_start", "on_episode_start", "bool", default=True,
          tip="Send the episode_start code with the first frame of each episode."),
]

CODE_FIELDS = [
    Field("frame_bits", "frame_bits", "int",
          tip="Low bits reserved for the frame counter: 3 -> frame codes 1..7. "
              "Lifecycle codes must live above these bits."),
    Field("task_start", "task_start", "int", tip="Sent when the session clock anchors (t=0)."),
    Field("task_stop", "task_stop", "int", tip="Sent when the session ends."),
    Field("episode_start", "episode_start", "int", tip="Sent with the first frame of an episode."),
]

_GAME_FIELDS = [
    Field("backend", "backend", "combo", BACKENDS,
          tip="Game engine adapter (fmri_gym/adapters). Type another name for a custom one."),
    Field("game", "game", "combo",
          tip="Env id for that backend. The list holds the games the configs in configs/ already "
              "set up for this backend; any other id can be typed."),
    Field("mode", "mode", "choice", ("duration", "episode"),
          tip="duration = play (and replay) until the time is up; episode = play n_episodes."),
    Field("duration", "duration (s)", "float", tip="duration mode: block length in seconds."),
    Field("n_episodes", "n_episodes", "int", tip="episode mode: how many episodes."),
    Field("max_duration", "max_duration (s)", "float",
          tip="episode mode: hard wall-clock cap for the block."),
    Field("fps", "fps", "float",
          tip="Steps per second. Required: every block states its own rate. The engine's own "
              "rate plays the game at its real speed and fits its sound (60 for consoles and "
              "Atari, 35 / frame_skip for Doom); Controls > backend defaults shows what this "
              "engine runs at. Pick one that divides the monitor's refresh."),
    Field("turn_based", "turn_based", "bool", default=False,
          tip="Step only on a key press instead of every frame (grid / text games)."),
    Field("seed", "seed", "int",
          tip="The phase's base seed: episode 1 plays with it, episode 2 with seed+1, and so "
              "on. Blank (the usual): derived from the run (subject, session, task, run) and the "
              "phase, so no two runs or participants replay each other's episodes; the line "
              "below shows it. A number pins it: the same episodes for every participant and "
              "run, for designs that need identical stimuli (Pin copies the derived one)."),
    Field("state_stride", "state_stride", "int",
          tip="Save a full emulator savestate every K frames (1 = every frame)."),
    Field("text", "text", tip="Optional label shown on the loading screen instead of the env id."),
    Field("audio", "audio", "bool", default=True,
          tip="Play the game's sound in this block. Off mutes the speakers; audio the env "
              "returns is still logged."),
]

PHASE_FIELDS: dict[str, list[Field]] = {
    "fixation": [Field("duration", "duration (s)", "float", tip="Seconds the cross is shown.")],
    "message": [
        Field("text", "text", "text", tip="The message. Lines are kept as typed."),
        Field("duration", "duration (s)", "float",
              tip="Seconds shown. Blank = stay until the participant presses 'key'."),
        Field("key", "key", tip="Key that dismisses an untimed message (default: space)."),
        Field("align", "align", "choice", ("center", "left"), tip="Text alignment."),
    ],
    "survey": [
        Field("questions", "questions", "lines", tip="One question per line."),
        Field("n_points", "n_points", "int", tip="Likert scale points (default 7)."),
    ],
    "game": _GAME_FIELDS,
    # The rig check's phases (fmri_gym/checks.py): blank = the quick check's value.
    "check_display": [
        Field("n", "flips", "int", tip="Flip intervals to measure (default 60)."),
        Field("seconds", "seconds", "float",
              tip="Flip for this long instead (the long check: 600, missed refreshes)."),
    ],
    "check_frames": [
        Field("rates", "rates (fps)", "list",
              tip="Frame rates to play the test pattern at, through the session's own game "
                  "loop: the rates your games use, and one that does not divide the refresh."),
        Field("seconds", "seconds per rate", "float", tip="Per rate and load (default 2)."),
        Field("loads", "loads", "list",
              tip="none, and/or cpu: every core kept busy while the frames play, to see "
                  "how the pacing and the frame triggers hold up."),
        Field("recording_hz", "recording (Hz)", "float",
              tip="The recording's sampling rate (MEG/EEG): each frame code must last 2 "
                  "samples. Blank: not checked."),
    ],
    "check_triggers": [
        Field("repeat", "passes", "int", tip="Passes of every line and code sent to the "
                                             "recording (default 1); only with a backend."),
        Field("hold_ms", "hold (ms)", "float", tip="How long each code is held (default 40)."),
        Field("gap_ms", "gap (ms)", "float", tip="Silence after each code (default 40)."),
        Field("pulses", "scanner pulses", "int",
              tip="Scanner pulses to count, when the run waits for the scanner (default 3)."),
        Field("seconds", "pulse seconds", "float",
              tip="Count pulses for this long instead (the long check: 600, clock drift)."),
        Field("timeout_s", "timeout (s)", "float",
              tip="Seconds allowed without a pulse before the test fails (default 60)."),
    ],
    "check_controls": [
        Field("timeout_s", "timeout (s)", "float",
              tip="Seconds to press each key when asked (default 10). The keys are on the "
                  "Controls tab: a device's buttons, each with what it stands for."),
    ],
    "check_photodiode": [
        Field("readout", "readout", "choice", ("soundcard", "recording"),
              tip="soundcard: the diode on this PC's input 0, offsets printed here. "
                  "recording: the diode in the MEG/EEG recording, matched offline to the "
                  "frame triggers (the Triggers tab's line)."),
        Field("input_device", "input device", tip="Sound-card input, index or name "
                                                  "(blank: the default)."),
        Field("mic", "microphone", "bool", default=False,
              tip="A microphone at the ear on input 1: the clicks are timed when heard."),
        Field("audio", "audio test", "bool", default=True,
              tip="A tone burst on every other flash through the session's audio output, "
                  "timed at the DAC."),
        Field("n", "flashes", "int", tip="Flashes (default 10; the long check 850)."),
        Field("on_ms", "on (ms)", "float", tip="White time per flash (default 50)."),
        Field("gap_min_ms", "gap min (ms)", "float", tip="Shortest black gap (default 150)."),
        Field("gap_max_ms", "gap max (ms)", "float", tip="Longest black gap (default 250)."),
        Field("settle_ms", "settle (ms)", "float",
              tip="Black before the first flash, for the input to settle (default 300)."),
        Field("corner", "corner", "choice", ("br", "bl", "tr", "tl"),
              tip="Where the patch is: under the diode."),
        Field("patch_px", "patch (px)", "int", tip="Patch side (default 120)."),
    ],
}

def parse_value(field: Field, raw: Any) -> Any:
    """Turn a widget's raw value into the config value (``None`` = not set).

    :raises ValueError: naming the field, when a number does not parse.
    """
    if field.kind == "bool":
        return bool(raw)
    text = str(raw)
    if field.kind == "text":
        return text  # a message's text is always stated, even when empty
    if not text.strip():
        return None
    if field.kind == "lines":
        return [line for line in text.split("\n") if line.strip()]
    if field.kind == "list":
        return [_item(item) for item in re.split(r"[,\s]+", text.strip()) if item]
    if field.kind not in ("int", "float"):
        return text.strip()
    try:
        return _number(text, field.kind)
    except ValueError as exc:
        raise ValueError(f"{field.label}: expected a number, got {text.strip()!r}") from exc


def _number(text: str, kind: str) -> int | float:
    """``"30"`` stays the int 30 in a float field, so Save does not turn it into 30.0."""
    try:
        return int(text)
    except ValueError:
        if kind == "int":
            raise
        return float(text)


def _item(text: str) -> int | float | str:
    """A ``list`` item: a number when it reads as one."""
    try:
        return _number(text, "float")
    except ValueError:
        return text


def format_value(field: Field, value: Any) -> Any:
    """The inverse of :func:`parse_value`: config value -> widget value."""
    if field.kind == "bool":
        return bool(field.default if value is None else value)
    if value is None:
        return ""
    if field.kind == "list" and isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, (list, tuple)):
        return "\n".join(str(v) for v in value)
    return str(value)


def field_values(fields: Sequence[Field], raw: dict[str, Any]) -> dict[str, Any]:
    """Parse a whole form; blanks are dropped, bools only kept when non-default."""
    out: dict[str, Any] = {}
    for f in fields:
        value = parse_value(f, raw.get(f.key, ""))
        if value is None or (f.kind == "bool" and value == f.default):
            continue
        out[f.key] = value
    return out


#: The repo's ``configs/`` folder (next to the package; absent from an installed wheel).
CONFIGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def curated_games(root: str = CONFIGS_DIR) -> tuple[dict[str, dict[str, tuple[str, dict]]],
                                                    list[str]]:
    """The game phases the configs under ``root`` already hold, per backend.

    Configs, not the engines' registries, feed the game drop-down: half the
    backends have no registry, and a config is a game set up with keys and fps
    that work, whether or not its engine is installed here.

    :param root: folder searched recursively for ``*.json``.
    :return: ``({backend: {game: (config path, phase)}}, unreadable paths)``;
        the first config in path order wins.
    """
    games: dict[str, dict[str, tuple[str, dict]]] = {}
    unreadable: list[str] = []
    for path in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
        try:
            config = cfg.load_config(path)
        except (OSError, ValueError):
            unreadable.append(os.path.relpath(path))
            continue
        for phase in (p for p in config["curriculum"] if p["type"] == "game"):
            backend = games.setdefault(phase.get("backend", "gym"), {})
            backend.setdefault(phase["game"], (os.path.relpath(path), phase))
    return games, unreadable


def listed_files(ext: str, root: str = CONFIGS_DIR) -> list[str]:
    """The ``*.<ext>`` files under ``root``, as the editor's drop-downs list them.

    Sessions (``sh``) and runs (``json``) are picked from here rather than found
    through a file dialog, so the repo's examples are one click away. A folder
    named ``unsupported`` holds configs that do not play; it is left out.

    :param ext: ``"sh"`` for session scripts, ``"json"`` for run configs.
    :param root: folder searched recursively.
    :return: paths relative to the working directory, in path order.
    """
    paths = sorted(glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True))
    return [os.path.relpath(p) for p in paths
            if "unsupported" not in os.path.relpath(p, root).split(os.sep)[:-1]]


def suggest_name(stem: str, taken: set[str]) -> str:
    """A free name counting up from ``stem``: ``ale``, ``ale_2``, ``ale_3`` -- never ``ale_2_2``.

    :param stem: the wanted name; a trailing ``_N`` is where counting resumes.
    :param taken: names in use.
    """
    if stem not in taken:
        return stem
    numbered = re.fullmatch(r"(.+)_(\d+)", stem)
    base, n = (numbered[1], int(numbered[2]) + 1) if numbered else (stem, 2)
    while f"{base}_{n}" in taken:
        n += 1
    return f"{base}_{n}"


def first_backend(config: dict) -> str:
    """What a new config's file is named after: the backend of its first game (``run`` if none)."""
    games = [p for p in config["curriculum"] if p["type"] == "game"]
    stem = re.sub(r"[^A-Za-z0-9_-]", "", games[0].get("backend", "gym")) if games else ""
    return stem or "run"


#: Window sizes offered when not fullscreen (those that fit the monitor).
WINDOW_SIZES = ((800, 600), (1024, 768), (1280, 720), (1280, 1024), (1600, 900), (1920, 1080))


def window_sizes(width: int, height: int) -> list[str]:
    """The window sizes to offer on a ``width`` x ``height`` monitor, its own size last."""
    fits = [f"{w}x{h}" for w, h in WINDOW_SIZES if w <= width and h <= height]
    return [*fits, f"{width}x{height}"] if f"{width}x{height}" not in fits else fits


#: Response devices that type keys, and the game key each button stands for. Every input is
#: a key press, so a device is a translation of the game's own keyboard map; the Controls tab
#: adds its keys to a phase's ``keys``, to edit there. A device's scanner trigger (Current
#: Designs: 5 in digit mode, t in letter mode) is left out: it arrives at every volume.
DEVICE_LAYOUTS: dict[str, dict[str, str]] = {
    # The keyboard needs no translation: every game plays with its own map, always.
    "Keyboard (the game's own keys)": {},
    "Button box (1 2 3 4 5)": {"1": "LEFT", "2": "DOWN", "3": "UP", "4": "RIGHT", "5": "SPACE"},
    "Current Designs fORP, HID KEY 12345 (1 2 3 4)": {"1": "LEFT", "2": "DOWN", "3": "UP",
                                                      "4": "RIGHT"},
    "Current Designs fORP, HID KEY BYGRT (b y g r)": {"B": "LEFT", "Y": "DOWN", "G": "UP",
                                                      "R": "RIGHT"},
}


def combo_name(combo: str) -> str:
    """``"space+Left"`` -> ``"LEFT+SPACE"``: the one spelling of a key combo."""
    return "+".join(sorted(key.strip().upper() for key in combo.split("+")))


def translate_keys(game_map: dict[str, Any], layout: dict[str, str]) -> dict[str, Any]:
    """The device's bindings: each combo of the game's map whose keys all have a button.

    :param game_map: combo -> action, as the phase plays on the keyboard now.
    :param layout: device key -> the game key it stands for (a :data:`DEVICE_LAYOUTS` value).
    :return: device combo -> the same action (``LEFT+SPACE`` -> ``1+5`` on a button box).
    """
    button = {game_key: device_key for device_key, game_key in layout.items()}
    out = {}
    for combo, action in game_map.items():
        keys = combo_name(combo).split("+")
        if all(key in button for key in keys):
            out[combo_name("+".join(button[key] for key in keys))] = action
    return out


def launch_values(form: dict) -> dict:
    """The Launch form as the flag values ``fmri_play`` reads, every flag present.

    :param form: what :func:`field_values` returned for :data:`LAUNCH_FIELDS`.
    :raises ValueError: if the subject is blank, or the size is not ``<w>x<h>``.
    """
    if "subject" not in form:
        raise ValueError("--subject: required")
    if not re.fullmatch(r"[1-9]\d*x[1-9]\d*", form.get("size", "")):
        raise ValueError(f"--size: expected <width>x<height> such as 1024x768, got "
                         f"{form.get('size', '')!r}")
    off = {f.key: False for f in LAUNCH_FIELDS if f.kind == "bool"}
    return {"ses": None, "data_root": "data", **off, **form}


def trigger_mismatches(steps: list[dict], configs: dict[str, dict]) -> list[str]:
    """Runs of one session whose triggers differ: a session has one scanner and one line.

    Each run keeps its own ``triggers`` section, so one edited on the Triggers
    tab changes that run only; a rig check set to ``send`` before a game run
    left at ``wait`` would test a line the session never uses.

    :param steps: the session's lines; skipped ones and commands are left out.
    :param configs: config path -> config.
    :return: one problem naming each run's settings, or nothing when they agree.
    """
    seen: dict[str, list[str]] = {}
    for step in steps:
        if step["skip"] or "config" not in step:
            continue
        try:
            s = asdict(TriggerSettings.from_dict(configs[step["config"]].get("triggers")))
        except (TypeError, ValueError):
            continue  # validate_config names what is wrong with it
        s.pop("defaulted")
        seen.setdefault(json.dumps(s, sort_keys=True), []).append(step["config"])
    if len(seen) < 2:
        return []
    groups = "; ".join(f"{', '.join(paths)}: {_short_triggers(json.loads(key))}"
                       for key, paths in seen.items())
    return [f"the session's runs have different triggers ({groups}); a session has one "
            "scanner and one trigger line: make them the same on the Triggers tab"]


def _short_triggers(s: dict) -> str:
    """``sync wait for '=', backend null`` / ``sync send, backend serial /dev/ttyUSB0``."""
    sync = s["sync"]
    start = f"sync wait for {sync['key']!r}" if sync["mode"] == "wait" else f"sync {sync['mode']}"
    line = {"lsl": s["lsl_stream_name"], "serial": s["port"], "parallel": s["port"]}
    where = f" {line[s['backend']]}" if line.get(s["backend"]) else ""
    return f"{start}, backend {s['backend']}{where}"


def phase_label(index: int, phase: dict) -> str:
    """One line for the curriculum list: ``"03  game  ale ALE/Pong-v5 (60 s)"``."""
    kind = phase["type"]
    if kind == "game":
        length = (f"{phase.get('duration', 30.0):g}s" if phase.get("mode", "duration") == "duration"
                  else f"{phase.get('n_episodes', 1)}ep")
        detail = f"{str(phase.get('game', '?')).split('/')[-1]} {length}"
    elif kind == "message":
        text = phase.get("text", "")
        detail = (text[0] if text else "") if isinstance(text, list) else text.split("\n")[0]
    elif kind == "survey":
        detail = f"{len(phase.get('questions', []))} questions"
    elif kind == "check_controls":
        detail = " ".join(phase.get("keys", {})) or "no keys"
    elif kind.startswith("check_"):
        detail = "rig check"
    else:
        detail = f"{phase.get('duration', 2.0)} s"
    return f"{index:02d} {kind:<8} {detail[:26]}"


def parse_action(text: str) -> Any:
    """``"2"`` -> 2, ``"[1,0,0]"`` -> list, anything else -> the string itself."""
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        return text


def format_action(action: Any) -> str:
    """The inverse of :func:`parse_action`.

    A string is shown bare only if it reads back as itself: the action ``"1"``
    is shown quoted, or it would come back as the int 1 and rebind the key.
    """
    if isinstance(action, str) and parse_action(action) == action:
        return action
    return json.dumps(action)


def split_phase(phase: dict, fields: Sequence[Field]) -> tuple[dict, dict]:
    """Split a phase into (form values, extra keys not on the form).

    ``type`` and ``keys`` belong to neither: the first is fixed, the second
    is edited on the Controls tab.
    """
    known = {f.key for f in fields} | {"type", "keys"}
    form = {k: v for k, v in phase.items() if k in known}
    extra = {k: v for k, v in phase.items() if k not in known}
    return form, extra


def triggers_section(sync: dict, trigger: dict, codes: dict) -> dict:
    """Assemble the ``triggers`` section from the three forms.

    Bools are written even at their default; the sync form's start code goes to
    ``codes``; ``sync`` keeps only its mode's rows (an unknown mode keeps all,
    for :class:`TriggerSettings` to refuse by name).
    """
    sync = dict(sync)
    start = sync.pop("scanner_start", None)
    if sync["mode"] in SYNC_ROWS:
        sync = {k: v for k, v in sync.items() if k == "mode" or k in SYNC_ROWS[sync["mode"]]}
    codes = codes if start is None else {**codes, "scanner_start": start}
    stated = {f.key: f.default for f in TRIGGER_FIELDS if f.kind == "bool"} | trigger
    ordered = {f.key: stated[f.key] for f in TRIGGER_FIELDS if f.key in stated}
    return {"sync": sync, **ordered, "codes": codes}


def describe_triggers(section: dict) -> str:
    """What a run will do on the trigger side, in a few plain lines.

    :raises ValueError: if the section does not validate.
    :raises TriggerError: if ``sync.mode`` is ``send`` with the ``null`` backend.
    """
    m = TriggerSettings.from_dict(section)
    sync = m.sync
    where = m.lsl_stream_name if m.backend == "lsl" else m.port
    lines = ["After the experimenter's SPACE:"]
    if sync.mode == "wait":
        lines.append(f"  wait for {sync.key!r} from the trigger box, then anchor the clock (t=0).")
    elif sync.mode == "send":
        lines.append(f"  send the start code {m.codes.scanner_start} on {m.backend} {where}, "
                     f"wait {sync.delay} s, then anchor the clock (t=0).")
    else:
        lines.append("  anchor the clock (t=0) immediately.")
    if m.backend == "null":
        lines.append("No trigger codes sent (backend null); only the log files carry timing.")
        return "\n".join(lines)
    frames = (f"frame codes 1..{m.codes.frame_mask} every {m.frame_every} frame(s)"
              if m.on_frame else "no frame codes")
    lines.append(f"Codes on {m.backend} {where}: task_start={m.codes.task_start}, "
                 f"episode_start={m.codes.episode_start if m.on_episode_start else 'off'}, "
                 f"{frames}, task_stop={m.codes.task_stop}.")
    lines.append(f"Codes share no bits: episode_start during frame code 2 is sent as "
                 f"{m.codes.episode_start | 2}.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session scripts (the .sh the Session manager edits; config.py has the .json)
# ---------------------------------------------------------------------------


HEADER = ["#!/bin/sh", "# fmri-gym session: one line per run, in order.", "set -e"]
PLAY = "uv run fmri-play"
#: What a script's first line asks for the session number when it is given none.
SES_TOOL = "uv run fmri-ses"
DATA_ROOT = "data"
#: The launch flags a session carries on every run's line (argparse names of ``fmri_play``).
SWITCHES = ("fullscreen", "no_vsync", "no_audio", "dummy_trigger")
#: Where a run's line takes its ``--ses``: the session the script picked once, in
#: its first line. Written as typed, not quoted -- the shell has to expand it.
SES_VAR = '"$SES"'
_PINNED = re.compile(r"SES=\$\{1:-(\d+)\}$")


def write_session(steps: list[dict], launch: dict) -> str:
    """A session script: one line per run, in order.

    ::

        #!/bin/sh
        # fmri-gym session: one line per run, in order.
        set -e
        SES=${1:-$(uv run fmri-ses --subject sub-01)}
        uv run fmri-play --curriculum configs/pong.json --subject sub-01 --ses "$SES" --run 1 ...
        ./scripts/localizer.sh "$SES"
        # uv run fmri-play --curriculum configs/mario.json --subject sub-01 --ses "$SES" --run 1 ...

    ``SES=`` picks the session once for all the runs: the script's own argument
    (``sh ses1.sh 003`` resumes that session), else the subject's next free one.
    Each run's ``--run`` is :func:`run_number`, its place among the lines that
    play its task -- fixed by the design, so skipping a line renumbers nothing.
    A commented line is a skipped run; any other line is a command of yours (it
    may use ``"$SES"``). ``set -e`` stops the session at the first line that
    fails or is quit.

    :param steps: ``{"config" | "command", "skip"}`` dicts, in order.
    :param launch: subject, ses (``None``: the next free one), data_root, size,
        monitor and the :data:`SWITCHES`; written on every run's line (the
        data root and monitor only when not the default).
    :return: the script.
    """
    lines = [*HEADER, _ses_line(launch)]
    for i, step in enumerate(steps):
        line = (step["command"] if "command" in step
                else _play_line(step["config"], launch, run_number(steps, i)))
        lines.append(f"# {line}" if step["skip"] else line)
    return "\n".join(lines) + "\n"


def run_number(steps: list[dict], index: int) -> int:
    """Which run of its task the line at ``index`` is: 1, 2, 3...

    Counted over the lines before it that play the same task, skipped ones
    included: a run's number belongs to the session's design, so resuming a
    session with some lines skipped leaves every other number where it was.

    :param steps: the session's lines (see :func:`write_session`).
    :param index: the line to number; it must be a run, not a command.
    :return: its ``--run``.
    """
    task = bids.task_label(steps[index]["config"])
    return 1 + sum(1 for step in steps[:index]
                   if "config" in step and bids.task_label(step["config"]) == task)


def read_session(text: str) -> tuple[list[dict], dict | None]:
    """The steps and launch flags of a session script (see :func:`write_session`).

    An ``fmri-play`` line the editor could not write back as it is -- other
    flags, flags unlike the session's, or a ``--run`` other than the one its
    place gives it (:func:`run_number`) -- is kept as a command, as typed.

    :param text: the script.
    :return: ``(steps, launch)``; ``launch`` is ``None`` if no line is a run
        (nothing to read the flags from).
    """
    body = [line for line in text.splitlines() if line.strip() and line not in HEADER]
    ses_line = next((line for line in body if line.startswith("SES=")), "")
    pinned = _PINNED.search(ses_line)
    launch: dict | None = None
    steps: list[dict] = []
    for line in body:
        if line is ses_line:
            continue
        skip = line.startswith("# ")
        bare = line[2:] if skip else line
        played = _parse_play_line(bare)
        if played is not None and launch in (None, played[1]):
            steps.append({"config": played[0], "skip": skip})
            if played[2] == run_number(steps, len(steps) - 1):
                launch = played[1]
                continue
            steps.pop()  # a run number of its own: not one the editor can write
        steps.append({"command": bare, "skip": skip})
    if launch is not None:
        launch["ses"] = int(pinned[1]) if pinned else None
    return steps, launch


def play_command(config_path: str, launch: dict, ses: str, run: int) -> list[str]:
    """The command that plays one run: ``fmri-play``, its config and the launch flags.

    The editor's Play runs this; a session script holds one per line.

    :param config_path: the run's config file.
    :param launch: the launch flags (see :data:`DEFAULT_LAUNCH`).
    :param ses: the ``--ses`` value -- a number, or :data:`SES_VAR` in a script.
    :param run: the ``--run`` value (see :func:`run_number`).
    :return: the command, word by word.
    """
    root = [] if launch["data_root"] == DATA_ROOT else ["--data-root", launch["data_root"]]
    monitor = ["--monitor", str(launch["monitor"])] if launch["monitor"] else []
    switches = [f"--{key.replace('_', '-')}" for key in SWITCHES if launch[key]]
    return [*PLAY.split(), "--curriculum", config_path, "--subject", launch["subject"], *root,
            "--ses", ses, "--run", str(run),
            "--size", launch["size"], *monitor, *switches]


def _ses_line(launch: dict) -> str:
    """``SES=``: the script's own argument, else the number pinned here or the next free one."""
    if launch["ses"] is not None:
        return f"SES=${{1:-{launch['ses']:03d}}}"
    root = [] if launch["data_root"] == DATA_ROOT else ["--data-root", launch["data_root"]]
    return f"SES=${{1:-$({_shell([*SES_TOOL.split(), '--subject', launch['subject'], *root])})}}"


def _play_line(config_path: str, launch: dict, run: int) -> str:
    return _shell(play_command(config_path, launch, SES_VAR, run))


def _shell(command: list[str]) -> str:
    """A command as a script line: each word quoted, but ``$SES`` left for the shell."""
    return " ".join(word if word == SES_VAR else shlex.quote(word) for word in command)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # argparse would print usage and exit
        raise ValueError(message)


def _parse_play_line(line: str) -> tuple[str, dict, int] | None:
    """``(config path, launch, run)`` if ``line`` is a run's line the editor can write.

    :return: ``None`` if it is not one -- another command, or flags the editor
        has no field for -- and the caller then keeps the line as typed.
    """
    if not line.startswith(PLAY + " "):
        return None
    parser = _Parser(add_help=False, allow_abbrev=False)
    for flag in ("--curriculum", "--subject", "--size", "--ses"):
        parser.add_argument(flag, required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--monitor", type=int, default=0)
    for key in SWITCHES:
        parser.add_argument(f"--{key.replace('_', '-')}", action="store_true")
    try:
        args = vars(parser.parse_args(shlex.split(line)[3:]))
    except ValueError:
        return None  # flags the editor has no field for: the caller keeps the line as typed
    if args.pop("ses") != "$SES":
        return None  # a run pinned to a session of its own: not one the editor can write
    run = args.pop("run")
    return args.pop("curriculum"), args, run


def edit_config(config: dict, path: str | None,
                session: str | None = None) -> list[str] | None:
    """Open the editor; return the command that plays what it showed, or ``None``.

    :param config: config dict (see :func:`fmri_gym.config.check_shape`).
    :param path: file it came from, for Save; ``None`` if new.
    :param session: a session script to open instead of ``config``.
    :return: what Play starts, word by word -- one ``fmri-play`` for a run
        (:func:`play_command`), ``sh <script>`` for a session of several. Both
        play files on disk: Play saves before it returns. ``None`` when the
        window was closed instead.
    :raises ImportError: if PySide6 is not installed, with the install line.
    """
    try:
        from .gui_qt import run_editor
    except ImportError as exc:
        if not (exc.name or "").startswith("PySide6"):
            raise
        raise ImportError("fmri-edit needs PySide6: `uv sync --extra gui` (name the other "
                          "extras you use too, or uv removes them), or `pip install "
                          "PySide6-Essentials`") from exc
    return run_editor(config, path, dict(DEFAULT_LAUNCH), session)


def main() -> None:
    """``fmri-edit``: design a run or a session, then become what plays it.

    The editor opens on a run config, a session script, or a new run. Play
    saves what is shown and this process turns into the command that plays it:
    one ``fmri-play`` for a run, the script itself for a session of several.
    Nothing about a launch is a flag here -- subject, window, the test switches
    are the Launch tab's, and a session writes them on every line of its script.
    """
    p = argparse.ArgumentParser(description="Design fmri-gym runs and sessions, then play one.")
    p.add_argument("--curriculum", help="run config (.json) to open; default: a new run")
    p.add_argument("--session", help="session script (.sh, one line per run) to open. To play "
                                     "one without the editor, run it: sh <session>.sh")
    args = p.parse_args()
    if args.session and args.curriculum:
        p.error("--session and --curriculum: the session already names its configs")
    config = cfg.load_config(args.curriculum) if args.curriculum else cfg.new_config()
    try:
        command = edit_config(config, args.curriculum, session=args.session)
    except KeyboardInterrupt:
        print("interrupted: the editor is closed and nothing was run; edits since the last "
              "Save are not written", file=sys.stderr)
        sys.exit(130)  # the shell's status for a Ctrl+C
    if command is not None:
        os.execvp(command[0], command)
