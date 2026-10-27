"""``session-editor``: design a session -- its runs and their curricula -- without JSON.

Two tabs. **Session** is the session script (:mod:`fmri_gym.session_script`):
who it is for and on which rig (each fixed here, or given when it is started),
then its lines -- each a run (a curriculum file) or an external command; add,
repeat, remove, reorder, skip -- as a list or as the script itself. **Run** is
one run: how it is played (its ``fmri-play`` flags), and its curriculum
(:mod:`fmri_gym.config.curriculum`) as a phase list with one form per phase
type -- a game's keys among them -- or as its JSON. How a rig plays it --
window, controller, audio, data folder, triggers -- is the rig file's
(:mod:`fmri_gym.gui.rig_config`). File > New / Open / Save / Save As.

Nothing is played from here: a session is played by its script (``sh
<script>``), a lone run by ``fmri-play``, and the bottom bar shows the command.
A lone curriculum is a session of one run with no script. File > Open takes
either kind of file, a curriculum (``.json``) or a session (``.sh``).

The model is ``self.steps`` -- the session's lines, each a run (a config file)
or an external command -- and ``self.configs``, the runs' configs by path. Each
tab has a form view and the text it stands for: the list of lines or the
script, the phases or the JSON. Whichever view is shown is the truth: every
handler first stores it into the model (:meth:`_Editor._store_current`,
:meth:`_Editor._store_session`), and leaving a text view that does not parse
is refused with the reason, never dropped.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from PySide6 import QtGui, QtWidgets
from PySide6.QtCore import Qt

from .. import bids, session_script
from ..config import curriculum
from ..config import rig as rig_file
from ..config.curriculum import Curriculum
from ..config.run import RunConfig
from ..run import END_FIXATION_S
from . import Field, rig_config
from . import widgets as ui


#: Suggestions for a game phase's ``backend`` (free text is accepted too; the
#: adapter registry, not this list, decides what exists).
BACKENDS = ("ale", "retro", "gym", "vgdl", "crafter", "minihack", "nethack", "vizdoom",
            "baba", "rushhour", "stk", "aigamestore")


#: The two views of the Session and Run tabs (sub-tab indices).
_LIST, _SCRIPT = 0, 1
_PHASES, _JSON = 0, 1


_GAME_KEYS_HINT = (
    "Game keys: which keyboard key does what in this game -- the same on every rig. Write "
    "keyboard keys only (UP, DOWN, LEFT, RIGHT, SPACE, RETURN, A-Z, 0-9, ..., joined with + "
    "for a combo), never a controller's buttons: a rig with a controller makes its buttons "
    "press these keys (its bindings, set in rig-config), and a button box types its keys "
    "itself (its response keys). There is no default map. The action is what the game takes: "
    "a Discrete index, or for a MultiBinary game the index of the button the key holds down. "
    "A real-time Discrete game also needs a row with a blank key: the action sent while no "
    "key is held.")

_GUIDE = ("A session is a script that plays its runs in order; a run plays one curriculum. "
          "1) Say who and on which rig (or leave them to whoever starts it). 2) List the runs. "
          "3) Open a run (Run tab) to see and change how it is played, its phases and each "
          "game's keys. 4) Check, then Save. Nothing plays from here: the line at the bottom is the command "
          "that plays what you see. Hover over anything for what it does.")

#: What a session script fixes for all of its runs; blank: given when it is started.
SESSION_FIELDS = [
    Field("subject", "subject", tip="The participant, as BIDS names them: sub-<letters/digits>, "
                                    "e.g. sub-01. It is the first folder of the data "
                                    "(sub-01/ses-001/...). Leave it blank to give it each time "
                                    "the session is started (SUBJECT=sub-01 sh <script>)."),
    Field("ses", "session", "int", tip="The session number, e.g. 2 for ses-002. Blank: the "
                                       "subject's next free one, counted from the folders "
                                       "already there. The script's own argument always wins "
                                       "(sh <script> 003): that is how a stopped session is "
                                       "resumed into the session it stopped in."),
]


#: How one run is played: the ``fmri-play`` flags a run's line carries
#: (:data:`fmri_gym.session_script.RUN_FLAGS`).
RUN_FLAG_FIELDS = [
    Field("duration", "run length (s)", "float",
          tip="How long the run lasts, in seconds from its start, closing fixation included: "
              "whatever is playing stops in time for it. Blank: the run lasts as long as its "
              "phases take."),
    Field("end_fixation", "closing fixation (s)", "float",
          tip=f"Seconds of fixation cross the run ends on, after the games stop. Blank: "
              f"{END_FIXATION_S:g} s."),
    Field("loop", "start again when it ends", "bool", default=False,
          tip="When the phases end before the run length, play them again from the first."),
    Field("resume", "carry on where the last run stopped", "bool", default=False,
          tip="Start where this participant's last run of this curriculum stopped (in any "
              "session), in the middle of a game if need be, instead of at the beginning."),
]

#: The phase fields that apply, given what the phase says so far; a kind not here shows all.
PHASE_ROWS: dict[str, Callable[[dict], list[str]]] = {
    "game": lambda v: [f.key for f in _GAME_FIELDS
                       if v.get("mode") == "episode" or f.key not in ("n_episodes", "advance_on")],
    "message": lambda v: ["text", "duration", "align"] + ["key"] * (not v.get("duration")),
    "check_display": lambda v: ["seconds"] + ["n"] * (not v.get("seconds")),
}

_GAME_FIELDS = [
    Field("backend", "backend", "combo", BACKENDS,
          tip="The game engine the game runs on. The list holds the engines fmri-gym plays."),
    Field("game", "game", "combo",
          tip="The game, by its name in that engine. The list holds the games already set up "
              "for it, with keys and speed that work; pick one, then 'Fill this phase from that "
              "config' below."),
    Field("mode", "mode", "choice", ("duration", "episode"),
          tip="duration: the block lasts a set time, starting a new game whenever one ends. "
              "episode: the block lasts a set number of games (episodes)."),
    Field("duration", "duration (s)", "float",
          tip="duration mode: how long the block lasts, in seconds (default 30). episode mode: the "
              "longest it may last, in seconds (default 300)."),
    Field("n_episodes", "n_episodes", "int", tip="episode mode: how many episodes."),
    Field("advance_on", "advance_on", "list",
          tip="Which endings count as a played episode, e.g. won: any other ending replays the "
              "same game. Names: won, lost, terminated, truncated, playing. Blank: every ending "
              "counts."),
    Field("fps", "fps", "float",
          tip="Game steps per second (required). The game's own speed plays it as designed and "
              "keeps its sound right: 60 for Atari and consoles, 35 for Doom; Game keys > Check "
              "with the engine shows it. Best if it divides the screen's refresh rate."),
    Field("is_turn_based", "is_turn_based", "bool", default=False,
          tip="Step only on a key press instead of every frame (grid / text games)."),
    Field("seed", "seed", "int",
          tip="The phase's base seed: episode 1 plays with it, episode 2 with seed+1, and so "
              "on. Blank (the usual): derived from the run (subject, session, task, run) and the "
              "phase, so no two runs or participants replay each other's episodes; the line "
              "below shows it. A number pins it: the same episodes for every participant and "
              "run, for designs that need identical stimuli (Pin copies the derived one)."),
    Field("state_stride", "state_stride", "int",
          tip="How often the game's full state is saved in the data, to replay it exactly: every K "
              "frames (1: every frame)."),
    Field("frame_stride", "frame_stride", "int",
          tip="How often the screen image is saved in the data: every K frames (1: every frame, "
              "0: never)."),
    Field("text", "text", tip="Optional label shown on the loading screen instead of the env id."),
    Field("has_audio", "has_audio", "bool", default=True,
          tip="Play this game's sound. Off: silent for the participant (its sound is still saved "
              "in the data). A rig with game sound off is silent whatever this says."),
    Field("show_outcome_duration", "show_outcome_duration (s)", "float",
          tip="Seconds the final score and outcome (won / lost / time is up ...) are shown after "
              "each episode (default 2; 0 = none)."),
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
    ],
    "check_triggers": [
        Field("repeat", "passes", "int", tip="How many times every code is sent to the "
                                             "recording (default 1); only on a rig that sends "
                                             "codes."),
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
              tip="Seconds to press each key when asked (default 10). The keys asked for are "
                  "the rig's: its controller bindings and its response keys (rig-config, "
                  "Response devices step)."),
    ],
    "check_photodiode": [
        Field("n", "flashes", "int", tip="Flashes (default 10; the long check 850)."),
        Field("on_ms", "on (ms)", "float", tip="White time per flash (default 50)."),
        Field("gap_min_ms", "gap min (ms)", "float", tip="Shortest black gap (default 150)."),
        Field("gap_max_ms", "gap max (ms)", "float", tip="Longest black gap (default 250)."),
        Field("settle_ms", "settle (ms)", "float",
              tip="Black before the first flash, for the input to settle (default 300)."),
    ],
}


#: The repo's ``configs/`` folder (next to the package; absent from an installed wheel).
CONFIGS_DIR = str(Path(__file__).resolve().parents[2] / "configs")
#: Its curricula, one folder per backend; the rest of ``configs/`` is rigs and sessions.
CURRICULA_DIR = os.path.join(CONFIGS_DIR, "curricula")


def curated_games(root: str = CURRICULA_DIR) -> tuple[dict[str, dict[str, tuple[str, dict]]],
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
        if curriculum.is_common_file(path):
            continue
        try:
            config = curriculum.load_curriculum(path)
        except (OSError, ValueError):
            unreadable.append(os.path.relpath(path))
            continue
        for phase in (p for p in config["curriculum"] if p["type"] == "game"):
            backend = games.setdefault(phase.get("backend", "gym"), {})
            backend.setdefault(phase["game"], (os.path.relpath(path), phase))
    return games, unreadable


def listed_files(ext: str) -> list[str]:
    """The files the editor's drop-downs list: session scripts in ``configs/``, or curricula.

    Sessions (``sh``) and runs (``json``) are picked from here rather than found
    through a file dialog, so the repo's examples are one click away. Shared phases
    (:func:`fmri_gym.config.curriculum.is_common_file`) are not runs; they are left out.

    :param ext: ``"sh"`` for session scripts, ``"json"`` for run configs.
    :return: paths relative to the working directory, in path order.
    """
    root = CURRICULA_DIR if ext == "json" else CONFIGS_DIR
    paths = sorted(glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True))
    return [os.path.relpath(p) for p in paths if not curriculum.is_common_file(p)]


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
    elif kind.startswith("check_"):
        detail = "rig check"
    else:
        detail = f"{phase.get('duration', 2.0)} s"
    return f"{index:02d} {kind:<8} {detail[:26]}"


def split_phase(phase: dict, fields: Sequence[Field]) -> tuple[dict, dict]:
    """Split a phase into (form values, extra keys not on the form).

    ``type`` and ``keys`` belong to neither: the first is fixed, the second
    is edited in the phase's Keys table.
    """
    known = {f.key for f in fields} | {"type", "keys"}
    form = {k: v for k, v in phase.items() if k in known}
    extra = {k: v for k, v in phase.items() if k not in known}
    return form, extra


def main() -> None:
    """``session-editor``: design a session or a run; play it with its script or ``fmri-play``."""
    p = argparse.ArgumentParser(description="Design fmri-gym sessions and their runs. Nothing "
                                            "is played from here: run a session's script "
                                            "(sh <script>), or a lone run with fmri-play.")
    p.add_argument("--curriculum", help="run curriculum (.json) to open; default: a new run")
    p.add_argument("--session", help="session script (.sh, one line per run) to open")
    args = p.parse_args()
    if args.session and args.curriculum:
        p.error("--session and --curriculum: the session already names its curricula")
    config = (curriculum.load_curriculum(args.curriculum, keep_includes=True) if args.curriculum
              else curriculum.new_curriculum())
    try:
        edit_config(config, args.curriculum, session=args.session)
    except KeyboardInterrupt:
        print("interrupted: the editor is closed; edits since the last Save are not written",
              file=sys.stderr)
        sys.exit(130)  # the shell's status for a Ctrl+C


def edit_config(config: dict, path: str | None, session: str | None = None) -> None:
    """Show the editor and block until it closes.

    :param config: config dict (see :func:`fmri_gym.config.curriculum.check_shape`).
    :param path: file it came from, for Save; ``None`` if new.
    :param session: a session script to open instead of ``config``.
    :raises KeyboardInterrupt: on Ctrl+C in the terminal, once the window is closed.
    """
    app = ui.app()
    editor = _Editor(config, path, session)
    editor.show()
    with ui.closed_by_ctrl_c(app) as interrupted:
        app.exec()
    if interrupted:
        editor.hide()
        raise KeyboardInterrupt


class _Editor(QtWidgets.QMainWindow):
    """The window: menu, tabs, bottom bar."""

    def __init__(self, config: dict, path: str | None, session: str | None = None) -> None:
        super().__init__()
        self.session_path: str | None = None
        self.steps: list[dict] = []     # {"config": path} | {"command": text}, each + "skip"
        self.step_index = 0
        self.configs: dict[str, dict] = {}   # the runs' configs by path
        self.opened: set[str] = set()        # ... those shown here, which Save writes
        # ... those created here: named at once (so the list and the script can show them) but
        # not on disk until Save, which asks where they go.
        self.new: set[str] = set()
        self.phases: list[dict] = []    # the selected run's curriculum
        self.notes: dict = {}           # top-level keys the editor does not own (its "##" comments)
        self.edit_index: int | None = None
        self.key_table: ui.KeyTable | None = None
        self.check_text: QtWidgets.QPlainTextEdit | None = None
        self.phase_form: ui.Form | None = None
        self.seed_label: QtWidgets.QLabel | None = None
        self.extra_text: QtWidgets.QPlainTextEdit | None = None
        # The view each panel shows, and the text a text view was last filled with: a
        # text view is stored back only when edited, so an untouched one changes nothing.
        self._session_view, self._run_view = _LIST, _PHASES
        self._script_shown = self._json_shown = ""
        self.setWindowTitle("fmri-gym session editor")
        self.resize(1250, 850)
        self._build_menu()
        bottom = self._build_bottom()
        self.tabs = QtWidgets.QTabWidget()
        self._build_tabs()
        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        guide = QtWidgets.QLabel(_GUIDE)
        guide.setObjectName("hint")
        guide.setWordWrap(True)
        layout.addWidget(guide)
        layout.addWidget(self.tabs)
        layout.addLayout(bottom)
        self.setCentralWidget(body)
        self.tabs.currentChanged.connect(lambda _index: self._on_tab_change())
        self.curated, unreadable = curated_games()
        if session is None:
            self.open_config(config, path)
        else:
            self.open_session(session)
        if unreadable:
            self._set_status("game lists: skipped config(s) that do not load: "
                             + ", ".join(unreadable))

    # -- construction ------------------------------------------------------

    def _build_tabs(self) -> None:
        self.tabs.addTab(self._session_design(), "Session")
        self.tabs.addTab(self._run_design(), "Run")
        self.tabs.setTabToolTip(0, "Who the session is for, on which rig, and its runs in order.")
        self.tabs.setTabToolTip(1, "One run: how it is played, its phases, and each game's keys.")

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("&File")
        keys = QtGui.QKeySequence.StandardKey
        for label, slot, shortcut in (("&New", self._new, keys.New), ("&Open...", self._open, keys.Open),
                                      ("&Save", self._save, keys.Save),
                                      ("Save &As...", self._save_as, keys.SaveAs),
                                      ("&Quit", self.close, keys.Quit)):
            action = menu.addAction(label)
            action.setShortcut(shortcut)
            action.triggered.connect(lambda _checked=False, s=slot: s())

    def _build_bottom(self) -> QtWidgets.QVBoxLayout:
        """File buttons and Check; under them, the command that plays what is shown."""
        check = ui.button("Check", self._check)
        check.setToolTip("Look for what would stop a run before anyone is in the scanner: "
                         "every curriculum's phases and keys, the rig file, empty lines.")
        bar = ui.row(ui.button("New", self._new), ui.button("Open...", self._open),
                     ui.button("Save", self._save), ui.button("Save As...", self._save_as))
        bar.addWidget(check)
        self.play_hint = QtWidgets.QLabel()
        self.play_hint.setObjectName("report")
        self.play_hint.setWordWrap(True)
        self.play_hint.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        bottom = QtWidgets.QVBoxLayout()
        bottom.addLayout(bar)
        bottom.addWidget(self.play_hint)
        return bottom

    def _run_buttons(self) -> QtWidgets.QVBoxLayout:
        """What is done to the session's lines: add, repeat, remove, reorder, skip."""
        add = QtWidgets.QPushButton("Add run")
        add.setMenu(QtWidgets.QMenu(add))
        for text, slot in (("existing config (.json)...", self._add_existing),
                           ("new config...", self._add_new),
                           ("external script...", self._add_command)):
            add.menu().addAction(text).triggered.connect(lambda _checked=False, s=slot: s())
        self.skip_box = QtWidgets.QCheckBox("skip")
        self.skip_box.setToolTip("Written commented out in the session script: not played.")
        self.skip_box.toggled.connect(self._toggle_skip)
        start = ui.button("Start here", self._start_here)
        start.setToolTip("Skip every run before this one and none after: how a stopped "
                         "session is resumed. The runs after it keep the numbers they have "
                         "here. To resume into the session that stopped, give its number to "
                         "the script (sh ses1.sh 003) or set it in the session field above.")
        rows = QtWidgets.QVBoxLayout()
        repeat = ui.button("Repeat", self._duplicate_step)
        repeat.setToolTip("The same line again. A config played twice is two runs of its task "
                          "(run-001, run-002), each with its own episode seeds.")
        edit = ui.button("Open this run >", lambda: self.tabs.setCurrentIndex(1))
        edit.setToolTip("See and change the selected run on the Run tab: how it is played, its "
                        "phases and each game's keys.")
        rows.addLayout(ui.row(add, repeat, ui.button("Remove", self._remove_step), edit))
        rows.addLayout(ui.row(ui.button("Up", lambda: self._move_step(-1)),
                            ui.button("Down", lambda: self._move_step(1)), self.skip_box, start))
        return rows

    def _session_design(self) -> QtWidgets.QTabWidget:
        """The session's lines, as a list or as the script itself."""
        listing = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(listing)
        self.session_pick = self._file_pick("sh", "the session scripts in configs/ (File > Open "
                                                  "for one elsewhere)", self._pick_session)
        layout.addLayout(ui.row(QtWidgets.QLabel("Session"), self.session_pick, stretch=False))
        layout.addWidget(self._session_head())
        runs = QtWidgets.QLabel("Runs, in the order they play:")
        runs.setObjectName("hint")
        layout.addWidget(runs)
        self.run_list = QtWidgets.QListWidget()
        self.run_list.setFont(ui.mono())
        self.run_list.currentRowChanged.connect(self._select_step)
        self.run_list.itemDoubleClicked.connect(lambda _item: self.tabs.setCurrentIndex(1))
        self.run_list.setToolTip("Double-click a run to open it on the Run tab.")
        layout.addWidget(self.run_list, 1)
        layout.addLayout(self._run_buttons())
        self.script_text = QtWidgets.QPlainTextEdit()
        self.script_text.setFont(ui.mono())
        self.script_text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        script = self._text_page(self.script_text, (
            "The session script, one line per run. Edit it as in any text editor; it is read "
            "back when you leave this view or save. SUBJECT= and RIG= are fixed here or, as "
            "${VAR:?...}, required when the script is started; SES= picks the session once for "
            "all the runs. A commented line is a skipped run; a line that is not an fmri-play "
            "run is any command of yours."))
        self.session_views = QtWidgets.QTabWidget()
        self.session_views.addTab(listing, "List")
        self.session_views.addTab(script, "Script (.sh)")
        self.session_views.currentChanged.connect(self._on_session_view)
        return self.session_views

    def _run_design(self) -> QtWidgets.QWidget:
        """The selected line: a run's phases or its JSON, or an external command."""
        # One page per kind of line: a run's phases, or an external command.
        self.run_pages = QtWidgets.QStackedWidget()
        self.run_pages.addWidget(self._phase_columns())
        self.run_pages.addWidget(self._command_page())
        self.json_text = QtWidgets.QPlainTextEdit()
        self.json_text.setFont(ui.mono())
        self.json_text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        json_page = self._text_page(self.json_text, (
            "The run's config file, as it will be saved. Edit it as in any text editor; it "
            "is read back when you leave this view, pick another run, or save."))
        self.run_views = QtWidgets.QTabWidget()
        self.run_views.addTab(self.run_pages, "Phases")
        self.run_views.addTab(json_page, "JSON")
        self.run_views.currentChanged.connect(self._on_run_view)
        self.run_title = QtWidgets.QLabel()
        self.run_title.setObjectName("hint")
        self.run_pick = self._file_pick("json", "the run configs in configs/curricula/: open one, "
                                                "or in a session add it after the selected line",
                                        self._pick_run)
        design = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(design)
        layout.setContentsMargins(0, 0, 0, 0)
        self.run_switch = QtWidgets.QComboBox()
        self.run_switch.setFont(ui.mono())
        self.run_switch.setMinimumWidth(320)
        self.run_switch.setToolTip("The session's run shown here.")
        self.run_switch.activated.connect(self._switch_run)
        layout.addLayout(ui.row(QtWidgets.QLabel("Run"), self.run_switch, self.run_title))
        layout.addLayout(ui.row(QtWidgets.QLabel("Open or add a curriculum:"), self.run_pick))
        self.flags_form = ui.Form(RUN_FLAG_FIELDS)
        self.flags_form.changed.connect(self._flags_edited)
        self.flags_box = ui.group("How this run is played", self.flags_form)
        layout.addWidget(self.flags_box)
        layout.addWidget(self.run_views, 1)
        return design

    def _phase_columns(self) -> QtWidgets.QWidget:
        """The phase list, its buttons, and the selected phase's form."""
        self.phase_list = QtWidgets.QListWidget()
        self.phase_list.setFont(ui.mono())
        self.phase_list.setFixedWidth(230)
        self.phase_list.currentRowChanged.connect(self._select_phase)
        add = QtWidgets.QPushButton("Add")
        add.setMenu(QtWidgets.QMenu(add))
        for kind in curriculum.PHASE_TYPES:
            add.menu().addAction(kind).triggered.connect(
                lambda _checked=False, k=kind: self._add_phase(k))
        buttons = QtWidgets.QVBoxLayout()
        for b in (add, ui.button("Duplicate", self._duplicate_phase),
                  ui.button("Remove", self._remove_phase),
                  ui.button("Move up", lambda: self._move_phase(-1)),
                  ui.button("Move down", lambda: self._move_phase(1))):
            buttons.addWidget(b)
        buttons.addStretch(1)
        self.phase_pane = QtWidgets.QGroupBox("phase")
        self.phase_pane.setLayout(QtWidgets.QVBoxLayout())
        self.phase_body: QtWidgets.QWidget | None = None
        phases = QtWidgets.QWidget()
        columns = QtWidgets.QHBoxLayout(phases)
        columns.addWidget(self.phase_list)
        columns.addLayout(buttons)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setWidget(self.phase_pane)
        columns.addWidget(scroll, 1)
        return phases

    @staticmethod
    def _text_page(editor: QtWidgets.QPlainTextEdit, hint: str) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        label = QtWidgets.QLabel(hint)
        label.setObjectName("hint")
        label.setWordWrap(True)
        layout.addWidget(label)
        layout.addWidget(editor, 1)
        return page

    def _command_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QGroupBox("external command")
        layout = QtWidgets.QVBoxLayout(page)
        self.command = QtWidgets.QLineEdit()
        self.command.setFont(ui.mono())
        self.command.setPlaceholderText("./scripts/localizer.sh --arg   (any shell command)")
        layout.addLayout(ui.row(self.command, ui.button("Browse...", self._pick_command),
                              stretch=False))
        hint = QtWidgets.QLabel(
            "Not played by fmri-play: this is one line of the session script, as typed, run in "
            "its turn from the folder the script is started in. It may use \"$SES\", the "
            "session's number. A non-zero exit status stops the session. The editor "
            "does not check that the command exists.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return page

    @staticmethod
    def _file_pick(ext: str, tip: str, slot: Callable[[str], Any]) -> QtWidgets.QComboBox:
        """A drop-down of the ``configs/`` files of one kind; choosing one calls ``slot``."""
        pick = QtWidgets.QComboBox()
        pick.setFont(ui.mono())
        pick.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        pick.setToolTip(tip)
        pick.addItem("pick from configs/...", None)
        for path in listed_files(ext):
            pick.addItem(path, path)
        pick.activated.connect(lambda index: slot(pick.itemData(index)))
        return pick

    def _pick_session(self, path: str | None) -> None:
        if path is None or not self._discard_ok():
            self._sync_session_pick()
            return
        try:
            self.open_session(path)
        except (OSError, ValueError) as exc:
            self._error(f"cannot open {path}: {exc}")

    def _pick_run(self, path: str | None) -> None:
        """A lone config is replaced, as by File > Open; a session gets the run as a line."""
        self.run_pick.setCurrentIndex(0)  # an action, not a state: the title says what is shown
        if path is None:
            return
        if self._is_session():
            self._add_config(path)
            return
        if not self._discard_ok():
            return
        try:
            self.open_config(curriculum.load_curriculum(path, keep_includes=True), path)
        except (OSError, ValueError) as exc:
            self._error(f"cannot open {path}: {exc}")

    def _sync_session_pick(self) -> None:
        """The session drop-down shows the open script, or the prompt for a lone config."""
        self.session_pick.blockSignals(True)  # ours, not a click
        self.session_pick.setCurrentIndex(max(0, self.session_pick.findData(self.session_path)))
        self.session_pick.blockSignals(False)

    def _session_head(self) -> QtWidgets.QWidget:
        """Who the session is for and which rig it plays on: fixed in the script, or blank to
        be given each time it is started."""
        box = QtWidgets.QGroupBox("Who and where (every run of the session)")
        layout = QtWidgets.QVBoxLayout(box)
        hint = QtWidgets.QLabel("Fix them here, or leave one blank to give it each time the "
                                "session is started: SUBJECT=sub-01 RIG=<rig file> sh <script>.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.session_form = ui.Form(SESSION_FIELDS)
        self.session_form.changed.connect(self._head_edited)
        self.rig_pick = rig_config.RigPick()
        self.rig_pick.setToolTip("The rig every run plays on: its window, controller, audio, "
                                 "triggers and data folder. Blank: given when the session is "
                                 "started (RIG=<rig file> sh <script>).")
        self.rig_pick.changed.connect(lambda _path: self._head_edited())
        self.session_form.layout().addRow("rig", self.rig_pick)
        layout.addWidget(self.session_form)
        edit = ui.button("Edit rig...", self._edit_rig)
        edit.setToolTip("Open this rig file in rig-config: see or change it, or make it.")
        self.rig_summary = QtWidgets.QLabel()
        self.rig_summary.setObjectName("hint")
        self.rig_summary.setWordWrap(True)
        layout.addLayout(ui.row(edit))
        layout.addWidget(self.rig_summary)
        return box

    def _head(self) -> dict:
        """The session form and the rig, as :class:`~fmri_gym.session_script.Session` fields.

        :raises ValueError: a session number that is not one.
        """
        values = self.session_form.get()
        return {"subject": values.get("subject"), "ses": values.get("ses"),
                "rig": self.rig_pick.path() or None}

    def _set_head(self, session: session_script.Session) -> None:
        self.session_form.set({"subject": session.subject, "ses": session.ses})
        self.rig_pick.set_path(session.rig)

    def _head_edited(self) -> None:
        self._show_rig()
        self._refresh_play_hint()
        if self.seed_label is not None:  # subject and session change a derived seed
            self._refresh_seed()

    def _edit_rig(self) -> None:
        """The rig form on the picked rig file (made if it is missing)."""
        path = self.rig_pick.path()
        if not path:
            self._error("Pick a rig file first, or type the path of a new one "
                        "(configs/<team>/rigs/<rig>.json) to make it.")
            return
        if rig_config.fill_rig(rig_file.draft(path), path) is None:
            return
        self.rig_pick.set_path(path)
        self._show_rig()

    def _show_rig(self) -> None:
        """One line on the picked rig; all of it in the line's tooltip."""
        path = self.rig_pick.path()
        if not path:
            self.rig_summary.setText("No rig fixed: whoever starts the session names it (RIG=...).")
            self.rig_summary.setToolTip("")
            return
        try:
            line = "Plays as: " + rig_file.RigConfig.open(path).status().split(" | ", 1)[1]
        except ValueError as exc:
            line = f"A run would refuse this rig file: {exc}"
        self.rig_summary.setText(line)
        self.rig_summary.setToolTip(rig_config.rig_summary(path))

    def _refresh_play_hint(self) -> None:
        """Say how what is shown is played: nothing is played from the editor."""
        try:
            head = self._head()
        except ValueError:
            return
        subject, rig = head["subject"] or "<sub-XX>", head["rig"] or "<rig file>"
        if self._is_session():
            given = " ".join(f"{var}={value}" for var, value, fixed in (
                ("SUBJECT", subject, head["subject"]), ("RIG", rig, head["rig"])) if not fixed)
            where = self.session_path or "<the script, once saved>"
            text = f"To play this session, from the repo root:  {given + ' ' if given else ''}sh {where}"
        else:
            command = session_script.play_command(
                self.steps[0]["config"], subject,
                f"{head['ses']:03d}" if head["ses"] else "<session number>", 1, rig,
                self.steps[0].get("flags", {}))
            text = "To play this run, from the repo root:  " + " ".join(command)
        self.play_hint.setText(text)

    # -- model <-> widgets -------------------------------------------------

    def open_config(self, config: dict, path: str | None) -> None:
        """Edit one config: a session of one run, with no script. ``path=None``: a new one."""
        self.new = set()
        if path is None:
            self.configs = {}
            path = self._provisional_path(config)
            self.new = {path}
        self.session_path = None
        self.steps = [{"config": path, "skip": False}]
        self.configs, self.opened = {path: config}, set()
        self._show_step(0)
        self._refresh_views()
        self._clean = self._state()

    def open_session(self, path: str) -> None:
        """Edit a session script and the configs its runs name.

        :raises OSError: if the script or one of its configs cannot be read.
        :raises ValueError: if a config does not load, or the script has no line to play.
        """
        with open(path) as f:
            session = session_script.read_session(f.read())
        steps = session.steps
        if not steps:
            raise ValueError(f"{path}: no run in it")
        # Every config now, so a missing one is named here and not at some later click.
        configs = {s["config"]: curriculum.load_curriculum(s["config"], keep_includes=True)
                   for s in steps if "config" in s}
        self.session_path, self.steps = path, steps
        self.configs, self.opened, self.new = configs, set(), set()
        self._set_head(session)
        self._show_step(0)
        self._refresh_views()
        self._clean = self._state()

    def _state(self) -> str:
        """Everything Save would write, as text: what tells an edit from none."""
        try:
            self._store_session()
            script = self._session_text()
        except ValueError as exc:
            return f"not readable: {exc}"
        return json.dumps([script, self.configs], sort_keys=True, default=str)

    def _discard_ok(self) -> bool:
        """Whether what is shown may be replaced: nothing unsaved, or the user says so."""
        if self._state() == self._clean:
            return True
        buttons = QtWidgets.QMessageBox.StandardButton
        answer = QtWidgets.QMessageBox.question(
            self, "fmri-gym session editor", "There are changes that are not saved. Save them?",
            buttons.Save | buttons.Discard | buttons.Cancel)
        if answer == buttons.Save:
            return self._save()
        return answer == buttons.Discard

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 -- Qt's name
        if self._discard_ok():
            event.accept()
        else:
            event.ignore()

    def _show_config(self, config: dict) -> None:
        """Load the selected run's config into the phase list and the config tabs."""
        self.notes = {k: v for k, v in config.items() if k not in curriculum.SECTIONS}
        self.phases = copy.deepcopy(config["curriculum"])
        self.edit_index = None
        # Opened on its game: a run plays one, and that is what is set up most.
        games = [i for i, p in enumerate(self.phases) if p["type"] == "game"]
        self._refresh_phase_list(games[0] if games else 0 if self.phases else None)
        self._show_json(config)

    def collect(self) -> dict:
        """Read the config tabs back into one config dict (the selected run's).

        :raises ValueError: naming the offending field.
        """
        self._commit()
        return {**copy.deepcopy(self.notes), "curriculum": copy.deepcopy(self.phases)}

    def _set_status(self, text: str = "") -> None:
        step = self.steps[self.step_index]
        where = self._where(step)
        here = f"{self.session_path}  >  {where}" if self.session_path else where
        self.statusBar().showMessage(text or here)

    def _commit(self) -> None:
        """Push the phase form and the key table into ``self.phases``.

        :raises ValueError: naming the offending field.
        """
        self._commit_phase()

    def _store_current(self) -> None:
        """Put the selected line back into the model: its forms or its JSON, or the command.

        :raises ValueError: naming the offending field, or why the JSON does not load.
        """
        step = self.steps[self.step_index]
        if "command" in step:
            step["command"] = self.command.text().strip()
            return
        self._store_flags(step)
        if self._run_view == _JSON and self.json_text.toPlainText() != self._json_shown:
            self._show_config(self._read_json())
        self.configs[step["config"]] = config = self.collect()
        if self._run_view == _JSON:
            self._show_json(config)  # the view shows what will be saved

    def _store_flags(self, step: dict) -> None:
        """The run's flags form into its line; only the flags that apply are kept.

        :raises ValueError: a number that is not one, named.
        """
        shown = self._flag_rows()
        flags = {k: v for k, v in self.flags_form.get().items() if k in shown}
        if flags:
            step["flags"] = flags
        else:
            step.pop("flags", None)

    def _flag_rows(self) -> list[str]:
        """Closing fixation and looping only mean something for a run with a length."""
        timed = bool(self.flags_form.widgets["duration"].text().strip())
        return ["duration", "resume"] + ["end_fixation", "loop"] * timed

    def _flags_edited(self) -> None:
        self.flags_form.show_rows(self._flag_rows())
        step = self.steps[self.step_index] if self.steps else {}
        if "config" in step:
            try:
                self._store_flags(step)
            except ValueError:
                return  # a number half typed: stored once it is one
            self._refresh_play_hint()

    def _read_json(self) -> dict:
        """:raises ValueError: if the JSON view does not hold a config (the reason says why)."""
        try:
            return curriculum.check_shape(json.loads(self.json_text.toPlainText()), "JSON")
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON: not valid JSON ({exc})") from exc

    def _show_json(self, config: dict) -> None:
        self._json_shown = json.dumps(config, indent=2)
        self.json_text.setPlainText(self._json_shown)

    def _store_session(self) -> None:
        """:meth:`_store_current`, then the script if it was edited: the whole editor, stored.

        :raises ValueError: as :meth:`_store_current`, or why the script does not load.
        """
        self._store_current()
        if self._session_view == _SCRIPT and self.script_text.toPlainText() != self._script_shown:
            self._apply_script(self.script_text.toPlainText())

    def _apply_script(self, text: str) -> None:
        """Make the session what the script says: its lines, its launch flags, its configs.

        :raises ValueError: if it has no line, or a config it names does not load.
        """
        session = session_script.read_session(text)
        steps = session.steps
        if not steps:
            raise ValueError("script: no line to play")
        try:  # configs shown here keep their unsaved edits; the others are read now
            new = {s["config"]: curriculum.load_curriculum(s["config"], keep_includes=True)
                   for s in steps if "config" in s and s["config"] not in self.configs}
        except OSError as exc:
            raise ValueError(f"script: {exc}") from exc
        self.configs.update(new)
        self.steps = steps
        self._set_head(session)
        self._show_step(min(self.step_index, len(steps) - 1))
        self._refresh_script()

    def _refresh_script(self) -> None:
        """:raises ValueError: if the script cannot be written yet (see :meth:`_session_text`)."""
        self._script_shown = self._session_text()
        self.script_text.setPlainText(self._script_shown)

    def _refresh_views(self) -> None:
        """Show the model again in the script view after it changed under it -- or the list,
        with the reason, while the script cannot be written."""
        if self._session_view != _SCRIPT:
            return
        try:
            self._refresh_script()
        except ValueError as exc:
            self._set_view(self.session_views, _LIST)
            self._session_view = _LIST
            self._set_status(f"script view closed: {exc}")

    @staticmethod
    def _set_view(views: QtWidgets.QTabWidget, index: int) -> None:
        views.blockSignals(True)  # ours, not a click
        views.setCurrentIndex(index)
        views.blockSignals(False)

    def _on_session_view(self, index: int) -> None:
        """Leaving a view stores it; the script is written when shown. Refused if either fails."""
        try:
            self._store_session()
            if index == _SCRIPT:
                self._refresh_script()
        except ValueError as exc:
            self._set_view(self.session_views, self._session_view)
            self._error(str(exc))
            return
        self._session_view = index

    def _on_run_view(self, index: int) -> None:
        """Leaving the JSON stores it into the forms, and leaving the forms into the JSON:
        refused, with the reason, if the view being left does not load."""
        try:
            self._store_current()
        except ValueError as exc:
            self._set_view(self.run_views, self._run_view)
            self._error(str(exc))
            return
        self._run_view = index
        step = self.steps[self.step_index]
        if index == _JSON:
            self._show_json(self.configs[step["config"]])

    def _commit_all(self) -> bool:
        """:meth:`_store_current` for a button or tab handler: ``False`` (after a dialog) on a
        bad field."""
        try:
            self._store_current()
        except ValueError as exc:
            self._error(str(exc))
            return False
        return True

    def _commit_session(self) -> bool:
        """:meth:`_store_session` for Save, Run, Check and the top tabs: ``False`` (after a
        dialog) if something shown does not load."""
        try:
            self._store_session()
        except ValueError as exc:
            self._error(str(exc))
            return False
        return True

    def _on_tab_change(self) -> None:
        if not self._commit_session():
            return
        self._refresh_views()
        step = self.steps[self.step_index]
        if "command" in step:
            return
        self._show_json(self.configs[step["config"]])
        if self.seed_label is not None:
            self._refresh_seed()

    # -- the session's lines -------------------------------------------------

    def _is_session(self) -> bool:
        """Several lines, a script on disk, or a lone command: saved as (and run from) a script."""
        return (self.session_path is not None or len(self.steps) > 1
                or "command" in self.steps[0])

    def _where(self, step: dict) -> str:
        """A line, in words: its config's path (and whether it is new), or that it is a command."""
        if "command" in step:
            return "an external command"
        return step["config"] + "  (new, not saved yet)" * (step["config"] in self.new)

    def _step_labels(self) -> list[str]:
        labels = []
        for i, step in enumerate(self.steps, start=1):
            what = (f"$ {step['command']}" if "command" in step
                    else _short(step["config"]) + "  (new)" * (step["config"] in self.new))
            labels.append(f"{i}  {'[skip] ' if step['skip'] else ''}{what}")
        return labels

    def _switch_run(self, index: int) -> None:
        if index != self.step_index and self._commit_all():
            self._show_step(index)
        else:
            self._refresh_run_list()

    def _refresh_run_list(self) -> None:
        self.run_list.blockSignals(True)  # the selection below is ours, not the user's
        self.run_list.clear()
        self.run_list.addItems(self._step_labels())
        self.run_list.setCurrentRow(self.step_index)
        self.run_list.blockSignals(False)
        self.run_switch.blockSignals(True)
        self.run_switch.clear()
        self.run_switch.addItems(self._step_labels())
        self.run_switch.setCurrentIndex(self.step_index)
        self.run_switch.blockSignals(False)

    def _show_step(self, index: int) -> None:
        """Point the editors at line ``index`` (nothing is stored first; see _select_step)."""
        self.step_index = index
        step = self.steps[index]
        is_command = "command" in step
        self.run_pages.setCurrentIndex(1 if is_command else 0)
        self.run_views.setTabEnabled(_JSON, not is_command)
        if is_command and self._run_view == _JSON:
            self._set_view(self.run_views, _PHASES)
            self._run_view = _PHASES
        self.run_title.setText(f"line {index + 1} of the session: {self._where(step)}")
        self._sync_session_pick()
        self.command.setText(step.get("command", ""))
        self.skip_box.blockSignals(True)  # ours, not a click
        self.skip_box.setChecked(step["skip"])
        self.skip_box.blockSignals(False)
        self.flags_box.setVisible(not is_command)
        self.flags_form.set(step.get("flags", {}))
        if not is_command:
            self._show_config(self.configs[step["config"]])
            self.opened.add(step["config"])
        self._refresh_run_list()
        self._refresh_play_hint()
        self._set_status()

    def _select_step(self, index: int) -> None:
        if index < 0 or index == self.step_index or not self._commit_all():
            self._refresh_run_list()
            return
        self._show_step(index)

    def _insert_step(self, step: dict) -> None:
        """Add a line after the selected one."""
        if not self._commit_all():
            return
        self.steps.insert(self.step_index + 1, step)
        self._show_step(self.step_index + 1)

    def _add_existing(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Add a run", "configs",
                                                        "JSON (*.json)")
        if path:
            self._add_config(os.path.relpath(path))

    def _add_config(self, path: str) -> None:
        if path not in self.configs:
            try:
                self.configs[path] = curriculum.load_curriculum(path, keep_includes=True)
            except (OSError, ValueError) as exc:
                self._error(f"cannot add {path}: {exc}")
                return
        self._insert_step({"config": path, "skip": False})

    def _add_new(self) -> None:
        """A new config, named after its first game's backend until Save asks where it goes."""
        config = curriculum.new_curriculum()
        path = self._provisional_path(config)
        self.configs[path] = config
        self.new.add(path)
        self._insert_step({"config": path, "skip": False})

    def _provisional_path(self, config: dict, own: str | None = None) -> str:
        """``<backend>.json`` in the curricula folder, a name not yet taken there or in this
        session (``own`` excepted, so it can keep its name)."""
        backend = first_backend(config)
        folder = os.path.relpath(CURRICULA_DIR)
        taken = {os.path.splitext(os.path.basename(p))[0] for p in self.configs if p != own}
        if os.path.isdir(folder):
            taken |= {os.path.splitext(name)[0] for name in os.listdir(folder)}
        return os.path.join(folder, suggest_name(backend, taken) + ".json")

    def _ask_config_path(self, suggested: str) -> str:
        """Where to save a config (``""`` if cancelled); the dialog starts at ``suggested``."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save the config", suggested,
                                                        "JSON (*.json)")
        if not path:
            return ""
        return os.path.relpath(path if path.endswith(".json") else path + ".json")

    def _add_command(self) -> None:
        command = self._ask_command()
        if command:
            self._insert_step({"command": command, "skip": False})

    def _ask_command(self) -> str:
        """A script picked from disk, as the shell command that runs it (``""`` if cancelled)."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "External script")
        if not path:
            return ""
        rel = os.path.relpath(path)
        # "./" because sh looks a bare file name up on PATH, not in the current folder.
        return shlex.quote(rel if rel.startswith("..") else os.path.join(".", rel))

    def _pick_command(self) -> None:
        command = self._ask_command()
        if command:
            self.command.setText(command)

    def _duplicate_step(self) -> None:
        """The same line again: a config played twice is two runs of its task (run-001, run-002)."""
        if self._commit_all():
            self._insert_step(copy.deepcopy(self.steps[self.step_index]))

    def _remove_step(self) -> None:
        if len(self.steps) < 2 or not self._commit_all():
            return
        del self.steps[self.step_index]
        self._show_step(min(self.step_index, len(self.steps) - 1))

    def _move_step(self, by: int) -> None:
        i = self.step_index
        if not 0 <= i + by < len(self.steps) or not self._commit_all():
            return
        self.steps[i], self.steps[i + by] = self.steps[i + by], self.steps[i]
        self._show_step(i + by)

    def _toggle_skip(self, skip: bool) -> None:
        self.steps[self.step_index]["skip"] = skip
        self._refresh_run_list()

    def _start_here(self) -> None:
        """Skip every line before the selected one and none from it on."""
        if not self._commit_all():
            return
        for i, step in enumerate(self.steps):
            step["skip"] = i < self.step_index
        self._show_step(self.step_index)

    # -- curriculum (phases of the current run) ------------------------------

    def _refresh_phase_list(self, select: int | None) -> None:
        self.phase_list.blockSignals(True)  # the selection below is ours, not the user's
        self.phase_list.clear()
        self.phase_list.addItems([phase_label(i, p) for i, p in enumerate(self.phases)])
        self.phase_list.setCurrentRow(-1 if select is None else select)
        self.phase_list.blockSignals(False)
        self._show_phase(select)

    def _select_phase(self, row: int) -> None:
        if row < 0 or row == self.edit_index:
            return
        if self._commit_all():
            self._show_phase(row)
            return
        self.phase_list.blockSignals(True)
        self.phase_list.setCurrentRow(-1 if self.edit_index is None else self.edit_index)
        self.phase_list.blockSignals(False)

    def _show_phase(self, index: int | None) -> None:
        if self.phase_body is not None:
            self.phase_body.setParent(None)
            self.phase_body.deleteLater()
        self.edit_index = index
        self.phase_form = self.extra_text = self.phase_body = self.seed_label = None
        self.key_table = self.check_text = None
        if index is None:
            self.phase_pane.setTitle("phase")
            return
        phase = self.phases[index]
        kind = phase["type"]
        self.phase_pane.setTitle(f"phase {index:02d}: {kind}")
        fields = PHASE_FIELDS[kind]
        form, extra = split_phase(phase, fields)
        self.phase_form = ui.Form(fields)
        self.phase_form.set(form)
        self.phase_form.changed.connect(self._relabel_phase)  # after set: the edits, not the load
        self.phase_form.changed.connect(self._show_phase_rows)
        self._show_phase_rows()
        self.phase_body = QtWidgets.QWidget()
        body = QtWidgets.QVBoxLayout(self.phase_body)
        body.setContentsMargins(0, 0, 0, 0)
        body.addWidget(self.phase_form)
        if kind == "game":
            self._add_curated_row(body)
            self._add_seed_row(body)
            self._add_game_extras(body, phase, extra)
        body.addStretch(1)
        self.phase_pane.layout().addWidget(self.phase_body)

    def _phase_rows(self) -> list[str]:
        """The fields of the selected phase that apply, given what it says so far."""
        kind = self.phases[self.edit_index]["type"]
        if kind not in PHASE_ROWS:
            return [f.key for f in self.phase_form.fields]
        values = {f.key: (w.currentText() if isinstance(w, QtWidgets.QComboBox)
                          else w.text() if isinstance(w, QtWidgets.QLineEdit) else None)
                  for f in self.phase_form.fields
                  for w in (self.phase_form.widgets[f.key],)}
        return PHASE_ROWS[kind]({k: v.strip() if isinstance(v, str) else v
                                 for k, v in values.items()})

    def _show_phase_rows(self) -> None:
        if self.phase_form is not None:
            self.phase_form.show_rows(self._phase_rows())

    def _relabel_phase(self) -> None:
        """The phase's line in the list, and a new config's name, follow the form as it is typed."""
        try:
            live = {"type": self.phases[self.edit_index]["type"], **self.phase_form.get()}
        except ValueError:
            return  # a number half typed: the label waits for the rest
        self.phase_list.item(self.edit_index).setText(phase_label(self.edit_index, live))
        self._rename_new_config(live)

    def _rename_new_config(self, live: dict) -> None:
        """A new config is named after its first game's backend: when that changes, so does the
        name (``ale.json`` -> ``gym.json``). A config on disk keeps its file."""
        path = self.steps[self.step_index]["config"]
        games = [i for i, p in enumerate(self.phases) if p["type"] == "game"]
        if path not in self.new or not games or games[0] != self.edit_index:
            return
        if self._session_view == _SCRIPT and self.script_text.toPlainText() != self._script_shown:
            return  # a script being edited names the old one: renaming would strand its lines
        phases = [*self.phases[:self.edit_index], live, *self.phases[self.edit_index + 1:]]
        wanted = self._provisional_path({"curriculum": phases}, own=path)
        if wanted == path:
            return
        self._rename_config(path, wanted)
        where = self._where(self.steps[self.step_index])
        self.run_title.setText(f"line {self.step_index + 1} of the session: {where}")
        self._refresh_run_list()
        self._refresh_views()
        self._set_status()

    def _add_curated_row(self, body: QtWidgets.QVBoxLayout) -> None:
        """Under a game phase's form: where its game is already set up, and a button to copy that.

        The ``game`` list follows ``backend`` (see :func:`fmri_gym.gui.curated_games`).
        """
        self.curated_label = QtWidgets.QLabel()
        self.curated_label.setObjectName("hint")
        self.curated_button = ui.button("Fill this phase from that config", self._use_curated)
        body.addLayout(ui.row(self.curated_button, self.curated_label))
        widgets = self.phase_form.widgets
        widgets["backend"].currentTextChanged.connect(lambda _text: self._refresh_games())
        widgets["game"].currentTextChanged.connect(lambda _text: self._refresh_curated())
        self._refresh_games()

    def _curated_phase(self) -> tuple[str, dict] | None:
        """``(config path, phase)`` of the form's backend + game, if a config holds it."""
        widgets = self.phase_form.widgets
        games = self.curated.get(widgets["backend"].currentText().strip(), {})
        return games.get(widgets["game"].currentText().strip())

    def _refresh_games(self) -> None:
        backend = self.phase_form.widgets["backend"].currentText().strip()
        self.phase_form.set_choices("game", sorted(self.curated.get(backend, {})))
        self._refresh_curated()

    def _refresh_curated(self) -> None:
        found = self._curated_phase()
        self.curated_button.setEnabled(found is not None)
        self.curated_label.setText(f"set up in {found[0]} (keys, fps, mode, ...)" if found else
                                   "no config here has this game: keys and fps are yours to set")

    def _use_curated(self) -> None:
        """Replace the phase with the one a config already holds for this game."""
        found = self._curated_phase()
        if found is None:
            return
        index = self.edit_index
        self.phases[index] = copy.deepcopy(found[1])
        self.edit_index = None  # the form shows the new phase, keys included
        self._refresh_phase_list(index)
        self._set_status(f"phase {index:02d} filled from {found[0]}")

    def _add_seed_row(self, body: QtWidgets.QVBoxLayout) -> None:
        """Under a game phase's form: the seed it plays with, to read, copy, or pin as it is."""
        self.seed_label = QtWidgets.QLabel()
        self.seed_label.setObjectName("hint")
        self.seed_label.setWordWrap(True)
        self.seed_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.pin_button = ui.button("Pin", self._pin_seed)
        self.pin_button.setToolTip("Write this seed into the seed field: the phase then plays "
                                   "these episodes for every participant and run.")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.pin_button)
        row.addWidget(self.seed_label, 1)
        body.addLayout(row)
        self.phase_form.widgets["seed"].textChanged.connect(lambda _text: self._refresh_seed())
        self._refresh_seed()

    def _refresh_seed(self) -> None:
        """Say which seed the phase plays with: the one typed, or the one a launch would derive."""
        field = self.phase_form.widgets["seed"]
        self._derived_seed = None
        typed = field.text().strip()
        if typed:
            self.pin_button.setEnabled(False)
            self.seed_label.setText(f"seed {typed}, pinned: episodes {typed}, {typed}+1, ... for "
                                    "every participant and run. Clear it to derive one per run.")
            return
        try:
            self._derived_seed, label = self._seed_preview()
        except ValueError as exc:
            self.pin_button.setEnabled(False)
            field.setPlaceholderText("derived per run")
            self.seed_label.setText(f"seed derived per run; to see it: {exc}")
            return
        self.pin_button.setEnabled(True)
        field.setPlaceholderText(f"derived: {self._derived_seed}")
        self.seed_label.setText(f"seed {self._derived_seed}, derived for {label} if launched now "
                                "(another participant, session or run gets another one).")

    def _seed_preview(self) -> tuple[int, str]:
        """The base seed this game phase will get, and the run it is for.

        The design says the run: which line of the session plays this task
        (:func:`fmri_gym.session_script.run_number`), in the session given or, blank,
        the subject's next free session. A re-acquisition of a run replays
        these same episodes -- the attempt is no part of the label.

        :raises ValueError: if the session's subject, its rig or the config's name give no
            run label.
        """
        step = self.steps[self.step_index]
        head = self._head()
        if not head["subject"]:
            raise ValueError("fix the subject above (it is given at launch)")
        subject = bids.subject_label(head["subject"])
        ses = head["ses"] or bids.next_session(self._data_root(head["rig"]), subject)
        label = bids.run_label(subject, ses, bids.task_label(step["config"]),
                               session_script.run_number(self.steps, self.step_index))
        return bids.phase_seed(label, self.edit_index), label

    def _pin_seed(self) -> None:
        if self._derived_seed is not None:
            self.phase_form.write("seed", str(self._derived_seed))

    def _add_game_keys(self, body: QtWidgets.QVBoxLayout, phase: dict) -> None:
        """Under a game phase's form: its keys, which keyboard key does what in it."""
        box = QtWidgets.QGroupBox("Keys")
        layout = QtWidgets.QVBoxLayout(box)
        hint = QtWidgets.QLabel(_GAME_KEYS_HINT)
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.key_table = ui.KeyTable()
        self.key_table.set(phase.get("keys", {}))
        self.key_table.setMinimumHeight(180)
        layout.addWidget(self.key_table)
        buttons = [ui.button("Add key", self.key_table.add_row),
                   ui.button("Press a key...", self._capture_key),
                   ui.button("Remove key", self.key_table.remove_selected),
                   ui.button("Check with the engine", self._check_keys)]
        for b, tip in zip(buttons, (
                "A blank row: type the key (or combo) and the action it sends.",
                "Press the key on the keyboard to add its row, spelled as the games read it.",
                "Remove the selected row.",
                "Start the game's engine with these keys: whether fmri-play would take them, "
                "and the engine's own speed (to set fps).")):
            b.setToolTip(tip)
        layout.addLayout(ui.row(*buttons))
        self.check_text = QtWidgets.QPlainTextEdit()
        self.check_text.setReadOnly(True)
        self.check_text.setFixedHeight(60)
        self.check_text.setFont(ui.mono())
        self.check_text.setPlaceholderText("Check with the engine says here whether fmri-play "
                                           "would take these keys, and how fast the game runs.")
        layout.addWidget(self.check_text)
        body.addWidget(box)

    def _add_game_extras(self, body: QtWidgets.QVBoxLayout, phase: dict, extra: dict) -> None:
        """Under a game phase's form: its keys, and the other fields as JSON."""
        self._add_game_keys(body, phase)
        body.addWidget(QtWidgets.QLabel("extra (JSON)"))
        self.extra_text = QtWidgets.QPlainTextEdit(json.dumps(extra, indent=1) if extra else "{}")
        self.extra_text.setFont(ui.mono())
        self.extra_text.setFixedHeight(120)
        self.extra_text.setToolTip("Backend-specific fields as JSON, e.g. retro \"state\"/"
                                   "\"scenario\", vgdl \"level\", vizdoom \"env_kwargs\", "
                                   "ale \"save_pixels\".")
        body.addWidget(self.extra_text)

    def _commit_phase(self) -> None:
        if self.edit_index is None or self.phase_form is None:
            return
        old = self.phases[self.edit_index]
        shown = self._phase_rows()
        new = {"type": old["type"],
               **{k: v for k, v in self.phase_form.get().items() if k in shown}}
        keys = self.key_table.get() if self.key_table is not None else old.get("keys")
        if keys:
            new["keys"] = keys
        if self.extra_text is None and curriculum.INCLUDE_KEY in old:  # else it is in extra
            new[curriculum.INCLUDE_KEY] = old[curriculum.INCLUDE_KEY]
        if self.extra_text is not None:
            try:
                extra = json.loads(self.extra_text.toPlainText() or "{}")
            except ValueError as exc:
                raise ValueError(f"extra (JSON): not valid JSON ({exc})") from exc
            if not isinstance(extra, dict):
                raise ValueError("extra (JSON): must be a JSON object")
            new.update(extra)
        # The file's key order first, so a saved file diffs only where it was edited.
        new = {k: new[k] for k in old if k in new} | new
        self.phases[self.edit_index] = new
        self.phase_list.item(self.edit_index).setText(phase_label(self.edit_index, new))

    def _add_phase(self, kind: str) -> None:
        templates = {"fixation": {"type": "fixation", "duration": 2.0},
                     "message": {"type": "message", "text": "", "duration": 2.0},
                     "survey": {"type": "survey", "n_points": 7, "questions": []},
                     # No fps: Check names it, as it does the empty game id -- the rate
                     # belongs to the game about to be picked, not to this template.
                     "game": {"type": "game", "backend": "gym", "game": "", "mode": "duration",
                              "duration": 30.0},
                     # The rig check's: the quick check's values; the rig says what each tests.
                     "check_display": {"type": "check_display", "n": 60},
                     "check_frames": {"type": "check_frames", "rates": [60, 30],
                                      "seconds": 2, "loads": ["none"]},
                     "check_triggers": {"type": "check_triggers", "pulses": 3},
                     "check_controls": {"type": "check_controls", "timeout_s": 10.0},
                     "check_photodiode": {"type": "check_photodiode", "n": 10}}
        self._insert_phase(templates[kind])

    def _duplicate_phase(self) -> None:
        if self.edit_index is not None and self._commit_all():
            self._insert_phase(copy.deepcopy(self.phases[self.edit_index]))

    def _insert_phase(self, phase: dict) -> None:
        if not self._commit_all():
            return
        at = len(self.phases) if self.edit_index is None else self.edit_index + 1
        self.phases.insert(at, phase)
        self._refresh_phase_list(at)

    def _remove_phase(self) -> None:
        if self.edit_index is None or not self._commit_all():
            return
        at = self.edit_index
        del self.phases[at]
        self.edit_index = None
        self._refresh_phase_list(min(at, len(self.phases) - 1) if self.phases else None)

    def _move_phase(self, step: int) -> None:
        i = self.edit_index
        if i is None or not 0 <= i + step < len(self.phases) or not self._commit_all():
            return
        self.phases[i], self.phases[i + step] = self.phases[i + step], self.phases[i]
        self.edit_index = None
        self._refresh_phase_list(i + step)

    def _capture_key(self) -> None:
        dialog = ui.KeyCapture(self)
        dialog.exec()
        if dialog.name is not None and self.key_table is not None:
            self.key_table.add_row(dialog.name, "")
        elif dialog.refused is not None:
            self._error(f"{dialog.refused!r} is not a key the games can read")

    def _check_keys(self) -> None:
        """Build the phase's env with the table's keys: what fmri-play would refuse, and its rate."""
        if self.key_table is None or self.edit_index is None:
            return
        try:
            phase = {**self.phases[self.edit_index], "keys": self.key_table.get()}
        except ValueError as exc:
            self._error(str(exc))
            return
        problems = curriculum.validate_curriculum({"curriculum": [phase]})
        if problems:
            self.check_text.setPlainText("\n".join(problems))
            return
        self.check_text.setPlainText(f"loading {phase.get('game')}...")
        QtWidgets.QApplication.processEvents()
        try:
            native = _engine_rate(phase)
        except Exception as exc:  # noqa: BLE001 -- any engine error: show it, keep the editor alive
            self.check_text.setPlainText(f"the engine refuses this phase: {exc}")
            return
        rate = (f"{native:g} steps/s is this engine's own rate: write it in fps to play the "
                "game at its real speed" if native is not None else
                "no rate of its own: fps is yours to pick (30 suits most)")
        self.check_text.setPlainText(f"{phase.get('backend')} {phase.get('game')}: the keys fit "
                                     f"its action space.\n{rate}")

    # -- file / run ----------------------------------------------------------

    def _new(self) -> None:
        if self._discard_ok():
            self.open_config(curriculum.new_curriculum(), None)

    def _open(self) -> None:
        if not self._discard_ok():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open a config or a session", "configs",
            "Configs and sessions (*.json *.sh);;All files (*)")
        if not path:
            return
        path = os.path.relpath(path)
        try:
            if path.endswith(".sh"):
                self.open_session(path)
            else:
                self.open_config(curriculum.load_curriculum(path, keep_includes=True), path)
        except (OSError, ValueError) as exc:
            self._error(f"cannot open {path}: {exc}")

    def _rename_config(self, old: str, new: str) -> None:
        """Point every line of ``old`` at ``new``; the config itself is unchanged.

        :raises ValueError: if ``new`` is already another of this session's configs.
        """
        if new != old and new in self.configs:
            raise ValueError(f"{new} is already a run of this session: pick another name")
        self.configs[new] = self.configs.pop(old)
        for step in self.steps:
            if step.get("config") == old:
                step["config"] = new
        for marks in (self.opened, self.new):
            if old in marks:
                marks.discard(old)
                marks.add(new)

    def _place_new_configs(self) -> bool:
        """Ask where each new config goes (its name so far suggested); ``False`` if cancelled."""
        news = list(dict.fromkeys(s["config"] for s in self.steps
                                  if "config" in s and s["config"] in self.new))
        for path in news:
            chosen = self._ask_config_path(path)
            if not chosen:
                return False
            try:
                self._rename_config(path, chosen)
            except ValueError as exc:
                self._error(str(exc))
                return False
            self.new.discard(chosen)
        return True

    def _save(self) -> bool:
        """Write the configs shown here and, for a session, its script.

        :return: ``False`` if nothing was written (a bad field, a cancelled dialog, an error).
        """
        if not self._commit_session() or not self._confirm_problems() \
                or not self._place_new_configs():
            return False
        try:
            script = self._session_text() if self._is_session() else None
        except ValueError as exc:
            self._error(str(exc))
            return False
        if script is not None and self.session_path is None and not self._ask_session_path():
            return False
        paths = sorted({s["config"] for s in self.steps if "config" in s} & self.opened)
        if not self._confirm_shared_overrides(paths):
            return False
        try:
            for path in paths:
                curriculum.save_curriculum(self.configs[path], path)
            if script is not None:
                with open(self.session_path, "w") as f:
                    f.write(script)
                os.chmod(self.session_path, 0o755)
        except (OSError, ValueError) as exc:
            self._error(str(exc))
            return False
        self._show_step(self.step_index)  # new configs are files now: names and marks change
        self._refresh_views()
        self._refresh_play_hint()
        self._clean = self._state()
        self._set_status("saved " + ", ".join(paths + [self.session_path] * (script is not None)))
        return True

    def _confirm_problems(self) -> bool:
        """Whether to save, when Check finds what would stop a run."""
        problems = self._problems()
        if not problems:
            return True
        answer = QtWidgets.QMessageBox.question(
            self, "fmri-gym session editor",
            "These would stop the session when it is played:\n\n" + "\n".join(
                f"• {p}" for p in problems) + "\n\nSave anyway, to finish it later?")
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    def _confirm_shared_overrides(self, paths: list[str]) -> bool:
        """Whether to save, when phases change what their shared phase in ``common/`` gives."""
        try:
            lines = [f"{path}, {line}" for path in paths
                     for line in curriculum.shared_overrides(self.configs[path])]
        except (OSError, ValueError) as exc:
            self._error(str(exc))
            return False
        if not lines:
            return True
        answer = QtWidgets.QMessageBox.question(
            self, "fmri-gym config",
            "These phases change what their shared phase in common/ gives them:\n\n"
            + "\n".join(lines) + "\n\nThe change is this curriculum's alone: the other curricula "
            "that include the same shared phase keep it as it is. If it should hold for all of "
            "them, edit the file in common/ instead, and look at every phase that includes it."
            "\n\nSave anyway?")
        return answer == QtWidgets.QMessageBox.StandardButton.Yes

    def _session_text(self) -> str:
        """:raises ValueError: a session number that is not one."""
        return session_script.write_session(session_script.Session(steps=self.steps,
                                                                   **self._head()))

    def _ask_session_path(self) -> bool:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save the session script",
                                                        self._sessions_folder(),
                                                        "Session script (*.sh)")
        if not path:
            return False
        self.session_path = os.path.relpath(path if path.endswith(".sh") else path + ".sh")
        return True

    def _sessions_folder(self) -> str:
        """Where a new session is saved first: its rig's team's ``subjects/`` folder, with the
        subject's when it is fixed; else ``configs/``."""
        rig = self.rig_pick.path()
        folder = os.path.dirname(os.path.abspath(rig)) if rig else ""
        if os.path.basename(folder) != rig_file.RIGS_DIRNAME:
            return os.path.relpath(CONFIGS_DIR)
        subjects = os.path.join(os.path.dirname(folder), "subjects")
        subject = self.session_form.widgets["subject"].text().strip()
        return os.path.relpath(os.path.join(subjects, subject) if subject else subjects)

    def _save_as(self) -> bool:
        """A new file for the session script -- or, for a lone config, for the config."""
        if not self._commit_session():
            return False
        if self._is_session():
            return self._ask_session_path() and self._save()
        path = self.steps[0]["config"]
        chosen = self._ask_config_path(path)
        if not chosen:
            return False
        try:
            self._rename_config(path, chosen)
        except ValueError as exc:
            self._error(str(exc))
            return False
        self.new.discard(chosen)  # placed: Save need not ask again
        return self._save()

    def _check(self) -> list[str]:
        try:
            self._store_session()
        except ValueError as exc:
            problems = [str(exc)]
        else:
            problems = self._problems()
        self._set_status("; ".join(problems) if problems else "looks runnable")
        return problems

    def _problems(self) -> list[str]:
        """What would stop the session, line by line (``validate_curriculum`` on each run)."""
        problems = []
        for i, step in enumerate(self.steps, start=1):
            label = f"run {i}: " if self._is_session() else ""
            if "command" in step:
                problems += [f"{label}the external command is empty"] * (not step["command"])
                continue
            problems += [label + p
                         for p in curriculum.validate_curriculum(self.configs[step["config"]])]
            problems += [label + p for p in _flag_problems(step.get("flags", {}))]
        if all(step["skip"] for step in self.steps):
            problems.append("every run is skipped: nothing would play")
        rig = self.rig_pick.path()
        if not rig:
            return problems
        try:
            problems += self._rig_problems(rig_file.RigConfig.open(rig))
        except ValueError as exc:
            problems.append(f"{exc}; Edit rig... fixes it")
        return problems

    def _rig_problems(self, rig: rig_file.RigConfig) -> list[str]:
        """What fmri-play would refuse in each run on the session's rig: a game key no
        button of its controller presses, a game key its scanner types at every volume."""
        problems = []
        for i, step in enumerate(self.steps, start=1):
            if "command" in step:
                continue
            config = self.configs[step["config"]]
            phases = [{k: v for k, v in p.items() if k != curriculum.INCLUDE_KEY}
                      for p in config["curriculum"]]
            try:
                played = Curriculum.from_dict({**config, "curriculum": phases}, step["config"])
            except ValueError:
                continue  # its own problems are listed already
            flags = step.get("flags", {})
            run = RunConfig(curriculum=played, rig=rig, subject="sub-check", ses=1, run=1,
                            duration=flags.get("duration"),
                            end_fixation=flags.get("end_fixation", END_FIXATION_S),
                            resume=flags.get("resume", False), loop=flags.get("loop", False),
                            dummy_trigger=False)
            label = f"run {i}: " if self._is_session() else ""
            problems += [f"{label}on {rig.path}: {p}" for p in run.problems()]
        return problems

    def _data_root(self, rig: str | None) -> str:
        """Where the session's rig files the data.

        :raises ValueError: when the session fixes no rig, or its file is not valid.
        """
        if not rig:
            raise ValueError("fix the rig above (it is given at launch)")
        return rig_file.RigConfig.open(rig).data_root

    def _error(self, text: str) -> None:
        QtWidgets.QMessageBox.critical(self, "fmri-gym session editor", text)


def _short(path: str) -> str:
    """A curriculum as the run lists show it: its path under ``configs/curricula/``."""
    inside = os.path.relpath(os.path.abspath(path), CURRICULA_DIR)
    return path if inside.startswith("..") else inside


def _flag_problems(flags: dict) -> list[str]:
    """What fmri-play refuses in a run's flags before it starts."""
    end = flags.get("end_fixation", END_FIXATION_S)
    out = [f"closing fixation {end:g} s: expected 0 or more seconds"] * (end < 0)
    if flags.get("duration") is not None and flags["duration"] <= end:
        out.append(f"run length {flags['duration']:g} s: it must be longer than its closing "
                   f"fixation ({end:g} s)")
    if flags.get("loop") and flags.get("duration") is None:
        out.append("start again when it ends: give the run a length")
    return out


def _engine_rate(phase: dict) -> float | None:
    """Build the phase's adapter (which checks its keys against the env) and read its own rate.

    :return: the native fps; ``None`` for an engine with no clock of its own.
    :raises Exception: whatever the engine or the :mod:`~.adapters.keymap` refuses.
    """
    from ..adapters import get_adapter

    adapter = get_adapter(phase.get("backend", "gym"), phase)
    try:
        return adapter.native_fps()
    finally:
        adapter.close()
