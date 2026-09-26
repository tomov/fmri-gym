"""The ``fmri-edit`` window (PySide6). What it edits and why is in :mod:`fmri_gym.gui`.

This module is the only one that imports Qt, and it does so at the top because
its classes subclass Qt widgets; :func:`fmri_gym.gui.edit_config` imports it
lazily, so the field tables and parsers stay usable without the ``gui`` extra.

Qt rather than tkinter: the Tk that ships with uv's Python has no font
rendering to speak of (Tk 9 without Xft), and an editor people avoid is an
editor that does not replace hand-edited JSON. PySide6 rather than PyQt6
because it is LGPL. The window is closed before ``fmri_play`` opens the pygame
display.

The model is ``self.steps`` -- the session's lines, each a run (a config file)
or an external command, see :func:`fmri_gym.gui.write_session` -- and ``self.configs``,
the runs' configs by path. Session manager has two panels, each a form view
and the text it stands for: Session design (the list of lines, or the script)
and Run design (the selected run's phases, or its JSON). Whichever view is
shown is the truth: every handler first stores it into the model
(:meth:`_Editor._store_current`, :meth:`_Editor._store_session`), and leaving
a text view that does not parse is refused with the reason, never dropped.
Controls and Triggers edit the selected run's config too. A lone config is a
session of one run with no script.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import shlex
import signal
from dataclasses import asdict
from typing import Any, Callable, Iterator, Sequence

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from . import bids
from . import config as cfg
from . import gui
from .display import list_monitors, monitor_label
from .triggers import Codes, SyncSettings, TriggerError, Triggers, TriggerSettings

#: The two views of each Session manager panel (sub-tab indices).
_LIST, _SCRIPT = 0, 1
_PHASES, _JSON = 0, 1

#: Qt keys -> the key NAMES used in ``keys`` (see fmri_gym/keys.py); letters
#: and digits are taken from the event's text.
_QT_KEYS = {Qt.Key.Key_Up: "UP", Qt.Key.Key_Down: "DOWN", Qt.Key.Key_Left: "LEFT",
            Qt.Key.Key_Right: "RIGHT", Qt.Key.Key_Space: "SPACE", Qt.Key.Key_Return: "RETURN",
            Qt.Key.Key_Tab: "TAB", Qt.Key.Key_Shift: "LSHIFT", Qt.Key.Key_Comma: "COMMA",
            Qt.Key.Key_Period: "PERIOD"}

_GAME_KEYS_HINT = (
    "Overrides of the backend's default keyboard map for this phase. Keys are pygame "
    "names (UP, DOWN, LEFT, RIGHT, SPACE, RETURN, A-Z, 0-9), joined with + for combos; "
    "the action is what the env expects (an int for Discrete; quote a string that "
    "looks like a number). Unmentioned keys keep the backend default.")
_CHECK_KEYS_HINT = (
    "The keys the rig check asks for, one at a time, on the participant's device: each "
    "button's key (a pygame name: 1, B, LEFT...), and what it stands for, shown when it is "
    "asked for. A key that never comes, or comes as another, fails the check.")

_STYLE = """
QGroupBox { font-weight: 600; border: 1px solid palette(mid); border-radius: 8px;
            margin-top: 14px; padding: 12px 8px 8px 8px; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }
QListWidget, QPlainTextEdit, QTableWidget { border: 1px solid palette(mid); border-radius: 6px; }
QLabel#hint { color: gray; }
QLabel#report { padding: 10px; border-radius: 6px; background: palette(alternate-base); }
QPushButton#run { background: #2f6fde; color: white; border: none; border-radius: 6px;
                  padding: 7px 22px; font-weight: 600; }
QPushButton#run:hover { background: #4a86ee; }
QPushButton#run:pressed { background: #2459b8; }
"""


def run_editor(config: dict, path: str | None, launch: dict,
               session: str | None = None) -> list[str] | None:
    """Show the editor and block until it closes; see :func:`fmri_gym.gui.edit_config`.

    :raises KeyboardInterrupt: on Ctrl+C in the terminal, once the window is closed.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["fmri-gym"])
    app.setStyle("Fusion")
    app.setStyleSheet(_STYLE)
    editor = _Editor(config, path, launch, session)
    editor.show()
    with _closed_by_ctrl_c(app) as interrupted:
        app.exec()
    if interrupted:
        editor.hide()
        raise KeyboardInterrupt
    return editor.to_run


@contextlib.contextmanager
def _closed_by_ctrl_c(app: QtWidgets.QApplication) -> Iterator[list[bool]]:
    """Let Ctrl+C in the terminal close the editor; yields a list that is non-empty if it did.

    Qt's event loop runs in C++, and Python only acts on a signal when it next
    runs Python code -- which Qt may not do until the mouse moves. So the
    handler quits the loop (every loop, a modal dialog's included), and a timer
    runs a no-op often enough for Ctrl+C to act at once. The previous handler is
    put back after, so Ctrl+C in the session that may follow behaves as before.
    """
    interrupted: list[bool] = []

    def on_sigint(_signum: int, _frame: Any) -> None:
        interrupted.append(True)
        app.quit()

    previous = signal.signal(signal.SIGINT, on_sigint)
    tick = QtCore.QTimer()
    tick.timeout.connect(lambda: None)
    tick.start(200)
    try:
        yield interrupted
    finally:
        tick.stop()
        signal.signal(signal.SIGINT, previous)


def key_name(event: QtGui.QKeyEvent) -> str | None:
    """A key press -> KeySpec key NAME, or ``None`` if it is not a game key."""
    if event.key() in _QT_KEYS:
        return _QT_KEYS[event.key()]
    text = event.text()
    if len(text) == 1 and text.isascii() and text.isalnum():
        return text.upper()
    return None


def _mono() -> QtGui.QFont:
    return QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)


def _button(text: str, slot: Callable[[], Any]) -> QtWidgets.QPushButton:
    b = QtWidgets.QPushButton(text)
    b.clicked.connect(lambda _checked=False: slot())
    return b


def _row(*widgets: QtWidgets.QWidget, stretch: bool = True) -> QtWidgets.QHBoxLayout:
    row = QtWidgets.QHBoxLayout()
    for w in widgets:
        row.addWidget(w)
    if stretch:
        row.addStretch(1)
    return row


class _Form(QtWidgets.QWidget):
    """One labelled row per :class:`~fmri_gym.gui.Field`; the tip is the tooltip."""

    #: Any row edited (typed, picked, ticked) -- also when :meth:`set` fills it.
    changed = QtCore.Signal()

    def __init__(self, fields: Sequence[gui.Field]) -> None:
        super().__init__()
        self.fields = list(fields)
        self.widgets: dict[str, Any] = {}
        #: ``text`` fields the file gave as a list of lines (the session takes
        #: either); they are handed back as a list, so Save keeps the file's form.
        self.as_lines: set[str] = set()
        layout = QtWidgets.QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for f in self.fields:
            label = QtWidgets.QLabel(f.label)
            self.widgets[f.key] = widget = self._make_widget(f)
            for w in (label, widget):
                w.setToolTip(f.tip)
            layout.addRow(label, widget)

    def _make_widget(self, f: gui.Field) -> Any:
        if f.kind == "bool":
            w = QtWidgets.QCheckBox()
            w.toggled.connect(self.changed)
            return w
        if f.kind in ("choice", "combo"):
            w = QtWidgets.QComboBox()
            w.setEditable(f.kind == "combo")
            w.addItems(list(f.choices))
            w.currentTextChanged.connect(self.changed)  # a pick, or typing in a combo
            return w
        if f.kind in ("text", "lines"):
            w = QtWidgets.QPlainTextEdit()
            w.setFixedHeight(110)
            w.textChanged.connect(self.changed)
            return w
        w = QtWidgets.QLineEdit()
        w.textChanged.connect(self.changed)
        return w

    def write(self, key: str, raw: Any) -> None:
        """Put a widget value (see :func:`~fmri_gym.gui.format_value`) into one row."""
        f = next(f for f in self.fields if f.key == key)
        w = self.widgets[key]
        if f.kind == "bool":
            w.setChecked(raw)
        elif f.kind == "combo":
            w.setEditText(raw)
        elif f.kind == "choice":
            # A choice has no blank entry, so a key the file left out (or gave a
            # value outside the list) gets one: shown as it is, not as a default.
            w.clear()
            w.addItems(list(f.choices) if raw in f.choices else [raw, *f.choices])
            w.setCurrentText(raw)
        elif f.kind in ("text", "lines"):
            w.setPlainText(raw)
        else:
            w.setText(raw)

    def show_rows(self, keys: Sequence[str]) -> None:
        """Show only the rows of ``keys`` (their values stay, and :meth:`get` reads them all)."""
        for f in self.fields:
            self.layout().setRowVisible(self.widgets[f.key], f.key in keys)

    def set_choices(self, key: str, choices: Sequence[str]) -> None:
        """Replace a ``combo`` row's list, keeping what is typed in it."""
        w = self.widgets[key]
        text = w.currentText()
        w.blockSignals(True)  # the text does not change, so nothing to announce
        w.clear()
        w.addItems(list(choices))
        w.setEditText(text)
        w.blockSignals(False)

    def _read(self, f: gui.Field) -> Any:
        w = self.widgets[f.key]
        if f.kind == "bool":
            return w.isChecked()
        if f.kind in ("choice", "combo"):
            return w.currentText()
        if f.kind in ("text", "lines"):
            return w.toPlainText()
        return w.text()

    def set(self, values: dict) -> None:
        """Fill the form from a dict (missing keys show as blank/default)."""
        self.as_lines = {f.key for f in self.fields
                         if f.kind == "text" and isinstance(values.get(f.key), list)}
        for f in self.fields:
            self.write(f.key, gui.format_value(f, values.get(f.key)))

    def get(self) -> dict:
        """Read the form back as config values (see :func:`~fmri_gym.gui.field_values`)."""
        values = gui.field_values(self.fields, {f.key: self._read(f) for f in self.fields})
        for key in self.as_lines & values.keys():
            values[key] = values[key].split("\n")
        return values


class _KeyTable(QtWidgets.QTableWidget):
    """Editable rows of ``key combo -> action`` for one game phase."""

    def __init__(self) -> None:
        super().__init__(0, 2)
        self.setHorizontalHeaderLabels(["key(s)   e.g. UP or LEFT+SPACE",
                                        "action   e.g. 2 or [1,0,0]"])
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.verticalHeader().hide()
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)

    def add_row(self, key: str = "", action: str = "") -> None:
        row = self.rowCount()
        self.insertRow(row)
        self.setItem(row, 0, QtWidgets.QTableWidgetItem(key))
        self.setItem(row, 1, QtWidgets.QTableWidgetItem(action))
        self.setCurrentCell(row, 1 if key else 0)

    def remove_selected(self) -> None:
        if self.currentRow() >= 0:
            self.removeRow(self.currentRow())

    def set(self, keys: dict) -> None:
        self.setRowCount(0)
        for key, action in keys.items():
            self.add_row(key, gui.format_action(action))
        self.setCurrentCell(-1, -1)

    def get(self) -> dict:
        """The bindings; rows with a blank key are ignored.

        :raises ValueError: if a key has no action.
        """
        self.setCurrentCell(-1, -1)  # closes a cell still being typed in, keeping its text
        out = {}
        for row in range(self.rowCount()):
            key, action = (self.item(row, col).text().strip() for col in (0, 1))
            if not key:
                continue
            if not action:
                raise ValueError(f"keys: {key.upper()} has no action")
            out[key.upper()] = gui.parse_action(action)
        return out


class _KeyCapture(QtWidgets.QDialog):
    """A modal "press a key": ``name`` is the game key pressed, ``None`` on Esc."""

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("press a key")
        self.name: str | None = None
        self.refused: str | None = None
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(36, 28, 36, 28)
        layout.addWidget(QtWidgets.QLabel("Press the key to bind (Esc cancels)"))

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802 -- Qt's name
        self.name = key_name(event)
        if self.name is None and event.key() != Qt.Key.Key_Escape:
            self.refused = QtGui.QKeySequence(event.key()).toString() or "that key"
        self.accept()


class _Editor(QtWidgets.QMainWindow):
    """The window: menu, tabs, bottom bar. ``to_run`` is what Play left to run."""

    def __init__(self, config: dict, path: str | None, launch: dict,
                 session: str | None = None) -> None:
        super().__init__()
        #: What Play left for fmri-edit to become (gui.edit_config); None: closed instead.
        self.to_run: list[str] | None = None
        self.session_path: str | None = None
        self.steps: list[dict] = []     # {"config": path} | {"command": text}, each + "skip"
        self.step_index = 0
        self.configs: dict[str, dict] = {}   # the runs' configs by path
        self.opened: set[str] = set()        # ... those shown here, which Save writes
        # ... those created here: named at once (so the list and the script can show them) but
        # not on disk until Save, which asks where they go.
        self.new: set[str] = set()
        self.phases: list[dict] = []    # the selected run's curriculum
        self.notes: dict = {}           # top-level keys the editor does not own (e.g. "_note")
        self.edit_index: int | None = None
        self.ctl_index: int | None = None
        self.phase_form: _Form | None = None
        self.extra_text: QtWidgets.QPlainTextEdit | None = None
        self._defaults: dict = {}
        # The view each panel shows, and the text a text view was last filled with: a
        # text view is stored back only when edited, so an untouched one changes nothing.
        self._session_view, self._run_view = _LIST, _PHASES
        self._script_shown = self._json_shown = ""
        self.setWindowTitle("fmri-gym config")
        self.resize(1250, 850)
        self._build_menu()
        bottom = self._build_bottom()  # first: the Launch tab labels its Run button
        self.tabs = QtWidgets.QTabWidget()
        self._build_tabs(launch)
        body = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(body)
        layout.addWidget(self.tabs)
        layout.addLayout(bottom)
        self.setCentralWidget(body)
        self.tabs.currentChanged.connect(lambda _index: self._on_tab_change())
        self.curated, unreadable = gui.curated_games()
        if session is None:
            self.open_config(config, path)
        else:
            self.open_session(session)
        if unreadable:
            self._set_status("game lists: skipped config(s) that do not load: "
                             + ", ".join(unreadable))

    # -- construction ------------------------------------------------------

    def _build_tabs(self, launch: dict) -> None:
        # Launch first: who, where and on which screen is the first thing set at the desk.
        pages = {title: build() for title, build in (
            ("Launch", lambda: self._tab_launch(launch)),
            ("Session manager", self._tab_session),
            ("Controls", self._tab_controls), ("Triggers", self._tab_triggers))}
        for title, page in pages.items():
            self.tabs.addTab(page, title)
        self.config_pages = (pages["Controls"], pages["Triggers"])  # a command line has no config

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

    def _build_bottom(self) -> QtWidgets.QHBoxLayout:
        self.play_button = _button("Play", self._run)
        self.play_button.setObjectName("run")
        bar = _row(_button("New", self._new), _button("Open...", self._open),
                   _button("Save", self._save), _button("Save As...", self._save_as))
        for w in (_button("Check", self._check), self.play_button):
            bar.addWidget(w)
        return bar

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
        start = _button("Start here", self._start_here)
        start.setToolTip("Skip every run before this one and none after: how a stopped "
                         "session is resumed. The runs after it keep the numbers they have "
                         "here. To resume into the session that stopped, give its number to "
                         "the script (sh ses1.sh 003) or write it in --ses on the Launch tab.")
        rows = QtWidgets.QVBoxLayout()
        repeat = _button("Repeat", self._duplicate_step)
        repeat.setToolTip("The same line again. A config played twice is two runs of its task "
                          "(run-001, run-002), each with its own episode seeds.")
        rows.addLayout(_row(add, repeat, _button("Remove", self._remove_step)))
        rows.addLayout(_row(_button("Up", lambda: self._move_step(-1)),
                            _button("Down", lambda: self._move_step(1)), self.skip_box, start))
        return rows

    def _tab_session(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        panels = QtWidgets.QSplitter()
        panels.addWidget(self._panel("Session design", self._session_design()))
        panels.addWidget(self._panel("Run design", self._run_design()))
        panels.setStretchFactor(1, 1)
        panels.setSizes([380, 870])
        QtWidgets.QVBoxLayout(page).addWidget(panels)
        return page

    @staticmethod
    def _panel(title: str, inner: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(title)
        layout = QtWidgets.QVBoxLayout(box)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(inner)
        return box

    def _session_design(self) -> QtWidgets.QTabWidget:
        """The session's lines, as a list or as the script itself."""
        listing = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(listing)
        self.session_pick = self._file_pick("sh", "the session scripts in configs/ (File > Open "
                                                  "for one elsewhere)", self._pick_session)
        layout.addLayout(_row(QtWidgets.QLabel("Session"), self.session_pick, stretch=False))
        self.run_list = QtWidgets.QListWidget()
        self.run_list.setFont(_mono())
        self.run_list.currentRowChanged.connect(self._select_step)
        layout.addWidget(self.run_list, 1)
        layout.addLayout(self._run_buttons())
        self.script_text = QtWidgets.QPlainTextEdit()
        self.script_text.setFont(_mono())
        self.script_text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        script = self._text_page(self.script_text, (
            "The session script, one line per run. Edit it as in any text editor; it is read "
            "back when you leave this view, save or run. SES= picks the session once for all "
            "the runs (the next free one, or a number). A commented line is a skipped run; a "
            "line that is not an fmri-play run is any command of yours."))
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
        self.json_text.setFont(_mono())
        self.json_text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        json_page = self._text_page(self.json_text, (
            "The run's config file, as it will be saved. Edit it as in any text editor; it "
            "is read back when you leave this view, pick another run, save or run."))
        self.run_views = QtWidgets.QTabWidget()
        self.run_views.addTab(self.run_pages, "Phases")
        self.run_views.addTab(json_page, "JSON")
        self.run_views.currentChanged.connect(self._on_run_view)
        self.run_title = QtWidgets.QLabel()
        self.run_title.setObjectName("hint")
        self.run_pick = self._file_pick("json", "the run configs in configs/: open one, or in a "
                                                "session add it after the selected line",
                                        self._pick_run)
        design = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(design)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(_row(self.run_title, QtWidgets.QLabel("Run"), self.run_pick,
                              stretch=False))
        layout.addWidget(self.run_views, 1)
        return design

    def _phase_columns(self) -> QtWidgets.QWidget:
        """The phase list, its buttons, and the selected phase's form."""
        self.phase_list = QtWidgets.QListWidget()
        self.phase_list.setFont(_mono())
        self.phase_list.setFixedWidth(230)
        self.phase_list.currentRowChanged.connect(self._select_phase)
        add = QtWidgets.QPushButton("Add")
        add.setMenu(QtWidgets.QMenu(add))
        for kind in cfg.PHASE_TYPES:
            add.menu().addAction(kind).triggered.connect(
                lambda _checked=False, k=kind: self._add_phase(k))
        buttons = QtWidgets.QVBoxLayout()
        for b in (add, _button("Duplicate", self._duplicate_phase),
                  _button("Remove", self._remove_phase),
                  _button("Move up", lambda: self._move_phase(-1)),
                  _button("Move down", lambda: self._move_phase(1))):
            buttons.addWidget(b)
        buttons.addStretch(1)
        self.phase_pane = QtWidgets.QGroupBox("phase")
        self.phase_pane.setLayout(QtWidgets.QVBoxLayout())
        self.phase_body: QtWidgets.QWidget | None = None
        phases = QtWidgets.QWidget()
        columns = QtWidgets.QHBoxLayout(phases)
        columns.addWidget(self.phase_list)
        columns.addLayout(buttons)
        columns.addWidget(self.phase_pane, 1)
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
        self.command.setFont(_mono())
        self.command.setPlaceholderText("./scripts/localizer.sh --arg   (any shell command)")
        layout.addLayout(_row(self.command, _button("Browse...", self._pick_command),
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
        pick.setFont(_mono())
        pick.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        pick.setToolTip(tip)
        pick.addItem("pick from configs/...", None)
        for path in gui.listed_files(ext):
            pick.addItem(path, path)
        pick.activated.connect(lambda index: slot(pick.itemData(index)))
        return pick

    def _pick_session(self, path: str | None) -> None:
        if path is None:
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
        try:
            self.open_config(cfg.load_config(path), path)
        except (OSError, ValueError) as exc:
            self._error(f"cannot open {path}: {exc}")

    def _sync_session_pick(self) -> None:
        """The session drop-down shows the open script, or the prompt for a lone config."""
        self.session_pick.blockSignals(True)  # ours, not a click
        self.session_pick.setCurrentIndex(max(0, self.session_pick.findData(self.session_path)))
        self.session_pick.blockSignals(False)

    def _tab_controls(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self.ctl_run = QtWidgets.QLabel()
        self.ctl_run.setObjectName("hint")
        layout.addWidget(self.ctl_run)
        self.ctl_phase = QtWidgets.QComboBox()
        self.ctl_phase.setMinimumWidth(420)
        self.ctl_phase.setFont(_mono())
        self.ctl_phase.activated.connect(self._select_ctl_phase)
        layout.addLayout(_row(QtWidgets.QLabel("phase"), self.ctl_phase))
        self.ctl_hint = QtWidgets.QLabel(_GAME_KEYS_HINT)
        self.ctl_hint.setObjectName("hint")
        self.ctl_hint.setWordWrap(True)
        layout.addWidget(self.ctl_hint)
        self.key_table = _KeyTable()
        layout.addWidget(self.key_table, 2)
        layout.addLayout(_row(
            _button("Add binding", self.key_table.add_row),
            _button("Press a key...", self._capture_key),
            _button("Remove binding", self.key_table.remove_selected),
            _button("Show backend defaults", self._show_defaults),
            _button("Use defaults as bindings", self._use_defaults)))
        layout.addLayout(self._layout_row())
        self.defaults_text = QtWidgets.QPlainTextEdit()
        self.defaults_text.setReadOnly(True)
        self.defaults_text.setFont(_mono())
        layout.addWidget(self.defaults_text, 1)
        return page

    def _layout_row(self) -> QtWidgets.QHBoxLayout:
        self.layout_pick = QtWidgets.QComboBox()
        self.layout_pick.addItems(list(gui.DEVICE_LAYOUTS))
        self.layout_pick.setToolTip("Keyboard: every game plays with its own keys, always -- "
                                    "Use shows them. A device is read as a keyboard too: Use adds "
                                    "its buttons to this phase, each with the action of the game "
                                    "key it stands for (1 = LEFT on a button box). The keyboard "
                                    "keeps working; edit the added rows like any other.")
        return _row(QtWidgets.QLabel("layout"), self.layout_pick, _button("Use", self._add_layout))

    def _tab_triggers(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        self.preset = QtWidgets.QComboBox()
        self.preset.addItems(list(cfg.TRIGGER_PRESETS))
        # Nothing picked: the forms show the file's settings, not a preset's.
        self.preset.setPlaceholderText("choose a preset...")
        self.preset.setCurrentIndex(-1)
        self.preset.setMinimumWidth(360)
        layout.addLayout(_row(QtWidgets.QLabel("preset"), self.preset,
                              _button("Apply preset", self._apply_preset)))
        self.sync_form = _Form(gui.SYNC_FIELDS)
        self.sync_form.widgets["mode"].currentTextChanged.connect(self._show_sync_rows)
        self.code_form = _Form(gui.CODE_FIELDS)
        self.trigger_form = _Form(gui.TRIGGER_FIELDS)
        left = QtWidgets.QVBoxLayout()
        left.addWidget(self._group("sync (run start)", self.sync_form))
        left.addWidget(self._group("codes", self.code_form))
        forms = QtWidgets.QHBoxLayout()
        forms.addLayout(left, 1)
        forms.addWidget(self._group("codes go out over", self.trigger_form), 1)
        layout.addLayout(forms)
        layout.addLayout(_row(_button("Check", self._refresh_triggers),
                              _button("Test (send task_start)", self._test_triggers)))
        self.trigger_text = QtWidgets.QLabel()
        self.trigger_text.setObjectName("report")
        self.trigger_text.setWordWrap(True)
        self.trigger_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.trigger_text)
        layout.addStretch(1)
        return page

    @staticmethod
    def _group(title: str, inner: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(title)
        QtWidgets.QVBoxLayout(box).addWidget(inner)
        return box

    def _tab_launch(self, launch: dict) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        self.launch_page = page
        layout = QtWidgets.QVBoxLayout(page)
        self.launch_form = _Form(gui.LAUNCH_FIELDS)
        self.monitors = list_monitors()
        self.monitor_pick = QtWidgets.QComboBox()
        for index, monitor in enumerate(self.monitors):
            self.monitor_pick.addItem(monitor_label(index, monitor), index)
        self.monitor_pick.setToolTip("Which screen the participant sees. SDL numbers the "
                                     "monitors and gives no names: pick by resolution and "
                                     "refresh rate. Calibrate the photodiode on the same one.")
        after_fullscreen = [f.key for f in gui.LAUNCH_FIELDS].index("fullscreen") + 1
        self.launch_form.layout().insertRow(after_fullscreen, "--monitor", self.monitor_pick)
        self._window_size = launch["size"]  # kept while fullscreen shows the monitor's instead
        self._loading_launch = False
        self._set_launch(launch)
        self.launch_form.widgets["fullscreen"].toggled.connect(lambda _on: self._sync_size())
        self.monitor_pick.currentIndexChanged.connect(lambda _index: self._sync_size())
        dummy = self.launch_form.widgets["dummy_trigger"]
        dummy.toggled.connect(self._label_play_button)
        self._label_play_button(dummy.isChecked())
        layout.addWidget(self._group("flags of this launch", self.launch_form))
        layout.addLayout(_row(_button("Choose --data-root...", self._pick_data_root)))
        note = QtWidgets.QLabel("These are the command-line flags of this one launch. They are "
                                "not saved to a config file; a session script carries them "
                                "on every run's line.")
        note.setObjectName("hint")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)
        return page

    def _set_launch(self, launch: dict) -> None:
        """Show launch flags, a monitor this machine lacks included (Check then says so)."""
        # Setting the fullscreen box and the monitor fires _sync_size half-way, when the size
        # row still shows the previous launch's: suspended here, and done once at the end.
        self._loading_launch = True
        try:
            self.launch_form.set(launch)
            at = self.monitor_pick.findData(launch["monitor"])
            if at < 0:
                self.monitor_pick.addItem(f"{launch['monitor']}: not connected here",
                                          launch["monitor"])
                at = self.monitor_pick.count() - 1
            self.monitor_pick.setCurrentIndex(at)
        finally:
            self._loading_launch = False
        self._window_size = launch["size"]
        self._sync_size(keep_typed=False)

    def _monitor(self) -> tuple[int, int, int] | None:
        """The picked monitor's ``(width, height, Hz)``, or ``None`` if not connected here."""
        index = self.monitor_pick.currentData()
        return self.monitors[index] if index < len(self.monitors) else None

    def _sync_size(self, keep_typed: bool = True) -> None:
        """Fullscreen: the size row shows the monitor's resolution, greyed, since that is what a
        fullscreen window gets. Windowed: the window's own size, with sizes that fit the monitor.

        :param keep_typed: first remember what the row shows, if it is the window's size.
        """
        if self._loading_launch:
            return
        box = self.launch_form.widgets["size"]
        if keep_typed and box.isEnabled():
            self._window_size = box.currentText().strip()
        monitor = self._monitor()
        fullscreen = self.launch_form.widgets["fullscreen"].isChecked()
        box.setEnabled(not fullscreen)
        if fullscreen:
            self.launch_form.set_choices("size", [])
            box.setEditText(f"{monitor[0]}x{monitor[1]}" if monitor else "monitor not connected")
            return
        fits = gui.window_sizes(*monitor[:2]) if monitor else [f"{w}x{h}" for w, h in
                                                               gui.WINDOW_SIZES]
        self.launch_form.set_choices("size", fits)
        box.setEditText(self._window_size)

    def _launch_values(self) -> dict:
        """The launch flags. ``--size`` is the window's, also while fullscreen shows the monitor's.

        :raises ValueError: if the subject is blank or the size malformed (gui.launch_values).
        """
        self._sync_size()  # remembers a size being typed
        form = self.launch_form.get() | {"size": self._window_size}
        return gui.launch_values(form) | {"monitor": self.monitor_pick.currentData()}

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

    def open_session(self, path: str) -> None:
        """Edit a session script and the configs its runs name.

        :raises OSError: if the script or one of its configs cannot be read.
        :raises ValueError: if a config does not load, or the script has no line to play.
        """
        with open(path) as f:
            steps, launch = gui.read_session(f.read())
        if not steps:
            raise ValueError(f"{path}: no run in it")
        # Every config now, so a missing one is named here and not at some later click.
        configs = {s["config"]: cfg.load_config(s["config"]) for s in steps if "config" in s}
        self.session_path, self.steps = path, steps
        self.configs, self.opened, self.new = configs, set(), set()
        if launch is not None:
            self._set_launch(launch)
        self._show_step(0)
        self._refresh_views()

    def _show_config(self, config: dict) -> None:
        """Load the selected run's config into the phase list and the config tabs."""
        self.notes = {k: v for k, v in config.items() if k not in cfg.SECTIONS}
        self.phases = copy.deepcopy(config["curriculum"])
        self.edit_index = self.ctl_index = None
        self._refresh_phase_list(0 if self.phases else None)
        self._refresh_ctl_phases()
        section = dict(config.get("triggers") or {})  # null: not set up yet (a rig check)
        codes = {**asdict(Codes()), **section.pop("codes", {})}
        self._set_sync(section.pop("sync", {}), codes["scanner_start"])
        self.code_form.set(codes)
        self.trigger_form.set({**_trigger_defaults(), **section})
        self._refresh_triggers()
        self._show_json(config)

    def collect(self) -> dict:
        """Read the config tabs back into one config dict (the selected run's).

        :raises ValueError: naming the offending field.
        """
        self._commit()
        return {**copy.deepcopy(self.notes), "triggers": self._current_triggers(),
                "curriculum": copy.deepcopy(self.phases)}

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
        self._commit_keys()

    def _store_current(self) -> None:
        """Put the selected line back into the model: its forms or its JSON, or the command.

        :raises ValueError: naming the offending field, or why the JSON does not load.
        """
        step = self.steps[self.step_index]
        if "command" in step:
            step["command"] = self.command.text().strip()
            return
        if self._run_view == _JSON and self.json_text.toPlainText() != self._json_shown:
            self._show_config(self._read_json())
        self.configs[step["config"]] = config = self.collect()
        if self._run_view == _JSON:
            self._show_json(config)  # the view shows what will be saved

    def _read_json(self) -> dict:
        """:raises ValueError: if the JSON view does not hold a config (the reason says why)."""
        try:
            return cfg.check_shape(json.loads(self.json_text.toPlainText()), "JSON")
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
        steps, launch = gui.read_session(text)
        if not steps:
            raise ValueError("script: no line to play")
        try:  # configs shown here keep their unsaved edits; the others are read now
            new = {s["config"]: cfg.load_config(s["config"]) for s in steps
                   if "config" in s and s["config"] not in self.configs}
        except OSError as exc:
            raise ValueError(f"script: {exc}") from exc
        self.configs.update(new)
        self.steps = steps
        if launch is not None:
            self._set_launch(launch)
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
        self._refresh_ctl_phases()
        self._refresh_triggers()
        self._show_json(self.configs[step["config"]])
        if self.seed_label is not None:  # Launch edits (subject, --ses) change a derived seed
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
                    else os.path.basename(step["config"]) + "  (new)" * (step["config"] in self.new))
            labels.append(f"{i}  {'[skip] ' if step['skip'] else ''}{what}")
        return labels

    def _refresh_run_list(self) -> None:
        self.run_list.blockSignals(True)  # the selection below is ours, not the user's
        self.run_list.clear()
        self.run_list.addItems(self._step_labels())
        self.run_list.setCurrentRow(self.step_index)
        self.run_list.blockSignals(False)

    def _label_play_button(self, dummy_trigger: bool) -> None:
        """A test run says so on the button that starts it."""
        self.play_button.setText("Test play (--dummy-trigger)" if dummy_trigger else "Play")

    def _show_step(self, index: int) -> None:
        """Point the editors at line ``index`` (nothing is stored first; see _select_step)."""
        self.step_index = index
        step = self.steps[index]
        is_command = "command" in step
        self.run_pages.setCurrentIndex(1 if is_command else 0)
        for page in self.config_pages:
            self.tabs.setTabEnabled(self.tabs.indexOf(page), not is_command)
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
        if not is_command:
            self._show_config(self.configs[step["config"]])
            self.opened.add(step["config"])
            self.ctl_run.setText(f"run {index + 1}: {self._where(step)}")
        self._refresh_run_list()
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
                self.configs[path] = cfg.load_config(path)
            except (OSError, ValueError) as exc:
                self._error(f"cannot add {path}: {exc}")
                return
        self._insert_step({"config": path, "skip": False})

    def _add_new(self) -> None:
        """A new config, named after its first game's backend until Save asks where it goes."""
        config = cfg.new_config()
        path = self._provisional_path(config)
        self.configs[path] = config
        self.new.add(path)
        self._insert_step({"config": path, "skip": False})

    def _provisional_path(self, config: dict, own: str | None = None) -> str:
        """``configs/<backend>.json``, counting up past the files there and this session's
        (but not ``own``, the name it has now, so it can keep it)."""
        taken = {os.path.splitext(os.path.basename(p))[0] for p in self.configs if p != own}
        if os.path.isdir("configs"):
            taken |= {os.path.splitext(name)[0] for name in os.listdir("configs")}
        return os.path.join("configs", gui.suggest_name(gui.first_backend(config), taken) + ".json")

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
        self.phase_list.addItems([gui.phase_label(i, p) for i, p in enumerate(self.phases)])
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
        if index is None:
            self.phase_pane.setTitle("phase")
            return
        phase = self.phases[index]
        kind = phase["type"]
        self.phase_pane.setTitle(f"phase {index:02d}: {kind}")
        fields = gui.PHASE_FIELDS[kind]
        form, extra = gui.split_phase(phase, fields)
        self.phase_form = _Form(fields)
        self.phase_form.set(form)
        self.phase_form.changed.connect(self._relabel_phase)  # after set: the edits, not the load
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

    def _relabel_phase(self) -> None:
        """The phase's line in the list, and a new config's name, follow the form as it is typed."""
        try:
            live = {"type": self.phases[self.edit_index]["type"], **self.phase_form.get()}
        except ValueError:
            return  # a number half typed: the label waits for the rest
        self.phase_list.item(self.edit_index).setText(gui.phase_label(self.edit_index, live))
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
        self.ctl_run.setText(f"run {self.step_index + 1}: {where}")
        self._refresh_run_list()
        self._refresh_views()
        self._set_status()

    def _add_curated_row(self, body: QtWidgets.QVBoxLayout) -> None:
        """Under a game phase's form: where its game is already set up, and a button to copy that.

        The ``game`` list follows ``backend`` (see :func:`fmri_gym.gui.curated_games`).
        """
        self.curated_label = QtWidgets.QLabel()
        self.curated_label.setObjectName("hint")
        self.curated_button = _button("Fill this phase from that config", self._use_curated)
        body.addLayout(_row(self.curated_button, self.curated_label))
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
        self.ctl_index = None  # the Controls tab reloads the new keys
        self._refresh_phase_list(index)
        self._refresh_ctl_phases()
        self._set_status(f"phase {index:02d} filled from {found[0]}")

    def _add_seed_row(self, body: QtWidgets.QVBoxLayout) -> None:
        """Under a game phase's form: the seed it plays with, to read, copy, or pin as it is."""
        self.seed_label = QtWidgets.QLabel()
        self.seed_label.setObjectName("hint")
        self.seed_label.setWordWrap(True)
        self.seed_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.pin_button = _button("Pin", self._pin_seed)
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
        (:func:`fmri_gym.gui.run_number`), in the ``--ses`` given or, blank,
        the subject's next free session. A re-acquisition of a run replays
        these same episodes -- the attempt is no part of the label.

        :raises ValueError: if the Launch flags or the config's name give no run label.
        """
        step = self.steps[self.step_index]
        launch = self._launch_values()
        subject, root = bids.subject_label(launch["subject"]), launch["data_root"]
        ses = launch["ses"] or bids.next_session(root, subject)
        label = bids.run_label(subject, ses, bids.task_label(step["config"]),
                               gui.run_number(self.steps, self.step_index))
        return bids.phase_seed(label, self.edit_index), label

    def _pin_seed(self) -> None:
        if self._derived_seed is not None:
            self.phase_form.write("seed", str(self._derived_seed))

    def _add_game_extras(self, body: QtWidgets.QVBoxLayout, phase: dict, extra: dict) -> None:
        """Under a game phase's form: its ``keys`` (read-only here) and the other fields as JSON."""
        keys = ", ".join(f"{k}={gui.format_action(v)}" for k, v in phase.get("keys", {}).items())
        label = QtWidgets.QLabel(f"keys: {keys or 'backend defaults'}   (edit on the Controls tab)")
        label.setObjectName("hint")
        label.setWordWrap(True)
        body.addWidget(label)
        body.addWidget(QtWidgets.QLabel("extra (JSON)"))
        self.extra_text = QtWidgets.QPlainTextEdit(json.dumps(extra, indent=1) if extra else "{}")
        self.extra_text.setFont(_mono())
        self.extra_text.setFixedHeight(120)
        self.extra_text.setToolTip("Backend-specific fields as JSON, e.g. retro \"state\"/"
                                   "\"scenario\", vgdl \"level\", vizdoom \"env_kwargs\", "
                                   "ale \"save_pixels\".")
        body.addWidget(self.extra_text)

    def _commit_phase(self) -> None:
        if self.edit_index is None or self.phase_form is None:
            return
        old = self.phases[self.edit_index]
        new = {"type": old["type"], **self.phase_form.get()}
        if "keys" in old:
            new["keys"] = old["keys"]
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
        self.phase_list.item(self.edit_index).setText(gui.phase_label(self.edit_index, new))

    def _add_phase(self, kind: str) -> None:
        templates = {"fixation": {"type": "fixation", "duration": 2.0},
                     "message": {"type": "message", "text": "", "duration": 2.0},
                     "survey": {"type": "survey", "n_points": 7, "questions": []},
                     # No fps: Check names it, as it does the empty game id -- the rate
                     # belongs to the game about to be picked, not to this template.
                     "game": {"type": "game", "backend": "gym", "game": "", "mode": "duration",
                              "duration": 30.0},
                     # The rig check's: the quick check's values stated, keys to fill in.
                     "check_display": {"type": "check_display", "n": 60},
                     "check_frames": {"type": "check_frames", "rates": [60, 30],
                                      "seconds": 2, "loads": ["none"]},
                     "check_triggers": {"type": "check_triggers", "pulses": 3},
                     "check_controls": {"type": "check_controls", "timeout_s": 10.0,
                                        "keys": {"1": "LEFT", "2": "DOWN", "3": "UP",
                                                 "4": "RIGHT"}},
                     "check_photodiode": {"type": "check_photodiode", "readout": "soundcard",
                                          "n": 10}}
        self._insert_phase(templates[kind])

    def _duplicate_phase(self) -> None:
        if self.edit_index is not None and self._commit_all():
            self._insert_phase(copy.deepcopy(self.phases[self.edit_index]))

    def _insert_phase(self, phase: dict) -> None:
        # Indices shift, so the Controls tab re-picks its phase on next view.
        if not self._commit_all():
            return
        at = len(self.phases) if self.edit_index is None else self.edit_index + 1
        self.phases.insert(at, phase)
        self.ctl_index = None
        self._refresh_phase_list(at)

    def _remove_phase(self) -> None:
        if self.edit_index is None or not self._commit_all():
            return
        at = self.edit_index
        del self.phases[at]
        self.edit_index = self.ctl_index = None
        self._refresh_phase_list(min(at, len(self.phases) - 1) if self.phases else None)

    def _move_phase(self, step: int) -> None:
        i = self.edit_index
        if i is None or not 0 <= i + step < len(self.phases) or not self._commit_all():
            return
        self.phases[i], self.phases[i + step] = self.phases[i + step], self.phases[i]
        self.edit_index = self.ctl_index = None
        self._refresh_phase_list(i + step)

    # -- controls tab ------------------------------------------------------

    def _game_indices(self) -> list[int]:
        """The phases with keys: games, and a rig check's controls check."""
        return [i for i, p in enumerate(self.phases) if p["type"] in ("game", "check_controls")]

    def _ctl_is_check(self) -> bool:
        """The Controls tab shows a controls check: its keys are a device's buttons, each with
        what it stands for, and there is no game to take defaults from."""
        return (self.ctl_index is not None
                and self.phases[self.ctl_index]["type"] == "check_controls")

    def _refresh_ctl_phases(self) -> None:
        indices = self._game_indices()
        self.ctl_phase.clear()
        self.ctl_phase.addItems([gui.phase_label(i, self.phases[i]) for i in indices])
        if self.ctl_index not in indices:
            self.ctl_index = indices[0] if indices else None
            self.key_table.set(self.phases[self.ctl_index].get("keys", {})
                               if self.ctl_index is not None else {})
        if self.ctl_index is not None:
            self.ctl_phase.setCurrentIndex(indices.index(self.ctl_index))
        self.ctl_hint.setText(_CHECK_KEYS_HINT if self._ctl_is_check() else _GAME_KEYS_HINT)

    def _select_ctl_phase(self, index: int) -> None:
        if not self._commit_all():
            self._refresh_ctl_phases()
            return
        self.ctl_index = self._game_indices()[index]
        self.key_table.set(self.phases[self.ctl_index].get("keys", {}))
        self.ctl_hint.setText(_CHECK_KEYS_HINT if self._ctl_is_check() else _GAME_KEYS_HINT)
        self.defaults_text.setPlainText("")
        self._defaults = {}

    def _commit_keys(self) -> None:
        if self.ctl_index is None or self.ctl_index >= len(self.phases):
            return
        keys = self.key_table.get()
        phase = self.phases[self.ctl_index]
        if keys:
            phase["keys"] = keys  # in place, so the key keeps its position in the file
        else:
            phase.pop("keys", None)
        if self.edit_index == self.ctl_index and self.phase_form is not None:
            self._show_phase(self.edit_index)

    def _capture_key(self) -> None:
        dialog = _KeyCapture(self)
        dialog.exec()
        if dialog.name is not None:
            self.key_table.add_row(dialog.name, "")
        elif dialog.refused is not None:
            self._error(f"{dialog.refused!r} is not a key the games can read")

    def _show_defaults(self) -> None:
        if self.ctl_index is None:
            return
        if self._ctl_is_check():
            self.defaults_text.setPlainText(
                "A controls check has no game, so no defaults: its keys are the buttons to "
                "press, each with what it stands for (shown on screen when asked). Pick the "
                "device below and Use it, or add keys with Press a key...")
            return
        phase = {k: v for k, v in self.phases[self.ctl_index].items() if k != "keys"}
        self.defaults_text.setPlainText(f"loading {phase.get('game')}...")
        QtWidgets.QApplication.processEvents()
        try:
            self._defaults, native = _backend_defaults(phase)
        except Exception as exc:  # noqa: BLE001 -- any engine error: show it, keep the editor alive
            self._defaults = {}
            self.defaults_text.setPlainText(f"could not build the adapter: {exc}")
            return
        body = "\n".join(f"{k:<16} {gui.format_action(v)}" for k, v in self._defaults.items())
        rate = (f"{native:g} steps/s, this engine's own rate: write it in fps to play the game "
                "at its real speed" if native is not None else
                "no rate of its own: fps is yours to pick (30 suits most)")
        self.defaults_text.setPlainText(f"{phase.get('backend')} defaults for "
                                        f"{phase.get('game')}:\n{body}\n{rate}")

    def _add_layout(self) -> None:
        """Add the picked device's keys: the game's keyboard map, translated to its buttons."""
        if self.ctl_index is None:
            return
        name = self.layout_pick.currentText()
        if self._ctl_is_check():
            self._check_layout(name)
            return
        if not self._defaults:
            self._show_defaults()  # builds the adapter: the game's own map
        if not self._defaults:
            return  # _show_defaults said why
        try:
            table = {gui.combo_name(k): v for k, v in self.key_table.get().items()}
        except ValueError as exc:
            self._error(str(exc))
            return
        game_map = {**self._defaults, **table}
        keyboard = not gui.DEVICE_LAYOUTS[name]
        if keyboard:
            self.defaults_text.setPlainText(
                "Keyboard: this phase plays with\n" + "\n".join(
                    f"  {k:<14} {gui.format_action(v)}" for k, v in game_map.items())
                + "\n(the game's own map, with this phase's keys over it; it always works, "
                "beside any device)")
            return
        added = gui.translate_keys(game_map, gui.DEVICE_LAYOUTS[name])
        self.key_table.set({**table, **added})
        buttons = set(gui.DEVICE_LAYOUTS[name].values())
        unbound = [k for k in self._defaults if not set(k.split("+")) <= buttons]
        text = f"{name}: added " + ", ".join(f"{k}={gui.format_action(v)}" for k, v in added.items())
        if unbound:
            text += f"\nno button for: {', '.join(unbound)} (still on the keyboard)"
        self.defaults_text.setPlainText(text)

    def _check_layout(self, name: str) -> None:
        """A controls check tests the device's buttons themselves, each named for its key."""
        layout = gui.DEVICE_LAYOUTS[name]
        if not layout:
            self.defaults_text.setPlainText("Keyboard: add the keys to test with Press a key..., "
                                            "each with what it stands for")
            return
        self.key_table.set(dict(layout))
        self.defaults_text.setPlainText(f"{name}: the check asks for "
                                        + ", ".join(f"{k} ({v})" for k, v in layout.items()))

    def _use_defaults(self) -> None:
        if not self._defaults:
            self._show_defaults()
        if self._defaults:
            self.key_table.set(self._defaults)

    # -- triggers tab ------------------------------------------------------

    def _apply_preset(self) -> None:
        if self.preset.currentIndex() < 0:
            return
        section = copy.deepcopy(cfg.TRIGGER_PRESETS[self.preset.currentText()])
        start = self.sync_form.get().get("scanner_start", Codes().scanner_start)  # a preset keeps it
        self._set_sync(section.pop("sync"), start)
        self.trigger_form.set({**_trigger_defaults(), **section})
        self._refresh_triggers()

    def _set_sync(self, sync: dict, start: int) -> None:
        """Fill the sync box: the section's ``sync`` (defaults for what it leaves out) and the
        start code, which the file keeps in ``codes``."""
        self.sync_form.set({**asdict(SyncSettings()), **sync, "scanner_start": start})
        self._show_sync_rows(self.sync_form.widgets["mode"].currentText())

    def _show_sync_rows(self, mode: str) -> None:
        """Wait shows the trigger box's key; send the start code and its delay; none nothing."""
        rows = gui.SYNC_ROWS.get(mode, [f.key for f in gui.SYNC_FIELDS])  # unknown: all, to fix
        self.sync_form.show_rows(["mode", *rows])

    def _current_triggers(self) -> dict:
        return gui.triggers_section(self.sync_form.get(), self.trigger_form.get(),
                                    self.code_form.get())

    def _refresh_triggers(self) -> None:
        try:
            section = self._current_triggers()
        except ValueError as exc:  # a non-numeric field, named in the message
            self.trigger_text.setText(str(exc))
            return
        problems = cfg.trigger_problems(section)
        self.trigger_text.setText("\n".join(problems) if problems
                                  else gui.describe_triggers(section))

    def _test_triggers(self) -> None:
        """Open the backend exactly as a run would and send one ``task_start``."""
        try:
            triggers = Triggers.from_config(self._current_triggers())
        except (TriggerError, ValueError, TypeError) as exc:
            self.trigger_text.setText(f"test failed: {exc}")
            return
        if not triggers.enabled:
            self.trigger_text.setText("backend is null: nothing to send. Pick serial, parallel "
                                      "or lsl first.")
            return
        value = triggers.lifecycle("task_start")
        triggers.close()
        self.trigger_text.setText(f"sent {value} on {triggers.active}; check that the recording "
                                  "shows one trigger.")

    # -- file / run ----------------------------------------------------------

    def _new(self) -> None:
        self.open_config(cfg.new_config(), None)

    def _open(self) -> None:
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
                self.open_config(cfg.load_config(path), path)
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
        if not self._commit_session() or not self._place_new_configs():
            return False
        try:
            script = self._session_text() if self._is_session() else None
        except ValueError as exc:
            self.tabs.setCurrentWidget(self.launch_page)
            self._error(str(exc))
            return False
        if script is not None and self.session_path is None and not self._ask_session_path():
            return False
        paths = sorted({s["config"] for s in self.steps if "config" in s} & self.opened)
        try:
            for path in paths:
                cfg.save_config(self.configs[path], path)
            if script is not None:
                with open(self.session_path, "w") as f:
                    f.write(script)
                os.chmod(self.session_path, 0o755)
        except (OSError, ValueError) as exc:
            self._error(str(exc))
            return False
        self._show_step(self.step_index)  # new configs are files now: names and marks change
        self._refresh_views()
        self._set_status("saved " + ", ".join(paths + [self.session_path] * (script is not None)))
        return True

    def _session_text(self) -> str:
        """:raises ValueError: if the launch flags every line carries are incomplete."""
        return gui.write_session(self.steps, self._launch_values())

    def _ask_session_path(self) -> bool:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save the session script",
                                                        "configs", "Session script (*.sh)")
        if not path:
            return False
        self.session_path = os.path.relpath(path if path.endswith(".sh") else path + ".sh")
        return True

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

    def _pick_data_root(self) -> None:
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Where the BIDS tree goes")
        if path:
            self.launch_form.write("data_root", os.path.relpath(path))

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
        """What would stop the session, line by line (see :func:`fmri_gym.config.validate_config`)."""
        problems = []
        for i, step in enumerate(self.steps, start=1):
            label = f"run {i}: " if self._is_session() else ""
            if "command" in step:
                problems += [f"{label}the external command is empty"] * (not step["command"])
                continue
            problems += [label + p for p in cfg.validate_config(self.configs[step["config"]])]
        if all(step["skip"] for step in self.steps):
            problems.append("every run is skipped: nothing would play")
        problems += gui.trigger_mismatches(self.steps, self.configs)
        monitor = self.monitor_pick.currentData()
        if monitor >= len(self.monitors):
            labels = "; ".join(monitor_label(i, m) for i, m in enumerate(self.monitors))
            problems.append(f"--monitor {monitor}: this machine has {len(self.monitors)} "
                            f"({labels}); pick one on the Launch tab")
        return problems

    def _checked_launch(self) -> dict | None:
        """The launch flags if the session and they are runnable, else ``None`` (after a dialog)."""
        problems = self._check()
        if problems:
            self._error("\n".join(problems))
            return None
        try:
            return self._launch_values()
        except ValueError as exc:
            self.tabs.setCurrentWidget(self.launch_page)
            self._error(str(exc))
            return None

    def _run(self) -> None:
        """Play: save, then leave the command that plays what is shown (see ``to_run``).

        Both kinds play files on disk -- a session its script, a lone run its
        config -- so what was edited here is saved first, a new config being
        asked for a name. ``fmri-edit`` then becomes that command.
        """
        launch = self._checked_launch()
        if launch is None:
            return
        if not self._save():
            return
        if self._is_session():
            self.to_run = ["sh", self.session_path]
        else:
            # fmri-play states its numbers; a blank --ses is the next free one,
            # resolved here, as the script's SES= line resolves it for a session.
            ses = launch["ses"] or bids.next_session(launch["data_root"],
                                                     bids.subject_label(launch["subject"]))
            self.to_run = gui.play_command(self.steps[self.step_index]["config"], launch,
                                           f"{ses:03d}", gui.run_number(self.steps, 0))
        self.close()

    def _error(self, text: str) -> None:
        QtWidgets.QMessageBox.critical(self, "fmri-gym config", text)


_RIG_INTRO = ("Describe this rig once; every rig check copies it into its results, so the "
              "checks of all sites can be pooled and compared. Software cannot see these: "
              "say what is plugged in.")


class _RigForm(QtWidgets.QDialog):
    def __init__(self, path: str, values: dict) -> None:
        super().__init__()
        self.setWindowTitle(f"Rig file: {path}")
        self.setMinimumWidth(640)
        self.fields: dict[str, QtWidgets.QWidget] = {}
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(_RIG_INTRO)
        intro.setWordWrap(True)
        layout.addWidget(intro)
        from . import checks  # its rig file's fields; checks.py is not the editor's to import
        self._problems = checks.rig_problems
        form = QtWidgets.QFormLayout()
        examples = checks.RIG_EXAMPLES
        for key, what in checks.RIG_FIELDS.items():
            form.addRow(key, self._field(key, str(values.get(key, "")), examples.get(key, "")))
            hint = QtWidgets.QLabel(what)
            hint.setStyleSheet("color: gray")
            form.addRow("", hint)
        layout.addLayout(form)
        self.problems = QtWidgets.QLabel()
        self.problems.setWordWrap(True)
        self.problems.setStyleSheet("color: #c0392b")
        layout.addWidget(self.problems)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Save
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.save = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Save)
        layout.addWidget(buttons)
        self._check()

    def _field(self, key: str, value: str, example: str) -> QtWidgets.QWidget:
        """The widget for one key: a choice for ``modality``, a text box for ``notes``."""
        if key == "modality":
            w = QtWidgets.QComboBox()
            from .checks import MODALITIES
            w.addItems(["", *MODALITIES])
            w.setCurrentText(value if value in MODALITIES else "")
            w.currentTextChanged.connect(self._check)
        elif key == "notes":
            w = QtWidgets.QPlainTextEdit(value)
            w.setFixedHeight(60)
            w.textChanged.connect(self._check)
        else:
            w = QtWidgets.QLineEdit(value)
            w.setPlaceholderText(f"e.g. {example}" if example else "")
            w.textChanged.connect(self._check)
        self.fields[key] = w
        return w

    @property
    def rig(self) -> dict[str, str]:
        """The values as typed, trimmed."""
        out = {}
        for key, w in self.fields.items():
            text = (w.currentText() if isinstance(w, QtWidgets.QComboBox)
                    else w.toPlainText() if isinstance(w, QtWidgets.QPlainTextEdit)
                    else w.text())
            out[key] = text.strip()
        return out

    def accept(self) -> None:
        """Close with Save only when valid: a disabled button does not stop Enter."""
        if not self._problems(self.rig):
            super().accept()

    def _check(self) -> None:
        problems = self._problems(self.rig)
        self.problems.setText("\n".join(f"• {p}" for p in problems))
        self.save.setEnabled(not problems)


def fill_rig(path: str, values: dict) -> dict[str, str] | None:
    """The rig file (:mod:`fmri_gym.checks`), filled in a form instead of by hand; on Save,
    written.

    One window rather than a prompt per field: all of it stays in view, and the
    problems are listed live -- the same :func:`~fmri_gym.checks.rig_problems` a
    loaded file is held to, so the form cannot save a file the check would refuse.

    :param path: the rig file to write.
    :param values: what is known so far (an invalid file's content, or nothing).
    :return: the rig written, or ``None`` if the form was cancelled.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["fmri-gym"])
    app.setStyle("Fusion")
    form = _RigForm(path, values)
    if form.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None
    rig = form.rig
    with open(path, "w") as f:
        json.dump(rig, f, indent=2)
        f.write("\n")
    return rig


class _TriggerChoice(QtWidgets.QDialog):
    """The trigger settings of a config that has none: a preset, or one's own."""

    _PICK = "(pick a setup, or set your own below)"

    def __init__(self, path: str) -> None:
        super().__init__()
        self.setWindowTitle(f"Triggers: {path}")
        self.setMinimumWidth(620)
        self.touched = False
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            f"{path} has no trigger settings yet: how does this rig start the recording, and "
            "what does it send? Pick a setup, or set your own. They are saved in the file; "
            "the editor's Triggers tab changes them later, and the session's other runs "
            "must use the same.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.preset = QtWidgets.QComboBox()
        self.preset.addItems([self._PICK, *cfg.TRIGGER_PRESETS])
        self.preset.activated.connect(self._apply_preset)
        layout.addLayout(_row(QtWidgets.QLabel("setup"), self.preset))
        self.sync_form, self.trigger_form = _Form(gui.SYNC_FIELDS), _Form(gui.TRIGGER_FIELDS)
        self.sync_form.set({**asdict(SyncSettings()), "scanner_start": Codes().scanner_start})
        self.trigger_form.set(_trigger_defaults())
        for form in (self.sync_form, self.trigger_form):
            layout.addWidget(form)
            form.changed.connect(self._edited)
        self.text = QtWidgets.QLabel()
        self.text.setWordWrap(True)
        layout.addWidget(self.text)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Save
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.save = buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Save)
        layout.addWidget(buttons)
        self.touched = False  # filling the forms above is not a choice
        self._refresh()

    def _apply_preset(self) -> None:
        if self.preset.currentText() == self._PICK:
            return
        section = copy.deepcopy(cfg.TRIGGER_PRESETS[self.preset.currentText()])
        sync = {**asdict(SyncSettings()), **section.pop("sync"),
                "scanner_start": Codes().scanner_start}
        self.sync_form.set(sync)
        self.trigger_form.set({**_trigger_defaults(), **section})
        self.touched = True
        self._refresh()

    def _edited(self) -> None:
        self.touched = True
        self._refresh()

    def section(self) -> dict:
        """The triggers section the forms say (see :func:`fmri_gym.gui.triggers_section`)."""
        codes = {k: v for k, v in asdict(Codes()).items() if k != "scanner_start"}
        return gui.triggers_section(self.sync_form.get(), self.trigger_form.get(), codes)

    def _refresh(self) -> None:
        mode = self.sync_form.widgets["mode"].currentText()
        self.sync_form.show_rows(["mode", *gui.SYNC_ROWS.get(mode, ())])
        try:
            section = self.section()
            problems = cfg.trigger_problems(section)
        except ValueError as exc:  # a number half typed, named in the message
            section, problems = None, [str(exc)]
        except KeyError:  # a form half refilled by a preset: its next change completes it
            return
        if not self.touched:
            problems = ["pick a setup, or change a field: the defaults are not a choice"]
        self.text.setText("\n".join(problems) if problems else gui.describe_triggers(section))
        self.save.setEnabled(not problems)

    def accept(self) -> None:
        """Close with Save only when chosen and valid: a disabled button does not stop Enter."""
        if self.save.isEnabled():
            super().accept()


def choose_triggers(path: str) -> dict | None:
    """Ask for the trigger settings of a config that has none.

    :param path: the config, named in the dialog (the caller saves the section).
    :return: the ``triggers`` section chosen, or ``None`` if cancelled.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(["fmri-gym"])
    app.setStyle("Fusion")
    dialog = _TriggerChoice(path)
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None
    return dialog.section()


def _trigger_defaults() -> dict:
    """The backend form's defaults: :class:`TriggerSettings` minus its sub-sections."""
    d = asdict(TriggerSettings())
    for key in ("sync", "codes", "defaulted"):
        d.pop(key)
    return d


def _backend_defaults(phase: dict) -> tuple[dict[str, Any], float | None]:
    """Build the phase's adapter: its default key map (combos included) and its own rate.

    :return: ``({combo: action}, native fps)``; the rate is ``None`` for an
        engine with no clock of its own, which is what a blank ``fps`` then means.
    """
    from .adapters import get_adapter

    adapter = get_adapter(phase.get("backend", "gym"), phase)
    try:
        combos = adapter.keyspec.combos
        return ({"+".join(sorted(ks)): adapter.keyspec.resolve(ks) for ks in combos},
                adapter.native_fps())
    finally:
        adapter.close()
