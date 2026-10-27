"""The Qt pieces the fmri-gym windows share: forms, key tables, buttons, the style.

Qt rather than tkinter: the Tk that ships with uv's Python has no font
rendering to speak of (Tk 9 without Xft), and an editor people avoid is an
editor that does not replace hand-edited JSON. PySide6 rather than PyQt6
because it is LGPL. Only the windows import this module, and they are
imported only when a command opens one (:mod:`fmri_gym.gui`).
"""

from __future__ import annotations

import contextlib
import signal
from typing import Any, Callable, Iterator, Sequence

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

from . import Field, field_values, format_action, format_value, parse_action


#: Qt keys -> the key NAMES used in ``keys`` (see fmri_gym/keys.py); letters
#: and digits are taken from the event's text.
_QT_KEYS = {Qt.Key.Key_Up: "UP", Qt.Key.Key_Down: "DOWN", Qt.Key.Key_Left: "LEFT",
            Qt.Key.Key_Right: "RIGHT", Qt.Key.Key_Space: "SPACE", Qt.Key.Key_Return: "RETURN",
            Qt.Key.Key_Tab: "TAB", Qt.Key.Key_Shift: "LSHIFT", Qt.Key.Key_Comma: "COMMA",
            Qt.Key.Key_Period: "PERIOD"}


STYLE = """
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
QPushButton#run:disabled { background: palette(mid); color: palette(base); }
"""


def app() -> QtWidgets.QApplication:
    """The one QApplication, made and styled the first time a window needs it."""
    existing = QtWidgets.QApplication.instance()
    if existing is not None:
        return existing
    made = QtWidgets.QApplication(["fmri-gym"])
    made.setStyle("Fusion")
    made.setStyleSheet(STYLE)
    return made


@contextlib.contextmanager
def closed_by_ctrl_c(app: QtWidgets.QApplication) -> Iterator[list[bool]]:
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
    """A key press -> the key NAME a phase's ``keys`` use, or ``None`` if it is not one."""
    if event.key() in _QT_KEYS:
        return _QT_KEYS[event.key()]
    text = event.text()
    if len(text) == 1 and text.isascii() and text.isalnum():
        return text.upper()
    return None


def mono() -> QtGui.QFont:
    return QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)


def button(text: str, slot: Callable[[], Any]) -> QtWidgets.QPushButton:
    b = QtWidgets.QPushButton(text)
    b.clicked.connect(lambda _checked=False: slot())
    return b


def row(*widgets: QtWidgets.QWidget, stretch: bool = True) -> QtWidgets.QHBoxLayout:
    row = QtWidgets.QHBoxLayout()
    for w in widgets:
        row.addWidget(w)
    if stretch:
        row.addStretch(1)
    return row


def group(title: str, inner: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
    """``inner`` in a titled box."""
    box = QtWidgets.QGroupBox(title)
    QtWidgets.QVBoxLayout(box).addWidget(inner)
    return box


class Form(QtWidgets.QWidget):
    """One labelled row per :class:`~fmri_gym.gui.Field`; the tip is the tooltip."""

    #: Any row edited (typed, picked, ticked) -- also when :meth:`set` fills it.
    changed = QtCore.Signal()

    def __init__(self, fields: Sequence[Field]) -> None:
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

    def _make_widget(self, f: Field) -> Any:
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

    def _read(self, f: Field) -> Any:
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
            self.write(f.key, format_value(f, values.get(f.key)))

    def get(self) -> dict:
        """Read the form back as config values (see :func:`~fmri_gym.gui.field_values`)."""
        values = field_values(self.fields, {f.key: self._read(f) for f in self.fields})
        for key in self.as_lines & values.keys():
            values[key] = values[key].split("\n")
        return values


class KeyTable(QtWidgets.QTableWidget):
    """Editable rows of ``key combo -> action`` for one game phase."""

    def __init__(self) -> None:
        super().__init__(0, 2)
        self.setHorizontalHeaderLabels(["key(s)   e.g. UP or LEFT+SPACE; blank = no key held",
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
            self.add_row(key, format_action(action))
        self.setCurrentCell(-1, -1)

    def get(self) -> dict:
        """The game keys; a blank key is the no-key action, and an empty row is ignored.

        :raises ValueError: if a key has no action.
        """
        self.setCurrentCell(-1, -1)  # closes a cell still being typed in, keeping its text
        out = {}
        for row in range(self.rowCount()):
            key, action = (self.item(row, col).text().strip() for col in (0, 1))
            if not key and not action:
                continue
            if not action:
                raise ValueError(f"keys: {key.upper()} has no action")
            out[key.upper()] = parse_action(action)
        return out


class KeyCapture(QtWidgets.QDialog):
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
