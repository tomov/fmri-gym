"""``fmri_play.py --gui``: a config editor so a rig can be set up without JSON.

Five tabs over one config dict (:mod:`fmri_gym.config`): Session (the CLI
flags), Runs (the session's runs, each a phase list with one form per phase
type), Controls (the
``keys`` remap of a game phase, with the backend's defaults on request),
Triggers (sync + markers with fMRI/MEG/EEG presets, a code check and a
live marker test) and JSON (the whole file, editable). File > New / Open /
Save / Save As, and Run hands the config back to ``fmri_play``.

tkinter is the stdlib toolkit, so the editor adds no dependency; it is
imported inside :func:`edit_config` so this module stays importable on a
headless machine and the field tables and parsers above the ``# tk`` line
can be tested there. Field labels are the config keys and the (?) tip says
what each does, so the dialog and the file read alike.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from . import config as cfg
from .triggers import (MARKER_BACKENDS, SYNC_MODES, Codes, MarkerSettings, Markers,
                       SyncSettings, TriggerError)


# ---------------------------------------------------------------------------
# Field tables (pure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Field:
    """One labelled row of a form.

    ``kind``: ``str`` | ``int`` | ``float`` | ``bool`` | ``choice`` (fixed
    list) | ``combo`` (list plus free text) | ``text`` (multi-line string) |
    ``lines`` (multi-line, one list item per line). A blank entry means "not
    set", so the engine default applies; a ``bool`` is only written when it
    differs from ``default``.
    """

    key: str
    label: str
    kind: str = "str"
    choices: Sequence[str] = ()
    tip: str = ""
    default: Any = None


SESSION_FIELDS = [
    Field("subject", "subject", tip="Subject id; names the output folder (data/<subject>_<stamp>). "
                                    "Not written by Save: a config file describes the rig and "
                                    "the task, not one participant."),
    Field("outdir", "outdir", tip="Output directory. Blank = data/<subject>_<timestamp>."),
    Field("size", "size", "combo", ("1024x768", "1280x720", "1920x1080"),
          tip="Window size as <w>x<h>. Ignored in fullscreen, where the desktop resolution is used."),
    Field("fullscreen", "fullscreen", "bool", default=False,
          tip="Fill the screen. Windowed is handier for piloting at the desk."),
    Field("vsync", "vsync", "bool", default=True,
          tip="Lock flips to the monitor refresh for exact frame onsets. Turn off only if the "
              "display self-test (python -m fmri_gym.display) reports it cannot lock."),
    Field("dummy_trigger", "dummy_trigger", "bool", default=False,
          tip="Skip the experimenter and scanner waits. For testing only."),
]

SYNC_FIELDS = [
    Field("mode", "mode", "choice", SYNC_MODES,
          tip="After the experimenter's SPACE: wait = wait for 'key' from the trigger box (fMRI); "
              "send = send the scanner_start code on the marker line (MEG started from the "
              "trigger input); none = start immediately."),
    Field("key", "key", tip="wait only: the character the trigger box types (usually '=' or '5')."),
    Field("delay", "delay (s)", "float",
          tip="send only: seconds between the scanner_start code and the clock anchor."),
]

MARKER_FIELDS = [
    Field("backend", "backend", "choice", MARKER_BACKENDS,
          tip="null = no markers (fMRI). serial (pyserial), parallel (pyparallel) or lsl (pylsl). "
              "A backend that cannot be opened stops the run at the desk with the reason."),
    Field("port", "port", tip="serial/parallel device, e.g. /dev/ttyUSB0, COM3, /dev/parport0."),
    Field("lsl_stream_name", "lsl_stream_name", tip="lsl only: name of the marker stream."),
    Field("pulse_ms", "pulse_ms", "float",
          tip="parallel only: how long a code stays on the port between blocks "
              "(inside a block the next frame code replaces it)."),
    Field("on_frame", "on_frame", "bool", default=True,
          tip="Send a code on every displayed game frame (cycling 1..2^frame_bits-1)."),
    Field("frame_every", "frame_every", "int",
          tip="Send the frame code every N frames (1 = all). Thin it for slow marker links."),
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
    Field("scanner_start", "scanner_start", "int",
          tip="sync.mode 'send' only: the code that starts the acquisition."),
]

_GAME_FIELDS = [
    Field("backend", "backend", "combo", cfg.BACKENDS,
          tip="Game engine adapter (fmri_gym/adapters). Type another name for a custom one."),
    Field("game", "game", tip="Env id for that backend, e.g. ALE/Pong-v5, Airstriker-Genesis-v0, "
                              "CartPole-v1."),
    Field("mode", "mode", "choice", ("duration", "episode"),
          tip="duration = play (and replay) until the time is up; episode = play n_episodes."),
    Field("duration", "duration (s)", "float", tip="duration mode: block length in seconds."),
    Field("n_episodes", "n_episodes", "int", tip="episode mode: how many episodes."),
    Field("max_duration", "max_duration (s)", "float",
          tip="episode mode: hard wall-clock cap for the block."),
    Field("fps", "fps", "int", tip="Game frames per second shown (30 for Atari, 60 for consoles)."),
    Field("turn_based", "turn_based", "bool", default=False,
          tip="Step only on a key press instead of every frame (grid / text games)."),
    Field("seed", "seed", "int", tip="Base RNG seed. Blank = 1000 + phase index."),
    Field("state_stride", "state_stride", "int",
          tip="Save a full emulator savestate every K frames (1 = every frame)."),
    Field("text", "text", tip="Optional label shown on the loading screen instead of the env id."),
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
}

#: tkinter keysyms -> the key NAMES used in ``keys`` (see fmri_gym/keys.py).
_KEYSYMS = {"Up": "UP", "Down": "DOWN", "Left": "LEFT", "Right": "RIGHT", "space": "SPACE",
            "Return": "RETURN", "Tab": "TAB", "Shift_L": "LSHIFT", "comma": "COMMA",
            "period": "PERIOD"}


def parse_value(field: Field, raw: Any) -> Any:
    """Turn a widget's raw value into the config value (``None`` = not set).

    :raises ValueError: naming the field, when a number does not parse.
    """
    if field.kind == "bool":
        return bool(raw)
    text = str(raw)
    if not text.strip():
        return None
    if field.kind == "lines":
        return [line for line in text.split("\n") if line.strip()]
    if field.kind not in ("int", "float"):
        return text if field.kind == "text" else text.strip()
    try:
        return int(text) if field.kind == "int" else float(text)
    except ValueError as exc:
        raise ValueError(f"{field.label}: expected a number, got {text.strip()!r}") from exc


def format_value(field: Field, value: Any) -> Any:
    """The inverse of :func:`parse_value`: config value -> widget value."""
    if field.kind == "bool":
        return bool(field.default if value is None else value)
    if value is None:
        return ""
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


def phase_label(index: int, phase: dict) -> str:
    """One line for the curriculum list: ``"03  game  ale ALE/Pong-v5 (60 s)"``."""
    kind = phase.get("type", "?")
    if kind == "game":
        length = (f"{phase.get('duration', 30.0):g}s" if phase.get("mode", "duration") == "duration"
                  else f"{phase.get('n_episodes', 1)}ep")
        detail = f"{str(phase.get('game', '?')).split('/')[-1]} {length}"
    elif kind == "message":
        detail = str(phase.get("text", "")).split("\n")[0]
    elif kind == "survey":
        detail = f"{len(phase.get('questions', []))} questions"
    else:
        detail = f"{phase.get('duration', 2.0)} s"
    return f"{index:02d} {kind:<8} {detail[:14]}"


def parse_action(text: str) -> Any:
    """``"2"`` -> 2, ``"[1,0,0]"`` -> list, anything else -> the string itself."""
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        return text


def format_action(action: Any) -> str:
    """The inverse of :func:`parse_action`."""
    return action if isinstance(action, str) else json.dumps(action)


def key_name(keysym: str) -> str | None:
    """tkinter keysym -> KeySpec key NAME, or ``None`` if it is not a game key."""
    if keysym in _KEYSYMS:
        return _KEYSYMS[keysym]
    if len(keysym) == 1 and keysym.isalnum():
        return keysym.upper()
    return None


def split_phase(phase: dict, fields: Sequence[Field]) -> tuple[dict, dict]:
    """Split a phase into (form values, extra keys not on the form).

    ``type`` and ``keys`` belong to neither: the first is fixed, the second
    is edited on the Controls tab.
    """
    known = {f.key for f in fields} | {"type", "keys"}
    form = {k: v for k, v in phase.items() if k in known}
    extra = {k: v for k, v in phase.items() if k not in known}
    return form, extra


def subject_free(config: dict) -> dict:
    """A copy of ``config`` without ``session.subject`` (what Save writes).

    A config file describes a rig and a task; the participant comes from the
    editor or ``--subject`` at run time.
    """
    out = dict(config)
    session = {k: v for k, v in (config.get("session") or {}).items() if k != "subject"}
    out.pop("session", None)
    if session:
        out["session"] = session
    return out


def triggers_section(sync: dict, markers: dict, codes: dict) -> dict | None:
    """Assemble the ``triggers`` section, or ``None`` when it is all defaults.

    A default sync (wait for ``=``) with the ``null`` backend is what a
    missing section means, so nothing is written in that case.
    """
    section: dict = {}
    if sync and SyncSettings.from_dict(sync) != SyncSettings():
        section["sync"] = sync
    if markers.get("backend", "null") != "null":
        section["markers"] = dict(markers, codes=codes) if codes else markers
    return section or None


def describe_triggers(section: dict | None) -> str:
    """What a run will do on the trigger side, in a few plain lines.

    :raises ValueError: if the section does not validate.
    """
    sync = SyncSettings.from_dict((section or {}).get("sync"))
    m = MarkerSettings.from_dict((section or {}).get("markers"))
    where = m.lsl_stream_name if m.backend == "lsl" else m.port
    lines = ["After the experimenter's SPACE:"]
    if sync.mode == "wait":
        lines.append(f"  wait for {sync.key!r} from the trigger box, then anchor the clock (t=0).")
    elif sync.mode == "send":
        lines.append(f"  send scanner_start={m.codes.scanner_start} on {m.backend} {where}, "
                     f"wait {sync.delay} s, then anchor the clock (t=0).")
    else:
        lines.append("  anchor the clock (t=0) immediately.")
    if m.backend == "null":
        lines.append("Markers: none (backend null); only the log files carry timing.")
        return "\n".join(lines)
    frames = (f"frame codes 1..{m.codes.frame_mask} every {m.frame_every} frame(s)"
              if m.on_frame else "no frame codes")
    lines.append(f"Markers on {m.backend} {where}: task_start={m.codes.task_start}, "
                 f"episode_start={m.codes.episode_start if m.on_episode_start else 'off'}, "
                 f"{frames}, task_stop={m.codes.task_stop}.")
    lines.append(f"Codes share no bits: episode_start during frame code 2 is sent as "
                 f"{m.codes.episode_start | 2}.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# tk
# ---------------------------------------------------------------------------


def edit_config(config: dict, path: str | None = None) -> tuple[dict, str | None] | None:
    """Open the editor on ``config``; return what to run, or ``None``.

    :param config: full config dict (with a resolved ``session`` section).
    :param path: file it came from, for the title and Save.
    :return: ``(config, only)`` when Run is pressed -- ``only`` is a 1-based
        run index as text, or ``None`` for every run; ``None`` when closed.
    """
    import tkinter as tk

    root = tk.Tk()
    editor = _Editor(root, config, path)
    root.mainloop()
    return editor.result


class _Tooltip:
    """Hover text on a widget (a (?) label)."""

    def __init__(self, widget: Any, text: str) -> None:
        import tkinter as tk

        self.tk, self.widget, self.text, self.tip = tk, widget, text, None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event: Any) -> None:
        if not self.text or self.tip:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + 16
        self.tip = self.tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        self.tk.Label(self.tip, text=self.text, justify="left", wraplength=360,
                      background="#ffffe0", relief="solid", borderwidth=1, padx=6,
                      pady=4).pack()

    def _hide(self, _event: Any) -> None:
        if self.tip:
            self.tip.destroy()
            self.tip = None


class _Form:
    """A grid of :class:`Field` rows: label, widget, (?) tip."""

    def __init__(self, parent: Any, fields: Sequence[Field]) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk, self.fields = tk, ttk, list(fields)
        self.vars: dict[str, Any] = {}
        self.widgets: dict[str, Any] = {}
        for row, f in enumerate(self.fields):
            ttk.Label(parent, text=f.label).grid(row=row, column=0, sticky="nw", padx=4, pady=1)
            w = self._make_widget(parent, f)
            w.grid(row=row, column=1, sticky="ew", padx=4, pady=1)
            info = ttk.Label(parent, text="(?)", foreground="#3465a4", cursor="question_arrow")
            info.grid(row=row, column=2, sticky="n", padx=(0, 4))
            _Tooltip(info, f.tip)
        parent.columnconfigure(1, weight=1)

    def _make_widget(self, parent: Any, f: Field) -> Any:
        tk, ttk = self.tk, self.ttk
        var: Any = None
        if f.kind == "bool":
            var = tk.BooleanVar(value=bool(f.default))
            w = ttk.Checkbutton(parent, variable=var)
        elif f.kind in ("choice", "combo"):
            var = tk.StringVar()
            w = ttk.Combobox(parent, textvariable=var, values=list(f.choices),
                             width=10 if f.kind == "choice" else 24,
                             state="readonly" if f.kind == "choice" else "normal")
        elif f.kind in ("text", "lines"):
            w = _text(parent, height=4, width=40, undo=True)
        else:
            var = tk.StringVar()
            w = ttk.Entry(parent, textvariable=var, width=8 if f.kind in ("int", "float") else 26)
        self.vars[f.key], self.widgets[f.key] = var, w
        return w

    def set(self, values: dict) -> None:
        """Fill the form from a dict (missing keys show as blank/default)."""
        for f in self.fields:
            raw = format_value(f, values.get(f.key))
            if self.vars[f.key] is None:
                self.widgets[f.key].delete("1.0", "end")
                self.widgets[f.key].insert("1.0", raw)
            else:
                self.vars[f.key].set(raw)

    def get(self) -> dict:
        """Read the form back as config values (see :func:`field_values`)."""
        raw = {f.key: (self.widgets[f.key].get("1.0", "end-1c") if self.vars[f.key] is None
                       else self.vars[f.key].get()) for f in self.fields}
        return field_values(self.fields, raw)


class _KeyTable:
    """Editable rows of ``key combo -> action`` for one game phase."""

    def __init__(self, parent: Any) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk, self.parent = tk, ttk, parent
        self.rows: list[tuple[Any, Any, Any]] = []
        ttk.Label(parent, text="key(s)  e.g. UP or LEFT+SPACE").grid(row=0, column=0, sticky="w")
        ttk.Label(parent, text="action  e.g. 2 or [1,0,0]").grid(row=0, column=1, sticky="w")

    def add_row(self, key: str = "", action: str = "") -> None:
        kvar, avar = self.tk.StringVar(value=key), self.tk.StringVar(value=action)
        row = len(self.rows) + 1
        widgets = (self.ttk.Entry(self.parent, textvariable=kvar, width=18),
                   self.ttk.Entry(self.parent, textvariable=avar, width=18),
                   self.ttk.Button(self.parent, text="x", width=2,
                                   command=lambda: self._remove(kvar)))
        for col, w in enumerate(widgets):
            w.grid(row=row, column=col, sticky="ew", padx=2, pady=1)
        self.rows.append((kvar, avar, widgets))
        widgets[0 if not key else 1].focus_set()

    def _remove(self, kvar: Any) -> None:
        keep = []
        for row in self.rows:
            if row[0] is kvar:
                for w in row[2]:
                    w.destroy()
                continue
            keep.append(row)
        self.rows = keep

    def set(self, keys: dict) -> None:
        for row in self.rows:
            for w in row[2]:
                w.destroy()
        self.rows = []
        for k, a in keys.items():
            self.add_row(k, format_action(a))

    def get(self) -> dict:
        """The bindings; rows with a blank key are ignored.

        :raises ValueError: if a key has no action.
        """
        out = {}
        for kvar, avar, _ in self.rows:
            key, action = kvar.get().strip().upper(), avar.get().strip()
            if not key:
                continue
            if not action:
                raise ValueError(f"keys: {key} has no action")
            out[key] = parse_action(action)
        return out


class _Editor:
    """The window: menu, notebook, bottom bar. ``result`` is set by Run."""

    def __init__(self, root: Any, config: dict, path: str | None) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk, self.ttk, self.root = tk, ttk, root
        self.path, self.result = path, None
        self.runs: list[dict] = []      # [{"name", "curriculum"}, ...]
        self.run_index = 0
        self.multi = False              # write "runs" (else a single "curriculum")
        self.phases: list[dict] = []    # alias of self.runs[self.run_index]["curriculum"]
        self.notes: dict = {}  # top-level keys the editor does not own (e.g. "_note")
        self.edit_index: int | None = None
        self.ctl_index: int | None = None
        self.phase_form = self.extra_text = None
        root.title("fmri-gym config")
        root.geometry("1100x900")
        _style(root)
        self._build_menu()
        self._build_bottom()
        self.nb = ttk.Notebook(root)
        self.nb.pack(fill="both", expand=True, padx=8, pady=(8, 4))
        self._tab_session()
        self._tab_runs()
        self._tab_controls()
        self._tab_triggers()
        self._tab_json()
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_change)
        self.populate(config)

    # -- construction ------------------------------------------------------

    def _build_menu(self) -> None:
        menu = self.tk.Menu(self.root)
        file_menu = self.tk.Menu(menu, tearoff=0)
        for label, cmd, acc in (("New", self._new, "Ctrl+N"), ("Open...", self._open, "Ctrl+O"),
                                ("Save", self._save, "Ctrl+S"), ("Save As...", self._save_as, ""),
                                (None, None, ""), ("Quit", self.root.destroy, "")):
            if label is None:
                file_menu.add_separator()
                continue
            file_menu.add_command(label=label, command=cmd, accelerator=acc)
        menu.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menu)
        self.root.bind("<Control-n>", lambda e: self._new())
        self.root.bind("<Control-o>", lambda e: self._open())
        self.root.bind("<Control-s>", lambda e: self._save())

    def _build_bottom(self) -> None:
        bar = self.ttk.Frame(self.root, padding=(8, 4))
        bar.pack(side="bottom", fill="x")
        for text, cmd in (("New", self._new), ("Open...", self._open), ("Save", self._save),
                          ("Save As...", self._save_as)):
            self.ttk.Button(bar, text=text, command=cmd).pack(side="left", padx=(0, 4))
        self.ttk.Button(bar, text="Run", command=self._run, style="Accent.TButton").pack(
            side="right")
        self.run_pick = self.ttk.Combobox(bar, state="readonly", width=22)
        self.run_pick.pack(side="right", padx=4)
        self.ttk.Button(bar, text="Check", command=self._check).pack(side="right", padx=4)
        self.status = self.ttk.Label(self.root, text="", padding=(8, 2), anchor="w",
                                     style="Status.TLabel")
        self.status.pack(side="bottom", fill="x")

    def _page(self, title: str) -> Any:
        page = self.ttk.Frame(self.nb, padding=12)
        self.nb.add(page, text=title)
        return page

    def _tab_session(self) -> None:
        page = self._page("Session")
        self.session_form = _Form(page, SESSION_FIELDS)
        self.ttk.Button(page, text="...", width=3, command=self._pick_outdir).grid(
            row=1, column=3, padx=2)

    def _run_bar(self, parent: Any, with_buttons: bool) -> Any:
        """A row with the run picker (and, on the Runs tab, name + run buttons)."""
        bar = self.ttk.Frame(parent)
        bar.pack(side="top", fill="x", pady=(0, 8))
        self.ttk.Label(bar, text="run").pack(side="left")
        pick = self.ttk.Combobox(bar, state="readonly", width=26)
        pick.pack(side="left", padx=6)
        pick.bind("<<ComboboxSelected>>", lambda e: self._select_run(pick.current()))
        if not with_buttons:
            return pick
        self.ttk.Label(bar, text="name").pack(side="left", padx=(12, 0))
        self.run_name = self.tk.StringVar()
        self.ttk.Entry(bar, textvariable=self.run_name, width=16).pack(side="left", padx=6)
        for text, cmd in (("Add run", self._add_run), ("Duplicate", self._duplicate_run),
                          ("Remove", self._remove_run), ("Up", lambda: self._move_run(-1)),
                          ("Down", lambda: self._move_run(1))):
            self.ttk.Button(bar, text=text, command=cmd).pack(side="left", padx=1)
        return pick

    def _tab_runs(self) -> None:
        page = self._page("Runs")
        self.run_pick_a = self._run_bar(page, with_buttons=True)
        self.phase_list = self.tk.Listbox(page, width=26, height=22, exportselection=False,
                                          font="TkFixedFont", relief="flat", highlightthickness=1,
                                          highlightbackground="#c0c0c0", activestyle="none",
                                          selectbackground="#3465a4", selectforeground="white")
        self.phase_list.pack(side="left", fill="y")
        self.phase_list.bind("<<ListboxSelect>>", lambda e: self._select_phase())
        buttons = self.ttk.Frame(page)
        buttons.pack(side="left", fill="y", padx=4)
        add = self.ttk.Menubutton(buttons, text="Add")
        add.menu = self.tk.Menu(add, tearoff=0)
        add["menu"] = add.menu
        for kind in cfg.PHASE_TYPES:
            add.menu.add_command(label=kind, command=lambda k=kind: self._add_phase(k))
        add.pack(fill="x", pady=1)
        for text, cmd in (("Duplicate", self._duplicate_phase), ("Remove", self._remove_phase),
                          ("Move up", lambda: self._move_phase(-1)),
                          ("Move down", lambda: self._move_phase(1))):
            self.ttk.Button(buttons, text=text, command=cmd).pack(fill="x", pady=1)
        self.phase_pane = self.ttk.LabelFrame(page, text="phase", padding=8)
        self.phase_pane.pack(side="left", fill="both", expand=True, padx=(4, 0))

    def _tab_controls(self) -> None:
        page = self._page("Controls")
        self.run_pick_b = self._run_bar(page, with_buttons=False)
        top = self.ttk.Frame(page)
        top.pack(fill="x")
        self.ttk.Label(top, text="game phase").pack(side="left")
        self.ctl_phase = self.ttk.Combobox(top, state="readonly", width=48)
        self.ctl_phase.pack(side="left", padx=6)
        self.ctl_phase.bind("<<ComboboxSelected>>", lambda e: self._select_ctl_phase())
        hint = ("Overrides of the backend's default keyboard map for this phase. Keys are "
                "pygame names (UP, DOWN, LEFT, RIGHT, SPACE, RETURN, A-Z, 0-9), joined with + "
                "for combos; the action is what the env expects (an int for Discrete). "
                "Unmentioned keys keep the backend default.")
        self.ttk.Label(page, text=hint, wraplength=760, justify="left").pack(
            fill="x", pady=(6, 8))
        table = self.ttk.Frame(page)
        table.pack(fill="x")
        self.key_table = _KeyTable(table)
        row = self.ttk.Frame(page)
        row.pack(fill="x", pady=6)
        self.ttk.Button(row, text="Add binding", command=self.key_table.add_row).pack(side="left")
        self.ttk.Button(row, text="Press a key...", command=self._capture_key).pack(
            side="left", padx=4)
        self.ttk.Button(row, text="Show backend defaults", command=self._show_defaults).pack(
            side="left", padx=4)
        self.ttk.Button(row, text="Use defaults as bindings", command=self._use_defaults).pack(
            side="left")
        self.defaults_text = _text(page, height=8, state="disabled")
        self.defaults_text.pack(fill="both", expand=True)
        self._defaults: dict = {}

    def _tab_triggers(self) -> None:
        page = self._page("Triggers")
        self.trigger_text = _text(page, height=3, state="disabled")
        self.trigger_text.pack(side="bottom", fill="x")
        row = self.ttk.Frame(page)
        row.pack(side="bottom", fill="x", pady=6)
        self.ttk.Button(row, text="Check", command=self._refresh_triggers).pack(side="left")
        self.ttk.Button(row, text="Test markers (send task_start)",
                        command=self._test_markers).pack(side="left", padx=6)
        top = self.ttk.Frame(page)
        top.pack(fill="x", pady=(0, 6))
        self.ttk.Label(top, text="preset").pack(side="left")
        self.preset = self.ttk.Combobox(top, state="readonly", width=40,
                                        values=list(cfg.TRIGGER_PRESETS))
        self.preset.pack(side="left", padx=6)
        self.ttk.Button(top, text="Apply preset", command=self._apply_preset).pack(side="left")
        forms = self.ttk.Frame(page)
        forms.pack(fill="x")
        self.sync_form = _Form(self._group(forms, "sync (run start)", 0, 0), SYNC_FIELDS)
        self.code_form = _Form(self._group(forms, "codes", 1, 0), CODE_FIELDS)
        self.marker_form = _Form(self._group(forms, "markers", 0, 1, rowspan=2), MARKER_FIELDS)

    def _group(self, parent: Any, title: str, row: int, column: int, rowspan: int = 1) -> Any:
        box = self.ttk.LabelFrame(parent, text=title, padding=6)
        box.grid(row=row, column=column, rowspan=rowspan, sticky="nsew", padx=4, pady=2)
        parent.columnconfigure(column, weight=1)
        return box

    def _tab_json(self) -> None:
        page = self._page("JSON")
        row = self.ttk.Frame(page)
        row.pack(fill="x", pady=(0, 6))
        self.ttk.Button(row, text="Refresh from form", command=self._refresh_json).pack(
            side="left")
        self.ttk.Button(row, text="Apply to form", command=self._apply_json).pack(
            side="left", padx=6)
        self.json_text = _text(page, wrap="none", undo=True, font="TkFixedFont")
        self.json_text.pack(fill="both", expand=True)

    # -- model <-> widgets -------------------------------------------------

    def populate(self, config: dict) -> None:
        """Load a whole config into every tab."""
        self.notes = {k: v for k, v in config.items()
                      if k not in ("session", "triggers", "curriculum", "runs")}
        self.session_form.set(cfg.resolve_session(config))
        self.runs = copy.deepcopy(cfg.runs_of(config)) or [{"name": "", "curriculum": []}]
        self.multi = "runs" in config
        self.edit_index = self.ctl_index = None
        self._show_run(0)
        section = config.get("triggers") or {}
        markers = dict(section.get("markers") or {})
        codes = markers.pop("codes", {})
        self.sync_form.set({**asdict(SyncSettings()), **(section.get("sync") or {})})
        self.marker_form.set({**_marker_defaults(), **markers})
        self.code_form.set({**asdict(Codes()), **codes})
        self._refresh_ctl_phases()
        self._refresh_triggers()
        self._refresh_json()
        self._set_status()

    def collect(self) -> dict:
        """Read every tab back into one config dict.

        :raises ValueError: naming the offending field.
        """
        self._commit_phase()
        self._commit_keys()
        config: dict = {**copy.deepcopy(self.notes), "session": self.session_form.get()}
        section = triggers_section(self.sync_form.get(), self.marker_form.get(),
                                   self.code_form.get())
        if section:
            config["triggers"] = section
        if self.multi or len(self.runs) > 1:
            config["runs"] = copy.deepcopy(self.runs)
        else:
            config["curriculum"] = copy.deepcopy(self.phases)
        return config

    def _set_status(self, text: str = "") -> None:
        self.status.config(text=text or (self.path or "unsaved config"))

    def _commit_all(self) -> bool:
        """Push both editors into ``self.phases``; ``False`` (after a dialog) on a bad field."""
        try:
            self._commit_phase()
            self._commit_keys()
        except ValueError as exc:
            self._error(str(exc))
            return False
        self.runs[self.run_index]["name"] = self.run_name.get().strip()
        return True

    def _on_tab_change(self, _event: Any) -> None:
        if not self._commit_all():
            return
        self._refresh_run_pickers()
        self._refresh_ctl_phases()
        self._refresh_triggers()
        self._refresh_json()

    # -- runs ----------------------------------------------------------------

    def _run_labels(self) -> list[str]:
        return [f"{i + 1}  {r['name'] or 'run'}  ({len(r['curriculum'])} phases)"
                for i, r in enumerate(self.runs)]

    def _refresh_run_pickers(self) -> None:
        labels = self._run_labels()
        for pick in (self.run_pick_a, self.run_pick_b):
            pick["values"] = labels
            pick.current(self.run_index)
        self.run_pick["values"] = ["all runs"] + labels
        if self.run_pick.get() not in self.run_pick["values"]:
            self.run_pick.current(0)

    def _show_run(self, index: int) -> None:
        """Point the phase editors at run ``index`` (no commit; see _select_run)."""
        self.run_index = index
        self.phases = self.runs[index]["curriculum"]
        self.run_name.set(self.runs[index]["name"])
        self.edit_index = self.ctl_index = None
        self._refresh_phase_list(0 if self.phases else None)
        self._refresh_run_pickers()
        self._refresh_ctl_phases()

    def _select_run(self, index: int) -> None:
        if index == self.run_index or not self._commit_all():
            self._refresh_run_pickers()
            return
        self._show_run(index)

    def _add_run(self) -> None:
        self._insert_run({"name": "", "curriculum": [{"type": "fixation", "duration": 2.0}]})

    def _duplicate_run(self) -> None:
        if self._commit_all():
            self._insert_run(copy.deepcopy(self.runs[self.run_index]))

    def _insert_run(self, run: dict) -> None:
        if not self._commit_all():
            return
        self.multi = True
        self.runs.insert(self.run_index + 1, run)
        self._show_run(self.run_index + 1)

    def _remove_run(self) -> None:
        if len(self.runs) < 2 or not self._commit_all():
            return
        del self.runs[self.run_index]
        self._show_run(min(self.run_index, len(self.runs) - 1))

    def _move_run(self, step: int) -> None:
        i = self.run_index
        if not 0 <= i + step < len(self.runs) or not self._commit_all():
            return
        self.runs[i], self.runs[i + step] = self.runs[i + step], self.runs[i]
        self._show_run(i + step)

    # -- curriculum (phases of the current run) ------------------------------

    def _refresh_phase_list(self, select: int | None) -> None:
        self.phase_list.delete(0, "end")
        for i, phase in enumerate(self.phases):
            self.phase_list.insert("end", phase_label(i, phase))
        if select is None:
            self._show_phase(None)
            return
        self.phase_list.selection_clear(0, "end")
        self.phase_list.selection_set(select)
        self.phase_list.see(select)
        self._show_phase(select)

    def _select_phase(self) -> None:
        picked = self.phase_list.curselection()
        if not picked or picked[0] == self.edit_index:
            return
        if not self._commit_all():
            self.phase_list.selection_clear(0, "end")
            if self.edit_index is not None:
                self.phase_list.selection_set(self.edit_index)
            return
        self._show_phase(picked[0])

    def _show_phase(self, index: int | None) -> None:
        for child in self.phase_pane.winfo_children():
            child.destroy()
        self.edit_index = index
        self.phase_form = self.extra_text = None
        if index is None:
            self.phase_pane.config(text="phase")
            return
        phase = self.phases[index]
        kind = phase.get("type", "game")
        self.phase_pane.config(text=f"phase {index:02d}: {kind}")
        fields = PHASE_FIELDS.get(kind, [])
        form, extra = split_phase(phase, fields)
        self.phase_form = _Form(self.phase_pane, fields)
        self.phase_form.set(form)
        if kind != "game":
            return
        n = len(fields)
        keys = ", ".join(f"{k}={format_action(v)}" for k, v in phase.get("keys", {}).items())
        self.ttk.Label(self.phase_pane, text="keys").grid(row=n, column=0, sticky="nw", padx=4)
        self.ttk.Label(self.phase_pane, text=(keys or "backend defaults") + "  (Controls tab)",
                       wraplength=300, justify="left").grid(row=n, column=1, columnspan=2,
                                                            sticky="w")
        self.ttk.Label(self.phase_pane, text="extra (JSON)").grid(
            row=n + 1, column=0, sticky="nw", padx=4, pady=(6, 0))
        self.extra_text = _text(self.phase_pane, height=4, width=40, undo=True)
        self.extra_text.grid(row=n + 1, column=1, sticky="ew", padx=4, pady=(6, 0))
        self.extra_text.insert("1.0", json.dumps(extra, indent=1) if extra else "{}")
        info = self.ttk.Label(self.phase_pane, text="(?)", foreground="#3465a4")
        info.grid(row=n + 1, column=2, sticky="n")
        _Tooltip(info, "Backend-specific fields as JSON, e.g. retro \"state\"/\"scenario\", "
                       "vgdl \"level\", vizdoom \"env_kwargs\", ale \"save_pixels\".")

    def _commit_phase(self) -> None:
        if self.edit_index is None or self.phase_form is None:
            return
        old = self.phases[self.edit_index]
        new = {"type": old.get("type", "game"), **self.phase_form.get()}
        if "keys" in old:
            new["keys"] = old["keys"]
        if self.extra_text is not None:
            try:
                extra = json.loads(self.extra_text.get("1.0", "end-1c") or "{}")
            except ValueError as exc:
                raise ValueError(f"other fields: not valid JSON ({exc})") from exc
            if not isinstance(extra, dict):
                raise ValueError("other fields: must be a JSON object")
            new.update(extra)
        self.phases[self.edit_index] = new
        self.phase_list.delete(self.edit_index)
        self.phase_list.insert(self.edit_index, phase_label(self.edit_index, new))
        self.phase_list.selection_set(self.edit_index)

    def _add_phase(self, kind: str) -> None:
        templates = {"fixation": {"type": "fixation", "duration": 2.0},
                     "message": {"type": "message", "text": "", "duration": 2.0},
                     "survey": {"type": "survey", "n_points": 7, "questions": []},
                     "game": {"type": "game", "backend": "gym", "game": "", "mode": "duration",
                              "duration": 30.0, "fps": 30}}
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
        return [i for i, p in enumerate(self.phases) if p.get("type") == "game"]

    def _refresh_ctl_phases(self) -> None:
        indices = self._game_indices()
        self.ctl_phase["values"] = [phase_label(i, self.phases[i]) for i in indices]
        if self.ctl_index not in indices:
            self.ctl_index = indices[0] if indices else None
            self.key_table.set(self.phases[self.ctl_index].get("keys", {})
                               if self.ctl_index is not None else {})
        if self.ctl_index is not None:
            self.ctl_phase.current(indices.index(self.ctl_index))

    def _select_ctl_phase(self) -> None:
        try:
            self._commit_keys()
        except ValueError as exc:
            self._error(str(exc))
            return
        self.ctl_index = self._game_indices()[self.ctl_phase.current()]
        self.key_table.set(self.phases[self.ctl_index].get("keys", {}))
        self._show_text(self.defaults_text, "")

    def _commit_keys(self) -> None:
        if self.ctl_index is None or self.ctl_index >= len(self.phases):
            return
        keys = self.key_table.get()
        phase = self.phases[self.ctl_index]
        phase.pop("keys", None)
        if keys:
            phase["keys"] = keys
        if self.edit_index == self.ctl_index and self.phase_form is not None:
            self._show_phase(self.edit_index)

    def _capture_key(self) -> None:
        win = self.tk.Toplevel(self.root)
        win.title("press a key")
        self.tk.Label(win, text="Press the key to bind (Esc cancels)", padx=24, pady=24).pack()
        win.grab_set()
        win.focus_force()

        def on_key(event: Any) -> None:
            win.destroy()
            name = key_name(event.keysym)
            if name is not None:
                self.key_table.add_row(name, "")
                return
            if event.keysym != "Escape":
                self._error(f"{event.keysym!r} is not a key the games can read")

        win.bind("<KeyPress>", on_key)

    def _show_defaults(self) -> None:
        if self.ctl_index is None:
            return
        phase = {k: v for k, v in self.phases[self.ctl_index].items() if k != "keys"}
        self._show_text(self.defaults_text, f"loading {phase.get('game')}...")
        self.root.update_idletasks()
        try:
            self._defaults = _backend_defaults(phase)
        except Exception as exc:  # any engine error: show it, keep the editor alive
            self._defaults = {}
            self._show_text(self.defaults_text, f"could not build the adapter: {exc}")
            return
        body = "\n".join(f"{k:<16} {format_action(v)}" for k, v in self._defaults.items())
        self._show_text(self.defaults_text, f"{phase.get('backend')} defaults for "
                                            f"{phase.get('game')}:\n{body}")

    def _use_defaults(self) -> None:
        if not self._defaults:
            self._show_defaults()
        if self._defaults:
            self.key_table.set(self._defaults)

    # -- triggers tab ------------------------------------------------------

    def _apply_preset(self) -> None:
        name = self.preset.get()
        if not name:
            return
        section = copy.deepcopy(cfg.TRIGGER_PRESETS[name]) or {}
        self.sync_form.set({**asdict(SyncSettings()), **section.get("sync", {})})
        self.marker_form.set({**_marker_defaults(), **section.get("markers", {})})
        self._refresh_triggers()

    def _current_triggers(self) -> dict | None:
        return triggers_section(self.sync_form.get(), self.marker_form.get(),
                                self.code_form.get())

    def _refresh_triggers(self) -> None:
        try:
            section = self._current_triggers()
            problems = cfg.trigger_problems(section)
            text = "\n".join(problems) if problems else describe_triggers(section)
        except (ValueError, TypeError) as exc:
            text = str(exc)
        self._show_text(self.trigger_text, text)

    def _test_markers(self) -> None:
        try:
            section = self._current_triggers() or {}
            markers = Markers.from_config(section.get("markers"))
            if not markers.enabled:
                raise TriggerError("markers backend is null; pick serial, parallel or lsl first")
            value = markers.lifecycle("task_start")
            markers.close()
        except (TriggerError, ValueError) as exc:
            self._show_text(self.trigger_text, f"test failed: {exc}")
            return
        self._show_text(self.trigger_text, f"sent {value} on {markers.active}; check that the "
                                           "recording shows one marker.")

    # -- json tab ----------------------------------------------------------

    def _refresh_json(self) -> None:
        try:
            config = self.collect()
        except ValueError as exc:
            self._error(str(exc))
            return
        self.json_text.delete("1.0", "end")
        self.json_text.insert("1.0", json.dumps(config, indent=2))

    def _apply_json(self) -> None:
        try:
            config = json.loads(self.json_text.get("1.0", "end-1c"))
        except ValueError as exc:
            self._error(f"not valid JSON: {exc}")
            return
        if isinstance(config, list):
            config = {"curriculum": config}
        self.populate(config)

    # -- file / run ----------------------------------------------------------

    def _new(self) -> None:
        self.path = None
        self.populate(cfg.new_config())

    def _open(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("all", "*")],
                                          initialdir="configs")
        if not path:
            return
        try:
            config = cfg.load_config(path)
        except (OSError, ValueError) as exc:
            self._error(f"cannot open {path}: {exc}")
            return
        self.path = path
        self.populate(config)

    def _save(self) -> None:
        if self.path is None:
            self._save_as()
            return
        try:
            cfg.save_config(subject_free(self.collect()), self.path)
        except (OSError, ValueError) as exc:
            self._error(str(exc))
            return
        self._set_status(f"saved {self.path} (subject is not written to the file)")

    def _save_as(self) -> None:
        from tkinter import filedialog

        path = filedialog.asksaveasfilename(defaultextension=".json", initialdir="configs",
                                            filetypes=[("JSON", "*.json")])
        if path:
            self.path = path
            self._save()

    def _pick_outdir(self) -> None:
        from tkinter import filedialog

        path = filedialog.askdirectory()
        if path:
            self.session_form.vars["outdir"].set(path)

    def _check(self) -> list[str]:
        try:
            problems = cfg.validate_config(self.collect())
        except ValueError as exc:
            problems = [str(exc)]
        self._set_status("; ".join(problems) if problems else "config looks runnable")
        return problems

    def _run(self) -> None:
        if self._check():
            self.nb.select(0)
            return
        only = self.run_pick.current()
        self.result = (self.collect(), str(only) if only > 0 else None)
        self.root.destroy()

    # -- small helpers -------------------------------------------------------

    def _show_text(self, widget: Any, text: str) -> None:
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.config(state="disabled")

    def _error(self, text: str) -> None:
        from tkinter import messagebox

        messagebox.showerror("fmri-gym config", text, parent=self.root)


def _marker_defaults() -> dict:
    d = asdict(MarkerSettings())
    d.pop("codes")
    d["port"] = ""
    return d


def _backend_defaults(phase: dict) -> dict[str, Any]:
    """Build the phase's adapter and read its default key map (combos included)."""
    from .adapters import get_adapter

    adapter = get_adapter(phase.get("backend", "gym"), phase)
    try:
        combos = adapter.keyspec.combos
        return {"+".join(sorted(ks)): adapter.keyspec.resolve(ks) for ks in combos}
    finally:
        adapter.close()


def _style(root: Any) -> None:
    """A flat, consistent look: the ``clam`` theme, a blue accent, even padding.

    The system fonts are left alone: overriding the family makes Tk fall back
    to a symbol font for non-ASCII glyphs on some X servers, so labels stay ASCII.
    """
    from tkinter import ttk

    style = ttk.Style(root)
    style.theme_use("clam")
    bg, accent = "#f4f4f4", "#3465a4"
    root.configure(background=bg)
    style.configure(".", background=bg)
    style.configure("TNotebook", tabmargins=(4, 4, 4, 0))
    style.configure("TNotebook.Tab", padding=(14, 6))
    style.configure("TLabelframe", padding=6)
    style.configure("TLabelframe.Label", foreground="#555555")
    style.configure("TButton", padding=(10, 4))
    style.configure("Accent.TButton", foreground="white", background=accent,
                    bordercolor=accent, lightcolor=accent, darkcolor=accent)
    style.map("Accent.TButton", background=[("active", "#2a5387"), ("pressed", "#1f3f68")])
    style.configure("Status.TLabel", background="#e8e8e8", foreground="#333333")
    style.configure("TEntry", padding=3)
    # Tk 8.6 does not scale these indicators with the screen DPI; do it by hand.
    px = round(6 * float(root.tk.call("tk", "scaling")))
    style.configure("TCombobox", padding=3, arrowsize=px * 2)
    style.configure("TCheckbutton", indicatorsize=px * 2, indicatormargin=(2, 2, 6, 2))


def _text(parent: Any, **kw: Any) -> Any:
    """A ``tk.Text`` that matches the ttk widgets (flat, thin border, wrapped)."""
    import tkinter as tk

    kw.setdefault("wrap", "word")
    return tk.Text(parent, relief="flat", highlightthickness=1, highlightbackground="#c0c0c0",
                   highlightcolor="#3465a4", padx=6, pady=4, **kw)
