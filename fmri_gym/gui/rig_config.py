"""``rig-config``: make, check and change a rig file, one step at a time, and run its rig checks.

A rig file (:mod:`fmri_gym.config.rig`) says how one rig plays every run: what
the rig is, its screen, sound and data folder, how the participant responds,
its triggers, and the equipment its rig check measures with. The window walks
through these in order; each step must be complete before the next, and a rig
check can only be run on the file once it is saved. The rig check opens the
same window on a rig whose file is missing or incomplete
(:func:`fmri_gym.checks.ensure_rig`).
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Any, Callable

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt

from ..config import rig as rig_file
from ..display import list_monitors, monitor_label
from ..keys import KEY_NAMES
from ..pad import CONTROL_LABELS, CONTROLS
from ..triggers import (HID_DEVICES, SYNC_MODES, TRIGGER_BACKENDS, Codes, SyncSettings,
                        TriggerError, Triggers, TriggerSettings)
from . import Field
from . import widgets as ui

SYNC_FIELDS = [
    Field("mode", "how a run starts", "choice", SYNC_MODES,
          tip="What happens after the experimenter presses SPACE. wait: the run waits for the "
              "key your scanner's trigger box types at each volume (fMRI). send: the run sends "
              "a start code on the trigger line, which starts your recording (MEG, EEG). none: "
              "the run starts at once (no scanner)."),
    Field("key", "trigger key", tip="The key your trigger box types at each volume, usually = "
                                    "or 5. The run starts at the first one."),
    Field("scanner_start", "start code", "int",
          tip="The code sent on the trigger line to start the recording. It must share no bit "
              "with the other codes."),
    Field("delay", "delay (s)", "float",
          tip="Seconds to wait after the start code before the run starts, for a recording "
              "that needs a moment to come up. 0 if it starts at once."),
]

#: Which of the start fields each way of starting uses.
SYNC_ROWS = {"wait": ("key",), "send": ("scanner_start", "delay"), "none": ()}

TRIGGER_FIELDS = [
    Field("backend", "trigger line", "choice", TRIGGER_BACKENDS,
          tip="How codes reach your recording. null: no codes are sent (fMRI, or no recording). "
              "serial: a serial or USB-serial port. parallel: a parallel port. hid: a USB box's "
              "digital port (on a computer with no parallel port, such as a Mac). lsl: a Lab "
              "Streaming Layer stream on the network."),
    Field("hid_device", "USB box", "choice", tuple(HID_DEVICES),
          tip="The USB box the trigger cable is plugged into. mcc-1208fs: a Measurement "
              "Computing USB-1208FS."),
    Field("port", "port", "combo", tip="The port the trigger cable is plugged into, e.g. "
                                       "/dev/ttyUSB0, COM3 or /dev/parport0; on a USB box, its "
                                       "digital port, e.g. A. The list shows the ports found on "
                                       "this computer."),
    Field("lsl_stream_name", "LSL stream", tip="The name your recording software listens to."),
    Field("pulse_ms", "pulse length (ms)", "float",
          tip="How long a code stays on the port before it is cleared."),
    Field("on_frame", "a code on every frame", "bool", default=True,
          tip="Mark every game frame shown on screen in the recording, so frames can be lined "
              "up with the data offline."),
    Field("frame_every", "frame code every N frames", "int",
          tip="Send a frame code every N frames only (1: every frame), for a trigger line too "
              "slow to keep up."),
    Field("on_episode_start", "a code at each episode start", "bool", default=True,
          tip="Mark the first frame of every episode (each new game or level) in the recording."),
]

CODE_FIELDS = [
    Field("frame_bits", "frame code bits", "int",
          tip="How many bits the frame codes count with: 3 gives frame codes 1 to 7. The other "
              "codes must use higher bits."),
    Field("task_start", "run start code", "int", tip="Sent when the run starts."),
    Field("task_stop", "run end code", "int", tip="Sent when the run ends."),
    Field("episode_start", "episode start code", "int",
          tip="Sent with the first frame of each episode."),
]

#: Which trigger-line fields each line uses.
BACKEND_ROWS = {"null": (), "serial": ("port",), "parallel": ("port", "pulse_ms"),
                "hid": ("hid_device", "port", "pulse_ms"), "lsl": ("lsl_stream_name",)}

#: The rig file's ``"rig_check"`` (:class:`~fmri_gym.config.rig.RigCheckConfig`), as a form.
RIG_CHECK_FIELDS = [
    Field("photodiode", "photodiode", "choice", rig_file.PHOTODIODE_READOUTS,
          tip="A light sensor taped to the screen measures when an image really appears. "
              "none: this rig has none. soundcard: it is plugged into this computer's "
              "sound-card input, and the delay is measured during the check. recording: your "
              "MEG/EEG system records it, and you compare it to the frame codes afterwards."),
    Field("input_device", "sound-card input", "combo",
          tip="The sound-card input the photodiode (and the microphone) is plugged into. The "
              "list shows this computer's inputs. Blank: the default input."),
    Field("has_mic", "microphone at the ear", "bool", default=False,
          tip="A microphone at the participant's ear, on the sound card's second input: the "
              "check then measures when the sound is heard, not only when it is sent."),
    Field("corner", "photodiode corner", "choice", rig_file.CORNERS,
          tip="The corner of the screen the photodiode is taped to: br bottom right, bl bottom "
              "left, tr top right, tl top left. The check flashes a square there."),
    Field("patch_px", "flashing square (px)", "int",
          tip="The side of the flashing square, in pixels: big enough to cover the photodiode."),
    Field("recording_hz", "recording rate (Hz)", "float",
          tip="Your MEG/EEG system's sampling rate. The check makes sure each frame code lasts "
              "long enough to be seen in the recording. Blank if this rig records nothing."),
    Field("frame_rates", "more frame rates (fps)", "list",
          tip="Frame rates your games play at that the check should also test, separated by "
              "commas (e.g. 35). Every check already tests 60 and 30 fps."),
]

#: The map every curriculum here is written for: a six-button controller's buttons and
#: stick on the ten keys the games use (README, "Rigs").
STANDARD_BINDINGS = {"A": "SPACE", "X": "X", "B": "C", "Y": "Z", "LT": "A", "RT": "S",
                     "UP": "UP", "DOWN": "DOWN", "LEFT": "LEFT", "RIGHT": "RIGHT"}

#: Window sizes offered when not fullscreen (those that fit the monitor).
WINDOW_SIZES = ((800, 600), (1024, 768), (1280, 720), (1280, 1024), (1600, 900), (1920, 1080))
#: How the window names the fields describing the rig.
_ABOUT_LABELS = {"pi": "PI", "modality": "modality", "monitor": "monitor or projector",
                 "photodiode": "photodiode", "audio_path": "sound path to the ear",
                 "trigger_hardware": "trigger hardware", "notes": "notes (optional)"}
#: The fields describing the rig that may be left blank in the window.
_OPTIONAL = ("notes",)


def window_sizes(width: int, height: int) -> list[str]:
    """The window sizes to offer on a ``width`` x ``height`` monitor, its own size last."""
    fits = [f"{w}x{h}" for w, h in WINDOW_SIZES if w <= width and h <= height]
    return [*fits, f"{width}x{height}"] if f"{width}x{height}" not in fits else fits


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
    where = {"lsl": m.lsl_stream_name,
             "hid": f"{m.hid_device} port {m.port}"}.get(m.backend, m.port)
    if sync.mode == "wait":
        start = f"After SPACE, the run waits for {sync.key!r} from the trigger box, then starts."
    elif sync.mode == "send":
        start = (f"After SPACE, the run sends the start code {m.codes.scanner_start} on "
                 f"{m.backend} {where}, waits {sync.delay:g} s, then starts.")
    else:
        start = "After SPACE, the run starts at once."
    if m.backend == "null":
        return start + "\nNo codes are sent: the timing is in the run's log files only."
    frames = (f"a frame code (1 to {m.codes.frame_mask}) every {m.frame_every} frame(s)"
              if m.on_frame else "no frame codes")
    return "\n".join([
        start,
        f"Codes on {m.backend} {where}: run start {m.codes.task_start}, episode start "
        f"{m.codes.episode_start if m.on_episode_start else '(off)'}, {frames}, "
        f"run end {m.codes.task_stop}."])


def rig_summary(path: str | None) -> str:
    """A rig file in a few lines: what every run on it gets, or why no run would start on it.

    :param path: the rig file; ``None`` or blank: none picked yet.
    """
    if not path:
        return "No rig picked yet."
    if not os.path.isfile(path):
        return f"{path} does not exist yet."
    try:
        rig = rig_file.RigConfig.open(path)
    except ValueError as exc:
        return f"This rig file is not complete, and no run would start on it:\n{exc}"
    from ..checks import measured_files  # pygame and numpy: only once a rig is shown

    checks = len(measured_files(path))
    return "\n".join([_describe(rig, rig_file.load(path).get("triggers")),
                      f"Rig checks run on this computer: {checks}." if checks else
                      "No rig check has been run on this computer yet: run the long one once."])


def _describe(rig: rig_file.RigConfig, triggers: dict | None) -> str:
    """What every run on ``rig`` gets, in a few lines."""
    screen = rig.screen
    window = ("fullscreen" if screen.is_fullscreen
              else f"a {screen.size[0]}x{screen.size[1]} window")
    controller = ("not used" if not rig.has_pad
                  else f"{len(rig.bindings)} control(s) bound to keys" if rig.bindings
                  else "used, but no control is bound to a key: it presses nothing")
    responses = (f"; response keys {', '.join(rig.response_keys)}" if rig.response_keys
                 else "")
    return "\n".join([
        f"Screen: {window} on monitor {screen.monitor}, vsync {'on' if screen.has_vsync else 'off'}",
        f"Game controller: {controller}{responses}",
        f"Game sound: {'on' if rig.has_audio else 'off'}.   Data saved in: {rig.data_root}",
        describe_triggers(triggers) if triggers else
        "Triggers: NOT SET -- a run waits for '=' and sends no codes."])


def check_plan(rig: rig_file.RigConfig) -> str:
    """What a rig check tests on ``rig``, and what it leaves out and why, in plain lines."""
    from ..checks import RIG_CHECKS
    from ..config.curriculum import load_curriculum

    def rates(which: str) -> str:
        found = [r for p in load_curriculum(RIG_CHECKS[which])["curriculum"]
                 if p["type"] == "check_frames" for r in p.get("rates", [60])]
        return ", ".join(f"{r:g}" for r in dict.fromkeys([*found, *rig.rig_check.frame_rates]))

    t, c = rig.triggers, rig.rig_check
    codes = t.backend != "null"
    lines = ["• Screen: whether images change exactly with the screen's refresh.",
             f"• Frame timing: a test pattern played like a game at {rates('q')} fps "
             f"(long check: {rates('l')})"
             + (", with its frame codes" + (f", each lasting at least 2 samples at "
                                            f"{c.recording_hz:g} Hz" if c.recording_hz else "")
                if codes and t.on_frame else "") + "."]
    line = {"lsl": t.lsl_stream_name, "hid": f"{t.hid_device} port {t.port}"}.get(t.backend,
                                                                                t.port or "")
    sent = (f"every code sent on {t.backend} {line}".strip()
            if codes else "")
    pulses = (f"the scanner's {t.sync.key!r} counted at each volume" if t.sync.mode == "wait"
              else "")
    lines.append(f"• Triggers: {'; '.join(w for w in (sent, pulses) if w)}." if sent or pulses
                 else "• Triggers: not tested -- this rig sends no codes and waits for no "
                      "scanner.")
    keys = [f"{CONTROL_LABELS.get(k, k)} ({v})" for k, v in rig.bindings.items()
            if rig.has_pad] + [f"{k} ({v or 'response key'})" for k, v in rig.response_keys.items()]
    lines.append(f"• Responses: the participant presses each when asked -- {', '.join(keys)}."
                 if keys else "• Responses: not tested -- no controller bound and no response "
                              "keys.")
    lines.append({"none": "• Photodiode: not tested -- this rig has none.",
                  "soundcard": f"• Photodiode: the delay before each image appears, measured "
                               f"on the {c.input_device or 'default'} input (corner {c.corner}).",
                  "recording": "• Photodiode: flashes for you to compare with the frame codes "
                               "in your recording."}[c.photodiode])
    lines.append(("• Sound: the delay before each click leaves the sound card"
                  + (", and before it reaches the ear." if c.has_mic else "."))
                 if rig.has_audio else "• Sound: not tested -- game sound is off on this rig.")
    return "\n".join(lines)


def run_rig_check(path: str, which: str) -> str:
    """Play a rig check on the rig file at ``path``, as ``fmri-play`` would, and say how it went.

    :param which: ``"q"`` the short check, ``"l"`` the long one.
    :return: a few lines for the person who started it: the verdict and the report's path.
    """
    from ..bids import next_session
    from ..checks import RIG_CHECKS

    rig = rig_file.RigConfig.open(path)
    ses = next_session(rig.data_root, "sub-rig")
    command = [sys.executable, "-m", "fmri_play", "--curriculum", RIG_CHECKS[which],
               "--subject", "sub-rig", "--ses", str(ses), "--run", "1", "--rig", path]
    print("rig-config: " + " ".join(command), file=sys.stderr)
    status = subprocess.run(command, check=False).returncode
    reports = glob.glob(os.path.join(rig.data_root, "sub-rig", f"ses-{ses:03d}", "*", "*",
                                     "report.html"))
    if not reports:
        return (f"The rig check stopped before it finished (exit status {status}). The terminal "
                "says why.")
    with open(os.path.join(os.path.dirname(reports[0]), "manifest.json")) as f:
        manifest = json.load(f)
    verdict = ("every test passed" if manifest.get("status") == "pass"
               else f"failed: {manifest.get('failed_tests') or 'see the report'}")
    return f"Rig check {verdict}.\n\nThe report: {os.path.abspath(reports[0])}"


def trigger_defaults() -> dict:
    """The backend form's defaults: :class:`TriggerSettings` minus its sub-sections."""
    d = asdict(TriggerSettings())
    for key in ("sync", "codes", "defaulted"):
        d.pop(key)
    return d


def _sound_inputs() -> list[str]:
    """This computer's sound-card inputs, by name; none when sound cannot be queried."""
    try:
        import sounddevice
        return [d["name"] for d in sounddevice.query_devices() if d["max_input_channels"] > 0]
    except Exception:  # noqa: BLE001 -- no PortAudio here: nothing to offer, the name is typed
        return []


#: Asks, in a process of its own, which controller is plugged in and its newest calibration:
#: finding a controller starts SDL's display, which must not run inside this window.
_CONTROLLER_PROBE = """
import json
from fmri_gym import pad
found = pad.find()
out = {"name": found.name if found else None, "calibrated": None, "error": None}
if found:
    try:
        newest = pad.newest(found.name)
    except pad.CalibrationError as exc:
        out["error"] = str(exc)
    else:
        out["calibrated"] = newest[0].stat().st_mtime if newest else None
print(json.dumps(out))
"""


def controller_status() -> dict[str, Any]:
    """The controller plugged in now and when it was last calibrated on this computer.

    :return: ``name`` (``None``: none plugged in), ``calibrated`` (a time, or ``None``:
        never), ``error`` (why its calibration cannot be read, or ``None``).
    """
    try:
        done = subprocess.run([sys.executable, "-c", _CONTROLLER_PROBE], capture_output=True,
                              text=True, timeout=15)
        return json.loads(done.stdout.strip().splitlines()[-1])
    except (subprocess.TimeoutExpired, ValueError, IndexError) as exc:
        return {"name": None, "calibrated": None, "error": f"could not look: {exc}"}


def _controller_line(status: dict[str, Any]) -> str:
    """The controller plugged in, in words, and what to do about its calibration."""
    if status["error"]:
        return f"Game controller: {status['error']}"
    if status["name"] is None:
        return ("No game controller is plugged in now. Plug in the participant's controller "
                "to calibrate it, then press Look again.")
    if status["calibrated"] is None:
        return (f"Plugged in now: {status['name']}. It has never been calibrated on this "
                "computer: calibrate it before a session.")
    when = time.strftime("%d %b %Y, %H:%M", time.localtime(status["calibrated"]))
    return (f"Plugged in now: {status['name']}, last calibrated on this computer on {when}. "
            "Calibrate it again if its sticks feel wrong.")


def _trigger_ports() -> list[str]:
    """The serial and parallel ports found on this computer, and its USB boxes' ports."""
    from ..checks import detect_outputs
    return list(dict.fromkeys(p["port"] for p in detect_outputs()))


class RigPick(QtWidgets.QWidget):
    """A rig file to pick: the repo's in a drop-down, or any other with Browse...;
    ``changed`` says which file is picked now."""

    changed = QtCore.Signal(str)

    def __init__(self, path: str | None = None) -> None:
        super().__init__()
        self.combo = QtWidgets.QComboBox()
        self.combo.setEditable(True)
        self.combo.setMinimumWidth(380)
        self.combo.lineEdit().setPlaceholderText("configs/<team>/rigs/<rig>.json")
        self.combo.setToolTip("The rigs of this repository. Browse... to pick a rig file "
                              "elsewhere.")
        self.combo.addItems(rig_file.rig_files())
        self.combo.setCurrentText(path or "")
        self.combo.currentTextChanged.connect(lambda text: self.changed.emit(text.strip()))
        browse = ui.button("Browse...", self._browse)
        browse.setToolTip("Pick a rig file anywhere on this computer.")
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.combo, 1)
        layout.addWidget(browse)

    def path(self) -> str:
        """The rig file picked; blank when none."""
        return self.combo.currentText().strip()

    def set_path(self, path: str | None) -> None:
        """Pick ``path`` (a new file is added to the list)."""
        if path and self.combo.findText(path) < 0:
            self.combo.addItem(path)
        self.combo.setCurrentText(path or "")

    def _browse(self) -> None:
        start = os.path.dirname(self.path()) or str(rig_file.CONFIGS)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Pick a rig file", start,
                                                        "Rig file (*.json)")
        if path:
            self.set_path(os.path.relpath(path) if not os.path.relpath(path).startswith("..")
                          else path)


#: Each step of the window: its title and what the user has to do on it.
_STEPS = [
    ("About this rig",
     "Describe this setup so that you, and anyone reading its rig check reports, can tell it "
     "apart from others. Say what is actually plugged in; write \"none\" for a part this rig "
     "does not have."),
    ("Screen, sound and data",
     "Which screen the participant sees and how the games are shown on it, whether game sound "
     "is played, and where the data of every run on this rig is saved."),
    ("Response devices",
     "How the participant answers. Games are played with keyboard keys. A game controller "
     "(Xbox, Current Designs gamepad) sends no keys by itself: choose the keyboard key each of "
     "its buttons presses. A button box that types keys by itself needs no binding: list its "
     "keys under Response keys. A keyboard alone needs nothing here."),
    ("Triggers",
     "How a run starts with your scanner or recording, and which codes it sends to the "
     "recording. Pick the preset that matches your setup and press Apply, then adjust: only the "
     "options that apply to your choices are shown."),
    ("Rig check equipment",
     "The rig check measures this rig's timing with this equipment. Set what this rig has; "
     "what the check will then test is listed below."),
    ("Save and check",
     "Read the summary, then Save. A rig check can be run once the file is saved: the short "
     "one (under a minute) before each session, the long one (about 25 minutes) once when the "
     "rig is set up, and again after any change of hardware, drivers or system."),
]
_NO_TRIGGERS = "pick a preset and press Apply, or set the fields"


class _RigWindow(QtWidgets.QDialog):
    """One rig file, filled in step by step: Next is off until the step is complete, and the
    rig checks are off until the file is saved as shown."""

    def __init__(self, values: dict, path: str, need_triggers: bool = False) -> None:
        super().__init__()
        self.path = path
        #: the rig as last written to ``path`` here; ``None`` until Save
        self.saved: dict | None = None
        #: the file's place, kept as written
        self._kept = {k: values[k] for k in ("site", "rig") if k in values}
        #: triggers are written once the file has them or the user chose them here
        self.has_triggers = "triggers" in values
        #: the controller plugged in (:func:`controller_status`); ``None`` until looked for
        self.controller: dict[str, Any] | None = None
        self.setWindowTitle(f"rig-config: {path}")
        self.setMinimumSize(800, 700)
        self.title = QtWidgets.QLabel()
        self.title.setStyleSheet("font-size: 16px; font-weight: 600")
        self.intro = QtWidgets.QLabel()
        self.intro.setWordWrap(True)
        self.stack = QtWidgets.QStackedWidget()
        builders: list[Callable[[], QtWidgets.QWidget]] = [
            lambda: self._about_page(values), lambda: self._screen_page(values),
            lambda: self._responses_page(values),
            lambda: self._triggers_page(values.get("triggers", {})),
            lambda: self._check_page(values.get("rig_check", {})), self._save_page]
        for build in builders:
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            scroll.setWidget(build())
            self.stack.addWidget(scroll)
        #: per step, what keeps it from being complete
        self._step_problems: list[Callable[[], list[str]]] = [
            self._about_problems, self._screen_problems, self._responses_problems,
            self._triggers_problems, self._check_problems, self._all_problems]
        self.problems = QtWidgets.QLabel()
        self.problems.setWordWrap(True)
        self.problems.setStyleSheet("color: #c0392b")
        self.previous = ui.button("< Previous", lambda: self._go(-1))
        self.next = ui.button("Next >", lambda: self._go(1))
        self.next.setObjectName("run")
        cancel = ui.button("Close", self.reject)
        for b in (self.previous, self.next, cancel):
            b.setAutoDefault(False)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.title)
        layout.addWidget(self.intro)
        layout.addWidget(self.stack, 1)
        layout.addWidget(self.problems)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(cancel)
        bar.addStretch(1)
        bar.addWidget(self.previous)
        bar.addWidget(self.next)
        layout.addLayout(bar)
        for form in (self.sync_form, self.code_form, self.trigger_form):
            form.changed.connect(self._triggers_edited)  # after filling: filling is not a choice
        self._show_step(3 if need_triggers else 0)
        self._opened = self._current()

    def _current(self) -> dict | None:
        """The rig the steps say; ``None`` while a number is half typed."""
        try:
            return self.rig
        except ValueError:
            return None

    # -- steps -----------------------------------------------------------------

    def _show_step(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        if index == 2 and self.controller is None:
            QtCore.QTimer.singleShot(0, self._look_for_controller)  # once the step is drawn
        title, intro = _STEPS[index]
        self.title.setText(f"Step {index + 1} of {len(_STEPS)}: {title}")
        self.intro.setText(intro)
        self.previous.setEnabled(index > 0)
        self.next.setVisible(index < len(_STEPS) - 1)
        self._check()

    def _go(self, by: int) -> None:
        index = self.stack.currentIndex()
        if by > 0 and self._step_problems[index]():
            return  # Next is off: Enter on a field does not get past it either
        self._show_step(index + by)

    def _check(self) -> None:
        """Show only the options that apply, and what keeps this step from being complete."""
        self._show_applicable()
        problems = self._step_problems[self.stack.currentIndex()]()
        last = self.stack.currentIndex() == len(_STEPS) - 1
        self.problems.setText(("To save: " if last else "To go on: ") + "; ".join(problems)
                              if problems else "")
        self.next.setEnabled(not problems)
        self._refresh_reports()

    def _show_applicable(self) -> None:
        """Hide, or turn off, what the choices made so far make irrelevant."""
        self.size_pick.setEnabled(not self.switches["is_fullscreen"].isChecked())
        has_pad = self.switches["has_pad"].isChecked()
        for widget in (self.bindings, self.bind_key, self.standard):
            widget.setEnabled(has_pad)
        self.calibrate.setEnabled(has_pad and (self.controller or {}).get("name") is not None)
        self.sync_form.show_rows(["mode", *SYNC_ROWS.get(self._value(self.sync_form, "mode"),
                                                         [f.key for f in SYNC_FIELDS])])
        backend = self._value(self.trigger_form, "backend")
        codes = backend != "null"
        on_frame = codes and bool(self._value(self.trigger_form, "on_frame"))
        self.trigger_form.show_rows(["backend", *BACKEND_ROWS.get(backend, ())]
                                    + ["on_frame", "on_episode_start"] * codes
                                    + ["frame_every"] * on_frame)
        self.code_box.setVisible(codes)
        self.code_form.show_rows(["task_start", "task_stop"] + ["frame_bits"] * on_frame
                                 + ["episode_start"] * bool(
                                     self._value(self.trigger_form, "on_episode_start")))
        self.test_button.setVisible(codes)
        diode = self._value(self.check_form, "photodiode")
        if diode != "soundcard" and self._value(self.check_form, "has_mic"):
            self.check_form.write("has_mic", False)  # a microphone is read beside the diode
        sound = self.switches["has_audio"].isChecked()
        self.check_form.show_rows(["photodiode", "frame_rates"]
                                  + ["corner", "patch_px"] * (diode != "none")
                                  + ["input_device"] * (diode == "soundcard")
                                  + ["has_mic"] * (diode == "soundcard" and sound)
                                  + ["recording_hz"] * on_frame)

    @staticmethod
    def _value(form: ui.Form, key: str) -> Any:
        widget = form.widgets[key]
        return (widget.isChecked() if isinstance(widget, QtWidgets.QCheckBox)
                else widget.currentText() if isinstance(widget, QtWidgets.QComboBox)
                else widget.text())

    def _refresh_reports(self) -> None:
        """The trigger summary, the check plan and the save step's summary, as things stand."""
        self.trigger_text.setText(self._trigger_report())
        try:
            rig = rig_file.RigConfig.from_dict(self.rig, self.path)
        except ValueError as exc:
            self.plan.setText(f"Complete the steps before to see it: {exc}")
            self.summary.setText(f"Not complete yet: {exc}")
        else:
            self.plan.setText(check_plan(rig))
            self.summary.setText(_describe(rig, self.rig.get("triggers")) + "\n\nA rig check "
                                 "will test:\n" + check_plan(rig))
        self._refresh_saved()

    # -- step 1: about ----------------------------------------------------------

    @staticmethod
    def _page() -> tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout]:
        page = QtWidgets.QWidget()
        return page, QtWidgets.QVBoxLayout(page)

    def _about_page(self, values: dict) -> QtWidgets.QWidget:
        page, layout = self._page()
        self.fields: dict[str, QtWidgets.QWidget] = {}
        form = QtWidgets.QFormLayout()
        for key, what in rig_file.FIELDS.items():
            if key in ("site", "rig"):   # the file's folder and name
                continue
            form.addRow(_ABOUT_LABELS.get(key, key), self._field(key, str(values.get(key, "")),
                                                          rig_file.EXAMPLES.get(key, ""), what))
        layout.addLayout(form)
        layout.addStretch(1)
        return page

    def _field(self, key: str, value: str, example: str, what: str) -> QtWidgets.QWidget:
        """The widget for one key: a choice for ``modality``, a text box for ``notes``."""
        if key == "modality":
            w = QtWidgets.QComboBox()
            w.addItems(["", *rig_file.MODALITIES])
            w.setCurrentText(value if value in rig_file.MODALITIES else "")
            w.currentTextChanged.connect(self._check)
        elif key == "notes":
            w = QtWidgets.QPlainTextEdit(value)
            w.setFixedHeight(60)
            w.textChanged.connect(self._check)
        else:
            w = QtWidgets.QLineEdit(value)
            w.setPlaceholderText(f"e.g. {example}" if example else "")
            w.textChanged.connect(self._check)
        w.setToolTip(what[0].upper() + what[1:] + ".")
        self.fields[key] = w
        return w

    def _about_problems(self) -> list[str]:
        about = self._about()
        missing = [k for k in self.fields if k not in _OPTIONAL and k not in about]
        names = [_ABOUT_LABELS.get(k, k) for k in missing]
        return ([f"fill in {', '.join(names)}"] if missing else []) + rig_file.problems(about)

    def _about(self) -> dict[str, str]:
        out = {}
        for key, w in self.fields.items():
            text = (w.currentText() if isinstance(w, QtWidgets.QComboBox)
                    else w.toPlainText() if isinstance(w, QtWidgets.QPlainTextEdit)
                    else w.text())
            if text.strip():   # a blank field is left out
                out[key] = text.strip()
        return out

    # -- step 2: screen, sound and data ---------------------------------------------

    def _screen_page(self, values: dict) -> QtWidgets.QWidget:
        page, layout = self._page()
        form = QtWidgets.QFormLayout()
        given = values.get("screen", {})  # an invalid file's may be anything: the form redoes it
        screen = {**rig_file.DEFAULTS["screen"], **(given if isinstance(given, dict) else {})}
        self.switches: dict[str, QtWidgets.QCheckBox] = {}
        form.addRow("participant's screen", self._monitor_pick(screen["monitor"]))
        form.addRow("fullscreen", self._switch(
            "is_fullscreen", screen["is_fullscreen"],
            "Fill the whole screen, as a session in the scanner needs. Off: the games are "
            "shown in a window of the size below, handy at a desk."))
        self.size_pick = QtWidgets.QComboBox()
        self.size_pick.setEditable(True)
        self._fill_sizes()
        self.size_pick.setEditText(str(screen["size"]))
        self.size_pick.setToolTip("The window's width x height in pixels. Not used in "
                                  "fullscreen, which takes the screen's own resolution.")
        self.monitor_pick.currentIndexChanged.connect(lambda _i: self._fill_sizes())
        self.size_pick.editTextChanged.connect(self._check)
        form.addRow("window size", self.size_pick)
        form.addRow("vsync", self._switch(
            "has_vsync", screen["has_vsync"],
            "Show each image exactly on the screen's refresh. Leave it on, unless the display "
            "test (python -m fmri_gym.display) says this computer cannot do it."))
        form.addRow("game sound", self._switch(
            "has_audio", values.get("has_audio", rig_file.DEFAULTS["has_audio"]),
            "Play the games' sound to the participant. Off: every game is silent on this rig."))
        self.data_root = QtWidgets.QLineEdit(str(values.get("data_root",
                                                            rig_file.DEFAULTS["data_root"])))
        self.data_root.setToolTip("The folder every run's data is saved in, as "
                                  "<folder>/sub-XX/ses-NNN/... A relative folder is taken from "
                                  "where the run is started (the repository's folder).")
        self.data_root.textChanged.connect(self._check)
        browse = ui.button("Browse...", self._browse_data_root)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.data_root, 1)
        row.addWidget(browse)
        form.addRow("data folder", row)
        layout.addLayout(form)
        layout.addStretch(1)
        return page

    def _browse_data_root(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Data folder",
                                                            self.data_root.text() or ".")
        if folder:
            rel = os.path.relpath(folder)
            self.data_root.setText(folder if rel.startswith("..") else rel)

    def _switch(self, key: str, on: Any, tip: str) -> QtWidgets.QCheckBox:
        switch = QtWidgets.QCheckBox()
        switch.setChecked(bool(on))
        switch.setToolTip(tip)
        switch.toggled.connect(lambda _on: self._check())
        self.switches[key] = switch
        return switch

    def _monitor_pick(self, monitor: Any) -> QtWidgets.QComboBox:
        """This computer's monitors, the rig's own shown even when not connected here."""
        self.monitors = list_monitors()
        self.monitor_pick = QtWidgets.QComboBox()
        for index, size in enumerate(self.monitors):
            self.monitor_pick.addItem(monitor_label(index, size), index)
        if self.monitor_pick.findData(monitor) < 0:
            self.monitor_pick.addItem(f"{monitor}: not connected now", monitor)
        self.monitor_pick.setCurrentIndex(self.monitor_pick.findData(monitor))
        self.monitor_pick.setToolTip("The screen the participant sees, among the screens "
                                     "connected now. Screens are numbered, not named: recognise "
                                     "it by its resolution and refresh rate.")
        return self.monitor_pick

    def _fill_sizes(self) -> None:
        """Offer the window sizes that fit the picked monitor, keeping the size typed."""
        index = self.monitor_pick.currentData()
        connected = isinstance(index, int) and 0 <= index < len(self.monitors)
        typed = self.size_pick.currentText()
        self.size_pick.blockSignals(True)
        self.size_pick.clear()
        self.size_pick.addItems(window_sizes(*self.monitors[index][:2]) if connected
                                else [f"{w}x{h}" for w, h in WINDOW_SIZES])
        self.size_pick.setEditText(typed)
        self.size_pick.blockSignals(False)

    def _screen(self) -> dict[str, Any]:
        return {"screen": {"size": self.size_pick.currentText().strip(),
                           "is_fullscreen": self.switches["is_fullscreen"].isChecked(),
                           "monitor": self.monitor_pick.currentData(),
                           "has_vsync": self.switches["has_vsync"].isChecked()},
                "has_audio": self.switches["has_audio"].isChecked(),
                "data_root": self.data_root.text().strip()}

    def _screen_problems(self) -> list[str]:
        return rig_file.problems(self._screen())

    # -- step 3: response devices -------------------------------------------------

    def _responses_page(self, values: dict) -> QtWidgets.QWidget:
        page, layout = self._page()
        self.controller_text = QtWidgets.QLabel()
        self.controller_text.setWordWrap(True)
        self.calibrate = ui.button("Calibrate controller...", self._calibrate)
        self.calibrate.setToolTip("Open pad-calibration on the controller plugged in: measure "
                                  "its sticks and check every button (about 20 s a stick), and "
                                  "see what it sends.")
        look = ui.button("Look again", self._look_for_controller)
        look.setToolTip("Look again for a controller plugged in, and its calibration.")
        layout.addWidget(ui.group("Controller plugged in", self._column(
            self.controller_text, ui.row(self.calibrate, look))))
        form = QtWidgets.QFormLayout()
        # A new rig starts without one: a map on a rig answered with a button box would have
        # its runs and its rig check expect controller buttons nobody has.
        form.addRow("the participant uses a game controller", self._switch(
            "has_pad", values.get("has_pad", rig_file.DEFAULTS["has_pad"] if values else False),
            "On for an Xbox controller or a Current Designs gamepad: its buttons then press "
            "the keys below. Off for a keyboard or a button box that types keys."))
        self.switches["has_pad"].toggled.connect(self._controller_turned)
        layout.addLayout(form)
        layout.addWidget(self._bindings_table(values.get("bindings", {})), 1)
        self.bind_key = ui.button("Press a key...", self._capture_binding)
        self.bind_key.setToolTip("Select a control in the list, then press the keyboard key it "
                                 "should press.")
        self.standard = ui.button("Use the standard map", self._use_standard_bindings)
        self.standard.setToolTip("The map every game here is written for: A SPACE, X X, B C, "
                                 "Y Z, LT A, RT S, and the stick on the arrows.")
        self.bindings_note = QtWidgets.QLabel()
        self.bindings_note.setObjectName("hint")
        layout.addLayout(ui.row(self.standard, self.bind_key, self.bindings_note))
        layout.addWidget(self._response_keys_box(values.get("response_keys", {})))
        return page

    @staticmethod
    def _column(*parts: Any) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        for part in parts:
            if isinstance(part, QtWidgets.QLayout):
                column.addLayout(part)
            else:
                column.addWidget(part)
        return box

    def _look_for_controller(self) -> None:
        self.controller_text.setText("Looking for a controller...")
        QtWidgets.QApplication.processEvents()
        self.controller = controller_status()
        self.controller_text.setText(_controller_line(self.controller))
        self._check()

    def _calibrate(self) -> None:
        """Run pad-calibration in its own window, with this rig's bindings shown; then look
        again, so the new calibration shows here."""
        command = [sys.executable, "-m", "fmri_gym.pad_scope"]
        if os.path.isfile(self.path) and self.saved is not None:
            command += ["--rig", self.path]
        self.hide()
        try:
            subprocess.run(command, check=False)
        finally:
            self.show()
        self._look_for_controller()

    def _bindings_table(self, given: Any) -> QtWidgets.QTableWidget:
        given = given if isinstance(given, dict) else {}
        controls = [*CONTROLS, *(c for c in given if c not in CONTROLS)]  # a wrong one, shown
        self.bindings = QtWidgets.QTableWidget(len(controls), 2)
        self.bindings.setHorizontalHeaderLabels(["controller button", "presses the key"])
        self.bindings.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.bindings.verticalHeader().hide()
        self.bindings.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.bindings.setMinimumHeight(260)
        self.bindings.setToolTip("The keyboard key each button of the controller presses, for "
                                 "every game on this rig. Blank: the button does nothing. Most "
                                 "games here use the arrows, SPACE, X, C, Z, A and S.")
        for row, control in enumerate(controls):
            name = QtWidgets.QTableWidgetItem(CONTROL_LABELS.get(control, control))
            name.setData(Qt.ItemDataRole.UserRole, control)
            name.setFlags(name.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.bindings.setItem(row, 0, name)
            key = QtWidgets.QComboBox()
            bound = str(given.get(control, ""))
            key.addItems(["", *sorted(KEY_NAMES | ({bound} - {""}))])
            key.setCurrentText(bound)
            key.currentTextChanged.connect(lambda _text: self._check())
            self.bindings.setCellWidget(row, 1, key)
        return self.bindings

    def _response_keys_box(self, given: Any) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Response keys: a button box that types keys by itself")
        layout = QtWidgets.QVBoxLayout(box)
        hint = QtWidgets.QLabel("Press each button of the box: its key is added. Then say what "
                                "it is (e.g. index finger); the rig check shows it when it asks "
                                "for that button. The games you play on this rig must use these "
                                "keys.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.response_keys = QtWidgets.QTableWidget(0, 2)
        self.response_keys.setHorizontalHeaderLabels(["key it types", "what it is"])
        self.response_keys.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.response_keys.verticalHeader().hide()
        for key, label in (given.items() if isinstance(given, dict) else ()):
            self._add_response_key(str(key), str(label))
        self.response_keys.itemChanged.connect(lambda _item: self._check())
        add = ui.button("Press a button...", self._capture_response_key)
        add.setToolTip("Press a button of the box: the key it types is added to the list.")
        remove = ui.button("Remove", self._remove_response_key)
        remove.setToolTip("Remove the selected key.")
        self.response_keys.setMaximumHeight(130)
        layout.addWidget(self.response_keys)
        layout.addLayout(ui.row(add, remove))
        return box

    def _add_response_key(self, key: str, label: str = "") -> None:
        row = self.response_keys.rowCount()
        self.response_keys.insertRow(row)
        item = QtWidgets.QTableWidgetItem(key)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.response_keys.setItem(row, 0, item)
        self.response_keys.setItem(row, 1, QtWidgets.QTableWidgetItem(label))

    def _remove_response_key(self) -> None:
        if self.response_keys.currentRow() >= 0:
            self.response_keys.removeRow(self.response_keys.currentRow())
            self._check()

    def _capture_response_key(self) -> None:
        dialog = ui.KeyCapture(self)
        dialog.exec()
        if dialog.name is not None:
            if dialog.name in self._response_keys():
                self.problems.setText(f"{dialog.name} is in the list already")
                return
            self._add_response_key(dialog.name)
            self.response_keys.setCurrentCell(self.response_keys.rowCount() - 1, 1)
            self.response_keys.editItem(self.response_keys.item(self.response_keys.rowCount()
                                                                - 1, 1))
            self._check()
        elif dialog.refused is not None:
            self.problems.setText(f"{dialog.refused!r} is not a key the games can read")

    def _controller_turned(self, on: bool) -> None:
        """A controller turned on with nothing bound gets the standard map, and says so."""
        if on and not self._bindings():
            self._use_standard_bindings()

    def _use_standard_bindings(self) -> None:
        for row in range(self.bindings.rowCount()):
            control = self.bindings.item(row, 0).data(Qt.ItemDataRole.UserRole)
            self.bindings.cellWidget(row, 1).setCurrentText(STANDARD_BINDINGS.get(control, ""))
        self.bindings_note.setText("The standard map: change any button you need to.")

    def _capture_binding(self) -> None:
        row = self.bindings.currentRow()
        if row < 0:
            self.problems.setText("select a controller button in the list first")
            return
        dialog = ui.KeyCapture(self)
        dialog.exec()
        if dialog.name is not None:
            self.bindings.cellWidget(row, 1).setCurrentText(dialog.name)

    def _bindings(self) -> dict[str, str]:
        out = {}
        for row in range(self.bindings.rowCount()):
            key = self.bindings.cellWidget(row, 1).currentText()
            if key:
                out[self.bindings.item(row, 0).data(Qt.ItemDataRole.UserRole)] = key
        return out

    def _response_keys(self) -> dict[str, str]:
        out = {}
        for row in range(self.response_keys.rowCount()):
            key, label = (self.response_keys.item(row, c) for c in (0, 1))
            if key is not None and key.text().strip():
                out[key.text().strip().upper()] = label.text().strip() if label else ""
        return out

    def _responses(self) -> dict[str, Any]:
        has_pad = self.switches["has_pad"].isChecked()
        out: dict[str, Any] = {"has_pad": has_pad}
        for key, value in (("bindings", self._bindings() if has_pad else {}),
                           ("response_keys", self._response_keys())):
            if value:
                out[key] = value
        return out

    def _responses_problems(self) -> list[str]:
        unnamed = [k for k, v in self._response_keys().items() if not v]
        return ([f"say what response key {', '.join(unnamed)} is"] if unnamed else []) \
            + rig_file.problems(self._responses())

    # -- step 4: triggers -----------------------------------------------------------

    def _triggers_page(self, section: Any) -> QtWidgets.QWidget:
        page, layout = self._page()
        self.preset = QtWidgets.QComboBox()
        self.preset.addItems(list(rig_file.TRIGGER_PRESETS))
        self.preset.setPlaceholderText("choose your setup...")
        self.preset.setCurrentIndex(-1)
        self.preset.setMinimumWidth(360)
        apply = ui.button("Apply", self._apply_preset)
        apply.setToolTip("Fill the fields below for this setup; then adjust them to yours.")
        layout.addLayout(ui.row(QtWidgets.QLabel("preset"), self.preset, apply))
        self.sync_form = ui.Form(SYNC_FIELDS)
        self.code_form = ui.Form(CODE_FIELDS)
        self.trigger_form = ui.Form(TRIGGER_FIELDS)
        self.trigger_form.set_choices("port", _trigger_ports())
        layout.addWidget(ui.group("Start of a run", self.sync_form))
        layout.addWidget(ui.group("Codes sent to the recording", self.trigger_form))
        self.code_box = ui.group("Code values", self.code_form)
        layout.addWidget(self.code_box)
        self.test_button = ui.button("Test: send the run start code", self._test_triggers)
        self.test_button.setToolTip("Open the trigger line as a run would and send the run start "
                                    "code once: check that your recording shows it.")
        layout.addLayout(ui.row(self.test_button))
        self.trigger_text = QtWidgets.QLabel()
        self.trigger_text.setObjectName("report")
        self.trigger_text.setWordWrap(True)
        self.trigger_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.trigger_text)
        layout.addStretch(1)
        section = dict(section) if isinstance(section, dict) else {}
        codes = {**asdict(Codes()), **section.pop("codes", {})}
        self._set_sync(section.pop("sync", {}), codes["scanner_start"])
        self.code_form.set(codes)
        self.trigger_form.set({**trigger_defaults(), **section})
        return page

    def _apply_preset(self) -> None:
        if self.preset.currentIndex() < 0:
            return
        section = copy.deepcopy(rig_file.TRIGGER_PRESETS[self.preset.currentText()])
        start = self.sync_form.get().get("scanner_start", Codes().scanner_start)  # a preset keeps it
        self._set_sync(section.pop("sync"), start)
        self.trigger_form.set({**trigger_defaults(), **section})
        self._triggers_edited()

    def _set_sync(self, sync: dict, start: int) -> None:
        self.sync_form.set({**asdict(SyncSettings()), **sync, "scanner_start": start})

    def _triggers_edited(self) -> None:
        self.has_triggers = True
        self._check()

    def _triggers(self) -> dict:
        """:raises ValueError: a number half typed, named in the message."""
        return triggers_section(self.sync_form.get(), self.trigger_form.get(),
                                self.code_form.get())

    def _triggers_problems(self) -> list[str]:
        if not self.has_triggers:
            return [_NO_TRIGGERS]
        try:
            return rig_file.trigger_problems(self._triggers())
        except ValueError as exc:
            return [str(exc)]

    def _trigger_report(self) -> str:
        if not self.has_triggers:
            return "No triggers chosen yet."
        try:
            section = self._triggers()
        except ValueError as exc:
            return str(exc)
        problems = rig_file.trigger_problems(section)
        return "\n".join(problems) if problems else describe_triggers(section)

    def _test_triggers(self) -> None:
        """Open the trigger line exactly as a run would and send one run start code."""
        try:
            triggers = Triggers.from_dict(self._triggers())
        except (TriggerError, ValueError, TypeError) as exc:
            self.trigger_text.setText(f"The test failed: {exc}")
            return
        value = triggers.lifecycle("task_start")
        triggers.close()
        self.trigger_text.setText(f"Sent {value} on {triggers.active}: check that your "
                                  "recording shows one code.")

    # -- step 5: rig check equipment ------------------------------------------------

    def _check_page(self, section: Any) -> QtWidgets.QWidget:
        page, layout = self._page()
        self.check_form = ui.Form(RIG_CHECK_FIELDS)
        self.check_form.set_choices("input_device", _sound_inputs())
        given = section if isinstance(section, dict) else {}
        self.check_form.set({**rig_file.RIG_CHECK_DEFAULTS, **given})
        self.check_form.changed.connect(self._check)
        layout.addWidget(ui.group("This rig's equipment", self.check_form))
        self.plan = QtWidgets.QLabel()
        self.plan.setObjectName("report")
        self.plan.setWordWrap(True)
        layout.addWidget(ui.group("What a rig check will test on this rig", self.plan))
        layout.addStretch(1)
        return page

    def _rig_check(self) -> dict:
        """The equipment form as the file's ``"rig_check"``: what differs from the defaults.

        :raises ValueError: a number half typed, named in the message.
        """
        values = self.check_form.get()
        return {k: v for k, v in values.items() if v != rig_file.RIG_CHECK_DEFAULTS.get(k)}

    def _check_problems(self) -> list[str]:
        try:
            return rig_file.problems({"rig_check": self._rig_check()})
        except ValueError as exc:
            return [str(exc)]

    # -- step 6: save and check -----------------------------------------------------

    def _save_page(self) -> QtWidgets.QWidget:
        page, layout = self._page()
        self.summary = QtWidgets.QLabel()
        self.summary.setObjectName("report")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        where = QtWidgets.QLabel(f"Rig file: {self.path}")
        where.setObjectName("hint")
        where.setWordWrap(True)
        layout.addWidget(where)
        layout.addWidget(ui.group("Every run on this rig", self.summary))
        self.save = ui.button("Save", self._save)
        self.save.setObjectName("run")
        self.save.setToolTip(f"Write the rig file {self.path}.")
        self.save_state = QtWidgets.QLabel()
        self.save_state.setObjectName("hint")
        self.save_state.setWordWrap(True)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.save)
        row.addWidget(self.save_state, 1)
        layout.addLayout(row)
        self.short_check = ui.button("Run the short rig check", lambda: self._run_check("q"))
        self.long_check = ui.button("Run the long rig check", lambda: self._run_check("l"))
        self.short_check.setToolTip("Under a minute: before each session, to see the rig is as "
                                    "it was measured. Have the participant's device at hand.")
        self.long_check.setToolTip("About 25 minutes: measures this rig. Run it once when the "
                                   "rig is set up, and again after any change.")
        layout.addWidget(ui.group("Rig check", self._check_buttons()))
        layout.addStretch(1)
        return page

    def _check_buttons(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(box)
        hint = QtWidgets.QLabel("The check opens in a window of its own, as a run would, and "
                                "files a report. Place the photodiode and the microphone first, "
                                "if this rig has them; the check asks for each response button "
                                "in turn.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        column.addWidget(hint)
        column.addLayout(ui.row(self.short_check, self.long_check))
        return box

    def _all_problems(self) -> list[str]:
        out = []
        for problems in self._step_problems[:-1]:
            out += problems()
        return out

    def _refresh_saved(self) -> None:
        """Save is on when the file is complete and differs from what is on disk; the rig
        checks only when what is shown is what is saved."""
        complete = not self._all_problems()
        try:
            current = self.rig if complete else None
        except ValueError:
            current = None
        on_disk = current is not None and current == self.saved
        self.save.setEnabled(complete and not on_disk)
        for button in (self.short_check, self.long_check):
            button.setEnabled(on_disk)
        self.save_state.setText(
            "Complete every step to save." if not complete else
            f"Saved in {self.path}." if on_disk else
            "Not saved yet: Save before running a rig check." if self.saved is None else
            "Changed since it was saved: Save again before running a rig check.")

    def _save(self) -> None:
        if self._all_problems():
            return
        rig = self.rig
        try:
            rig_file.save(rig, self.path)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "rig-config", f"Could not save: {exc}")
            return
        self.saved = rig
        self._refresh_saved()

    def _run_check(self, which: str) -> None:
        """Play a rig check on the saved file in its own window; say how it went."""
        self.hide()
        try:
            outcome = run_rig_check(self.path, which)
        finally:
            self.show()
        QtWidgets.QMessageBox.information(self, "Rig check", outcome)

    # -- the file --------------------------------------------------------------

    @property
    def rig(self) -> dict[str, Any]:
        """The rig file the steps say.

        :raises ValueError: a number half typed, named in the message.
        """
        out = {**self._about(), **self._screen(), **self._responses()}
        if self.has_triggers:
            out["triggers"] = self._triggers()
        rig_check = self._rig_check()
        if rig_check:
            out["rig_check"] = rig_check
        return {**out, **self._kept}

    def accept(self) -> None:
        """Enter on a field does not close the window: Close does."""

    def reject(self) -> None:
        """Close, after asking when there are changes not saved."""
        if self._current() != (self.saved if self.saved is not None else self._opened):
            answer = QtWidgets.QMessageBox.question(
                self, "rig-config", "Close without saving your changes to the rig file?")
            if answer != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        super().reject()


def fill_rig(values: dict, path: str, need_triggers: bool = False) -> dict[str, Any] | None:
    """The rig window on ``path``, started from ``values``; it saves the file itself.

    :param values: what is known so far (an invalid file's content, or nothing).
    :param path: the rig file.
    :param need_triggers: start on the Triggers step.
    :return: the rig as last saved, or ``None`` if the window was closed without saving.
    """
    ui.app()
    window = _RigWindow(values, path, need_triggers)
    window.exec()
    return window.saved


_CHOOSER_INTRO = (
    "Pick the rig you are about to use or set up, or make a new one. A rig is one setup: the "
    "participant's screen, response device and sound, the trigger line, and the folder the "
    "data is saved in. Every run names the rig it is played on.")


class _RigChooser(QtWidgets.QDialog):
    """Which rig file to open: the repo's, one elsewhere, or a new one; what each settles."""

    def __init__(self) -> None:
        super().__init__()
        #: the rig file to open: the one picked, or a new one (not on disk until its Save)
        self.chosen = ""
        self.setWindowTitle("rig-config: pick a rig")
        self.setMinimumWidth(680)
        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(_CHOOSER_INTRO)
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.pick = RigPick()
        self.pick.changed.connect(lambda _path: self._refresh())
        layout.addLayout(ui.row(QtWidgets.QLabel("rig file"), self.pick, stretch=False))
        self.summary = QtWidgets.QLabel()
        self.summary.setObjectName("report")
        self.summary.setWordWrap(True)
        layout.addWidget(ui.group("Every run on this rig", self.summary))
        self.open_button = ui.button("Open", self._open)
        self.open_button.setObjectName("run")
        self.open_button.setToolTip("Go through this rig's settings, change them, and run its "
                                    "rig checks.")
        new = ui.button("New rig...", self._new)
        new.setToolTip("Make a rig file for a setup that has none yet.")
        layout.addLayout(ui.row(new, ui.button("Close", self.reject), self.open_button))
        self._refresh()

    def _refresh(self) -> None:
        path = self.pick.path()
        self.summary.setText(rig_summary(path))
        self.open_button.setEnabled(bool(path) and os.path.isfile(path))

    def _open(self) -> None:
        self.chosen = self.pick.path()
        self.accept()

    def _new(self) -> None:
        path = ask_new_rig(self)
        if path is not None:
            self.chosen = path
            self.accept()


def ask_new_rig(parent: QtWidgets.QWidget) -> str | None:
    """Ask the team and the rig's name of a new rig file; ``None`` if cancelled.

    :return: ``configs/<team>/rigs/<rig>.json``, which does not exist yet.
    """
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("New rig")
    layout = QtWidgets.QVBoxLayout(dialog)
    intro = QtWidgets.QLabel("The new rig file is saved as configs/<team>/rigs/<rig>.json in "
                             "this repository; commit it, so the other teams see it. The team "
                             "is your PI's initials; name the rig after its place and setup "
                             "(mgh-prisma, mit-desk). Both appear in every run's data and "
                             "every rig check report.")
    intro.setWordWrap(True)
    layout.addWidget(intro)
    form = QtWidgets.QFormLayout()
    site, rig = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
    site.setPlaceholderText(f"e.g. {rig_file.EXAMPLES['site']} (letters, digits, _ and -)")
    rig.setPlaceholderText(f"e.g. {rig_file.EXAMPLES['rig']} (letters, digits, _ and -)")
    teams = sorted(p.parent.name for p in rig_file.CONFIGS.glob(f"*/{rig_file.RIGS_DIRNAME}")
                   if p.parent.name != "examples")
    if teams:
        site.setText(teams[0] if len(teams) == 1 else "")
        site.setToolTip("Teams with rigs already: " + ", ".join(teams))
    form.addRow("team", site)
    form.addRow("rig", rig)
    layout.addLayout(form)
    problem = QtWidgets.QLabel()
    layout.addWidget(problem)
    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                         | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    def path() -> str:
        return os.path.relpath(rig_file.CONFIGS / site.text().strip() / rig_file.RIGS_DIRNAME
                               / f"{rig.text().strip()}.json")

    def check() -> None:
        names = (site.text().strip(), rig.text().strip())
        why = ("" if not all(names) else
               "use letters, digits, _ and - only" if not all(rig_file.LABEL.fullmatch(n)
                                                              for n in names)
               else f"{path()} exists already: pick it instead" if os.path.exists(path()) else "")
        problem.setText(why or (f"Will make {path()}" if all(names) else ""))
        problem.setStyleSheet("color: #c0392b" if why else "color: gray")
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(
            all(names) and not why)

    site.textChanged.connect(check)
    rig.textChanged.connect(check)
    check()
    if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
        return None
    return path()


def main() -> None:
    """``rig-config [--rig <file>]``: pick a rig file (or make one), then go through it."""
    p = argparse.ArgumentParser(description="Make, check or change a rig file, and run its rig "
                                            "checks. Without --rig, a window asks which rig.")
    p.add_argument("--rig", help="the rig file, e.g. configs/<team>/rigs/<rig>.json; made when "
                                 "missing")
    args = p.parse_args()
    if args.rig:
        _edit(args.rig)
        return
    ui.app()
    chooser = _RigChooser()
    while chooser.exec() == QtWidgets.QDialog.DialogCode.Accepted:
        _edit(chooser.chosen)
        if os.path.isfile(chooser.chosen):  # a new rig saved: picked, so the summary shows it
            chooser.pick.set_path(chooser.chosen)
        chooser._refresh()


def _edit(path: str) -> None:
    """The rig window on ``path``; says on the console what became of the file."""
    if fill_rig(rig_file.draft(path), path) is None:
        print(f"rig-config: {path} not saved", file=sys.stderr)
        return
    print(f"saved: {path} ({time.strftime('%H:%M')})")
