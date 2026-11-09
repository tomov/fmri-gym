"""Rig measurement: make a rig's file, and measure the rig with the window, trigger line and audio
its runs open.

``rig-measure --rig <file>`` (``--long`` for the long one) is not a run and plays no
curriculum: what it measures, for how long and how many, is fixed here (:data:`QUICK`,
:data:`LONG`), and **with what is the rig file's** (:mod:`fmri_gym.config.rig`), so a
rig measurement is complete for the rig it runs on and asks nothing of what the rig does not have.
A test the rig gives no reason to run is recorded as "not run", with why. In order:

* ``display`` -- are flips locked to the refresh; missed refreshes.
* ``frames`` -- a test pattern (:class:`TestPattern`) played at each frame rate (and
  the rig's ``rig_measurement.frame_rates``) under each of :data:`LOADS` (``cpu``: every
  core kept busy), paced as a game block is: frames lost, frames shown a refresh
  late, pacing resets, and -- when the rig sends frame codes -- whether they marked
  every frame, in order, and given the rig's ``recording_hz``, for two samples each.
  The pacing is a copy of the game loop's, so a change to that loop is made here too.
* ``triggers`` -- the trigger outputs this PC has (serial ports with hardware behind
  them, parallel ports, HID trigger boxes, whether LSL reads back), with a warning
  when none is usable; then the rig's line: every bit alone and every code of its
  scheme sent (read back, over LSL), and scanner pulses counted when the rig waits
  for the scanner (a sequence, or the trigger box's test mode). Only the configured
  port is written to: another may be wired to a response box or an eye tracker, and a
  write that succeeds says nothing of where it went.
* ``controls`` -- the response devices: what the rig's participant can press, i.e.
  each control of the devices its ``"inputs"`` list (each stick axis calibrated
  first, :mod:`fmri_gym.standalone.pad_calibration`) and each of its ``"response_keys"``. Every
  response key is first checked automatically -- one that comes back through the
  event queue as itself -- and the input devices plugged in are listed; then each
  control is asked for on screen and pressed on the real device.
* ``photodiode`` -- the flip-to-photon offset, read as the rig's
  ``rig_measurement.photodiode`` says, and on the same flashes the ``audio`` test when the
  rig plays audio: a tone burst on every other flash, timed at the DAC, and at a
  microphone at the ear when the rig has one.

The screen says what each measurement does while it does it, then its verdict. A
measurement does not raise: it records a verdict for each of its tests (pass, fail,
offline, not run, and why), so one failure hides none of the others.

**The rig file** (:mod:`fmri_gym.config.rig`, one per rig) says what software cannot
see: the site and rig, the PI, the monitor or projector, the photodiode, the sound
path to the ear, the trigger hardware. A rig measurement is how a rig gets its file: when
it is missing or invalid, the rig measurement opens a form to fill it in (:func:`ensure_rig`);
a rig whose file says nothing of its triggers is asked for them, and the answer is
saved in the rig file: no rig measurement is anyone's default.

**What is filed.** A folder in the rig's data root, under :data:`SUBJECT` and its next
session: ``manifest.json``, one ``.npz`` per measurement, ``report.html`` for the
people who run the rig (a verdict per test, the numbers, charts, this rig's previous
rig measurements; one offline file), ``report.md``, and ``rigmeasurement.tsv`` -- one row of fixed
columns (:data:`COLUMNS`) with its BIDS sidecar. The data root's ``rigmeasurements.tsv``
holds every rig measurement filed there, read from their manifests.

**What the rig measured.** Every rig measurement that measured a latency also files it beside
the rig file (:func:`file_measured`, :func:`fmri_gym.config.rig.measured_latency_files`):
flip to light at the screen, flip to the sound card, sound card to the ear, with
every measurement's summary, the verdicts, the rig file as played, the software and
the machine (:func:`machine`: processor, memory, displays, GPUs, sound and input
devices, what is plugged into USB). The rig file is the rig's settings, shared; these
are one machine's measurements, and stay on it (``.gitignore``)::

    rig-measure --rig <rig file> [--long]
    rig-measure report <rig measurement folder>...   # re-file rig measurements already run
    rig-measure pool data/ /mnt/siteB/data/ --out rigmeasurements.tsv
"""

from __future__ import annotations

import argparse
import datetime
import glob
import html
import json
import math
import os
import platform
import random
import re
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterator, Sequence

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from ...outputs.audio import Audio
from ...save.bids import next_session, run_output
from ...config import rig as rig_file
from ...outputs.display import Display, check_monitor, is_locked, quit_like_esc
from ...inputs import (QUIT, ControlNames, Controller, Edge, InputDevice, InputDevices,
                     open_input_devices)
from ...inputs.keyboard import KEYCODES
from ..pad_calibration import calibrate, cue_order
from ..photodiode import (AudioRecorder, Verdict, _clicker, _light_verdict,
                         _log_clicks, _readout, _sound_verdict, run_flashes)
from ...run import EXPERIMENTER_KEY, _wait_for_control, _wait_for_duration, read_input_events
from ...sync.triggers import HID_DEVICES, LIFECYCLE_EVENTS, Codes, TriggerError, Triggers

#: The subject every rig measurement is filed under in the data root.
SUBJECT = "sub-rig"
#: What the frame test plays its rates under: nothing else, or every core kept busy.
LOADS = ("none", "cpu")
#: Seconds each measurement's verdict stays on screen.
_VERDICT_S = 1.5
#: Seconds the list of every test's verdict stays up at the end.
_SUMMARY_S = 5.0
_PASS = Verdict("pass", None)


@dataclass(frozen=True, slots=True)
class RigMeasurementSettings:
    """How long and how many each measurement of a rig measurement takes; what it measures with
    is the rig file's."""

    #: the BIDS task label the rig measurement is filed under
    task: str
    #: how long it takes, for the screen that offers it
    about: str
    #: flip intervals the display test times; for ``display_seconds`` instead, if given
    display_flips: int
    display_seconds: float | None
    #: the frame test's rates (the rig's own are added) and loads (:data:`LOADS`)
    frame_rates: tuple[float, ...]
    frame_loads: tuple[str, ...]
    #: seconds of frames per rate and load
    frame_seconds: float
    #: passes of every line and code; how long each is held, and the gap after it
    trigger_repeat: int
    trigger_hold_ms: float
    trigger_gap_ms: float
    #: scanner pulses to count; for ``pulse_seconds`` after the first instead, if given
    pulses: int
    pulse_seconds: float | None
    #: seconds allowed without a pulse, before the first or between two
    pulse_timeout_s: float
    #: seconds to press each control asked for
    control_timeout_s: float
    #: flashes; how long each is on, the range its gap is drawn from, and the settle first
    flashes: int
    flash_on_ms: float
    flash_gap_ms: tuple[float, float]
    flash_settle_ms: float
    #: the seed the flash gaps are drawn with
    flash_seed: int
    #: the sound-card input's rate
    samplerate: int

    def record(self) -> dict[str, Any]:
        """As the manifest records it."""
        return asdict(self)


#: Before a session: under a minute, most of it pressing the buttons.
QUICK = RigMeasurementSettings(
    task="rigMeasurement", about="under a minute", display_flips=60, display_seconds=None,
    frame_rates=(60, 30), frame_loads=("none",), frame_seconds=2, trigger_repeat=1,
    trigger_hold_ms=40, trigger_gap_ms=40, pulses=3, pulse_seconds=None,
    pulse_timeout_s=60, control_timeout_s=10, flashes=10, flash_on_ms=50,
    flash_gap_ms=(150, 250), flash_settle_ms=300, flash_seed=0, samplerate=48000)
#: Measuring the rig: once per rig, and after any hardware, driver or OS change.
LONG = RigMeasurementSettings(
    task="rigMeasurementLong", about="about 25 minutes", display_flips=60, display_seconds=120,
    frame_rates=(60, 30, 20, 50), frame_loads=("none", "cpu"), frame_seconds=60,
    trigger_repeat=20, trigger_hold_ms=100, trigger_gap_ms=400, pulses=3,
    pulse_seconds=600, pulse_timeout_s=60, control_timeout_s=10, flashes=850,
    flash_on_ms=100, flash_gap_ms=(400, 800), flash_settle_ms=1000, flash_seed=0,
    samplerate=48000)


@dataclass
class RigMeasurement:
    """One rig measurement under way: the rig, what it measures, the devices it opened, where it
    files, and what each measurement recorded."""

    rig: rig_file.RigConfig
    settings: RigMeasurementSettings
    display: Display
    triggers: Triggers
    audio: Audio
    inputs: InputDevices
    #: the folder it files in; the session and the BIDS label of it
    folder: str
    session: int
    label: str
    #: 1, or 2, 3... for a rig measurement filed again under the same label
    attempt: int
    #: why the trigger line did not open (:func:`prepare`); ``None`` when it did
    trigger_error: str | None
    #: each measurement's record, by its name, in the order taken
    measurements: dict[str, dict] = field(default_factory=dict)
    start_epoch: float = field(default_factory=time.time)
    t0: float = field(default_factory=time.perf_counter)

    @classmethod
    def open(cls, rig: rig_file.RigConfig, settings: RigMeasurementSettings,
             trigger_error: str | None) -> RigMeasurement:
        """Make its folder, then open the trigger line, the audio output, the input devices
        and the window, saying on stderr what each got.

        :raises ValueError: on a monitor this machine does not have, or an input device
            that is not plugged in.
        """
        session = next_session(rig.data_root, SUBJECT)
        out = run_output(rig.data_root, SUBJECT, settings.task, session, 1)
        os.makedirs(out.folder)
        print(f"output: {out.folder}", file=sys.stderr)
        check_monitor(rig.screen.monitor)
        triggers = Triggers(rig.triggers)
        print(f"triggers: {triggers.status()}", file=sys.stderr)
        audio = Audio(enabled=rig.has_audio)
        print(f"audio: {audio.status()}", file=sys.stderr)
        inputs = open_input_devices(rig.inputs, frozenset())
        print(f"inputs: {inputs.status()}", file=sys.stderr)
        screen = rig.screen
        display = Display(size=screen.size, fullscreen=screen.is_fullscreen,
                          vsync=screen.has_vsync, monitor=screen.monitor)
        return cls(rig, settings, display, triggers, audio, inputs, out.folder, session,
                   out.label, out.attempt, trigger_error)

    def now(self) -> float:
        """Seconds since the rig measurement opened."""
        return time.perf_counter() - self.t0

    def close(self) -> None:
        """Close the window, the audio output, the trigger line and the inputs."""
        self.display.close()
        self.audio.close()
        self.triggers.close()
        self.inputs.close()


# ---------------------------------------------------------------------------
# Before and after the measurements
# ---------------------------------------------------------------------------


def prepare(rig: rig_file.RigConfig) -> tuple[rig_file.RigConfig, str | None]:
    """Before the window: the rig's trigger settings, and a line the rig measurement can open.

    A rig whose file says nothing of its triggers is asked for them -- a
    preset, or one's own -- and the answer is saved in its file
    (:func:`_ask_triggers`). A trigger line that does not open fails the
    triggers test, and the rig measurement goes on without the line (said on the console
    and in the manifest), so the other measurements still run.

    :param rig: the rig to set up.
    :raises ValueError: when the rig has no trigger settings and none is chosen.
    :return: the rig to play on, and why its line did not open (``None`` when it did).
    """
    if not rig.has_trigger_setup:
        rig = _ask_triggers(rig)
    try:
        Triggers(rig.triggers).close()
        return rig, None
    except (TriggerError, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    print(f"rig measurement: the trigger line did not open ({error}); the triggers test fails, "
          "the other measurements run without codes", file=sys.stderr)
    return rig.without_trigger_line(), error


def _ask_triggers(rig: rig_file.RigConfig) -> rig_file.RigConfig:
    """The rig once its file has trigger settings, chosen in rig-config's Triggers step.

    :raises ValueError: with no screen for the form, or when it is cancelled.
    """
    where = rig.path
    fix = (f'{where} has no "triggers": set them with rig-config --rig {where} or in the '
           "file, or in the form a rig measurement opens")
    if not can_show_form():
        raise ValueError(f"rig measurement: {fix}; there is no screen here for the form")
    from .window import fill_rig
    if fill_rig(rig_file.load(where), where, need_triggers=True) is None:
        raise ValueError(f"rig measurement: {fix}; it was cancelled")
    print(f"rig measurement: trigger settings saved in {where}", file=sys.stderr)
    return rig_file.RigConfig.open(where)


def measure_rig(rig_path: str, settings: RigMeasurementSettings) -> bool:
    """Set the rig up and measure it: its file (:func:`ensure_rig`, :func:`prepare`), the
    start screen, each measurement and its verdict, every verdict; then file it all
    (:func:`finish`). ESC or Ctrl+C stops the measurements; what they measured is filed.

    :param rig_path: the rig's file, made in the form when it is missing or invalid.
    :param settings: how long and how many (:data:`QUICK`, :data:`LONG`).
    :return: ``True`` if a test failed or did not finish.
    :raises ValueError: when the rig file is not made, or a device does not open.
    """
    rig, trigger_error = prepare(ensure_rig(rig_path))
    print(f"rig: {rig.status()}", file=sys.stderr)
    rig_measurement = RigMeasurement.open(rig, settings, trigger_error)
    previous_handler = signal.signal(signal.SIGINT, quit_like_esc)
    try:
        _start_screen(rig_measurement)
        for name, measure in MEASUREMENTS.items():
            _measure(rig_measurement, name, measure)
        _summary_screen(rig_measurement)
    except KeyboardInterrupt:
        print("rig measurement: stopped; filing what it measured", file=sys.stderr)
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        rig_measurement.close()
    return finish(rig_measurement)


def finish(rig_measurement: RigMeasurement) -> bool:
    """File the rig measurement: its manifest, the report, the pooling row, the rig's data
    root's ``rigmeasurements.tsv`` rebuilt, and what it measured beside the rig file.

    :param rig_measurement: the rig measurement, its measurements taken (or stopped).
    :return: ``True`` if a test failed or did not finish (the report lists them).
    """
    tests = {test: {"status": "fail", "why": "did not finish (stopped before it)"}
             for test in TESTS}
    for entry in rig_measurement.measurements.values():
        tests.update(entry["tests"])
    m = {"subject": SUBJECT, "session": rig_measurement.session,
         "task": rig_measurement.settings.task,
         "run": {"label": rig_measurement.label, "attempt": rig_measurement.attempt},
         "start_time": datetime.datetime.fromtimestamp(rig_measurement.start_epoch).astimezone()
         .isoformat(timespec="seconds"),
         "rig": rig_measurement.rig.record(), "settings": rig_measurement.settings.record(),
         "trigger_error": rig_measurement.trigger_error,
         "measurements": rig_measurement.measurements,
         "summaries": {name: e["summary"] for name, e in rig_measurement.measurements.items()},
         "tests": tests, "versions": versions(), "platform": platform_info(),
         "machine": machine()}
    data_root = rig_measurement.rig.data_root
    write_outputs(rig_measurement.folder, m, data_root)
    file_measured(m, rig_measurement.folder, rig_measurement.rig.path)
    verdict = "PASS" if m["status"] == "pass" else f"FAIL ({m['failed_tests']})"
    print(f"rig measurement {verdict}: {os.path.join(rig_measurement.folder, 'report.html')} "
          f"(all rig measurements: {os.path.join(data_root, 'rigmeasurements.tsv')})",
          file=sys.stderr)
    return m["status"] != "pass"


# ---------------------------------------------------------------------------
# The measurements
# ---------------------------------------------------------------------------


def _start_screen(rig_measurement: RigMeasurement) -> None:
    """Say what this is and what to have ready; wait for the experimenter's key.

    :raises KeyboardInterrupt: on ESC or window close.
    """
    rig = rig_measurement.rig
    ready = ["Photodiode on the patch, bottom-right; microphone at the ear if there is one.",
             "The controls test asks for each button: press it on the participant's device."]
    if rig.triggers.sync.mode == "wait":
        ready.append("The triggers test counts scanner pulses: start a sequence, or switch "
                     "the trigger box to its test mode, when it asks.")
    rig_measurement.display.draw_text(
        f"Rig measurement ({rig_measurement.settings.about}) of {rig.path}\n"
        "This is a rig measurement, not a scanner run: no participant plays.\n\n"
        + "\n".join(ready) + f"\n\nPress {EXPERIMENTER_KEY} to start, ESC to stop.")
    _wait_for_control(rig_measurement.inputs, rig_measurement.display, EXPERIMENTER_KEY)


def _measure(rig_measurement: RigMeasurement, name: str, measure: Callable[..., None]) -> None:
    """Take one measurement: record its tests' verdicts, save its arrays, show the verdicts.

    :raises KeyboardInterrupt: on ESC or window close; the measurement is recorded first,
        with what it measured.
    """
    onset = rig_measurement.now()
    summary: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    tests: dict[str, Verdict] = {}
    try:
        measure(rig_measurement, summary, arrays, tests)
    except Exception as exc:  # noqa: BLE001 -- one that breaks fails its tests; the rest run
        for test in (t for t, m in TESTS.items() if m == name):
            tests.setdefault(test, Verdict("fail", f"{type(exc).__name__}: {exc}"))
    finally:
        if arrays:
            np.savez_compressed(os.path.join(rig_measurement.folder, f"{name}.npz"), **arrays)
        rig_measurement.measurements[name] = {
            "onset": onset, "offset": rig_measurement.now(), "summary": summary,
            "tests": {t: v._asdict() for t, v in tests.items()}}
        for test, v in tests.items():
            print(f"rig measurement: {test}: {v.status}" + (f" -- {v.why}" if v.why else ""),
                  file=sys.stderr)
    lines = [f"{t}: {v.status.upper()}" + (f"\n  {v.why[:160]}" if v.why else "")
             for t, v in tests.items()]
    rig_measurement.display.draw_text("Rig measurement\n\n" + "\n".join(lines))
    _wait_for_duration(rig_measurement.inputs, rig_measurement.display, _VERDICT_S)


def _summary_screen(rig_measurement: RigMeasurement) -> None:
    """Every test's verdict, for :data:`_SUMMARY_S`: a rig measurement often runs unattended, so
    nothing waits for a key; the report and the console keep the list."""
    tests = {t: v for e in rig_measurement.measurements.values() for t, v in e["tests"].items()}
    failed = [t for t, v in tests.items() if v["status"] == "fail"]
    lines = [f"{t}: {v['status'].upper()}" for t, v in tests.items()]
    head = (f"Rig measurement done: {len(failed)} test(s) failed, listed in the report"
            if failed else "Rig measurement done: every test passed")
    rig_measurement.display.draw_text(head + "\n\n" + "\n".join(lines))
    _wait_for_duration(rig_measurement.inputs, rig_measurement.display, _SUMMARY_S)


def _title(rig_measurement: RigMeasurement, text: str) -> None:
    rig_measurement.display.draw_text(f"Rig measurement: {text}")


def _measure_display(rig_measurement: RigMeasurement, summary: dict, arrays: dict,
                     tests: dict) -> None:
    info = rig_measurement.display.describe()
    n = rig_measurement.settings.display_flips
    if rig_measurement.settings.display_seconds is not None:
        n = int(rig_measurement.settings.display_seconds * (info["refresh_rate"] or 60))
    _title(rig_measurement,
           f"display\n\n{info['refresh_rate']} Hz, vsync {info['vsync']}: timing {n} "
                  f"flips (about {n / (info['refresh_rate'] or 60):.0f} s)\nthe screen "
                  "flickers grey meanwhile")
    _wait_for_duration(rig_measurement.inputs, rig_measurement.display, _VERDICT_S)
    stats = rig_measurement.display.measure_flips(n)
    arrays["intervals_ms"] = np.asarray(stats.pop("intervals_ms"))
    locked = is_locked(info, stats)
    summary.update(display=info, flips={"n": n, **stats}, locked=locked)
    tests["display"] = _PASS if locked else Verdict(
        "fail", f"flips are NOT locked to the refresh (vsync {info['vsync']}, "
                f"mean {stats['mean_ms']:.2f} ms, SD {stats['sd_ms']:.2f})")


def _measure_triggers(rig_measurement: RigMeasurement, summary: dict, arrays: dict,
                      tests: dict) -> None:
    _title(rig_measurement, "triggers")
    summary["outputs"] = discover()
    summary["triggers"] = rig_measurement.triggers.describe()
    if rig_measurement.trigger_error:
        tests["triggers"] = Verdict(
            "fail", f"the trigger line did not open: {rig_measurement.trigger_error}")
        return
    t = rig_measurement.triggers
    settings = rig_measurement.settings
    pulses = t.sync.mode == "wait"
    if not (t.enabled or pulses):
        tests["triggers"] = Verdict("not run", "this rig sends no codes and waits for no "
                                               "scanner (rig-config, Triggers step)")
        return
    if t.enabled:
        def on_send(label: str, value: int, number: int, total: int) -> None:
            _title(rig_measurement,
                   f"triggers\n\nsending {label} = {value}   ({number} of {total})\n"
                          f"on {t.active}")

        send_and_check(t, settings.trigger_repeat, settings.trigger_hold_ms / 1000,
                       settings.trigger_gap_ms / 1000, summary, arrays, on_send)
        readback = summary["sent"]["lsl_readback"] == "ok"
        _title(rig_measurement, f"triggers\n\nsent {summary['sent']['n']} codes on {t.active}\n"
                      + ("all read back from the network" if readback
                         else "check each in the recording: every line alone, then every code"))
        _wait_for_duration(rig_measurement.inputs, rig_measurement.display, _VERDICT_S)
    if pulses:
        _count_pulses(rig_measurement, summary, arrays)
    tests["triggers"] = _PASS


def _count_pulses(rig_measurement: RigMeasurement, summary: dict, arrays: dict) -> None:
    """Count the scanner's pulses, from the first.

    :raises TriggerError: when none comes in time, or the train is irregular.
    """
    settings = rig_measurement.settings
    times = receive_pulses(rig_measurement.display, rig_measurement.triggers.sync.key,
                           settings.pulses,
                           settings.pulse_timeout_s, settings.pulse_seconds)
    arrays["pulse_time"] = times - times[0]
    if len(times) < 3:
        raise TriggerError(f"only {len(times)} scanner pulses; need 3 for intervals")
    s = summary["pulses"] = pulse_summary(times)
    if s["missed"] or s["doubles"]:
        raise TriggerError(f"the pulse train is irregular: {s['missed']} missed, "
                           f"{s['doubles']} doubled (TR {s['tr_s']:.3f} s)")


# -- controls ----------------------------------------------------------------


class _Skipped(Exception):
    """ESC during the controls prompts: the keys left are skipped."""


#: Devices the kernel files as keyboards that no one presses: power keys, lid, video, sound.
_SYSTEM_INPUTS = re.compile(r"button|switch|video bus|hotkey|hda |headphone|speaker|wmi",
                            re.IGNORECASE)


def input_devices() -> list[str]:
    """The keyboards, button boxes and joysticks plugged in, by name.

    On Linux from ``/proc/bus/input/devices`` (every device, no permission
    needed): a button box presenting itself as a keyboard is listed by the
    name its maker gave it. Elsewhere, the joysticks pygame sees.
    """
    names = []
    if os.path.exists("/proc/bus/input/devices"):
        with open("/proc/bus/input/devices") as f:
            for block in f.read().split("\n\n"):
                name = re.search(r'^N: Name="(.*)"$', block, re.MULTILINE)
                handlers = re.search(r"^H: Handlers=(.*)$", block, re.MULTILINE)
                if not (name and handlers and re.search(r"\b(kbd|js\d+)\b", handlers[1])):
                    continue
                if not _SYSTEM_INPUTS.search(name[1]):
                    names.append(name[1])
        return names
    if not pygame.joystick.get_init():
        pygame.joystick.init()
    names += [f"joystick: {pygame.joystick.Joystick(i).get_name()}"
              for i in range(pygame.joystick.get_count())]
    return names


def _round_trip(inputs: InputDevice, name: str) -> bool:
    """A synthetic press of the keyboard key ``name`` comes back from ``inputs`` as ``name``."""
    read_input_events(inputs)
    for kind in (pygame.KEYDOWN, pygame.KEYUP):
        pygame.event.post(pygame.event.Event(kind, key=KEYCODES[name], mod=0, unicode="",
                                             scancode=0))
    return (name, Edge.DOWN) in read_input_events(inputs)


def _controllers(inputs: InputDevices) -> list[Controller]:
    return [d for d in inputs.devices if isinstance(d, Controller)]


def _controls_to_ask(inputs: InputDevices) -> dict[str, str]:
    """Each control of the controllers to press, buttons then stick directions: what it is on."""
    return {c: f"on the {d.name}" for d in _controllers(inputs)
            for c in (*cue_order(d.BUTTON_NAMES), *d.axis_controls())}


def _measure_controls(rig_measurement: RigMeasurement, summary: dict, arrays: dict,
                      tests: dict) -> None:
    rig = rig_measurement.rig
    keys = {**_controls_to_ask(rig_measurement.inputs), **rig.response_keys}
    if not keys:
        tests["controls"] = Verdict("not run", "this rig lists no controller and types no "
                                               "response keys (rig-config, Response devices step)")
        return
    summary["devices"] = input_devices()
    summary["inputs"] = rig_measurement.inputs.status()
    calibrations = [_calibrate(rig_measurement, d) for d in _controllers(rig_measurement.inputs)]
    summary["pad"] = calibrations[0] if calibrations else {"present": False}
    summary["automatic"] = {name: _round_trip(rig_measurement.inputs, name)
                            for name in rig.response_keys}
    sync = rig_measurement.triggers.sync
    ignore = sync.key if sync.mode == "wait" else None
    results: dict[str, dict] = {}
    timeout_s = rig_measurement.settings.control_timeout_s
    try:
        for name, label in keys.items():
            results[name] = _prompt(rig_measurement, name, label, keys, results, timeout_s, ignore)
    except _Skipped:
        pass
    summary["prompted"] = results
    latencies = [r["latency_s"] for r in results.values() if r.get("pressed")]
    summary.update(n_keys=len(keys), n_pressed=len(latencies),
                   median_latency_ms=float(np.median(latencies)) * 1000 if latencies else None)
    arrays["latency_s"] = np.array([results.get(k, {}).get("latency_s", np.nan) for k in keys])
    arrays["held_s"] = np.array([results.get(k, {}).get("held_s", np.nan) for k in keys])
    problems = [f"{k}: not read back as itself" for k, ok in summary["automatic"].items()
                if not ok]
    problems += [f"{c['device']}: {c['why']}" for c in calibrations if c.get("why")]
    problems += [_key_problem(k, results.get(k)) for k in keys
                 if not results.get(k, {}).get("pressed")]
    tests["controls"] = Verdict("fail", "; ".join(problems)) if problems else _PASS


def _calibrate(rig_measurement: RigMeasurement, device: Controller) -> dict[str, Any]:
    """Calibrate ``device``'s stick axes before asking for any control, and read it through that.

    Each axis has to be measured on the rig before a threshold means anything
    (see :mod:`fmri_gym.standalone.pad_calibration`), so the calibration runs here and is
    written for the device, which every later run on this rig with it reads.
    """
    if not device.AXIS_NAMES:
        return {"present": True, "device": device.device_name, "measured": False, "why": "",
                "note": "it has no stick"}
    result = calibrate(device, rig_measurement.display.screen, pygame.display.flip)
    if result is None:
        return {"present": True, "device": device.device_name, "measured": False,
                "why": "stopped before it finished; nothing written"}
    device.set_calibration((result.path, result.calibration))
    return {"present": True, "device": device.device_name, "why": "", **result.record()}


def _key_problem(name: str, result: dict | None) -> str:
    if result is None:
        return f"{name}: skipped (ESC)"
    got = f" (got {', '.join(result['strays'])})" if result["strays"] else ""
    return f"{name}: never pressed{got}"


def _prompt(rig_measurement: RigMeasurement, name: str, label: str, keys: dict, done: dict,
            timeout: float,
            ignore: str | None) -> dict[str, Any]:
    """Ask for ``name`` on screen; wait for it, noting any other key that comes instead.

    :raises _Skipped: on ESC.
    """
    marks = "   ".join(f"{k}{' ok' if done.get(k, {}).get('pressed') else ''}" for k in keys)
    _title(rig_measurement, f"controls\n\nPress  {name}"
                  f"{f'  ({label})' if label and label != name else ''}\n\n{marks}"
                  "\n\n(ESC: skip the rest; they fail)")
    read_input_events(rig_measurement.inputs)           # what was pressed before the prompt
    start = time.perf_counter()
    strays: list[str] = []
    while time.perf_counter() - start < timeout:
        events = _key_events(rig_measurement.inputs, ignore)
        for i, (down, got) in enumerate(events):
            if not down:
                continue
            if got != name:
                strays.append(got)
                continue
            pressed = time.perf_counter()
            return {"pressed": True, "latency_s": pressed - start,
                    "held_s": _held(rig_measurement.inputs, name, pressed, events[i + 1:]),
                    "strays": strays}
        rig_measurement.display.idle(time.perf_counter() + 0.002, poll=0.001)
    return {"pressed": False, "strays": strays}


def _key_events(inputs: InputDevice, ignore: str | None) -> list[tuple[bool, str]]:
    """Presses and releases since the last call, in order: ``(down, name)``; the
    scanner's key left out.

    :raises _Skipped: on ESC or window close.
    """
    out = []
    for name, edge in read_input_events(inputs):
        if name in (QUIT, ControlNames.K_ESCAPE) and edge is Edge.DOWN:
            raise _Skipped
        if name == ignore:
            continue
        out.append((edge is Edge.DOWN, name))
    return out


def _held(inputs: InputDevice, name: str, pressed: float, pending: list[tuple[bool, str]],
          limit: float = 2.0) -> float:
    """Seconds until ``name`` is released (``nan`` if not within ``limit``).

    :param pending: the events that came after the press in the same poll.
    """
    while time.perf_counter() - pressed < limit:
        if (False, name) in pending:
            return time.perf_counter() - pressed
        pending = _key_events(inputs, None)
        time.sleep(0.001)
    return float("nan")


# -- photodiode and audio --------------------------------------------------------


def _measure_photodiode(rig_measurement: RigMeasurement, summary: dict, arrays: dict,
                        tests: dict) -> None:
    rig = rig_measurement.rig.rig_measurement
    settings = rig_measurement.settings
    readout = rig.photodiode
    clicks = rig_measurement.audio.enabled
    if not clicks:
        tests["audio"] = Verdict("not run",
                                 "the rig plays no game audio (its \"has_audio\": false)")
    summary.update(display=rig_measurement.display.describe(), readout=readout, mic=rig.has_mic,
                   n=settings.flashes, on_ms=settings.flash_on_ms,
                   gap_ms=list(settings.flash_gap_ms), corner=rig.corner, patch_px=rig.patch_px)
    if readout == "none":
        tests["photodiode"] = Verdict("not run", "the rig has no photodiode (rig-config, "
                                                 "Rig measurement equipment step)")
        if not clicks:
            return
    recorder = _recorder(rig, settings.samplerate, tests) if readout == "soundcard" else None
    click, chunks, clicked = _clicker(rig_measurement.audio) if clicks else (None, [], [])
    what = " + ".join(w for w, on in (("photodiode", readout != "none"), ("audio", clicks)) if on)

    def caption(i: int) -> str:
        sound = ", with a click" if clicks and i % 2 == 0 else ""
        return f"Rig measurement: {what}, flash {i + 1} of {summary['n']}{sound}"

    try:
        arrays.update(run_flashes(
            rig_measurement.display, rig_measurement.triggers, summary["n"], summary["on_ms"],
            tuple(summary["gap_ms"]), random.Random(settings.flash_seed), summary["corner"],
            summary["patch_px"], click, settings.flash_settle_ms / 1000, bracket=False,
            caption=caption))
    finally:
        if recorder is not None:
            recorder.stop()
        if clicks:
            _log_clicks(rig_measurement.audio, chunks, clicked, arrays, summary, close=False)
    if recorder is not None:
        _readout(*recorder.signal(), arrays, summary)
    if "photodiode" not in tests:  # an input that did not open has failed it already
        tests["photodiode"] = _light_verdict(arrays, summary)
    if "audio" not in tests:
        tests["audio"] = _sound_verdict(arrays, clicks, None)


def _recorder(rig: rig_file.RigMeasurementConfig, samplerate: int,
              tests: dict) -> AudioRecorder | None:
    """The sound-card input, started; ``None`` (and the photodiode failed) if it will not open."""
    try:
        recorder = AudioRecorder(rig.input_device, samplerate, channels=2 if rig.has_mic else 1)
        recorder.start()
        return recorder
    except Exception as exc:  # noqa: BLE001 -- any device error fails the photodiode, not the rest
        tests["photodiode"] = Verdict("fail", f"the sound-card input did not open: {exc}")
        if rig.has_mic:
            tests["audio"] = Verdict("fail", f"the microphone's input did not open: {exc}")
        return None


# -- frames: a paced loop at each rate, under each load -----------------------------


def _measure_frames(rig_measurement: RigMeasurement, summary: dict, arrays: dict,
                    tests: dict) -> None:
    blocks = []
    rates = list(dict.fromkeys([*rig_measurement.settings.frame_rates,
                                *rig_measurement.rig.rig_measurement.frame_rates]))
    for load in rig_measurement.settings.frame_loads:
        for fps in rates:
            with _load(load):
                flips, codes, pacing_resets = _play_pattern(rig_measurement, fps, load)
            arrays[f"flip_time_{len(blocks)}"] = flips
            blocks.append(_judge_frames(rig_measurement, fps, load, flips, codes, pacing_resets))
    summary["blocks"] = blocks
    summary["played"] = [f"{b['fps']:g}/{b['load']}" for b in blocks]
    summary["refresh_rate"] = rig_measurement.display.refresh_rate
    summary["vsync"] = rig_measurement.display.vsync
    summary["recording_hz"] = rig_measurement.rig.rig_measurement.recording_hz
    for key in ("late", "lost", "pacing_resets"):
        summary[f"{key}_total"] = sum(b[key] for b in blocks)
    summary["triggers_ok"] = all(b["triggers"]["ok"] for b in blocks if b["triggers"])
    problems = [p for b in blocks for p in b["problems"]]
    tests["frames"] = Verdict("fail", "; ".join(problems)) if problems else _PASS


@contextmanager
def _load(kind: str) -> Iterator[None]:
    """``cpu``: a busy process on every core for the duration; ``none``: nothing."""
    if kind == "none":
        yield
        return
    busy = [subprocess.Popen([sys.executable, "-c", "while True: pass"])
            for _ in range(os.cpu_count() or 1)]
    try:
        time.sleep(0.5)  # let them take the cores before the frames start
        yield
    finally:
        for p in busy:
            p.kill()
            p.wait()


def _play_pattern(rig_measurement: RigMeasurement, fps: float,
                  load: str) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Play the test pattern at ``fps`` for the settings' seconds, paced as a real-time game
    block is, with a frame trigger on the flip of each step.

    :return: each step's flip time, its frame code (0: none; ``None`` when the rig sends
        no codes), and how many times the pacing dropped a debt of over a frame.
    :raises KeyboardInterrupt: on ESC or window close.
    """
    display, triggers = rig_measurement.display, rig_measurement.triggers
    env = TestPattern(caption=f"Rig measurement: frames at {fps:g} fps"
                              + ("" if load == "none" else f", {load} load"))
    env.reset(seed=0)
    dt = 1.0 / fps
    display.call_on_flip(triggers.episode_start)
    flip_t0 = display.draw_frame(env.render())
    end, next_t = flip_t0 + rig_measurement.settings.frame_seconds, flip_t0 + dt
    flips: list[float] = []
    codes: list[int] = []
    pacing_resets = 0
    try:
        while time.perf_counter() < end:
            _wait_for_duration(rig_measurement.inputs, display, math.inf, until=next_t)
            next_t += dt
            env.step(0)
            display.call_on_flip(triggers.frame)
            flip_t = display.draw_frame(env.render())
            if next_t + dt < flip_t:     # over a frame behind: drop the debt, as a block does
                pacing_resets += 1
                next_t = flip_t + dt
            flips.append(flip_t)
            codes.append(triggers.last_frame)
    finally:
        triggers.block_end()
    return (np.asarray(flips, dtype=float),
            np.asarray(codes, dtype=int) if triggers.enabled else None, pacing_resets)


def _judge_frames(rig_measurement: RigMeasurement, fps: float, load: str, flips: np.ndarray,
                  codes: np.ndarray | None, pacing_resets: int) -> dict[str, Any]:
    """Frames lost, frames a refresh late, pacing resets, and the frame triggers."""
    dt = 1.0 / fps
    display = rig_measurement.display
    period = 1.0 / display.refresh_rate if display.vsync and display.refresh_rate else None
    iv = np.diff(flips)
    # Locked, a frame lands on the refresh after its step: dt rounded up to refreshes,
    # alternating with dt rounded down when fps does not divide the refresh.
    slack = (np.ceil(dt / period - 1e-6) + 0.5) * period if period else 1.5 * dt
    # Expected from the frames' own span, not the seconds played: those also hold the
    # reset frame before the first step, which is no frame lost.
    expected = int(round((flips[-1] - flips[0]) / dt)) + 1 if len(flips) else 0
    b = {"fps": fps, "load": load, "n": int(len(flips)), "expected": expected,
         "lost": max(0, expected - len(flips)), "late": int(np.sum(iv > slack)),
         "pacing_resets": pacing_resets,
         "interval_median_ms": float(np.median(iv) * 1000) if len(iv) else None,
         "interval_max_ms": float(np.max(iv) * 1000) if len(iv) else None,
         "interval_sd_ms": float(np.std(iv) * 1000) if len(iv) else None,
         "locked": period is not None,
         "divides_refresh": (bool(abs(dt / period - round(dt / period)) < 1e-3)
                             if period else None),
         "triggers": _frame_marks(rig_measurement, codes, dt)}
    name = f"{fps:g} fps" + ("" if load == "none" else f" under {load} load")
    allowed = {"lost": 0, "late": 0, "pacing_resets": 0}
    what = {"lost": "frames lost", "late": "frames a refresh late",
            "pacing_resets": "pacing resets"}
    b["problems"] = [f"{name}: {b[k]} {what[k]}" for k in allowed if b[k] > allowed[k]]
    if b["triggers"] and not b["triggers"]["ok"]:
        b["problems"].append(f"{name}: {b['triggers']['why']}")
    return b


def _frame_marks(rig_measurement: RigMeasurement, codes: np.ndarray | None,
                 dt: float) -> dict[str, Any] | None:
    """Did the frame triggers mark every frame they should, in their cycle?

    ``frame_every`` thins them: every k-th frame gets the next code of the
    cycle ``1..2**frame_bits - 1``. A code that is missing, repeated or out of
    order would put the analysis a frame off. Given the recording's
    ``recording_hz``, each code must also last two samples, or a level-holding
    line (parallel) is read with codes merged.
    """
    s = rig_measurement.triggers.settings
    if codes is None or not rig_measurement.triggers.enabled or not s.on_frame:
        return None
    sent = np.flatnonzero(codes > 0)
    should = np.arange(0, len(codes), s.frame_every)
    cycle = 1 + np.arange(len(should)) % rig_measurement.triggers.codes.frame_mask
    out: dict[str, Any] = {"marked": int(len(sent)), "expected": int(len(should)),
                           "ok": True, "why": None}
    hold_ms = dt * s.frame_every * 1000
    out["code_ms"] = hold_ms
    if not (np.array_equal(sent, should) and np.array_equal(codes[sent], cycle)):
        out.update(ok=False, why=f"frame triggers marked {len(sent)} of {len(should)} frames "
                                 "or broke their cycle")
    hz = rig_measurement.rig.rig_measurement.recording_hz
    if hz and hold_ms * hz / 1000 < 2:
        out.update(ok=False, why=f"a frame code lasts {hold_ms:.1f} ms, under 2 samples at "
                                 f"{hz:g} Hz: raise frame_every")
    return out


#: Each measurement by its name, in the order a rig measurement takes them.
MEASUREMENTS: dict[str, Callable[[RigMeasurement, dict, dict, dict], None]] = {
    "display": _measure_display, "frames": _measure_frames, "triggers": _measure_triggers,
    "controls": _measure_controls, "photodiode": _measure_photodiode}


# ---------------------------------------------------------------------------
# Trigger outputs, codes and scanner pulses
# ---------------------------------------------------------------------------


def code_sequence(codes: Codes) -> list[tuple[str, int]]:
    """Every line alone, then every code of the scheme, as ``(label, value)``."""
    bits = [(f"bit {i}", 1 << i) for i in range(8)]
    frames = [(f"frame {v}", v) for v in range(1, codes.frame_mask + 1)]
    return bits + frames + [(name, codes.lifecycle(name)) for name in LIFECYCLE_EVENTS]


def send_codes(triggers: Triggers, repeat: int, hold_s: float, gap_s: float,
               on_send: Callable[[str, int, int, int], None] | None = None) -> list[dict]:
    """Send :func:`code_sequence` ``repeat`` times, each value held then cleared.

    :param on_send: called before each code with ``(label, value, number, total)``,
        to say on screen what goes out.
    :return: one record per value sent: label, value, wall time, send duration.
    """
    sequence = code_sequence(triggers.codes)
    print("sending, in this order:", ", ".join(f"{label}={v}" for label, v in sequence))
    sent = []
    for r in range(repeat):
        print(f"  pass {r + 1}/{repeat}")
        for label, value in sequence:
            if on_send is not None:
                on_send(label, value, len(sent) + 1, repeat * len(sequence))
            wall = time.time()
            took = triggers.pulse(value, hold_s)
            sent.append({"pass": r, "label": label, "value": value, "wall_time": wall,
                         "send_ms": took * 1000.0})
            time.sleep(gap_s)
    return sent


def detect_outputs() -> list[dict[str, Any]]:
    """The serial ports with hardware behind them, the parallel ports and the HID trigger
    boxes of :data:`~fmri_gym.sync.triggers.HID_DEVICES`; nothing is opened.

    Linux lists a ``ttyS`` for every slot its driver reserves, most with no
    UART behind them (``hwid`` ``n/a``); writes to those succeed and go
    nowhere, so they are left out.

    A HID box is listed once per digital port, as a rig names it (``device``,
    ``port``), with its device node in ``path``. Only Linux guards one with file
    permissions (``/dev/hidraw*``); elsewhere it is writable by anyone.

    :return: one dict per port: ``kind``, ``port``, ``what``, ``writable`` (by this user).
    """
    found = []
    try:
        from serial.tools import list_ports
    except ImportError:
        print("  serial: pyserial not installed, serial ports not listed (pip install pyserial)")
        list_ports = None
    for p in list_ports.comports() if list_ports else []:
        if p.hwid != "n/a":
            found.append({"kind": "serial", "port": p.device, "what": f"{p.description} {p.hwid}"})
    for dev in sorted(glob.glob("/dev/parport*")):
        found.append({"kind": "parallel", "port": dev, "what": ""})
    for f in found:
        f["writable"] = os.access(f["port"], os.R_OK | os.W_OK)
    try:
        import hid
    except ImportError:
        print("  hid: hidapi not installed, HID trigger boxes not listed (pip install hidapi)")
        hid = None
    for name, spec in HID_DEVICES.items() if hid else []:
        boxes = [d for d in hid.enumerate(spec.vendor_id, spec.product_id)
                 if d["interface_number"] in (spec.interface, -1)]
        for d in boxes:
            path = d["path"].decode(errors="replace")
            writable = not path.startswith("/dev/") or os.access(path, os.R_OK | os.W_OK)
            for port in spec.ports:
                found.append({"kind": "hid", "port": port, "device": name, "path": path,
                              "writable": writable,
                              "what": f"{d['product_string']} serial {d['serial_number']}"})
    return found


def _port_name(o: dict[str, Any]) -> str:
    """A port of :func:`detect_outputs` as a rig names it: ``/dev/ttyUSB0``, ``mcc-1208fs:A``."""
    return f"{o['device']}:{o['port']}" if o["kind"] == "hid" else o["port"]


def _lsl_inlet(source_id: str, timeout: float) -> Any:
    """An open inlet on the stream with ``source_id``, or ``None`` if none is found."""
    import pylsl
    found = pylsl.resolve_byprop("source_id", source_id, timeout=timeout)
    if not found:
        return None
    inlet = pylsl.StreamInlet(found[0])
    inlet.open_stream(timeout=timeout)
    return inlet


def _drain(inlet: Any, n: int, timeout: float) -> list[int]:
    """Up to ``n`` values from ``inlet``, waiting at most ``timeout`` seconds."""
    got: list[int] = []
    end = time.perf_counter() + timeout
    while len(got) < n and time.perf_counter() < end:
        sample, _ = inlet.pull_sample(timeout=0.05)
        if sample is not None:
            got.append(int(sample[0]))
    return got


def lsl_probe(timeout: float = 3.0) -> str:
    """Can a marker stream from this PC be found on the network and read back?

    The probe opens a stream of its own name, never the session's, so that a
    LabRecorder already recording the session's stream does not record it.

    :return: ``"OK"``, or what went wrong.
    """
    try:
        import pylsl
    except ImportError:
        return "pylsl not installed (pip install pylsl)"
    source = f"fmri_gym_check_{os.getpid()}"
    outlet = pylsl.StreamOutlet(pylsl.StreamInfo("fmri_gym_check", "Markers", 1, 0,
                                                 "int32", source))
    inlet = _lsl_inlet(source, timeout)
    if inlet is None:
        return "this PC's own test stream was not found (firewall, or no multicast?)"
    # An outlet drops what it pushes before the inlet's data connection lands.
    if not outlet.wait_for_consumers(timeout):
        return "this PC's own test stream was found but never connected"
    values = [1 << i for i in range(8)]
    for v in values:
        outlet.push_sample([v])
    got = _drain(inlet, len(values), timeout)
    inlet.close_stream()
    return "OK" if got == values else f"sent {values}, read back {got}"


def discover() -> dict[str, Any]:
    """List the trigger outputs of this PC; warn when none is usable."""
    print("trigger outputs on this PC:")
    outputs = detect_outputs()
    for o in outputs:
        fix = {"serial": "add yourself to the dialout group",
               "parallel": "add yourself to the lp group",
               "hid": f"give {o.get('path')} to your user (a udev rule)"}[o["kind"]]
        access = "writable" if o["writable"] else f"NOT writable by you: {fix}"
        print(f"  {o['kind']} {_port_name(o)}  {o['what']}  {access}")
    lsl = lsl_probe()
    print(f"  lsl: {lsl}")
    if not (lsl == "OK" or any(o["writable"] for o in outputs)):
        print("WARNING: no usable trigger output on this PC (no writable serial, parallel "
              "or HID port, no working LSL): nothing could send codes to a recording")
    return {"ports": outputs, "n_writable": sum(o["writable"] for o in outputs), "lsl": lsl}


def _pulses(key: str) -> list[float]:
    """Drain events; the arrival times of ``key`` presses.

    :raises KeyboardInterrupt: on ESC or window close.
    """
    now = time.perf_counter()
    times = []
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            raise KeyboardInterrupt
        if event.type == pygame.KEYDOWN and event.key == KEYCODES[key]:
            times.append(now)
    return times


def receive_pulses(display: Display, key: str, n: int, timeout: float,
                   seconds: float | None = None) -> np.ndarray:
    """Stamp ``n`` scanner pulses (``key`` presses) as they arrive.

    :param n: pulses to count (ignored when ``seconds`` is given).
    :param timeout: seconds allowed without a pulse, before the first or between two.
    :param seconds: count instead until this long after the first pulse.
    :return: ``perf_counter`` of each pulse, to the event poll (~1 ms).
    :raises TriggerError: when no pulse comes within ``timeout``.
    """
    times: list[float] = []
    display.draw_text(f"Waiting for scanner pulses (key {key!r})\n\nstart a sequence now, or "
                      "switch the trigger box to its test mode")
    last = time.perf_counter()
    goal = f"{seconds:g} s" if seconds else str(n)
    while not _done(times, n, seconds):
        new = _pulses(key)
        if new:
            times += new
            last = new[-1]
            display.draw_text(f"scanner pulses: {len(times)} (of {goal})")
        if time.perf_counter() - last > timeout:
            raise TriggerError(f"triggers test: {len(times)} of {goal} scanner pulses, then "
                               f"none for {timeout:g} s -- is a sequence running (or the "
                               f"trigger box in its test mode), the box plugged in, and does "
                               f"it type {key!r}?")
        display.idle(time.perf_counter() + 0.002, poll=0.001)
    return np.array(times if seconds else times[:n])


def _done(times: list[float], n: int, seconds: float | None) -> bool:
    if seconds:
        return bool(times) and time.perf_counter() - times[0] >= seconds
    return len(times) >= n


def pulse_summary(times: np.ndarray) -> dict[str, Any]:
    """TR and regularity of a pulse train: a long interval is a miss, a short one a double."""
    iv = np.diff(times)
    tr = float(np.median(iv))
    return {"n": int(len(times)), "tr_s": tr, "tr_train_s": float(np.mean(iv)),
            "sd_ms": float(np.std(iv) * 1000.0),
            "min_s": float(iv.min()), "max_s": float(iv.max()),
            "missed": int(np.sum(iv > 1.5 * tr)), "doubles": int(np.sum(iv < 0.5 * tr))}


def send_and_check(triggers: Triggers, repeat: int, hold_s: float, gap_s: float,
                   result: dict, arrays: dict,
                   on_send: Callable[[str, int, int, int], None] | None = None) -> None:
    """Send the code sequence; over LSL, read it back and refuse any difference.

    :param result: receives ``sent``: count, send-call timing, LSL read-back.
    :param arrays: receives ``sent_*``: every code sent, and when.
    :raises TriggerError: when the LSL stream is not visible or reads back otherwise.
    """
    s = triggers.settings
    # The session's own stream, found by the source id triggers._LSL gives it.
    inlet = _lsl_inlet(f"{s.lsl_stream_name}_markers", 3.0) if s.backend == "lsl" else None
    if s.backend == "lsl" and inlet is None:
        raise TriggerError(f"triggers test: the LSL stream {s.lsl_stream_name!r} is not "
                           "visible on the network: a recorder could not find it either")
    if inlet is not None:
        # An outlet drops what it pushes before the inlet's data connection
        # lands; the session's outlet is inside Triggers, so give it time
        # (a few ms here) instead of asking it. Too short fails below, loudly.
        time.sleep(0.5)
    sent = send_codes(triggers, repeat, hold_s, gap_s, on_send)
    for key in ("label", "value", "pass", "wall_time", "send_ms"):
        arrays[f"sent_{key}"] = np.array([x[key] for x in sent])
    ms = arrays["sent_send_ms"]
    result["sent"] = {"n": len(sent), "send_median_ms": float(np.median(ms)),
                      "send_max_ms": float(ms.max()), "lsl_readback": "n/a"}
    print(f"sent {len(ms)} codes over {triggers.active}; send call median "
          f"{np.median(ms):.3f} ms, max {ms.max():.3f} ms")
    if inlet is None:
        print("now check the recording: each bit alone and in order, then every code as "
              "printed above")
        return
    values = [x["value"] for x in sent]
    got = _drain(inlet, len(values), 3.0)
    inlet.close_stream()
    result["sent"]["lsl_readback"] = "ok" if got == values else "differs"
    if got != values:
        raise TriggerError(f"triggers test: LSL read back {len(got)} codes that differ from "
                           f"the {len(values)} sent: {got[:20]}")
    print(f"LSL: all {len(got)} codes read back from the network as sent")


# ---------------------------------------------------------------------------
# The test pattern the frame test plays
# ---------------------------------------------------------------------------


_PATTERN_SIZE = (320, 240)
_PATTERN_PATCH = 48


class TestPattern(gym.Env):
    """A frame counter with a toggling corner patch (see the module docstring)."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: str = "rgb_array", caption: str = "") -> None:
        self.render_mode = render_mode
        self.caption = caption
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(0, np.iinfo(np.int64).max, shape=(1,),
                                            dtype=np.int64)
        self.n = 0
        self._font: pygame.font.Font | None = None

    def reset(self, *, seed: int | None = None,
              options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.n = 0
        return np.array([0]), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.n += 1
        return np.array([self.n]), 0.0, False, False, {}

    def render(self) -> np.ndarray:
        if self._font is None:
            pygame.font.init()
            self._font = pygame.font.SysFont(None, 22)
        surface = pygame.Surface(_PATTERN_SIZE)
        for i, line in enumerate((self.caption, f"frame {self.n}")):
            surface.blit(self._font.render(line, True, (170, 170, 170)), (8, 8 + 22 * i))
        if self.n % 2:
            surface.fill((255, 255, 255), (_PATTERN_SIZE[0] - _PATTERN_PATCH, _PATTERN_SIZE[1] - _PATTERN_PATCH, _PATTERN_PATCH, _PATTERN_PATCH))
        return pygame.surfarray.array3d(surface).transpose(1, 0, 2)


# ---------------------------------------------------------------------------
# The rig file, the pooling row, report.md, and the command line
# ---------------------------------------------------------------------------


#: The tests a rig measurement reports on, and the measurement that runs each: the photodiode's
#: flashes carry the audio test's clicks too.
TESTS = {"display": "display", "triggers": "triggers", "controls": "controls",
         "photodiode": "photodiode", "audio": "photodiode", "frames": "frames"}
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

#: The pooled table: ``(column, path into the manifest, description, units)``.
COLUMNS: list[tuple[str, tuple[str, ...], str, str | None]] = [
    ("site", ("rig", "site"), "Site, from the rig file", None),
    ("rig", ("rig", "rig"), "Rig at the site, from the rig file", None),
    ("pi", ("rig", "pi"), "PI's initials, from the rig file", None),
    ("modality", ("rig", "modality"), "Recording modality of the rig", None),
    ("date", ("start_time",), "When the rig measurement started, ISO 8601 with UTC offset", None),
    ("task", ("task",), f"{QUICK.task} (before a session) or {LONG.task} (measuring the "
     "rig): the BIDS task label", None),
    ("subject", ("subject",), "BIDS subject the rig measurement was filed under", None),
    ("session", ("session",), "BIDS session number", None),
    ("label", ("run", "label"), "BIDS run label of the rig measurement", None),
    ("attempt", ("run", "attempt"), "1, or 2, 3... for one filed again under its label",
     None),
    ("status", ("status",), "pass, or fail", None),
    ("failed_tests", ("failed_tests",), "The tests that failed, comma-separated", None),
    ("failure", ("failure",), "Why each failed, test: reason; ...", None),
    *[(f"{t}_status", ("tests", t, "status"), f"The {t} test: pass, fail, offline (read "
       "in the recording) or not run", None) for t in TESTS],
    ("fmri_gym_version", ("versions", "fmri_gym"), "fmri-gym package version", None),
    ("git_commit", ("versions", "git"), "git describe of the checkout (-dirty: edited)", None),
    ("os", ("platform", "system"), "Operating system", None),
    ("python", ("versions", "python"), "Python version", None),
    ("cpu", ("machine", "cpu"), "Processor", None),
    ("cores", ("machine", "cores"), "Processor cores", None),
    ("memory_gb", ("machine", "memory_gb"), "Memory", "GB"),
    ("gpus", ("machine", "gpus"), "Graphics cards, separated by ;", None),
    ("monitor", ("rig", "monitor"), "Monitor or projector, from the rig file", None),
    ("photodiode", ("rig", "photodiode"), "Photodiode, from the rig file", None),
    ("audio_path", ("rig", "audio_path"), "Sound path to the ear, from the rig file", None),
    ("trigger_hardware", ("rig", "trigger_hardware"), "Trigger hardware, from the rig file",
     None),
    ("display_driver", ("summaries", "display", "display", "driver"), "SDL video driver", None),
    ("display_size", ("summaries", "display", "display", "size"), "Window size obtained", "px"),
    ("fullscreen", ("summaries", "display", "display", "fullscreen"), "Fullscreen window", None),
    ("refresh_hz", ("summaries", "display", "display", "refresh_rate"), "Refresh rate reported", "Hz"),
    ("vsync", ("summaries", "display", "display", "vsync"), "Flips wait for the vertical blank", None),
    ("flip_n", ("summaries", "display", "flips", "n"), "Flip intervals measured", None),
    ("flip_mean_ms", ("summaries", "display", "flips", "mean_ms"), "Mean flip interval", "ms"),
    ("flip_sd_ms", ("summaries", "display", "flips", "sd_ms"), "SD of the flip intervals", "ms"),
    ("flip_max_ms", ("summaries", "display", "flips", "max_ms"), "Longest flip interval", "ms"),
    ("flip_late", ("summaries", "display", "flips", "late"), "Intervals over 1.5 periods (missed refresh)",
     None),
    ("display_locked", ("summaries", "display", "locked"), "Flips locked to the refresh", None),
    ("trigger_backend", ("summaries", "triggers", "triggers", "settings", "backend"),
     "Trigger output of the config: null, lsl, serial, parallel", None),
    ("sync_mode", ("summaries", "triggers", "triggers", "settings", "sync", "mode"),
     "Run start: wait (for the scanner), send, none", None),
    ("ports_writable", ("summaries", "triggers", "outputs", "n_writable"),
     "Serial/parallel ports with hardware that this user may write", None),
    ("lsl_probe", ("summaries", "triggers", "outputs", "lsl"), "LSL stream found and read back: OK or why not",
     None),
    ("codes_sent", ("summaries", "triggers", "sent", "n"), "Trigger codes sent", None),
    ("send_median_ms", ("summaries", "triggers", "sent", "send_median_ms"), "Median send call duration",
     "ms"),
    ("send_max_ms", ("summaries", "triggers", "sent", "send_max_ms"), "Longest send call", "ms"),
    ("lsl_readback", ("summaries", "triggers", "sent", "lsl_readback"),
     "LSL codes read back as sent: ok, differs, n/a (not LSL)", None),
    ("pulses_n", ("summaries", "triggers", "pulses", "n"), "Scanner pulses counted", None),
    ("tr_median_s", ("summaries", "triggers", "pulses", "tr_s"), "Median pulse interval", "s"),
    ("tr_train_s", ("summaries", "triggers", "pulses", "tr_train_s"),
     "Mean pulse interval over the train, on this PC's clock", "s"),
    ("pulse_sd_ms", ("summaries", "triggers", "pulses", "sd_ms"), "SD of the pulse intervals", "ms"),
    ("pulses_missed", ("summaries", "triggers", "pulses", "missed"), "Intervals over 1.5 TR", None),
    ("pulses_doubled", ("summaries", "triggers", "pulses", "doubles"), "Intervals under 0.5 TR", None),
    ("controls_keys", ("summaries", "controls", "n_keys"), "Keys asked for", None),
    ("controls_pressed", ("summaries", "controls", "n_pressed"), "Keys pressed when asked",
     None),
    ("controls_latency_ms", ("summaries", "controls", "median_latency_ms"),
     "Median time from the prompt to the press (a person's, not the device's)", "ms"),
    ("pad_device", ("summaries", "controls", "pad", "device"),
     "The controller found, whose buttons are read as keys", None),
    ("pad_axis_on", ("summaries", "controls", "pad", "axes", ControlNames.CONTROLLER_AXIS_LEFTX, "on"),
     "Fraction of the left stick's horizontal travel that counts as pushed", None),
    ("pad_corrections", ("summaries", "controls", "pad", "corrections"),
     "Wrong controls pressed, and asked again, while the calibration confirmed each", None),
    ("input_devices", ("summaries", "controls", "devices"),
     "Keyboards, button boxes and joysticks plugged in", None),
    ("frames_blocks", ("summaries", "frames", "played"),
     "Frame-rate blocks played, as fps/load", None),
    ("frames_late", ("summaries", "frames", "late_total"),
     "Frames shown a refresh later than their rate allows, all blocks", None),
    ("frames_lost", ("summaries", "frames", "lost_total"), "Frames not played, all blocks",
     None),
    ("frames_pacing_resets", ("summaries", "frames", "pacing_resets_total"),
     "Stalls that restarted the frame schedule, all blocks", None),
    ("frames_triggers_ok", ("summaries", "frames", "triggers_ok"),
     "Every block's frame triggers marked every frame, in their cycle", None),
    ("diode_readout", ("summaries", "photodiode", "readout"),
     "Where the diode was read: soundcard (here) or recording (offline)", None),
    ("flashes_n", ("summaries", "photodiode", "n"), "Flashes shown", None),
    ("photon_matched", ("summaries", "photodiode", "offset", "n_matched"), "Flashes the diode saw", None),
    ("photon_median_ms", ("summaries", "photodiode", "offset", "median_ms"), "Flip to photodiode, median",
     "ms"),
    ("photon_sd_ms", ("summaries", "photodiode", "offset", "sd_ms"), "Flip to photodiode, SD", "ms"),
    ("photon_min_ms", ("summaries", "photodiode", "offset", "min_ms"), "Flip to photodiode, min", "ms"),
    ("photon_max_ms", ("summaries", "photodiode", "offset", "max_ms"), "Flip to photodiode, max", "ms"),
    ("photon_drift_ms_per_min", ("summaries", "photodiode", "offset", "drift_ms_per_min"),
     "Flip to photodiode, trend over the run (runs of a minute or more)", "ms/min"),
    ("audio_device", ("summaries", "photodiode", "audio_out", "device"), "Sound output device", None),
    ("audio_delay_ms", ("summaries", "photodiode", "audio_out", "delay_ms"),
     "Delay from flip to DAC the audio output chose", "ms"),
    ("click_dac_median_ms", ("summaries", "photodiode", "click_dac_offset", "median_ms"),
     "Flip to click at the DAC, median", "ms"),
    ("sound_matched", ("summaries", "photodiode", "sound_offset", "n_matched"), "Clicks the microphone heard",
     None),
    ("sound_median_ms", ("summaries", "photodiode", "sound_offset", "median_ms"),
     "Flip to sound at the microphone, median", "ms"),
    ("sound_sd_ms", ("summaries", "photodiode", "sound_offset", "sd_ms"), "Flip to sound, SD", "ms"),
    ("sound_min_ms", ("summaries", "photodiode", "sound_offset", "min_ms"), "Flip to sound, min", "ms"),
    ("sound_max_ms", ("summaries", "photodiode", "sound_offset", "max_ms"), "Flip to sound, max", "ms"),
    ("sound_drift_ms_per_min", ("summaries", "photodiode", "sound_offset", "drift_ms_per_min"),
     "Flip to sound, trend over the run (runs of a minute or more)", "ms/min"),
    ("acoustic_median_ms", ("summaries", "photodiode", "acoustic_offset", "median_ms"),
     "DAC to sound at the microphone (amplifier, tubes, earphones), median", "ms"),
    ("acoustic_sd_ms", ("summaries", "photodiode", "acoustic_offset", "sd_ms"),
     "DAC to sound, SD", "ms"),
    ("av_median_ms", ("summaries", "photodiode", "av_offset", "median_ms"),
     "Photodiode to sound (audio behind video), median", "ms"),
    ("av_sd_ms", ("summaries", "photodiode", "av_offset", "sd_ms"), "Photodiode to sound, SD", "ms"),
]
#: The table's column names, in order.
NAMES = tuple(c[0] for c in COLUMNS)


# ---------------------------------------------------------------------------
# Rig file
# ---------------------------------------------------------------------------


def ensure_rig(path: str) -> rig_file.RigConfig:
    """The rig to set up; when its file is missing or invalid, the form to
    fill it in first (:func:`fmri_gym.standalone.rig_setup.window.fill_rig`), then the file.

    The form needs the ``gui`` extra and a screen; without them, or when it is
    cancelled, the rig file's own error is raised.

    :param path: ``--rig``, the rig file.
    :raises ValueError: when there is still no valid rig file, saying how to make one.
    """
    try:
        return rig_file.RigConfig.open(path)
    except ValueError as error:
        if not can_show_form():
            raise
        print(f"{error}\nrig measurement: opening the rig form", file=sys.stderr)
        from .window import fill_rig
        if fill_rig(rig_file.draft(path), path) is None:
            raise
    return rig_file.RigConfig.open(path)


def can_show_form() -> bool:
    """PySide6 is installed and there is a screen to show it on."""
    try:
        import PySide6  # noqa: F401
    except ImportError:
        return False
    return sys.platform != "linux" or bool(os.environ.get("DISPLAY")
                                           or os.environ.get("WAYLAND_DISPLAY"))


# ---------------------------------------------------------------------------
# Filing
# ---------------------------------------------------------------------------


def write_outputs(folder: str, m: dict, data_root: str | None = None) -> None:
    """The verdict from ``m["tests"]``; then the manifest, the TSV row, its sidecar, the
    reports (``report.md``, ``report.html``), and the data root's integrated table.

    :param m: the rig measurement's manifest with ``tests``, ``rig`` and ``summaries``; the
        overall ``status``, ``failed_tests`` and ``failure`` are set here.
    :param data_root: the BIDS tree the rig measurement was filed in: its ``rigmeasurements.tsv``
        (every rig measurement under it, one row each) is rebuilt, and the HTML report
        compares this rig measurement with the rig's previous ones.
    """
    bad = [t for t, v in m["tests"].items() if v["status"] == "fail"]
    m["failed_tests"] = ",".join(bad) or None
    m["failure"] = "; ".join(f"{t}: {m['tests'][t]['why']}" for t in bad) or None
    m["status"] = "fail" if bad else "pass"
    _write_json(os.path.join(folder, "manifest.json"), m)
    with open(os.path.join(folder, "rigmeasurement.tsv"), "w") as f:
        f.write("\t".join(c[0] for c in COLUMNS) + "\n" + "\t".join(row(m)) + "\n")
    _write_json(os.path.join(folder, "rigmeasurement.json"), sidecar())
    with open(os.path.join(folder, "report.md"), "w") as f:
        f.write(report(m))
    history = rows([data_root]) if data_root else [dict(zip(NAMES, row(m)))]
    if data_root:
        write_table(os.path.join(data_root, "rigmeasurements.tsv"), history)
    with open(os.path.join(folder, "report.html"), "w") as f:
        f.write(render_html(m, folder, history))


def write_table(path: str, table: list[dict[str, str]]) -> None:
    """A TSV of rows (:func:`rows`) with its BIDS sidecar (``.json``) beside it."""
    with open(path, "w") as f:
        f.write("\n".join("\t".join(r) for r in _lines(table)) + "\n")
    _write_json(os.path.splitext(path)[0] + ".json", sidecar())


def _lines(table: list[dict[str, str]]) -> list[list[str]]:
    return [list(NAMES), *([r[n] for n in NAMES] for r in table)]


def rows(roots: list[str]) -> list[dict[str, str]]:
    """Every rig measurement filed under ``roots``, one row each, oldest first, read from each
    one's ``manifest.json``."""
    out = []
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, "**", "manifest.json"), recursive=True)):
            with open(path) as f:
                m = json.load(f)
            if "summaries" in m:
                out.append(dict(zip(NAMES, row(m))))
    return sorted(out, key=lambda r: r["date"])


def row(manifest: dict) -> list[str]:
    """The manifest's values for :data:`COLUMNS`, as TSV cells."""
    return [_cell(_get(manifest, path)) for _, path, _, _ in COLUMNS]


def sidecar() -> dict[str, dict[str, str]]:
    """The BIDS-style description of the TSV's columns."""
    out = {}
    for name, _, description, units in COLUMNS:
        out[name] = {"Description": description, **({"Units": units} if units else {})}
    return out


def pool(roots: list[str]) -> list[str]:
    """Every rig measurement under ``roots`` (:func:`rows`), as the lines of one TSV."""
    return ["\t".join(r) for r in _lines(rows(roots))]


def _get(d: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def _cell(v: Any) -> str:
    """A TSV cell: ``n/a`` for nothing, BIDS-style; no tab or newline inside."""
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        return format(v, ".6g")
    if isinstance(v, list):  # a size reads 1024x768; names read a; b
        numbers = all(isinstance(x, (int, float)) for x in v)
        return ("x" if numbers else "; ").join(map(str, v))
    return re.sub(r"\s+", " ", str(v))


def versions() -> dict[str, str]:
    """What ran the rig measurement: fmri-gym, its git checkout, and the libraries timing
    depends on."""
    from importlib.metadata import version

    import numpy
    import pygame
    try:
        git = subprocess.run(["git", "-C", _REPO, "describe", "--always", "--dirty"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        git = ""
    return {"fmri_gym": version("fmri-gym"), "git": git or "n/a",
            "python": platform.python_version(), "numpy": numpy.__version__,
            "pygame": pygame.version.ver, "sdl": ".".join(map(str, pygame.get_sdl_version()))}


def platform_info() -> dict[str, str]:
    """The machine: OS, architecture, host name."""
    return {"system": platform.platform(), "machine": platform.machine(),
            "hostname": socket.gethostname()}


def machine() -> dict[str, Any]:
    """The machine a rig measurement ran on, and what is plugged into it: for provenance, and to
    tell two rigs (or one rig before and after a change) apart when their numbers differ.

    Each part is read from the operating system without new dependencies, and is
    ``None`` (or empty) where the system does not say.
    """
    displays, inputs = _pygame_devices()
    return {**platform_info(), "python": platform.python_version(), "cpu": _cpu(),
            "cores": os.cpu_count(), "memory_gb": _memory_gb(), "displays": displays,
            "gpus": _gpus(), "audio_devices": _audio_devices(),
            "input_devices": inputs, "usb_devices": _usb_devices(),
            "trigger_ports": detect_outputs()}


def _pygame_devices() -> tuple[list[dict[str, Any]], list[str]]:
    """The monitors and the input devices, read with pygame's display up (it opens no
    window); pygame's display is left as it was found."""
    was_up = pygame.display.get_init()
    try:
        pygame.display.init()
        displays = _displays()
        pygame.joystick.init()
        settled = time.monotonic() + 0.5  # controllers arrive as events, ~0.1 s after init
        while time.monotonic() < settled:
            pygame.event.pump()
            time.sleep(0.05)
    except pygame.error:  # no screen here: no monitor to list
        displays = []
    try:
        return displays, input_devices()
    finally:
        if not was_up:
            pygame.joystick.quit()
            pygame.display.quit()


def _run(*command: str) -> str:
    """A system tool's output; empty when it is missing, fails or hangs."""
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout if out.returncode == 0 else ""


def _cpu() -> str | None:
    if sys.platform == "darwin":
        return _run("sysctl", "-n", "machdep.cpu.brand_string").strip() or None
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            model = re.search(r"^model name\s*:\s*(.+)$", f.read(), re.MULTILINE)
        if model:
            return model[1].strip()
    return platform.processor() or None


def _memory_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        return None


def _displays() -> list[dict[str, Any]]:
    """Every monitor the system has, its desktop size and refresh rate."""
    sizes = pygame.display.get_desktop_sizes()
    rates = (pygame.display.get_desktop_refresh_rates()
             if hasattr(pygame.display, "get_desktop_refresh_rates") else [None] * len(sizes))
    return [{"size": f"{w}x{h}", "refresh_hz": rate} for (w, h), rate in zip(sizes, rates)]


def _gpus() -> list[str]:
    if sys.platform == "darwin":
        out = _run("system_profiler", "SPDisplaysDataType", "-json")
        try:
            return [g.get("sppci_model", g.get("_name", "?"))
                    for g in json.loads(out)["SPDisplaysDataType"]] if out else []
        except (ValueError, KeyError, TypeError):
            return []
    lines = _run("lspci", "-mm").splitlines()
    return [line for line in lines if re.search(r'"(VGA|3D|Display)', line)]


def _audio_devices() -> list[dict[str, Any]]:
    """Every sound device, with its channels and default rate, the defaults marked."""
    try:
        import sounddevice
        devices = sounddevice.query_devices()
        default_in, default_out = sounddevice.default.device
    except Exception:  # noqa: BLE001 -- no PortAudio: nothing to list
        return []
    return [{"name": d["name"], "hostapi": sounddevice.query_hostapis(d["hostapi"])["name"],
             "inputs": d["max_input_channels"], "outputs": d["max_output_channels"],
             "default_samplerate": d["default_samplerate"],
             "default": [k for k, i in (("input", default_in), ("output", default_out))
                         if i == d["index"]]}
            for d in devices]


def _usb_devices() -> list[str]:
    """What is plugged into USB: maker, product and vendor:product ids."""
    if os.path.isdir("/sys/bus/usb/devices"):
        devices = sorted(glob.glob("/sys/bus/usb/devices/*"))
        return [f"{_sysfs(d, 'manufacturer')} {_sysfs(d, 'product')} "
                f"({_sysfs(d, 'idVendor')}:{_sysfs(d, 'idProduct')})".strip()
                for d in devices if _sysfs(d, "product")]
    if sys.platform == "darwin":
        # macOS 15 moved the USB tree to SPUSBHostDataType; earlier ones have SPUSBDataType.
        for kind in ("SPUSBHostDataType", "SPUSBDataType"):
            out = _run("system_profiler", kind, "-json")
            try:
                devices = _usb_items(json.loads(out)[kind]) if out else []
            except (ValueError, KeyError, TypeError):
                devices = []
            if devices:
                return devices
    return []


def _sysfs(device: str, name: str) -> str:
    """One attribute of a Linux USB device; empty when it has none."""
    try:
        with open(os.path.join(device, name)) as f:
            return f.read().strip()
    except OSError:
        return ""


def _usb_items(items: list[dict]) -> list[str]:
    """The devices in macOS's USB tree, depth first (either report's keys)."""
    out = []
    for item in items:
        vendor = item.get("vendor_id", item.get("USBDeviceKeyVendorID"))
        product = item.get("product_id", item.get("USBDeviceKeyProductID"))
        if vendor or product:
            maker = item.get("manufacturer", item.get("USBDeviceKeyVendorName", ""))
            out.append(f"{maker} {item.get('_name', '?')} ({vendor}:{product})".strip())
        out += _usb_items(item.get("_items", []))
    return out


def measured(m: dict, folder: str) -> dict[str, Any]:
    """What a rig measurement measured on its rig, as filed beside the rig file.

    ``latency`` holds when what a frame shows reaches the participant, from its
    flip: each the rig measurement's summary of its offsets (median, spread, count), and
    ``None`` when the rig measurement could not measure it (no photodiode, no microphone,
    a diode read in the MEG/EEG recording) or its test did not pass. The rest is
    how it was measured and on what: the rig measurement, its verdicts, every
    measurement's summary, the rig file as it played, the machine and the software.

    :param m: the rig measurement's manifest, filed (:func:`write_outputs`).
    :param folder: the rig measurement's folder.
    """
    photodiode = m["summaries"].get("photodiode", {})

    def latency(test: str, key: str) -> dict | None:
        summary = photodiode.get(key)
        passed = m["tests"][test]["status"] == "pass"
        return summary if passed and summary and "median_ms" in summary else None

    return {"date": m["start_time"],
            "rig_measurement": {"folder": folder, "label": m["run"]["label"],
                                "task": m["task"], "status": m["status"],
                                "failed_tests": m["failed_tests"], "tests": m["tests"]},
            "latency": {"flip_to_photon": latency("photodiode", "offset"),
                        "flip_to_dac": latency("audio", "click_dac_offset"),
                        "dac_to_ear": latency("audio", "acoustic_offset")},
            "summaries": m["summaries"], "rig": m["rig"], "machine": m["machine"],
            "versions": m["versions"]}


def file_measured(m: dict, folder: str, rig_path: str) -> str | None:
    """Write what the rig measurement measured beside the rig file
    (:func:`fmri_gym.config.rig.next_measured_latency_file`); nothing when it measured
    no latency.

    :param m: the rig measurement's manifest, filed.
    :param folder: the rig measurement's folder.
    :param rig_path: the rig file it set up.
    :return: the file written, or ``None``.
    """
    values = measured(m, folder)
    if not any(values["latency"].values()):
        print("rig measurement: no latency measured (no photodiode or microphone read, or their "
              "tests failed): nothing filed beside the rig", file=sys.stderr)
        return None
    path = rig_file.next_measured_latency_file(rig_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _write_json(path, values)
    print(f"rig measurement: what it measured on the rig: {path}", file=sys.stderr)
    return path


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.write("\n")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def report(m: dict) -> str:
    """``report.md``: the verdicts, then each test's numbers, for a person to read."""
    rig = m["rig"]
    verdict = "PASS" if m["status"] == "pass" else f"FAIL ({m['failed_tests']})"
    lines = [f"# Rig measurement {rig['site']}/{rig['rig']}: {verdict}", "",
             f"{m['start_time']} · `{m['run']['label']}` "
             f"(attempt {m['run']['attempt']}) · fmri-gym {m['versions']['fmri_gym']} "
             f"(`{m['versions']['git']}`)", "",
             "| test | verdict | why |", "|---|---|---|"]
    for test, v in m["tests"].items():
        status = f"**{v['status'].upper()}**" if v["status"] == "fail" else v["status"]
        lines.append(f"| {test} | {status} | {_cell(v['why']) if v['why'] else ''} |")
    lines += ["", "## Rig", "", "| | |", "|---|---|"]
    lines += [f"| {k} | {_cell(v)} |" for k, v in rig.items()]
    for test, section in (("display", _md_display), ("frames", _md_frames),
                          ("triggers", _md_triggers), ("controls", _md_controls),
                          ("photodiode", _md_photodiode), ("audio", _md_audio)):
        status = m["tests"][test]["status"]
        lines += ["", f"## {test.capitalize()}: {status}", ""]
        measured = m["summaries"].get(TESTS[test])
        if measured:
            lines += section(measured)
    lines += ["", "## This computer"]
    for title, head, table in computer(m):
        lines += ["", f"### {title}", "", "| " + " | ".join(head) + " |",
                  "|" + "---|" * len(head)]
        lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in table]
    return "\n".join(lines) + "\n"


def computer(m: dict) -> list[tuple[str, list[str], list[list[Any]]]]:
    """The computer a rig measurement ran on and what was plugged into it, as titled tables.

    :param m: the rig measurement's manifest.
    :return: ``(title, column names, rows)`` per table; a table with nothing found says so.
    """
    mc, p = m["machine"], m["platform"]
    none = [["none found"]]
    return [
        ("Computer", ["", ""], [[k, v] for k, v in (
            ("system", p.get("system")), ("architecture", p.get("machine")),
            ("host", p.get("hostname")), ("processor", mc.get("cpu")), ("cores", mc.get("cores")),
            ("memory (GB)", mc.get("memory_gb")), ("graphics", mc.get("gpus")))]),
        ("Screens", ["screen", "size", "refresh (Hz)"],
         [[i, d.get("size"), d.get("refresh_hz")] for i, d in enumerate(mc.get("displays", []))]
         or none),
        ("Sound devices", ["name", "system", "inputs", "outputs", "rate (Hz)", "default for"],
         [[d.get("name"), d.get("hostapi"), d.get("inputs"), d.get("outputs"),
           d.get("default_samplerate"), d.get("default")] for d in mc.get("audio_devices", [])]
         or none),
        ("Response devices (keyboards, button boxes, controllers)", ["name"],
         [[n] for n in mc.get("input_devices", [])] or none),
        ("Trigger ports", ["kind", "port", "what", "writable"],
         [[o.get("kind"), _port_name(o), o.get("what"), o.get("writable")]
          for o in mc.get("trigger_ports", [])] or none),
        ("USB devices", ["device"], [[d] for d in mc.get("usb_devices", [])] or none),
        ("Software", ["", "version"], [[k, v] for k, v in m["versions"].items()]),
    ]


def _md_display(d: dict) -> list[str]:
    s, f = d["display"], d["flips"]
    return [f"- {s['driver']}, {_cell(s['size'])} px, fullscreen {_cell(s['fullscreen'])}, "
            f"monitor {s['monitor']}, {s['refresh_rate']} Hz, vsync {_cell(s['vsync'])}",
            f"- {f['n']} flips: mean {f['mean_ms']:.2f} ms, SD {f['sd_ms']:.2f}, "
            f"max {f['max_ms']:.2f}; missed a refresh: {f['late']}",
            f"- **{'locked to the refresh' if d['locked'] else 'NOT locked to the refresh'}**"]


def _md_triggers(t: dict) -> list[str]:
    o = t["outputs"]
    lines = [f"- ports: {', '.join(map(_port_name, o['ports'])) or 'none with hardware'} "
             f"({o['n_writable']} writable); LSL: {o['lsl']}"]
    if not (o["n_writable"] or o["lsl"] == "OK"):
        lines.append("- **warning: no usable trigger output on this PC**")
    if "triggers" not in t:
        return lines
    s = t["triggers"]["settings"]
    lines.append(f"- config: backend {s['backend']} ({t['triggers']['active']}), "
                 f"start {s['sync']['mode']}")
    if "sent" in t:
        x = t["sent"]
        check = ("read back as sent" if x["lsl_readback"] == "ok"
                 else "**check them in the recording**")
        lines.append(f"- sent {x['n']} codes, send call median {x['send_median_ms']:.3f} ms, "
                     f"max {x['send_max_ms']:.3f} ms: {check}")
    if "pulses" in t:
        p = t["pulses"]
        lines.append(f"- scanner pulses: {p['n']}, TR {p['tr_s']:.3f} s (median), "
                     f"{p['tr_train_s']:.6f} s over the train, SD {p['sd_ms']:.1f} ms, "
                     f"missed {p['missed']}, doubled {p['doubles']}")
    return lines


def _md_frames(f: dict) -> list[str]:
    lines = [f"- the test pattern played at each rate, under each load, paced as a game; "
             f"refresh {f['refresh_rate']} Hz"
             + (f", recording at {f['recording_hz']:g} Hz" if f.get("recording_hz") else "")]
    if not f.get("vsync"):
        lines.append("- **vsync off: frames were paced by the clock alone, not the refresh "
                     "(see the display test); late frames cannot be told from jitter**")
    lines += ["", "| fps | load | frames | lost | late | pacing resets "
                  "| interval median / SD / max ms | frame triggers |",
              "|---|---|---|---|---|---|---|---|"]
    for b in f["blocks"]:
        t = b["triggers"]
        marks = ("off" if t is None else f"{t['marked']}/{t['expected']}, "
                 f"{t['code_ms']:.1f} ms each" + ("" if t["ok"] else f" **{t['why']}**"))
        fits = " (does not divide the refresh)" if b["divides_refresh"] is False else ""
        ms = " / ".join("n/a" if b[k] is None else f"{b[k]:.2f}"
                        for k in ("interval_median_ms", "interval_sd_ms", "interval_max_ms"))
        lines.append(f"| {b['fps']:g}{fits} | {b['load']} | {b['n']}/{b['expected']} | "
                     f"{b['lost']} | {b['late']} | {b['pacing_resets']} | {ms} | {marks} |")
    return lines


def _md_controls(c: dict) -> list[str]:
    lines = [f"- input devices: {', '.join(c['devices']) or 'none listed'}",
             f"- {_pad_line(c.get('pad') or {})}"]
    for key, auto in c["automatic"].items():
        p = c["prompted"].get(key)
        asked = ("skipped" if p is None else
                 f"pressed {p['latency_s']:.2f} s after asked, held "
                 f"{p['held_s'] * 1000:.0f} ms" 
                 if p["pressed"] else "**never pressed**")
        strays = f" (other keys: {', '.join(p['strays'])})" if p and p["strays"] else ""
        lines.append(f"- `{key}`: {'read back' if auto else '**not read back**'}; "
                     f"{asked}{strays}")
    return lines


def _diode_text(readout: str) -> str:
    """How the photodiode was read, in words."""
    return {"soundcard": "the diode read on the sound card",
            "recording": "the diode read in the recording (match offline)"}.get(
                readout, "no photodiode on this rig")


def _md_photodiode(p: dict) -> list[str]:
    lines = [f"- {p['n']} flashes of {p['on_ms']:g} ms, {_diode_text(p['readout'])}"]
    return lines + _md_offsets(p, (("offset", "flip → photodiode"),))


def _md_audio(p: dict) -> list[str]:
    if "audio_out" not in p:
        return ["- the audio output did not open"]
    a = p["audio_out"]
    lines = [f"- a tone burst on every other flash through {a.get('device')}, "
             f"delay chosen {a.get('delay_ms', 0):.1f} ms; microphone: "
             f"{'on input 1' if p.get('mic') else 'none'}"]
    return lines + _md_offsets(p, (("click_dac_offset", "flip → click at the DAC"),
                                ("sound_offset", "flip → sound at the microphone"),
                                ("acoustic_offset", "DAC → sound at the microphone"),
                                ("av_offset", "photodiode → sound")))


def _md_offsets(p: dict, keys: tuple[tuple[str, str], ...]) -> list[str]:
    """A table of the offsets under ``keys`` that were measured."""
    rows = [(label, p[key]) for key, label in keys if key in p]
    if not rows:
        return []
    lines = ["", "| offset | median ms | SD | min | max | drift ms/min | seen |",
             "|---|---|---|---|---|---|---|"]
    for label, o in rows:
        if not o.get("n_matched"):
            lines.append(f"| {label} | none seen | | | | | 0/{o['n']} |")
            continue
        lines.append(f"| {label} | {o['median_ms']:.2f} | {o['sd_ms']:.2f} | "
                     f"{o['min_ms']:.2f} | {o['max_ms']:.2f} | "
                     f"{_cell(o.get('drift_ms_per_min'))} | {o['n_matched']}/{o['n']} |")
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="rig-measure", description="Measure a rig, making its rig file first (a form, when "
        "it is missing or invalid). report and pool re-file and stack the "
        "rig measurements already filed.")
    p.add_argument("--rig", help="the rig file to set up")
    p.add_argument("--long", action="store_true",
                   help=f"measure the rig at length ({LONG.about}): once per rig, and after "
                        f"any hardware, driver or OS change; without it, {QUICK.about}")
    sub = p.add_subparsers(dest="command")
    f = sub.add_parser("report", help="re-file rig measurements already run: their reports and the "
                                      "data root's rigmeasurements.tsv, with this version")
    f.add_argument("folders", nargs="+", help="rig measurement folders")
    f.add_argument("--data-root", default="data")
    q = sub.add_parser("pool", help="stack every rig measurement under the roots into one table")
    q.add_argument("roots", nargs="+")
    q.add_argument("--out", help="file to write (default: stdout)")
    args = p.parse_args(argv)
    if args.command is None:
        if args.rig is None:
            p.error("give --rig <rig file> to set up, or a command: report, pool")
        sys.exit(1 if measure_rig(args.rig, LONG if args.long else QUICK) else 0)
    if args.command == "report":
        for folder in args.folders:
            with open(os.path.join(folder, "manifest.json")) as fh:
                m = json.load(fh)
            if "summaries" not in m:
                raise ValueError(f"{folder}: its manifest is not a rig measurement's")
            write_outputs(folder, m, args.data_root)
            print(f"{os.path.join(folder, 'report.html')}")
        print(f"all rig measurements: {os.path.join(args.data_root, 'rigmeasurements.tsv')}")
        return
    text = "\n".join(pool(args.roots)) + "\n"
    if args.out:
        with open(args.out, "w") as out:
            out.write(text)
    else:
        sys.stdout.write(text)


# ---------------------------------------------------------------------------
# report.html
# ---------------------------------------------------------------------------


_W, _H = 680, 220  # a chart's box, before its margins
_M = {"l": 52, "r": 16, "t": 12, "b": 34}

_CSS = """
:root { color-scheme: light;
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,0.10);
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
  --good:#0ca30c; --critical:#d03b3b; --good-ink:#006300; --critical-ink:#b42f2f; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,0.10);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --good-ink:#0ca30c; --critical-ink:#e66767; } }
:root[data-theme="dark"] { color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,0.10);
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --good-ink:#0ca30c; --critical-ink:#e66767; }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width: 760px; margin: 0 auto; padding: 32px 16px 64px; }
header .meta { color: var(--ink-2); font-size: 13px; }
h1 { font-size: 22px; margin: 0 0 4px; font-weight: 600; }
h3 { font-size: 15px; margin: 20px 0 6px; font-weight: 600; }
h2 { font-size: 17px; margin: 40px 0 8px; font-weight: 600; display:flex; gap:10px;
  align-items:center; }
.hero { font-size: 48px; font-weight: 600; line-height: 1.1; margin: 20px 0 4px; }
.card { background: var(--surface); border-radius: 10px; box-shadow: 0 0 0 1px var(--ring);
  padding: 16px; margin: 12px 0; overflow-x: auto; }
.note { color: var(--ink-2); font-size: 13px; margin: 6px 0; }
.status { display:inline-flex; align-items:center; gap:6px; font-weight:600; font-size:13px;
  white-space: nowrap; }
.status i { font-style: normal; width:18px; height:18px; border-radius:50%;
  display:inline-grid; place-items:center; color:#fff; font-size:12px; }
.pass i { background: var(--good); } .pass { color: var(--good-ink); }
.fail i { background: var(--critical); } .fail { color: var(--critical-ink); }
.other i { background: var(--axis); color: var(--ink); } .other { color: var(--ink-2); }
.tiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:12px; }
.tile { background: var(--surface); border-radius: 10px; box-shadow: 0 0 0 1px var(--ring);
  padding: 12px 14px; }
.tile .label { color: var(--ink-2); font-size: 13px; }
.tile .value { font-size: 24px; font-weight: 600; }
.tile .sub { color: var(--muted); font-size: 12px; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th { text-align: left; color: var(--ink-2); font-weight: 500; }
th, td { padding: 6px 8px; border-bottom: 1px solid var(--grid); vertical-align: top; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
details summary { cursor: pointer; color: var(--ink-2); font-size: 13px; margin-top: 8px; }
.legend { display:flex; gap:16px; font-size:13px; color:var(--ink-2); margin: 4px 0 8px;
  flex-wrap: wrap; }
.legend span::before { content:""; display:inline-block; width:10px; height:10px;
  border-radius:50%; margin-right:6px; background: var(--c); vertical-align: -1px; }
svg { display:block; max-width:100%; height:auto; }
svg text { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
svg .grid { stroke: var(--grid); stroke-width: 1; }
svg .axis { stroke: var(--axis); stroke-width: 1; }
svg [data-tip] { cursor: default; outline: none; }
svg [data-tip]:hover, svg [data-tip]:focus { opacity: 0.75; }
#tip { position: fixed; pointer-events: none; background: var(--surface); color: var(--ink);
  box-shadow: 0 0 0 1px var(--ring), 0 4px 16px rgba(0,0,0,0.12); border-radius: 8px;
  padding: 6px 10px; font-size: 13px; display: none; max-width: 280px; z-index: 2; }
code { font-size: 12px; }
"""

_JS = """
const tip = document.getElementById('tip');
function show(el, x, y) { tip.textContent = el.dataset.tip; tip.style.display = 'block';
  tip.style.left = Math.min(x + 14, innerWidth - tip.offsetWidth - 8) + 'px';
  tip.style.top = (y + 14) + 'px'; }
document.querySelectorAll('[data-tip]').forEach(el => {
  el.setAttribute('tabindex', '0');
  el.addEventListener('pointermove', e => show(el, e.clientX, e.clientY));
  el.addEventListener('pointerleave', () => tip.style.display = 'none');
  el.addEventListener('focus', () => { const r = el.getBoundingClientRect();
    show(el, r.left, r.bottom); });
  el.addEventListener('blur', () => tip.style.display = 'none');
});
"""


def _e(v: Any) -> str:
    return html.escape("" if v is None else str(v))


def _num(v: Any, fmt: str = ".2f", unit: str = "") -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "n/a"
    return f"{v:{fmt}}{unit}"


def _status(status: str) -> str:
    kind, icon = {"pass": ("pass", "✓"), "fail": ("fail", "✕")}.get(status, ("other", "–"))
    return f'<span class="status {kind}"><i aria-hidden="true">{icon}</i>{_e(status)}</span>'


# ---------------------------------------------------------------------------
# Charts (inline SVG)
# ---------------------------------------------------------------------------


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    """Round tick values covering ``lo..hi``."""
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    step = 10 ** np.floor(np.log10(raw))
    step *= next(m for m in (1, 2, 2.5, 5, 10) if m * step >= raw)
    start = np.floor(lo / step) * step
    count = int(np.ceil((hi - start) / step - 1e-9)) + 1  # the last tick at or past hi
    return [float(start + i * step) for i in range(count)]


def _frame(xt: Sequence[float], yt: Sequence[float], sx, sy, xlabel: str,
           ylabel: str) -> list[str]:
    """Gridlines, axis and tick labels of a chart."""
    out = []
    for y in yt:
        out.append(f'<line class="grid" x1="{_M["l"]}" x2="{_M["l"] + _W}" y1="{sy(y):.1f}" '
                   f'y2="{sy(y):.1f}"/><text x="{_M["l"] - 6}" y="{sy(y) + 4:.1f}" '
                   f'text-anchor="end">{y:g}</text>')
    base = _M["t"] + _H
    out.append(f'<line class="axis" x1="{_M["l"]}" x2="{_M["l"] + _W}" y1="{base}" y2="{base}"/>')
    for x in xt:
        out.append(f'<text x="{sx(x):.1f}" y="{base + 16}" text-anchor="middle">{x:g}</text>')
    out.append(f'<text x="{_M["l"] + _W / 2}" y="{base + 31}" text-anchor="middle">'
               f'{_e(xlabel)}</text>')
    out.append(f'<text x="12" y="{_M["t"] + _H / 2}" text-anchor="middle" '
               f'transform="rotate(-90 12 {_M["t"] + _H / 2})">{_e(ylabel)}</text>')
    return out


def _svg(parts: list[str], label: str) -> str:
    w, h = _M["l"] + _W + _M["r"], _M["t"] + _H + _M["b"]
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{_e(label)}">'
            + "".join(parts) + "</svg>")


def _span(values: np.ndarray) -> tuple[float, float, int]:
    """The 0.5th-99.5th percentile range, padded, and how many values fall outside it:
    a stray value would otherwise squash the rest into a line."""
    lo, hi = np.percentile(values, [0.5, 99.5])
    pad = (hi - lo) * 0.1 or 1.0
    lo, hi = lo - pad, hi + pad
    return float(lo), float(hi), int(np.sum((values < lo) | (values > hi)))


def _outside(n: int) -> str:
    return (f'<p class="note">{n} value{"s" if n > 1 else ""} beyond this range, left out of '
            f"the chart; the min and max are in the numbers given with it.</p>" if n else "")


def histogram(values: np.ndarray, unit: str, label: str, bins: int = 40) -> str:
    """Columns of how many values fall in each bin: one series, one hue."""
    values = values[np.isfinite(values)]
    if not len(values):
        return ""
    lo, hi, outside = _span(values)
    values = values[(values >= lo) & (values <= hi)]
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi + 1e-9))
    xt = _ticks(lo, hi)
    yt = _ticks(0, counts.max())
    sx = lambda v: _M["l"] + (v - xt[0]) / (xt[-1] - xt[0]) * _W  # noqa: E731
    sy = lambda v: _M["t"] + _H - v / yt[-1] * _H  # noqa: E731
    parts = _frame(xt, yt, sx, sy, unit, "count")
    for c, a, b in zip(counts, edges[:-1], edges[1:]):
        if not c:
            continue
        x0, x1 = sx(a) + 1, sx(b) - 1  # the 2px surface gap between neighbours
        w = min(24.0, max(1.0, x1 - x0))
        x0 += (x1 - x0 - w) / 2
        top, base = sy(c), _M["t"] + _H
        r = min(4.0, w / 2, base - top)
        parts.append(f'<path fill="var(--s1)" d="M{x0:.1f},{base} V{top + r:.1f} '
                     f'Q{x0:.1f},{top:.1f} {x0 + r:.1f},{top:.1f} H{x0 + w - r:.1f} '
                     f'Q{x0 + w:.1f},{top:.1f} {x0 + w:.1f},{top + r:.1f} V{base} Z" '
                     f'data-tip="{c} between {a:.2f} and {b:.2f} {_e(unit)}"/>')
    return _svg(parts, label) + _outside(outside)


def dots(series: list[tuple[str, str, np.ndarray, np.ndarray]], xlabel: str, ylabel: str,
         label: str) -> str:
    """Points per series, ``(name, colour var, x, y)``; at most three (the all-pairs limit)."""
    xs = np.concatenate([s[2] for s in series])
    ys = np.concatenate([s[3][np.isfinite(s[3])] for s in series])
    if not len(ys):
        return ""
    lo, hi, outside = _span(ys)
    xt, yt = _ticks(0, float(xs.max())), _ticks(lo, hi)
    # Many points: small and ringless, or the rings bury each other's fills.
    dense = len(ys) > 200
    r, ring = ("2.5", 'fill-opacity="0.8"') if dense else ("4", 'stroke="var(--surface)" '
                                                              'stroke-width="2"')
    sx = lambda v: _M["l"] + (v - xt[0]) / (xt[-1] - xt[0]) * _W  # noqa: E731
    sy = lambda v: _M["t"] + _H - (v - yt[0]) / (yt[-1] - yt[0]) * _H  # noqa: E731
    parts = _frame(xt, yt, sx, sy, xlabel, ylabel)
    for name, colour, x, y in series:
        for xi, yi in zip(x, y):
            if np.isfinite(yi) and lo <= yi <= hi:  # the rest is counted in _outside
                parts.append(f'<circle cx="{sx(xi):.1f}" cy="{sy(yi):.1f}" r="{r}" '
                             f'fill="var({colour})" {ring} '
                             f'data-tip="{_e(name)}, flash {int(xi) + 1}: {yi:.2f} ms"/>')
    return _svg(parts, label) + _outside(outside)


def sparkline(values: np.ndarray, width: int = 180, height: int = 28) -> str:
    """A line of ``values`` over time, down-sampled to the widest value per pixel."""
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return ""
    buckets = np.array_split(values, min(width, len(values)))
    peaks = np.array([b.max() for b in buckets])
    lo, hi = float(values.min()), float(peaks.max()) or 1.0
    span = hi - lo or 1.0
    pts = " ".join(f"{i / (len(peaks) - 1) * width:.1f},{height - 2 - (v - lo) / span * (height - 4):.1f}"
                   for i, v in enumerate(peaks))
    return (f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'aria-hidden="true"><polyline points="{pts}" fill="none" stroke="var(--s1)" '
            f'stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round"/></svg>')


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _html_npz(folder: str, pattern: str) -> Any:
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    return np.load(paths[-1], allow_pickle=True) if paths else None


def _html_tiles(m: dict) -> str:
    rc = m["summaries"]
    tiles = []
    d = rc.get("display")
    if d:
        tiles.append(("Refresh", f"{d['display']['refresh_rate']} Hz",
                      "locked" if d["locked"] else "NOT locked"))
        tiles.append(("Flip interval SD", _num(d["flips"]["sd_ms"], ".2f", " ms"),
                      f"{d['flips']['late']} missed refreshes"))
    f = rc.get("frames")
    if f:
        tiles.append(("Late frames", f"{f['late_total']}",
                      f"{f['lost_total']} lost, {len(f['blocks'])} blocks"))
    p = rc.get("photodiode", {})
    if p.get("offset", {}).get("n_matched"):
        tiles.append(("Flip → photodiode", _num(p["offset"]["median_ms"], ".1f", " ms"),
                      f"median, SD {_num(p['offset']['sd_ms'])} ms"))
    if p.get("sound_offset", {}).get("n_matched"):
        tiles.append(("Flip → sound", _num(p["sound_offset"]["median_ms"], ".1f", " ms"),
                      "at the microphone"))
    elif p.get("click_dac_offset", {}).get("n_matched"):
        tiles.append(("Flip → sound out", _num(p["click_dac_offset"]["median_ms"], ".1f", " ms"),
                      "at the DAC"))
    t = rc.get("triggers", {})
    if "sent" in t:
        tiles.append(("Codes sent", str(t["sent"]["n"]),
                      "read back" if t["sent"]["lsl_readback"] == "ok" else "check the recording"))
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="label">{_e(a)}</div><div class="value">{_e(b)}</div>'
        f'<div class="sub">{_e(c)}</div></div>' for a, b, c in tiles) + "</div>"


def _html_table(head: Sequence[str], rows: Sequence[Sequence[Any]], nums: Sequence[int] = ()) -> str:
    th = "".join(f'<th class="{"num" if i in nums else ""}">{_e(h)}</th>'
                 for i, h in enumerate(head))
    body = "".join("<tr>" + "".join(
        f'<td class="{"num" if i in nums else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>"
        for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def _html_display(m: dict, folder: str) -> str:
    d = m["summaries"]["display"]
    s, f = d["display"], d["flips"]
    out = [f'<p class="note">{_e(s["driver"])}, {_e("x".join(map(str, s["size"])))} px, '
           f'{"fullscreen" if s["fullscreen"] else "windowed"}, monitor {s["monitor"]}, '
           f'{s["refresh_rate"]} Hz, vsync {"on" if s["vsync"] else "off"}. {f["n"]} flips: '
           f'mean {f["mean_ms"]:.2f} ms, SD {f["sd_ms"]:.2f}, max {f["max_ms"]:.2f}; '
           f'{f["late"]} missed a refresh.</p>']
    z = _html_npz(folder, "display.npz")
    if z is not None:
        out.append('<p class="note">Flip intervals: one tight column at the refresh period '
                   'is a locked screen.</p>')
        out.append(histogram(np.asarray(z["intervals_ms"]), "ms between flips",
                             "Histogram of flip intervals"))
    return "".join(out)


def _html_section(section: Callable[[dict, str], str], m: dict, folder: str,
                  verdict: dict | None) -> str:
    """A measurement's section; for one that broke part-way and left its summary
    half-written (see :func:`_measure`), a note that it stopped, and why."""
    try:
        return section(m, folder)
    except (KeyError, IndexError, TypeError):
        why = (verdict or {}).get("why")
        return (f'<p class="note">This measurement stopped before it finished'
                f'{": " + _e(why) if why else ""}.</p>')


def _html_frames(m: dict, folder: str) -> str:
    f = m["summaries"]["frames"]
    out = [f'<p class="note">A test pattern played at each rate and load, paced as a game '
           f'block is; refresh {f["refresh_rate"]} Hz. Each line is that block\'s frame '
           f'intervals over its duration: flat is steady, spikes are late frames.</p>']
    if not f.get("vsync"):
        out.append('<p class="note"><b>vsync was off:</b> frames were paced by the clock '
                   'alone; late frames cannot be told from jitter.</p>')
    z = _html_npz(folder, "frames.npz")
    rows = []
    for i, b in enumerate(f["blocks"]):
        flips = f"flip_time_{i}"
        line = sparkline(np.diff(z[flips]) * 1000) if z is not None and flips in z else ""
        t = b["triggers"]
        marks = "off" if t is None else (f"{t['marked']}/{t['expected']}"
                                         + ("" if t["ok"] else f" — {_e(t['why'])}"))
        rows.append([f"{b['fps']:g}" + ("" if b["divides_refresh"] is not False else "*"),
                     _e(b["load"]), f"{b['n']}/{b['expected']}", b["lost"], b["late"],
                     b["pacing_resets"], _num(b["interval_median_ms"]),
                     _num(b.get("interval_sd_ms")), _num(b["interval_max_ms"]), line, marks])
    out.append(_html_table(["fps", "load", "frames", "lost", "late", "resets", "median ms", "SD ms",
                       "max ms", "intervals over the block", "frame triggers"], rows,
                      nums=(2, 3, 4, 5, 6, 7, 8)))
    out.append('<p class="note">* does not divide the refresh: frames alternate between two '
               'hold times by design.</p>')
    return "".join(out)


def _html_triggers(m: dict, folder: str) -> str:
    t = m["summaries"]["triggers"]
    o = t["outputs"]
    ports = ", ".join(map(_port_name, o["ports"])) or "none with hardware"
    out = [f'<p class="note">This PC: {_e(ports)} ({o["n_writable"]} writable); LSL: '
           f'{_e(o["lsl"])}.</p>']
    if "triggers" in t:
        s = t["triggers"]["settings"]
        out.append(f'<p class="note">Config: start <b>{_e(s["sync"]["mode"])}</b>'
                   + (f' (scanner key {_e(repr(s["sync"]["key"]))})' if s["sync"]["mode"] == "wait"
                      else "") + f', codes over <b>{_e(t["triggers"]["active"])}</b>.</p>')
    rows = []
    if "sent" in t:
        x = t["sent"]
        rows.append(["codes sent", x["n"], f"send call median {x['send_median_ms']:.3f} ms, max "
                     f"{x['send_max_ms']:.3f} ms; " + ("all read back from the network"
                                                      if x["lsl_readback"] == "ok"
                                                      else "check each in the recording")])
    if "pulses" in t:
        p = t["pulses"]
        rows.append(["scanner pulses", p["n"], f"TR {p['tr_s']:.3f} s (median), "
                     f"{p['tr_train_s']:.6f} s over the train; missed {p['missed']}, "
                     f"doubled {p['doubles']}"])
    return "".join(out) + (_html_table(["", "n", ""], rows, nums=(1,)) if rows else "")


def _html_controls(m: dict, folder: str) -> str:
    c = m["summaries"]["controls"]
    devices = _e(", ".join(c["devices"]) or "none listed")
    out = [(f'<p class="note">Input devices: {devices}. '
            f'{_e(_pad_line(c.get("pad") or {}))}</p>')]
    rows = []
    for key, auto in c["automatic"].items():
        p = c["prompted"].get(key)
        asked = ("skipped" if p is None else
                 f"{p['latency_s']:.2f} s after asked, held {p['held_s'] * 1000:.0f} ms"
                 if p["pressed"] else "never pressed")
        others = ", ".join(p["strays"]) if p and p["strays"] else ""
        rows.append([f"<code>{_e(key)}</code>", "read back" if auto else "not read back",
                     _e(asked), _e(others)])
    return "".join(out) + _html_table(["key", "automatic", "when asked", "other keys"], rows)


def _pad_line(pad_out: dict) -> str:
    """One line on what the controls test found of the controller, for either report."""
    if not pad_out.get("present"):
        return "The rig lists no controller; the device in use must send the keys below itself."
    device = pad_out.get("device", "a controller")
    if not pad_out.get("measured"):
        return f"{device} plugged in, not measured: {pad_out.get('note') or pad_out.get('why')}."
    axes = "; ".join(f"{name} on past {a['on']:.2f} of its travel and off under {a['off']:.2f}"
                     for name, a in pad_out["axes"].items())
    return (f"{device} measured, written to {pad_out['config']}: {axes}; "
            f"{pad_out['confirmed']} controls "
            f"confirmed, {pad_out['corrections']} wrong presses asked again. Every run on this "
            "rig with it reads what was written.")


def _html_photodiode(m: dict, folder: str) -> str:
    p = m["summaries"]["photodiode"]
    out = [f'<p class="note">{p["n"]} flashes of {p["on_ms"]:g} ms; {_diode_text(p["readout"])}; '
           f'a click on every other flash'
           + (f' through {_e(p["audio_out"].get("device"))}, delay chosen '
              f'{p["audio_out"].get("delay_ms", 0):.1f} ms' if "audio_out" in p else "")
           + (", a microphone on input 1" if p.get("mic") else "") + ".</p>"]
    z = _html_npz(folder, "photodiode.npz")
    series = []
    if z is not None and "offset_s" in z:
        series.append(("flip → photodiode", "--s1", np.arange(len(z["offset_s"])),
                       z["offset_s"] * 1000))
    if z is not None and "click_dac" in z:
        idx = z["click_flash"]
        on = z["flip_on"][idx]
        if "sound_offset_s" in z:
            series.append(("flip → sound at the mic", "--s2", idx, z["sound_offset_s"] * 1000))
        series.append(("flip → click at the DAC", "--s3", idx, (z["click_dac"] - on) * 1000))
    if series:
        out.append('<div class="legend">' + "".join(
            f'<span style="--c:var({c})">{_e(n)}</span>' for n, c, _, _ in series) + "</div>")
        out.append(dots(series, "flash", "ms after the flip", "Offsets per flash"))
    rows = [[_e(label), f"{o['n_matched']}/{o['n']}", _num(o.get("median_ms")),
             _num(o.get("sd_ms")), _num(o.get("min_ms")), _num(o.get("max_ms")),
             _num(o.get("drift_ms_per_min"), ".3f")]
            for key, label in (("offset", "flip → photodiode"),
                               ("sound_offset", "flip → sound at the mic"),
                               ("click_dac_offset", "flip → click at the DAC"),
                               ("av_offset", "photodiode → sound"))
            if (o := p.get(key))]
    if rows:
        out.append(_html_table(["offset", "seen", "median ms", "SD", "min", "max", "drift ms/min"],
                          rows, nums=(1, 2, 3, 4, 5, 6)))
    return "".join(out)


def _cellnum(text: str | None, fmt: str = ".2f") -> str:
    """A TSV cell shown as a number, or as it is (``n/a``)."""
    try:
        return f"{float(text):{fmt}}"
    except (TypeError, ValueError):
        return _e(text)


def _html_history(m: dict, history: list[dict[str, str]]) -> str:
    rig = m["rig"]
    mine = [r for r in history if r.get("site") == rig["site"] and r.get("rig") == rig["rig"]]
    mine = sorted(mine, key=lambda r: r.get("date", ""))[-10:]
    if len(mine) < 2:
        return '<p class="note">No earlier rig measurement of this rig to compare with yet.</p>'
    rows = [[f'<span style="white-space:nowrap">{_e(r.get("date", "")[:16].replace("T", " "))}'
             "</span>", _e(r.get("task")), _status(r.get("status", "")),
             _cellnum(r.get("flip_sd_ms")), _e(r.get("frames_late")),
             _cellnum(r.get("photon_median_ms"), ".1f"), _cellnum(r.get("click_dac_median_ms"), ".1f"),
             _e((r.get("failed_tests") or "").replace(",", ", "))] for r in mine]
    return (f'<p class="note">The last {len(mine)} rig measurements of '
            f'{_e(rig["site"])}/{_e(rig["rig"])} '
            f'(this one last): today\'s numbers should sit with the others.</p>'
            + _html_table(["date", "task", "status", "flip SD ms", "late frames", "photodiode ms",
                      "sound out ms", "failed"], rows, nums=(3, 4, 5, 6)))


_html_SECTIONS = (("display", "Display", _html_display), ("frames", "Frames", _html_frames),
             ("triggers", "Triggers", _html_triggers), ("controls", "Controls", _html_controls),
             ("photodiode", "Photodiode and audio", _html_photodiode))


def render_html(m: dict, folder: str, history: list[dict[str, str]]) -> str:
    """The whole page for one finished rig measurement.

    :param m: its manifest, with ``tests``, ``rig`` and ``summaries``.
    :param folder: its folder, for the ``.npz`` the charts are drawn from.
    :param history: pooled rows (:func:`rows`) of the rig measurements under
        the same data root, this one included.
    """
    rig, tests = m["rig"], m["tests"]
    failed = [t for t, v in tests.items() if v["status"] == "fail"]
    verdict = "Every test passed" if not failed else f"{len(failed)} of {len(tests)} tests failed"
    body = [f'<header><h1>Rig measurement · {_e(rig["site"])} / {_e(rig["rig"])}</h1>'
            f'<div class="meta">{_e(m["start_time"])} · {_e(m["run"]["label"])} · PI '
            f'{_e(rig.get("pi", "?"))} · {_e(rig.get("modality", "?"))} · '
            f'fmri-gym {_e(m["versions"]["fmri_gym"])} '
            f'({_e(m["versions"]["git"])})</div></header>',
            f'<div class="hero">{_status("pass" if not failed else "fail")} {_e(verdict)}</div>',
            '<div class="card">' + _html_table(["test", "verdict", "why"], [
                [_e(t), _status(v["status"]), _e(v["why"] or "")] for t, v in tests.items()])
            + "</div>", _html_tiles(m)]
    for key, title, section in _html_SECTIONS:
        test = "audio" if key == "photodiode" and "photodiode" not in tests else key
        body.append(f'<h2>{_e(title)} {_status(tests.get(test, {}).get("status", "not run"))}'
                    + (f' {_status(tests["audio"]["status"])}' if key == "photodiode"
                       and "audio" in tests else "") + "</h2>")
        body.append('<div class="card">' + (_html_section(section, m, folder, tests.get(test))
                                            if key in m["summaries"]
                                            else '<p class="note">Not run.</p>') + "</div>")
    body.append(f'<h2>This rig over time</h2><div class="card">{_html_history(m, history)}</div>')
    body.append('<h2>Rig</h2><div class="card">' + _html_table(["", ""], [
        [_e(k), _e(v)] for k, v in rig.items()]) + "</div>")
    body.append("<h2>This computer</h2>")
    for title, head, table in computer(m):
        body.append(f'<h3>{_e(title)}</h3><div class="card">'
                    + _html_table(head, [[_e(_cell(c)) for c in row] for row in table])
                    + "</div>")
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>Rig measurement {_e(rig['site'])}/{_e(rig['rig'])}</title>"
            f"<style>{_CSS}</style></head><body><main>{''.join(body)}</main>"
            f'<div id="tip" role="tooltip"></div><script>{_JS}</script></body></html>\n')


if __name__ == "__main__":
    main()
