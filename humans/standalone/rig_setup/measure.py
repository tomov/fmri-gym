"""The measurements of a rig, with the window, trigger line, audio output and input devices its
runs open.

What a rig measurement measures, for how long and how many, is fixed here (:data:`QUICK`,
:data:`LONG`), and **with what is the rig file's** (:mod:`humans.config.rig`), so a
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
  each control of the devices its ``"inputs"`` list and each of its ``"response_keys"``.
  A controller is calibrated first (:mod:`humans.standalone.gamepad_calibration`), which
  asks for each of its controls twice in a row. Every response key is checked
  automatically -- one that comes back through the event queue as itself -- and the
  input devices plugged in are listed; then each control no calibration asked for is
  asked for on screen and pressed on the real device.
* ``photodiode`` -- the flip-to-photon offset, read as the rig's
  ``rig_measurement.photodiode`` says, and on the same flashes the ``audio`` test when the
  rig plays audio: a tone burst on every other flash, timed at the DAC, and at a
  microphone at the ear when the rig has one.

The screen says what each measurement does while it does it, then its verdict. A
measurement does not raise: it records a verdict for each of its tests (pass, fail,
offline, not run, and why), so one failure hides none of the others.
"""

from __future__ import annotations

import glob
import math
import os
import random
import re
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterator

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from ...config import rig as rig_file
from ...inputs import (CONTROLLER_CLASSES_BY_NAME, QUIT, ControlNames, Controller, Edge,
                      InputDevice, InputDevices, open_input_devices)
from ...inputs.keyboard import KEYCODES
from ...outputs.audio import Audio
from ...outputs.display import Display, check_monitor, is_locked, quit_like_esc
from ...run import EXPERIMENTER_KEY, _wait_for_control, _wait_for_duration, read_input_events
from ...save.bids import next_session, run_output
from ...sync.triggers import HID_DEVICES, LIFECYCLE_EVENTS, Codes, TriggerError, Triggers
from ..gamepad_calibration import calibrate, cue_order
from ..photodiode import (AudioRecorder, Verdict, _clicker, _light_verdict,
                          _log_clicks, _readout, _sound_verdict, run_flashes)

#: The subject every rig measurement is filed under in the data root.
SUBJECT = "sub-rig"
#: What the frame test plays its rates under: nothing else, or every core kept busy.
LOADS = ("none", "cpu")
#: Seconds each measurement's verdict stays on screen.
_VERDICT_S = 1.5
#: Seconds the list of every test's verdict stays up at the end.
_SUMMARY_S = 5.0
_PASS = Verdict("pass", None)
#: Why a measurement left out by the experimenter was not run.
UNTICKED = "unticked before the rig measurement"


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
    #: why the trigger line did not open (:func:`~.cli.prepare`); ``None`` when it did
    trigger_error: str | None
    #: the measurements to take, by their :data:`MEASUREMENTS` names; the others are not run
    measurement_names: tuple[str, ...]
    #: each measurement's record, by its name, in the order taken
    measurements: dict[str, dict] = field(default_factory=dict)
    start_epoch: float = field(default_factory=time.time)
    t0: float = field(default_factory=time.perf_counter)

    @classmethod
    def open(cls, rig: rig_file.RigConfig, settings: RigMeasurementSettings,
             trigger_error: str | None, measurement_names: tuple[str, ...]) -> RigMeasurement:
        """Make its folder, then open the trigger line, the audio output, the input devices
        and the window, saying on stderr what each got. Without the controls measurement,
        the rig's controllers are not opened, so one not plugged in stops nothing.

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
        inputs = open_input_devices(
            rig.inputs if "controls" in measurement_names else
            tuple(n for n in rig.inputs if n not in CONTROLLER_CLASSES_BY_NAME), frozenset())
        print(f"inputs: {inputs.status()}", file=sys.stderr)
        screen = rig.screen
        display = Display(size=screen.size, fullscreen=screen.is_fullscreen,
                          vsync=screen.has_vsync, monitor=screen.monitor)
        return cls(rig, settings, display, triggers, audio, inputs, out.folder, session,
                   out.label, out.attempt, trigger_error, measurement_names)

    def now(self) -> float:
        """Seconds since the rig measurement opened."""
        return time.perf_counter() - self.t0

    def close(self) -> None:
        """Close the inputs, the trigger line, the audio output and the window.

        The window goes last, because closing it shuts down every SDL subsystem the inputs
        read through.
        """
        self.inputs.close()
        self.triggers.close()
        self.audio.close()
        self.display.close()


def take_measurements(rig_measurement: RigMeasurement) -> None:
    """The start screen, each measurement it takes and its verdict, every verdict; then close
    the devices. A measurement it does not take is recorded as not run. ESC or Ctrl+C stops the
    measurements, and what they measured stays recorded."""
    previous_handler = signal.signal(signal.SIGINT, quit_like_esc)
    try:
        _start_screen(rig_measurement)
        for name, measure in MEASUREMENTS.items():
            if name not in rig_measurement.measurement_names:
                _leave_out(rig_measurement, name)
                continue
            _measure(rig_measurement, name, measure)
        _summary_screen(rig_measurement)
    except KeyboardInterrupt:
        print("rig measurement: stopped; filing what it measured", file=sys.stderr)
    finally:
        signal.signal(signal.SIGINT, previous_handler)
        rig_measurement.close()


# ---------------------------------------------------------------------------
# The measurements
# ---------------------------------------------------------------------------


def _start_screen(rig_measurement: RigMeasurement) -> None:
    """Say what this is and what to have ready for the measurements it takes; wait for the
    experimenter's key.

    :raises KeyboardInterrupt: on ESC or window close.
    """
    rig, measurement_names = rig_measurement.rig, rig_measurement.measurement_names
    ready = []
    if "photodiode" in measurement_names:
        ready.append("Photodiode on the patch, bottom-right; microphone at the ear if there is "
                     "one.")
    if "controls" in measurement_names:
        ready.append("The controls test asks for each button: press it on the participant's "
                     "device.")
    if "triggers" in measurement_names and rig.triggers.sync.mode == "wait":
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


def _leave_out(rig_measurement: RigMeasurement, name: str) -> None:
    """Record each test of the measurement ``name`` as not run, unticked."""
    now = rig_measurement.now()
    rig_measurement.measurements[name] = {
        "onset": now, "offset": now, "summary": {},
        "tests": {t: Verdict("not run", UNTICKED)._asdict() for t, m in TESTS.items()
                  if m == name}}
    print(f"rig measurement: {name}: not run -- {UNTICKED}", file=sys.stderr)


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
                                               "scanner (rig-setup, Triggers step)")
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


def _controls_to_ask(calibrated: list[tuple[Controller, dict]]) -> dict[str, str]:
    """Each control, buttons then stick directions, of the controllers whose calibration did
    not ask for every control (:func:`_calibrate`'s ``measured``): what it is on."""
    return {c: f"on the {d.name}" for d, calibration in calibrated
            if not calibration["measured"]
            for c in (*cue_order(d.BUTTON_NAMES), *d.axis_controls())}


def _measure_controls(rig_measurement: RigMeasurement, summary: dict, arrays: dict,
                      tests: dict) -> None:
    rig = rig_measurement.rig
    controllers = _controllers(rig_measurement.inputs)
    if not controllers and not rig.response_keys:
        tests["controls"] = Verdict("not run", "this rig lists no controller and types no "
                                               "response keys (rig-setup, Response devices step)")
        return
    summary["devices"] = input_devices()
    summary["inputs"] = rig_measurement.inputs.status()
    calibrated = [(d, _calibrate(rig_measurement, d)) for d in controllers]
    calibrations = [calibration for _, calibration in calibrated]
    summary["pad"] = calibrations[0] if calibrations else {"present": False}
    keys = {**_controls_to_ask(calibrated), **rig.response_keys}
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
    (see :mod:`humans.standalone.gamepad_calibration`), so the calibration runs here and is
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
        tests["photodiode"] = Verdict("not run", "the rig has no photodiode (rig-setup, "
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
    boxes of :data:`~humans.sync.triggers.HID_DEVICES`; nothing is opened.

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


def port_name(o: dict[str, Any]) -> str:
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
        print(f"  {o['kind']} {port_name(o)}  {o['what']}  {access}")
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


#: The tests a rig measurement reports on, and the measurement that runs each: the photodiode's
#: flashes carry the audio test's clicks too.
TESTS = {"display": "display", "triggers": "triggers", "controls": "controls",
         "photodiode": "photodiode", "audio": "photodiode", "frames": "frames"}
