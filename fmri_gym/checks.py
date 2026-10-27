"""The rig check: measure the rig with the very window, trigger line and audio a run opens.

A rig check is a run like any other (``configs/curricula/checks/rig-check.json``): the editor
edits its triggers and its controls as it does any config's, ``fmri-play``
plays it -- first in a session, or on its own -- and it writes a run folder.
Its curriculum holds check phases where a game run holds games:

* ``check_display`` -- are flips locked to the refresh; missed refreshes.
* ``check_frames`` -- a test pattern (:class:`TestPattern`) played as a game,
  through the run's own game loop, at each of ``rates`` frames per second and
  under each of ``loads`` (``cpu``: every core kept busy): frames lost, frames
  shown a refresh late, pacing resets, and whether the frame triggers marked
  every frame, in order -- and, given ``recording_hz``, for two samples each.
  The run's own loop, not a copy of it: its pacing is what is measured.
* ``check_triggers`` -- the trigger outputs this PC has (serial ports with
  hardware behind them, parallel ports, HID trigger boxes, whether LSL reads
  back), with a warning when none is usable; then the config's line: every bit alone and every code
  of its scheme sent (read back, over LSL), and scanner pulses counted when the
  run waits for the scanner. Only the configured port is written to: another
  may be wired to a response box or an eye tracker, and a write that succeeds
  says nothing of where it went.
* ``check_controls`` -- the response device. The keys asked for are the
  phase's ``keys``, or without them each control of the rig's ``"bindings"``,
  expected as the key it is bound to. Every key is first checked
  automatically -- one that comes back through the event queue as itself --
  and the input devices plugged in are listed; then each key is asked for on
  screen and pressed on the real device.
* ``check_photodiode`` -- the flip-to-photon offset, and on the same flashes
  the audio test: a tone burst on every other flash, timed at the DAC, and at
  a microphone at the ear with ``"has_mic": true``.

The screen says what each check does while it does it, then its verdict. A
check does not raise: it records a verdict for each of its tests (pass, fail,
offline, not run, and why), so one failure hides none of the others, and a
failed test ends nothing, neither the check nor the session.

**The rig file** (:mod:`fmri_gym.config.rig`, one per rig) says what software cannot
see: the site and rig, the PI, the monitor or projector, the photodiode, the
sound path to the ear, the trigger hardware. A rig check is how a rig gets its
file: when it is missing or invalid, the check opens a form to fill it in
(:func:`ensure_rig`); a check that says nothing of where it ran could not be
pooled, so without it none starts. A rig whose file says nothing of its
triggers is asked for them on its first check, and the answer is saved in
the rig file: no setup is anyone's default.

**What is filed.** Beside the run's own ``manifest.json`` and one ``.npz`` per
check: ``report.html`` for the people who run the rig (a verdict per test, the
numbers, charts, this rig's previous checks; one offline file),
``report.md``, and ``rigcheck.tsv`` -- one row of fixed columns
(:data:`COLUMNS`) with its BIDS sidecar. The data root's ``rigchecks.tsv``
holds every check filed there, read from their manifests, so checks filed by
an earlier version still line up.

**What the rig measured.** Every check also files its measurements beside the
rig file, in ``measured/<rig>-<n>.json`` (:func:`file_measured`), ``n`` one
more than the last, when it measured any latency: the latencies -- flip to light at the screen, flip to the
sound card, sound card to the ear -- with every check's summary, the verdicts,
the rig file as played, the software and the machine (:func:`machine`:
processor, memory, displays, GPUs, sound and input devices, what is plugged
into USB). The rig file is the rig's settings,
shared; these are one machine's measurements, and stay on it (``.gitignore``).
A run on a rig with none says so, and the first run of a session, on a
terminal, offers to run a check first (:func:`offer_check`)::

    python -m fmri_gym.checks rig --rig configs/rigs/<site>/<rig>.json   # fill in / update a rig file
    python -m fmri_gym.checks report <run folder>...   # re-file checks already run
    python -m fmri_gym.checks pool data/ /mnt/siteB/data/ --out rigchecks.tsv
"""

from __future__ import annotations

import argparse
import datetime
import glob
import html
import json
import os
import platform
import random
import re
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from . import pad, pad_calibration
from .bids import has_session, next_session
from .config import rig as rig_file
from .config.phases import (CHECK_PHASES, CheckControls, CheckDisplay, CheckFrames, CheckPhase,
                            CheckPhotodiode, CheckTriggers, GamePhase)
from .display import Display, is_locked
from .keys import _PYGAME_KEY_NAMES, get_events, key_name
from .photodiode import (AudioRecorder, Verdict, _clicker, _light_verdict,
                         _log_clicks, _readout, _sound_verdict, run_flashes)
from .replay import frame_arrays
from .run import _wait_for_duration
from .triggers import HID_DEVICES, LIFECYCLE_EVENTS, Codes, TriggerError, Triggers

if TYPE_CHECKING:
    from .config.run import RunConfig
    from .run import Run


#: The rig checks a run on a rig never measured offers to play first, by the key that picks one.
_RIG_CHECKS = {key: str(Path(__file__).resolve().parents[1] / "configs" / "curricula" / "checks"
                        / name)
               for key, name in (("q", "rig-check.json"), ("l", "rig-check-long.json"))}
#: The check phases' types, as a manifest's phase entries name them.
CHECK_TYPES = tuple(c.type for c in CHECK_PHASES)
#: Seconds each check's verdict stays on screen.
_VERDICT_S = 1.5
#: Seconds the list of every test's verdict stays up at the end.
_SUMMARY_S = 5.0
_PASS = Verdict("pass", None)


# ---------------------------------------------------------------------------
# Before and after the run
# ---------------------------------------------------------------------------


def prepare(rig: rig_file.RigConfig) -> tuple[rig_file.RigConfig, str | None]:
    """Before the window: the rig's trigger settings, and a line the run can open.

    A rig whose file says nothing of its triggers is asked for them -- a
    preset, or one's own -- and the answer is saved in its file
    (:func:`_ask_triggers`). A trigger line that does not open would stop the
    run before any check; here it fails the triggers check instead, and the
    run plays on the rig without its line (said on the console and in the
    manifest), so the other checks still run.

    :param rig: the rig the check plays on.
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
    print(f"rig check: the trigger line did not open ({error}); the triggers check fails, "
          "the other checks run without codes", file=sys.stderr)
    return rig.without_trigger_line(), error


def _ask_triggers(rig: rig_file.RigConfig) -> rig_file.RigConfig:
    """The rig with the trigger settings the dialog gives, saved in its file.

    :raises ValueError: with no screen for the dialog, or when it is cancelled.
    """
    where = rig.path
    fix = (f'{where} has no "triggers": set them on the editor\'s Triggers tab (fmri-edit) '
           "or in the file, or answer the dialog a rig check opens")
    if not can_show_form():
        raise ValueError(f"rig check: {fix}; there is no screen here for the dialog")
    from .gui_qt import choose_triggers
    section = choose_triggers(where)
    if section is None:
        raise ValueError(f"rig check: {fix}; it was cancelled")
    rig_file.save({**rig_file.load(where), "triggers": section}, where)
    print(f"rig check: trigger settings saved in {where}", file=sys.stderr)
    return rig_file.RigConfig.open(where)


def offer_check(config: RunConfig) -> None:
    """Before a run that is not a rig check: say so when its rig has never been measured,
    and before the session's first run, on a terminal, offer to measure it first, as a
    rig check run of its own.

    :param config: the run.
    """
    rig = config.rig
    if measured_files(rig.path):
        return
    print(f"WARNING: {rig.path} has never been measured: no rig check has filed when a "
          "frame's light and sound reach the participant on it", file=sys.stderr)
    if has_session(rig.data_root, config.subject, config.ses):
        return
    args = ["--subject", "sub-rig", "--ses", str(next_session(rig.data_root, "sub-rig")),
            "--run", "1", "--rig", rig.path]
    if not sys.stdin.isatty():
        curriculum = os.path.relpath(_RIG_CHECKS["l"])
        print("  measure it: uv run fmri-play " + " ".join(args + ["--curriculum", curriculum]),
              file=sys.stderr)
        return
    try:
        answer = input("Run a rig check first? [q]uick (under a minute), [l]ong (about 25 "
                       "minutes), Enter to go on without: ").strip().lower()
    except EOFError:  # Ctrl+D: nothing typed
        answer = ""
    if answer not in _RIG_CHECKS:
        return
    subprocess.run([sys.executable, "-m", "fmri_play", *args,
                    "--curriculum", _RIG_CHECKS[answer]], check=False)
    print(f"rig check done; the run starts on {rig.path}", file=sys.stderr)


def finish(run: Run, session: int) -> bool:
    """File the rig check: the tests in the manifest, the report, the pooling row, and
    the rig's data root's ``rigchecks.tsv``, rebuilt.

    :param run: the run, played (its manifest saved).
    :param session: its BIDS session.
    :return: ``True`` if a test failed or did not finish (the report lists them; the
        session goes on).
    """
    path = os.path.join(run.outdir, "manifest.json")
    with open(path) as f:
        m = json.load(f)
    entries = [e for e in m["phases"] if e.get("type") in CHECK_TYPES]
    m["rig_check"] = {e["type"].removeprefix("check_"): e["summary"] for e in entries}
    tests = {test: {"status": "not run", "why": None} for test in TESTS}
    kinds = {p["type"] for p in m["curriculum"]}
    tests.update({t: {"status": "fail", "why": "did not finish (quit before it)"}
                  for t, kind in TESTS.items() if kind in kinds})
    for e in entries:
        tests.update(e["tests"])
    started = m.get("start_epoch") or time.time()
    m.update(tests=tests, session=session, task=m["run"]["label"].split("_task-")[1].split("_")[0],
             start_time=datetime.datetime.fromtimestamp(started).astimezone()
             .isoformat(timespec="seconds"),
             versions={**m.get("versions", {}), **versions()},
             platform=platform_info(), machine=machine())
    data_root = run.config.rig.data_root
    write_outputs(run.outdir, m, data_root)
    file_measured(m, run.outdir, run.config.rig.path)
    print(f"rig check {'PASS' if m['status'] == 'pass' else 'FAIL (' + m['failed_tests'] + ')'}"
          f": {os.path.join(run.outdir, 'report.html')} "
          f"(all checks: {os.path.join(data_root, 'rigchecks.tsv')})", file=sys.stderr)
    return m["status"] != "pass"


# ---------------------------------------------------------------------------
# The phase
# ---------------------------------------------------------------------------


def run_check(run: Run, phase: CheckPhase, index: int) -> None:
    """Play one check phase: measure, record its tests' verdicts, save its arrays.

    :raises KeyboardInterrupt: on ESC or window close (the run is quit); the
        phase is logged first, with what it measured.
    """
    kind = phase.type
    onset = run.clock.run_time()
    summary: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    tests: dict[str, Verdict] = {}
    try:
        _CHECKS[type(phase)](run, phase, summary, arrays, tests, index)
    except Exception as exc:  # noqa: BLE001 -- a check that breaks fails its tests; the rest run
        for test in (t for t, k in TESTS.items() if k == kind):
            tests.setdefault(test, Verdict("fail", f"{type(exc).__name__}: {exc}"))
    finally:
        if arrays:
            np.savez_compressed(os.path.join(run.outdir, f"block-{index:02d}_{kind}.npz"),
                                **arrays)
        run.logger.log_phase({"index": index, "type": kind, "onset": onset,
                              "offset": run.clock.run_time(), "summary": summary,
                              "tests": {t: v._asdict() for t, v in tests.items()}})
        for test, v in tests.items():
            print(f"rig check: {test}: {v.status}" + (f" -- {v.why}" if v.why else ""),
                  file=sys.stderr)
    lines = [f"{t}: {v.status.upper()}" + (f"\n  {v.why[:160]}" if v.why else "")
             for t, v in tests.items()]
    run.display.draw_text("Rig check\n\n" + "\n".join(lines))
    _wait_for_duration(run.display, _VERDICT_S)
    if index == max(i for i, p in enumerate(run.config.curriculum.phases)
                    if isinstance(p, CheckPhase)):
        _summary_screen(run)


def _summary_screen(run: Run) -> None:
    """Every test's verdict, for :data:`_SUMMARY_S`: a check often runs unattended, so
    nothing waits for a key; the report and the console keep the list."""
    tests = {t: v for e in run.logger.manifest["phases"] if e.get("type") in CHECK_TYPES
             for t, v in e["tests"].items()}
    failed = [t for t, v in tests.items() if v["status"] == "fail"]
    lines = [f"{t}: {v['status'].upper()}" for t, v in tests.items()]
    head = (f"Rig check done: {len(failed)} test(s) failed, listed in the report"
            if failed else "Rig check done: every test passed")
    run.display.draw_text(head + "\n\n" + "\n".join(lines))
    _wait_for_duration(run.display, _SUMMARY_S)


def _title(run: Run, text: str) -> None:
    run.display.draw_text(f"Rig check: {text}")


def _check_display(run: Run, phase: CheckDisplay, summary: dict, arrays: dict, tests: dict,
                   index: int) -> None:
    info = run.display.describe()
    n = phase.n
    if phase.seconds is not None:
        n = int(phase.seconds * (info["refresh_rate"] or 60))
    _title(run, f"display\n\n{info['refresh_rate']} Hz, vsync {info['vsync']}: timing {n} "
                f"flips (about {n / (info['refresh_rate'] or 60):.0f} s)\nthe screen flickers "
                "grey meanwhile")
    _wait_for_duration(run.display, _VERDICT_S)
    stats = run.display.measure_flips(n)
    arrays["intervals_ms"] = np.asarray(stats.pop("intervals_ms"))
    locked = is_locked(info, stats)
    summary.update(display=info, flips={"n": n, **stats}, locked=locked)
    tests["display"] = _PASS if locked else Verdict(
        "fail", f"flips are NOT locked to the refresh (vsync {info['vsync']}, "
                f"mean {stats['mean_ms']:.2f} ms, SD {stats['sd_ms']:.2f})")


def _check_triggers(run: Run, phase: CheckTriggers, summary: dict, arrays: dict, tests: dict,
                    index: int) -> None:
    _title(run, "triggers")
    summary["outputs"] = discover()
    summary["triggers"] = run.triggers.describe()
    error = run.logger.manifest.get("trigger_error")
    if error:
        tests["triggers"] = Verdict("fail", f"the trigger line did not open: {error}")
        return
    t = run.triggers
    pulses = t.sync.mode == "wait" and not run.config.dummy_trigger
    if not (t.enabled or pulses):
        tests["triggers"] = Verdict("not run", "this config neither sends codes nor waits "
                                               "for the scanner")
        return
    if t.enabled:
        def on_send(label: str, value: int, number: int, total: int) -> None:
            _title(run, f"triggers\n\nsending {label} = {value}   ({number} of {total})\n"
                        f"on {t.active}")

        send_and_check(t, phase.repeat, phase.hold_ms / 1000, phase.gap_ms / 1000, summary,
                       arrays, on_send)
        readback = summary["sent"]["lsl_readback"] == "ok"
        _title(run, f"triggers\n\nsent {summary['sent']['n']} codes on {t.active}\n"
                    + ("all read back from the network" if readback
                       else "check each in the recording: every line alone, then every code"))
        _wait_for_duration(run.display, _VERDICT_S)
    if pulses:
        _count_pulses(run, phase, summary, arrays)
    tests["triggers"] = _PASS


def _count_pulses(run: Run, phase: CheckTriggers, summary: dict, arrays: dict) -> None:
    """Count the scanner's pulses (the run already waited for the first).

    :raises TriggerError: when none comes in time, or the train is irregular.
    """
    times = receive_pulses(run.display, run.triggers.sync.key, phase.pulses, phase.timeout_s,
                           phase.seconds)
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


def _round_trip(name: str) -> bool:
    """A synthetic press of ``name`` comes back through the event queue as ``name``."""
    code = {v: k for k, v in _PYGAME_KEY_NAMES.items()}.get(name)
    if code is None:
        return False
    pygame.event.clear(pygame.KEYDOWN)
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=code, mod=0, unicode="",
                                         scancode=0))
    return any(key_name(e.key) == name for e in pygame.event.get(pygame.KEYDOWN))


def _check_controls(run: Run, phase: CheckControls, summary: dict, arrays: dict, tests: dict,
                    index: int) -> None:
    keys = phase.keys if phase.keys is not None else {
        key: f"the {control} control" for control, key in run.config.rig.bindings.items()}
    if not keys:
        tests["controls"] = Verdict("not run", "the phase lists no keys and the rig file has "
                                               'no "bindings" to test')
        return
    summary["devices"] = input_devices()
    summary["pad"] = _check_pad(run, phase)
    summary["automatic"] = {name: _round_trip(name) for name in keys}
    ignore = run.triggers.sync.key if run.triggers.sync.mode == "wait" else None
    results: dict[str, dict] = {}
    try:
        for name, label in keys.items():
            results[name] = _prompt(run, name, label, keys, results, phase.timeout_s, ignore)
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
    if summary["pad"].get("why"):
        problems.append(f"pad: {summary['pad']['why']}")
    problems += [_key_problem(k, results.get(k)) for k in keys
                 if not results.get(k, {}).get("pressed")]
    tests["controls"] = Verdict("fail", "; ".join(problems)) if problems else _PASS


def _check_pad(run: Run, phase: CheckControls) -> dict[str, Any]:
    """Calibrate the controller, if there is one, before asking for any keys.

    A pad on an fMRI interface sends joystick buttons, not keystrokes, and each of
    its sticks has to be measured on the rig before a threshold means anything (see
    :mod:`fmri_gym.pad_calibration`). So when one is plugged in the calibration runs
    here, and writes a calibration of that pad, which every later run on this rig
    with it reads. With no pad there is nothing to measure: whatever device is
    plugged in is expected to send the keys itself, which the prompts below test.
    ``"has_pad": false`` in the phase skips the measurement and leaves any existing
    calibration alone.
    """
    found = pad.find()
    if found is None:
        return {"present": False}
    if not phase.has_pad:
        return {"present": True, "device": found.name, "measured": False, "why": "",
                "note": 'skipped ("has_pad": false)'}
    result = pad_calibration.calibrate(found, run.display.screen, pygame.display.flip,
                                       found.sticks())
    if result is None:
        return {"present": True, "device": found.name, "measured": False,
                "why": "stopped before it finished; nothing written"}
    pad.use((result.path, result.calibration))      # the keys below are read through it
    return {"present": True, "why": "", **result.record()}


def _key_problem(name: str, result: dict | None) -> str:
    if result is None:
        return f"{name}: skipped (ESC)"
    got = f" (got {', '.join(result['strays'])})" if result["strays"] else ""
    return f"{name}: never pressed{got}"


def _prompt(run: Run, name: str, label: str, keys: dict, done: dict, timeout: float,
            ignore: str | None) -> dict[str, Any]:
    """Ask for ``name`` on screen; wait for it, noting any other key that comes instead.

    :raises _Skipped: on ESC.
    """
    marks = "   ".join(f"{k}{' ok' if done.get(k, {}).get('pressed') else ''}" for k in keys)
    run.display.draw_text(f"Rig check: controls\n\nPress  [ {name} ]"
                          f"{f'  ({label})' if label and label != name else ''}\n\n{marks}"
                          "\n\n(ESC: skip the rest; they fail)")
    pygame.event.clear((pygame.KEYDOWN, pygame.KEYUP))  # what was pressed before the prompt
    start = time.perf_counter()
    strays: list[str] = []
    while time.perf_counter() - start < timeout:
        events = _key_events(ignore)
        for i, (down, got) in enumerate(events):
            if not down:
                continue
            if got != name:
                strays.append(got)
                continue
            pressed = time.perf_counter()
            return {"pressed": True, "latency_s": pressed - start,
                    "held_s": _held(name, pressed, events[i + 1:]), "strays": strays}
        run.display.idle(time.perf_counter() + 0.002, poll=0.001)
    return {"pressed": False, "strays": strays}


def _key_events(ignore: str | None) -> list[tuple[bool, str]]:
    """Key presses and releases since the last call, in order: ``(down, name)``; the
    scanner's key left out.

    :raises _Skipped: on ESC or window close.
    """
    out = []
    for e in get_events(pygame.QUIT, pygame.KEYDOWN, pygame.KEYUP):
        if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
            raise _Skipped
        if ignore and getattr(e, "unicode", None) == ignore:
            continue
        out.append((e.type == pygame.KEYDOWN, key_name(e.key) or f"keycode {e.key}"))
    return out


def _held(name: str, pressed: float, pending: list[tuple[bool, str]],
          limit: float = 2.0) -> float:
    """Seconds until ``name`` is released (``nan`` if not within ``limit``).

    :param pending: the events that came after the press in the same poll.
    """
    while time.perf_counter() - pressed < limit:
        if (False, name) in pending:
            return time.perf_counter() - pressed
        pending = _key_events(None)
        time.sleep(0.001)
    return float("nan")


# -- photodiode and audio --------------------------------------------------------


def _check_photodiode(run: Run, phase: CheckPhotodiode, summary: dict, arrays: dict,
                      tests: dict, index: int) -> None:
    readout = phase.readout
    clicks = phase.has_audio and run.audio.enabled
    if phase.has_audio and not run.audio.enabled:
        tests["audio"] = Verdict("not run",
                                 "the audio output is off (the rig's \"has_audio\": false)")
    summary.update(display=run.display.describe(), readout=readout, mic=phase.has_mic,
                   n=phase.n, on_ms=phase.on_ms, gap_ms=[phase.gap_min_ms, phase.gap_max_ms],
                   corner=phase.corner, patch_px=phase.patch_px)
    recorder = _recorder(phase, tests) if readout == "soundcard" else None
    click, chunks, clicked = _clicker(run.audio) if clicks else (None, [], [])
    what = "photodiode + audio" if clicks else "photodiode"

    def caption(i: int) -> str:
        sound = ", with a click" if clicks and i % 2 == 0 else ""
        return f"Rig check: {what}, flash {i + 1} of {summary['n']}{sound}"

    try:
        arrays.update(run_flashes(
            run.display, run.triggers, summary["n"], summary["on_ms"], tuple(summary["gap_ms"]),
            random.Random(phase.seed), summary["corner"], summary["patch_px"], click,
            phase.settle_ms / 1000, bracket=False, caption=caption))
    finally:
        if recorder is not None:
            recorder.stop()
        if clicks:
            _log_clicks(run.audio, chunks, clicked, arrays, summary, close=False)
    if recorder is not None:
        _readout(*recorder.signal(), arrays, summary)
    if "photodiode" not in tests:  # an input that did not open has failed it already
        tests["photodiode"] = _light_verdict(arrays, summary)
    if "audio" not in tests:
        tests["audio"] = _sound_verdict(arrays, clicks, None)


def _recorder(phase: CheckPhotodiode, tests: dict) -> AudioRecorder | None:
    """The sound-card input, started; ``None`` (and the photodiode failed) if it will not open."""
    try:
        recorder = AudioRecorder(phase.input_device, phase.samplerate,
                                 channels=2 if phase.has_mic else 1)
        recorder.start()
        return recorder
    except Exception as exc:  # noqa: BLE001 -- any device error fails the photodiode, not the rest
        tests["photodiode"] = Verdict("fail", f"the sound-card input did not open: {exc}")
        if phase.has_mic:
            tests["audio"] = Verdict("fail", f"the microphone's input did not open: {exc}")
        return None


# -- frames: the game loop at each rate, under each load -----------------------------


def _check_frames(run: Run, phase: CheckFrames, summary: dict, arrays: dict, tests: dict,
                  index: int) -> None:
    blocks = []
    for load in phase.loads:
        for fps in phase.rates:
            with _load(load):
                blocks.append(_play_pattern(run, fps, load, phase, index))
    summary["blocks"] = blocks
    summary["played"] = [f"{b['fps']:g}/{b['load']}" for b in blocks]
    summary["refresh_rate"] = run.display.refresh_rate
    summary["vsync"] = run.display.vsync
    summary["recording_hz"] = phase.recording_hz
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


def _play_pattern(run: Run, fps: float, load: str, phase: CheckFrames,
                  index: int) -> dict[str, Any]:
    """Play the test pattern at ``fps`` as a game block of this run, and judge its frames."""
    rate = f"{fps:g}".replace(".", "p")
    env_id = _register_pattern(f"Frames{rate}fps{'' if load == 'none' else load.upper()}")
    caption = f"Rig check: frames at {fps:g} fps" + ("" if load == "none" else f", {load} load")
    # keys: the pattern takes no input; its one action is what a frame with no key sends.
    game = {"type": "game", "backend": "gym", "game": env_id, "mode": "duration",
            "duration": phase.seconds, "fps": fps, "has_audio": False, "seed": 1000 + index,
            "keys": {"": 0}, "env_kwargs": {"caption": caption}}
    # The run's own game loop, not a copy: its pacing and triggers are what is measured.
    run._game(GamePhase.from_dict(game, index), index)
    entry = run.logger.manifest["phases"][-1]
    flips, codes = _block_flips(os.path.join(run.outdir, entry["data_dir"]))
    return _judge_frames(run, fps, load, phase, flips, codes, entry)


def _block_flips(block: str) -> tuple[np.ndarray, np.ndarray | None]:
    """A logged block's flip times, and its frame trigger codes (0: none) if it sent any."""
    frames = frame_arrays(block)
    codes = frames.get("trigger")
    return (np.asarray(frames.get("flip_time", []), dtype=float),
            None if codes is None else np.array([c or 0 for c in codes], dtype=int))


def _judge_frames(run: Run, fps: float, load: str, phase: CheckFrames, flips: np.ndarray,
                  codes: np.ndarray | None, entry: dict) -> dict[str, Any]:
    """Frames lost, frames a refresh late, pacing resets, and the frame triggers."""
    dt = 1.0 / fps
    period = 1.0 / run.display.refresh_rate if run.display.vsync and run.display.refresh_rate \
        else None
    iv = np.diff(flips)
    # Locked, a frame lands on the refresh after its step: dt rounded up to refreshes,
    # alternating with dt rounded down when fps does not divide the refresh.
    slack = (np.ceil(dt / period - 1e-6) + 0.5) * period if period else 1.5 * dt
    # Expected from the frames' own span, not the block's duration: the block also
    # holds the env's reset before its first frame, which is no frame lost.
    expected = int(round((flips[-1] - flips[0]) / dt)) + 1 if len(flips) else 0
    b = {"fps": fps, "load": load, "n": int(len(flips)), "expected": expected,
         "lost": max(0, expected - len(flips)), "late": int(np.sum(iv > slack)),
         "pacing_resets": int(entry.get("n_pacing_resets", 0)),
         "interval_median_ms": float(np.median(iv) * 1000) if len(iv) else None,
         "interval_max_ms": float(np.max(iv) * 1000) if len(iv) else None,
         "interval_sd_ms": float(np.std(iv) * 1000) if len(iv) else None,
         "locked": period is not None,
         "divides_refresh": (bool(abs(dt / period - round(dt / period)) < 1e-3)
                             if period else None),
         "data_dir": entry["data_dir"], "triggers": _frame_marks(run, codes, dt, phase)}
    name = f"{fps:g} fps" + ("" if load == "none" else f" under {load} load")
    allowed = {"lost": 0, "late": 0, "pacing_resets": 0}
    what = {"lost": "frames lost", "late": "frames a refresh late",
            "pacing_resets": "pacing resets"}
    b["problems"] = [f"{name}: {b[k]} {what[k]}" for k in allowed if b[k] > allowed[k]]
    if b["triggers"] and not b["triggers"]["ok"]:
        b["problems"].append(f"{name}: {b['triggers']['why']}")
    return b


def _frame_marks(run: Run, codes: np.ndarray | None, dt: float,
                 phase: CheckFrames) -> dict[str, Any] | None:
    """Did the frame triggers mark every frame they should, in their cycle?

    ``frame_every`` thins them: every k-th frame gets the next code of the
    cycle ``1..2**frame_bits - 1``. A code that is missing, repeated or out of
    order would put the analysis a frame off. Given the recording's
    ``recording_hz``, each code must also last two samples, or a level-holding
    line (parallel) is read with codes merged.
    """
    s = run.triggers.settings
    if codes is None or not run.triggers.enabled or not s.on_frame:
        return None
    sent = np.flatnonzero(codes > 0)
    should = np.arange(0, len(codes), s.frame_every)
    cycle = 1 + np.arange(len(should)) % run.triggers.codes.frame_mask
    out: dict[str, Any] = {"marked": int(len(sent)), "expected": int(len(should)),
                           "ok": True, "why": None}
    hold_ms = dt * s.frame_every * 1000
    out["code_ms"] = hold_ms
    if not (np.array_equal(sent, should) and np.array_equal(codes[sent], cycle)):
        out.update(ok=False, why=f"frame triggers marked {len(sent)} of {len(should)} frames "
                                 "or broke their cycle")
    hz = phase.recording_hz
    if hz and hold_ms * hz / 1000 < 2:
        out.update(ok=False, why=f"a frame code lasts {hold_ms:.1f} ms, under 2 samples at "
                                 f"{hz:g} Hz: raise frame_every")
    return out


_CHECKS: dict[type, Callable[..., None]] = {
    CheckDisplay: _check_display, CheckTriggers: _check_triggers,
    CheckControls: _check_controls, CheckPhotodiode: _check_photodiode,
    CheckFrames: _check_frames}


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
    boxes of :data:`~fmri_gym.triggers.HID_DEVICES`; nothing is opened.

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
        if event.type == pygame.KEYDOWN and event.unicode == key:
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
    display.draw_text(f"Waiting for scanner pulses (key {key!r})\n\nstart the sequence now")
    last = time.perf_counter()
    goal = f"{seconds:g} s" if seconds else str(n)
    while not _done(times, n, seconds):
        new = _pulses(key)
        if new:
            times += new
            last = new[-1]
            display.draw_text(f"scanner pulses: {len(times)} (of {goal})")
        if time.perf_counter() - last > timeout:
            raise TriggerError(f"trigger check: {len(times)} of {goal} scanner pulses, then none "
                               f"for {timeout:g} s -- is the scanner acquiring, the trigger "
                               f"box plugged in, and does it type {key!r}?")
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
        raise TriggerError(f"trigger check: the LSL stream {s.lsl_stream_name!r} is not "
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
        raise TriggerError(f"trigger check: LSL read back {len(got)} codes that differ from "
                           f"the {len(values)} sent: {got[:20]}")
    print(f"LSL: all {len(got)} codes read back from the network as sent")




# ---------------------------------------------------------------------------
# The test pattern the frame check plays as a game
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


def _register_pattern(name: str) -> str:
    """Register a variant under its own id, once; returns the id."""
    env_id = f"fmri_gym/{name}-v0"
    if env_id not in gym.registry:
        gym.register(id=env_id, entry_point=TestPattern)
    return env_id


# ---------------------------------------------------------------------------
# The rig file, the pooling row, report.md, and the command line
# ---------------------------------------------------------------------------


#: The tests a check reports on, and the check phase that runs each: the flashes
#: of the photodiode check carry the audio test's clicks too.
TESTS = {"display": "check_display", "triggers": "check_triggers",
         "controls": "check_controls", "photodiode": "check_photodiode",
         "audio": "check_photodiode", "frames": "check_frames"}
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The pooled table: ``(column, path into the manifest, description, units)``.
COLUMNS: list[tuple[str, tuple[str, ...], str, str | None]] = [
    ("site", ("rig", "site"), "Site, from the rig file", None),
    ("rig", ("rig", "rig"), "Rig at the site, from the rig file", None),
    ("pi", ("rig", "pi"), "PI's initials, from the rig file", None),
    ("modality", ("rig", "modality"), "Recording modality of the rig", None),
    ("date", ("start_time",), "When the check started, ISO 8601 with UTC offset", None),
    ("task", ("task",), "checksrigcheck (before a session) or checksrigchecklong (measuring "
     "the rig): the curriculum's task label", None),
    ("subject", ("subject",), "BIDS subject the check was filed under (sub-rig: none)", None),
    ("session", ("session",), "BIDS session number", None),
    ("label", ("run", "label"), "BIDS run label of the check", None),
    ("attempt", ("run", "attempt"), "1, or 2, 3... for a check re-run in that session", None),
    ("status", ("status",), "pass, or fail", None),
    ("failed_tests", ("failed_tests",), "The tests that failed, comma-separated", None),
    ("failure", ("failure",), "Why each failed, test: reason; ...", None),
    *[(f"{t}_status", ("tests", t, "status"), f"The {t} test: pass, fail, offline (read "
       "in the recording) or not run", None) for t in TESTS],
    ("fmri_gym_version", ("versions", "fmri_gym"), "fmri-gym package version", None),
    ("git_commit", ("versions", "git"), "git describe of the checkout (-dirty: edited)", None),
    ("os", ("platform", "system"), "Operating system", None),
    ("monitor", ("rig", "monitor"), "Monitor or projector, from the rig file", None),
    ("photodiode", ("rig", "photodiode"), "Photodiode, from the rig file", None),
    ("audio_path", ("rig", "audio_path"), "Sound path to the ear, from the rig file", None),
    ("trigger_hardware", ("rig", "trigger_hardware"), "Trigger hardware, from the rig file",
     None),
    ("display_driver", ("rig_check", "display", "display", "driver"), "SDL video driver", None),
    ("display_size", ("rig_check", "display", "display", "size"), "Window size obtained", "px"),
    ("fullscreen", ("rig_check", "display", "display", "fullscreen"), "Fullscreen window", None),
    ("refresh_hz", ("rig_check", "display", "display", "refresh_rate"), "Refresh rate reported", "Hz"),
    ("vsync", ("rig_check", "display", "display", "vsync"), "Flips wait for the vertical blank", None),
    ("flip_n", ("rig_check", "display", "flips", "n"), "Flip intervals measured", None),
    ("flip_mean_ms", ("rig_check", "display", "flips", "mean_ms"), "Mean flip interval", "ms"),
    ("flip_sd_ms", ("rig_check", "display", "flips", "sd_ms"), "SD of the flip intervals", "ms"),
    ("flip_max_ms", ("rig_check", "display", "flips", "max_ms"), "Longest flip interval", "ms"),
    ("flip_late", ("rig_check", "display", "flips", "late"), "Intervals over 1.5 periods (missed refresh)",
     None),
    ("display_locked", ("rig_check", "display", "locked"), "Flips locked to the refresh", None),
    ("trigger_backend", ("rig_check", "triggers", "triggers", "settings", "backend"),
     "Trigger output of the config: null, lsl, serial, parallel", None),
    ("sync_mode", ("rig_check", "triggers", "triggers", "settings", "sync", "mode"),
     "Run start: wait (for the scanner), send, none", None),
    ("ports_writable", ("rig_check", "triggers", "outputs", "n_writable"),
     "Serial/parallel ports with hardware that this user may write", None),
    ("lsl_probe", ("rig_check", "triggers", "outputs", "lsl"), "LSL stream found and read back: OK or why not",
     None),
    ("codes_sent", ("rig_check", "triggers", "sent", "n"), "Trigger codes sent", None),
    ("send_median_ms", ("rig_check", "triggers", "sent", "send_median_ms"), "Median send call duration",
     "ms"),
    ("send_max_ms", ("rig_check", "triggers", "sent", "send_max_ms"), "Longest send call", "ms"),
    ("lsl_readback", ("rig_check", "triggers", "sent", "lsl_readback"),
     "LSL codes read back as sent: ok, differs, n/a (not LSL)", None),
    ("pulses_n", ("rig_check", "triggers", "pulses", "n"), "Scanner pulses counted", None),
    ("tr_median_s", ("rig_check", "triggers", "pulses", "tr_s"), "Median pulse interval", "s"),
    ("tr_train_s", ("rig_check", "triggers", "pulses", "tr_train_s"),
     "Mean pulse interval over the train, on this PC's clock", "s"),
    ("pulse_sd_ms", ("rig_check", "triggers", "pulses", "sd_ms"), "SD of the pulse intervals", "ms"),
    ("pulses_missed", ("rig_check", "triggers", "pulses", "missed"), "Intervals over 1.5 TR", None),
    ("pulses_doubled", ("rig_check", "triggers", "pulses", "doubles"), "Intervals under 0.5 TR", None),
    ("controls_keys", ("rig_check", "controls", "n_keys"), "Keys asked for", None),
    ("controls_pressed", ("rig_check", "controls", "n_pressed"), "Keys pressed when asked",
     None),
    ("controls_latency_ms", ("rig_check", "controls", "median_latency_ms"),
     "Median time from the prompt to the press (a person's, not the device's)", "ms"),
    ("pad_device", ("rig_check", "controls", "pad", "device"),
     "The controller found, whose buttons are read as keys", None),
    ("pad_axis_on", ("rig_check", "controls", "pad", "sticks", "left", "on"),
     "Fraction of the left stick's travel that counts as pushed", None),
    ("pad_corrections", ("rig_check", "controls", "pad", "corrections"),
     "Wrong controls pressed, and asked again, while the calibration confirmed each", None),
    ("input_devices", ("rig_check", "controls", "devices"),
     "Keyboards, button boxes and joysticks plugged in", None),
    ("frames_blocks", ("rig_check", "frames", "played"),
     "Frame-rate blocks played, as fps/load", None),
    ("frames_late", ("rig_check", "frames", "late_total"),
     "Frames shown a refresh later than their rate allows, all blocks", None),
    ("frames_lost", ("rig_check", "frames", "lost_total"), "Frames not played, all blocks",
     None),
    ("frames_pacing_resets", ("rig_check", "frames", "pacing_resets_total"),
     "Stalls that restarted the frame schedule, all blocks", None),
    ("frames_triggers_ok", ("rig_check", "frames", "triggers_ok"),
     "Every block's frame triggers marked every frame, in their cycle", None),
    ("diode_readout", ("rig_check", "photodiode", "readout"),
     "Where the diode was read: soundcard (here) or recording (offline)", None),
    ("flashes_n", ("rig_check", "photodiode", "n"), "Flashes shown", None),
    ("photon_matched", ("rig_check", "photodiode", "offset", "n_matched"), "Flashes the diode saw", None),
    ("photon_median_ms", ("rig_check", "photodiode", "offset", "median_ms"), "Flip to photodiode, median",
     "ms"),
    ("photon_sd_ms", ("rig_check", "photodiode", "offset", "sd_ms"), "Flip to photodiode, SD", "ms"),
    ("photon_min_ms", ("rig_check", "photodiode", "offset", "min_ms"), "Flip to photodiode, min", "ms"),
    ("photon_max_ms", ("rig_check", "photodiode", "offset", "max_ms"), "Flip to photodiode, max", "ms"),
    ("photon_drift_ms_per_min", ("rig_check", "photodiode", "offset", "drift_ms_per_min"),
     "Flip to photodiode, trend over the run (runs of a minute or more)", "ms/min"),
    ("audio_device", ("rig_check", "photodiode", "audio_out", "device"), "Sound output device", None),
    ("audio_delay_ms", ("rig_check", "photodiode", "audio_out", "delay_ms"),
     "Delay from flip to DAC the audio output chose", "ms"),
    ("click_dac_median_ms", ("rig_check", "photodiode", "click_dac_offset", "median_ms"),
     "Flip to click at the DAC, median", "ms"),
    ("sound_matched", ("rig_check", "photodiode", "sound_offset", "n_matched"), "Clicks the microphone heard",
     None),
    ("sound_median_ms", ("rig_check", "photodiode", "sound_offset", "median_ms"),
     "Flip to sound at the microphone, median", "ms"),
    ("sound_sd_ms", ("rig_check", "photodiode", "sound_offset", "sd_ms"), "Flip to sound, SD", "ms"),
    ("sound_min_ms", ("rig_check", "photodiode", "sound_offset", "min_ms"), "Flip to sound, min", "ms"),
    ("sound_max_ms", ("rig_check", "photodiode", "sound_offset", "max_ms"), "Flip to sound, max", "ms"),
    ("sound_drift_ms_per_min", ("rig_check", "photodiode", "sound_offset", "drift_ms_per_min"),
     "Flip to sound, trend over the run (runs of a minute or more)", "ms/min"),
    ("acoustic_median_ms", ("rig_check", "photodiode", "acoustic_offset", "median_ms"),
     "DAC to sound at the microphone (amplifier, tubes, earphones), median", "ms"),
    ("acoustic_sd_ms", ("rig_check", "photodiode", "acoustic_offset", "sd_ms"),
     "DAC to sound, SD", "ms"),
    ("av_median_ms", ("rig_check", "photodiode", "av_offset", "median_ms"),
     "Photodiode to sound (audio behind video), median", "ms"),
    ("av_sd_ms", ("rig_check", "photodiode", "av_offset", "sd_ms"), "Photodiode to sound, SD", "ms"),
]
#: The table's column names, in order.
NAMES = tuple(c[0] for c in COLUMNS)


# ---------------------------------------------------------------------------
# Rig file
# ---------------------------------------------------------------------------


def ensure_rig(path: str) -> rig_file.RigConfig:
    """The rig a rig check plays on; when its file is missing or invalid, the form to
    fill it in first (:func:`fmri_gym.gui_qt.fill_rig`), then the file.

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
        print(f"{error}\nrig check: opening the rig form", file=sys.stderr)
        from .gui_qt import fill_rig
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

    :param m: the run's manifest with ``tests``, ``rig`` and ``rig_check``; the
        overall ``status``, ``failed_tests`` and ``failure`` are set here.
    :param data_root: the BIDS tree the check was filed in: its ``rigchecks.tsv``
        (every rig check under it, one row each) is rebuilt, and the HTML report
        compares this check with the rig's previous ones.
    """
    bad = [t for t, v in m["tests"].items() if v["status"] == "fail"]
    m["failed_tests"] = ",".join(bad) or None
    m["failure"] = "; ".join(f"{t}: {m['tests'][t]['why']}" for t in bad) or None
    m["status"] = "fail" if bad else "pass"
    _write_json(os.path.join(folder, "manifest.json"), m)
    with open(os.path.join(folder, "rigcheck.tsv"), "w") as f:
        f.write("\t".join(c[0] for c in COLUMNS) + "\n" + "\t".join(row(m)) + "\n")
    _write_json(os.path.join(folder, "rigcheck.json"), sidecar())
    with open(os.path.join(folder, "report.md"), "w") as f:
        f.write(report(m))
    history = rows([data_root]) if data_root else [dict(zip(NAMES, row(m)))]
    if data_root:
        write_table(os.path.join(data_root, "rigchecks.tsv"), history)
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
    """Every rig check filed under ``roots``, one row each, oldest first.

    Read from each check's ``manifest.json``, not its ``rigcheck.tsv``: a check
    filed before a column existed gets ``n/a`` there, rather than a table that
    no longer lines up.
    """
    out = []
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, "**", "manifest.json"), recursive=True)):
            with open(path) as f:
                m = json.load(f)
            if "rig_check" in m and "tests" in m:
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
    """Every rig check under ``roots`` (:func:`rows`), as the lines of one TSV."""
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
    """What ran the check: fmri-gym, its git checkout, and the libraries timing depends on."""
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
    """The machine a check ran on, and what is plugged into it: for provenance, and to
    tell two rigs (or one rig before and after a change) apart when their numbers differ.

    Each part is read from the operating system without new dependencies, and is
    ``None`` (or empty) where the system does not say.
    """
    displays, inputs = _pygame_devices()
    return {**platform_info(), "python": platform.python_version(), "cpu": _cpu(),
            "cores": os.cpu_count(), "memory_gb": _memory_gb(), "displays": displays,
            "gpus": _gpus(), "audio_devices": _audio_devices(),
            "input_devices": inputs, "usb_devices": _usb_devices()}


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
    """What a rig check measured on its rig, as filed beside the rig file.

    ``latency`` holds when what a frame shows reaches the participant, from its
    flip: each the check's summary of its offsets (median, spread, count), and
    ``None`` when the check could not measure it (no photodiode, no microphone,
    a diode read in the MEG/EEG recording) or its test did not pass. The rest is
    how it was measured and on what: the check, its verdicts, every check's
    summary, the rig file as it played, the machine and the software.

    :param m: the check's manifest, filed (:func:`write_outputs`).
    :param folder: the check's run folder.
    """
    photodiode = m["rig_check"].get("photodiode", {})

    def latency(test: str, key: str) -> dict | None:
        summary = photodiode.get(key)
        passed = m["tests"].get(test, {}).get("status") == "pass"
        return summary if passed and summary and "median_ms" in summary else None

    return {"date": m["start_time"],
            "check": {"folder": folder, "label": m["run"]["label"], "task": m["task"],
                      "status": m["status"], "failed_tests": m["failed_tests"],
                      "tests": m["tests"]},
            "latency": {"flip_to_photon": latency("photodiode", "offset"),
                        "flip_to_dac": latency("audio", "click_dac_offset"),
                        "dac_to_ear": latency("audio", "acoustic_offset")},
            "rig_check": m["rig_check"], "rig": m["rig"], "machine": m["machine"],
            "versions": m["versions"]}


def measured_files(rig_path: str) -> list[str]:
    """What the rig checks measured on the rig file at ``rig_path``, in the order they
    were filed: ``measured/<rig>-<n>.json`` beside it, by ``n``."""
    folder = os.path.join(os.path.dirname(rig_path), "measured")
    rig = os.path.splitext(os.path.basename(rig_path))[0]
    numbered = {}
    for path in glob.glob(os.path.join(folder, f"{glob.escape(rig)}-*.json")):
        n = re.fullmatch(rf"{re.escape(rig)}-(\d+)\.json", os.path.basename(path))
        if n:
            numbered[int(n[1])] = path
    return [numbered[n] for n in sorted(numbered)]


def file_measured(m: dict, folder: str, rig_path: str) -> str | None:
    """Write what the check measured beside the rig file, under the next free number;
    nothing when it measured no latency.

    :param m: the check's manifest, filed.
    :param folder: the check's run folder.
    :param rig_path: the rig file the check played on.
    :return: the file written, or ``None``.
    """
    values = measured(m, folder)
    if not any(values["latency"].values()):
        print("rig check: no latency measured (no photodiode or microphone read, or their "
              "tests failed): nothing filed beside the rig", file=sys.stderr)
        return None
    rig = os.path.splitext(os.path.basename(rig_path))[0]
    path = os.path.join(os.path.dirname(rig_path), "measured",
                        f"{rig}-{len(measured_files(rig_path)) + 1}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _write_json(path, values)
    print(f"rig check: what it measured on the rig: {path}", file=sys.stderr)
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
    lines = [f"# Rig check {rig['site']}/{rig['rig']}: {verdict}", "",
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
        status = m["tests"].get(test, {"status": "not run"})["status"]  # an older check's
        lines += ["", f"## {test.capitalize()}: {status}", ""]
        measured = m["rig_check"].get(TESTS[test].removeprefix("check_"))
        if measured:
            lines += section(measured)
    lines += ["", "## Versions", ""]
    lines += [f"- {k}: {v}" for k, v in {**m["versions"], **m["platform"]}.items()]
    return "\n".join(lines) + "\n"


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
    lines = [f"- the test pattern played as a game at each rate, under each load; "
             f"refresh {f['refresh_rate']} Hz"
             + (f", recording at {f['recording_hz']:g} Hz" if f.get("recording_hz") else "")]
    if not f.get("vsync"):
        lines.append("- **vsync off: frames were paced by the clock alone, not the refresh "
                     "(see the display check); late frames cannot be told from jitter**")
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


def _md_photodiode(p: dict) -> list[str]:
    lines = [f"- {p['n']} flashes of {p['on_ms']:g} ms, diode read on the "
             f"{'sound card' if p['readout'] == 'soundcard' else 'recording (match offline)'}"]
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


def main() -> None:
    p = argparse.ArgumentParser(description="The rig file, and pooling rig checks.")
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("rig", help="fill in or update a rig file, in a form")
    r.add_argument("--rig", required=True, help="the rig file, made when it is missing")
    f = sub.add_parser("report", help="re-file checks already run: their reports and the "
                                      "data root's rigchecks.tsv, with this version")
    f.add_argument("folders", nargs="+", help="rig-check run folders")
    f.add_argument("--data-root", default="data")
    q = sub.add_parser("pool", help="stack every rig check under the roots into one table")
    q.add_argument("roots", nargs="+")
    q.add_argument("--out", help="file to write (default: stdout)")
    args = p.parse_args()
    if args.command == "rig":
        _edit_rig(args.rig)
        return
    if args.command == "report":
        for folder in args.folders:
            with open(os.path.join(folder, "manifest.json")) as fh:
                m = json.load(fh)
            if not ("rig_check" in m and "tests" in m):
                print(f"{folder}: skipped, filed by an earlier rig check without per-test "
                      "verdicts", file=sys.stderr)
                continue
            write_outputs(folder, m, args.data_root)
            print(f"{os.path.join(folder, 'report.html')}")
        print(f"all checks: {os.path.join(args.data_root, 'rigchecks.tsv')}")
        return
    text = "\n".join(pool(args.roots)) + "\n"
    if args.out:
        with open(args.out, "w") as out:
            out.write(text)
    else:
        sys.stdout.write(text)


def _edit_rig(path: str) -> None:
    """Open the form on the rig file at ``path`` (a new one if there is none), and save it."""
    if not can_show_form():
        raise RuntimeError("rig check: the rig form needs PySide6 (pip install "
                           "'fmri-gym[gui]') and a screen; or edit the file by hand")
    from .gui_qt import fill_rig
    if fill_rig(rig_file.draft(path), path) is None:
        print("rig check: cancelled, the rig file is unchanged", file=sys.stderr)
        return
    print(f"saved: {path}")


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
    rc = m["rig_check"]
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
    d = m["rig_check"]["display"]
    s, f = d["display"], d["flips"]
    out = [f'<p class="note">{_e(s["driver"])}, {_e("x".join(map(str, s["size"])))} px, '
           f'{"fullscreen" if s["fullscreen"] else "windowed"}, monitor {s["monitor"]}, '
           f'{s["refresh_rate"]} Hz, vsync {"on" if s["vsync"] else "off"}. {f["n"]} flips: '
           f'mean {f["mean_ms"]:.2f} ms, SD {f["sd_ms"]:.2f}, max {f["max_ms"]:.2f}; '
           f'{f["late"]} missed a refresh.</p>']
    z = _html_npz(folder, "block-*_check_display.npz")
    if z is not None:
        out.append('<p class="note">Flip intervals: one tight column at the refresh period '
                   'is a locked screen.</p>')
        out.append(histogram(np.asarray(z["intervals_ms"]), "ms between flips",
                             "Histogram of flip intervals"))
    return "".join(out)


def _html_section(section: Callable[[dict, str], str], m: dict, folder: str,
                  verdict: dict | None) -> str:
    """A check's section; for a check that broke part-way and left its summary
    half-written (see :func:`run_check`), a note that it stopped, and why."""
    try:
        return section(m, folder)
    except (KeyError, IndexError, TypeError):
        why = (verdict or {}).get("why")
        return (f'<p class="note">This check stopped before it finished'
                f'{": " + _e(why) if why else ""}.</p>')


def _html_frames(m: dict, folder: str) -> str:
    f = m["rig_check"]["frames"]
    out = [f'<p class="note">A test pattern played as a game through the session\'s own game '
           f'loop, at each rate and load; refresh {f["refresh_rate"]} Hz. Each line is that '
           f'block\'s frame intervals over its duration: flat is steady, spikes are late '
           f'frames.</p>']
    if not f.get("vsync"):
        out.append('<p class="note"><b>vsync was off:</b> frames were paced by the clock '
                   'alone; late frames cannot be told from jitter.</p>')
    rows = []
    for b in f["blocks"]:
        block = os.path.join(folder, b["data_dir"])
        line = (sparkline(np.diff(_block_flips(block)[0]) * 1000) if os.path.isdir(block)
                else "")
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
    t = m["rig_check"]["triggers"]
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
    c = m["rig_check"]["controls"]
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
    """One line on what the controller check found, for either report."""
    if not pad_out.get("present"):
        return "No controller plugged in; the device in use must send the keys below itself."
    device = pad_out.get("device", "a controller")
    if not pad_out.get("measured"):
        return f"{device} plugged in, not measured: {pad_out.get('note') or pad_out.get('why')}."
    sticks = "; ".join(f"{name} stick on past {s['on']:.2f} of its travel and off under "
                       f"{s['off']:.2f}" for name, s in pad_out["sticks"].items())
    return (f"{device} measured: {sticks or 'no stick'}; {pad_out['confirmed']} controls "
            f"confirmed, {pad_out['corrections']} wrong presses asked again. Written to "
            f"{pad_out['config']}, and every run on this rig with it reads it.")


def _html_photodiode(m: dict, folder: str) -> str:
    p = m["rig_check"]["photodiode"]
    out = [f'<p class="note">{p["n"]} flashes of {p["on_ms"]:g} ms; the diode read on the '
           f'{"sound card" if p["readout"] == "soundcard" else "recording (match offline)"}; '
           f'a click on every other flash'
           + (f' through {_e(p["audio_out"].get("device"))}, delay chosen '
              f'{p["audio_out"].get("delay_ms", 0):.1f} ms' if "audio_out" in p else "")
           + (", a microphone on input 1" if p.get("mic") else "") + ".</p>"]
    z = _html_npz(folder, "block-*_check_photodiode.npz")
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
        return '<p class="note">No earlier check of this rig to compare with yet.</p>'
    rows = [[f'<span style="white-space:nowrap">{_e(r.get("date", "")[:16].replace("T", " "))}'
             "</span>", _e(r.get("task")), _status(r.get("status", "")),
             _cellnum(r.get("flip_sd_ms")), _e(r.get("frames_late")),
             _cellnum(r.get("photon_median_ms"), ".1f"), _cellnum(r.get("click_dac_median_ms"), ".1f"),
             _e((r.get("failed_tests") or "").replace(",", ", "))] for r in mine]
    return (f'<p class="note">The last {len(mine)} checks of {_e(rig["site"])}/{_e(rig["rig"])} '
            f'(this one last): today\'s numbers should sit with the others.</p>'
            + _html_table(["date", "task", "status", "flip SD ms", "late frames", "photodiode ms",
                      "sound out ms", "failed"], rows, nums=(3, 4, 5, 6)))


_html_SECTIONS = (("display", "Display", _html_display), ("frames", "Frames", _html_frames),
             ("triggers", "Triggers", _html_triggers), ("controls", "Controls", _html_controls),
             ("photodiode", "Photodiode and audio", _html_photodiode))


def render_html(m: dict, folder: str, history: list[dict[str, str]]) -> str:
    """The whole page for one finished rig check.

    :param m: its manifest, with ``tests``, ``rig`` and ``rig_check``.
    :param folder: its run folder, for the ``.npz`` the charts are drawn from.
    :param history: pooled rows (:func:`rows`) of the checks under
        the same data root, this one included.
    """
    rig, tests = m["rig"], m["tests"]
    failed = [t for t, v in tests.items() if v["status"] == "fail"]
    verdict = "Every test passed" if not failed else f"{len(failed)} of {len(tests)} tests failed"
    body = [f'<header><h1>Rig check · {_e(rig["site"])} / {_e(rig["rig"])}</h1>'
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
                                            if key in m["rig_check"]
                                            else '<p class="note">Not run.</p>') + "</div>")
    body.append(f'<h2>This rig over time</h2><div class="card">{_html_history(m, history)}</div>')
    body.append('<h2>Rig</h2><div class="card">' + _html_table(["", ""], [
        [_e(k), _e(v)] for k, v in rig.items()]) + "</div>")
    body.append('<details><summary>Versions and platform</summary><div class="card">'
                + _html_table(["", ""], [[_e(k), _e(v)] for k, v in {**m["versions"],
                                                                **m["platform"]}.items()])
                + "</div></details>")
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            f"<title>Rig check {_e(rig['site'])}/{_e(rig['rig'])}</title>"
            f"<style>{_CSS}</style></head><body><main>{''.join(body)}</main>"
            f'<div id="tip" role="tooltip"></div><script>{_JS}</script></body></html>\n')


if __name__ == "__main__":
    main()
