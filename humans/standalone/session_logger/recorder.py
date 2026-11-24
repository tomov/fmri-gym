"""The recorder of one session: every input device, the ones plugged in during it too, which
application has the focus, the machine's stalls, the lines the commands ask it to add and the
clock, written to the recording's events file as they arrive until it is told to stop or the
session's own process ends.

It reads the devices with this OS's reader: Linux reads each ``/dev/input/event*`` node
(:mod:`.linux`); macOS, the IOKit HID manager (:mod:`.macos`); Windows, its Raw Input
(:mod:`.windows`).
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from typing import IO, Any, Protocol

from ...config.rig import RigConfig
from ...save.provenance import platform_info, versions
from . import (CLOCK_INTERVAL_S, FLUSH_INTERVAL_S, READ_TIMEOUT_S, FrontApplication,
               RecordingFiles, SessionLoggerError, is_process_running)

#: Seconds between two looks at the control file for a line the commands added.
CONTROL_POLL_S = 0.05
#: Seconds between two looks at whether the session's own process is still running.
SESSION_POLL_S = 0.5
#: Seconds between two looks at which application has the focus.
FOCUS_POLL_S = 0.5
#: The heartbeat's sleep, in seconds: how often the recorder looks whether it was run.
HEARTBEAT_S = 0.001
#: Seconds the heartbeat may wake late before it is a stall: the machine ran nothing of the
#: recorder for that long, and most likely little of anything else.
STALL_THRESHOLD_S = 0.010


class OpenDevices(Protocol):
    """An OS's reader of every input device, open (:mod:`.linux`, :mod:`.macos`,
    :mod:`.windows`)."""

    def record(self) -> dict[str, Any]: ...
    def read(self, timeout: float) -> list[dict[str, Any]]: ...
    def close(self) -> None: ...


def main(argv: list[str] | None = None) -> None:
    """Record into ``--events`` until the recording's control file says to stop, or the process
    ``--session-pid`` ends.

    :param argv: the command line; ``None`` reads the process's.
    :raises SystemExit: with the reason alone, when a device cannot be read.
    """
    parser = argparse.ArgumentParser(description="Record a session into a recording's events "
                                     "file until its control file says to stop.")
    parser.add_argument("--events", required=True, help="the recording's events file, made here")
    parser.add_argument("--rig", required=True, help="the rig file, recorded in the first line")
    parser.add_argument("--subject", required=True, help="sub-<label>")
    parser.add_argument("--ses", type=int, required=True, help="the session number, from 1")
    parser.add_argument("--session-pid", type=int, required=True,
                        help="the session's own process: the recording ends when it ends")
    args = parser.parse_args(argv)
    files = RecordingFiles.of_events_file(args.events)
    rig = RigConfig.open(args.rig)
    try:
        devices = open_devices()
    except SessionLoggerError as error:
        sys.exit(str(error))            # the message alone: the command that started it raises
    front_application = front_application_reader()
    side_lines: queue.Queue[dict[str, Any]] = queue.Queue()
    stop = threading.Event()
    sources: list[tuple[Callable[..., None], tuple[Any, ...]]] = [
        (follow_control_file, (files.control, side_lines, stop)),
        (watch_session_process, (args.session_pid, side_lines, stop)),
        (sample_clock, (side_lines, stop)), (watch_stalls, (side_lines, stop))]
    if front_application is not None:
        sources.append((watch_focus, (front_application, side_lines, stop)))
    for source, source_arguments in sources:
        threading.Thread(target=source, args=source_arguments, daemon=True).start()
    first_line = {"type": "recording_start", "perf_time": time.perf_counter(),
                  "wall_time": time.time(), "subject": args.subject, "session": args.ses,
                  **devices.record(), "sources": _sources(devices, front_application),
                  "rig": rig.record(), "recorder_pid": os.getpid(),
                  "session_pid": args.session_pid, "versions": versions(),
                  "platform": platform_info()}
    record_session_events(devices, files.events, first_line, side_lines, stop)


def open_devices() -> OpenDevices:
    """This OS's reader of every input device, its devices open.

    :return: the reader.
    :raises SessionLoggerError: on an OS with no reader, or a device the OS will not let this
        process read.
    """
    if sys.platform.startswith("linux"):
        from .linux import LinuxDevices
        return LinuxDevices()
    if sys.platform == "darwin":
        from .macos import MacDevices
        return MacDevices()
    if sys.platform == "win32":
        from .windows import WindowsDevices
        return WindowsDevices()
    raise SessionLoggerError(f"session-logger: no device reader for {sys.platform}")


def front_application_reader() -> Callable[[], FrontApplication | None] | None:
    """This OS's way of asking which application has the focus; ``None`` when it has none, or
    when it does not answer here (a Linux session with no X display)."""
    if sys.platform.startswith("linux"):
        from .linux import front_application
    elif sys.platform == "darwin":
        from .macos import front_application
    elif sys.platform == "win32":
        from .windows import front_application
    else:
        return None
    return front_application if front_application() is not None else None


def record_session_events(devices: OpenDevices, path: str, first_line: dict[str, Any],
                          side_lines: queue.Queue[dict[str, Any]],
                          stop: threading.Event) -> None:
    """Write ``first_line``, then every line ``devices`` read or ``side_lines`` holds, as they
    come, flushed to disk every :data:`FLUSH_INTERVAL_S`, until ``stop``; then the last ones and
    a ``recording_end`` line that says how it ended (``session_ended`` when a ``session_ended``
    line was written, else ``stopped``), counts the lines and lists the devices plugged in and
    unplugged.

    :param devices: the open devices; closed at the end.
    :param path: the events file, made here: a recording never writes over another.
    :param first_line: the ``recording_start`` line, on disk before any device is read.
    :param side_lines: the lines other threads add.
    :param stop: set to end the recording.
    """
    counts: Counter[str] = Counter()
    devices_changed: dict[str, list[dict[str, Any]]] = {"device_added": [], "device_gone": []}
    with open(path, "x") as out:
        _write_lines(out, [first_line], counts, devices_changed)
        _sync(out)
        last_sync = time.monotonic()
        while not stop.is_set():
            _write_lines(out, devices.read(READ_TIMEOUT_S) + _drain(side_lines), counts,
                         devices_changed)
            if time.monotonic() - last_sync >= FLUSH_INTERVAL_S:
                _sync(out)
                last_sync = time.monotonic()
        _write_lines(out, devices.read(0.0) + _drain(side_lines), counts, devices_changed)
        devices.close()
        del counts["recording_start"]
        how = "session_ended" if counts["session_ended"] else "stopped"
        out.write(json.dumps({"type": "recording_end", "how": how,
                              "perf_time": time.perf_counter(), "wall_time": time.time(),
                              "counts": dict(counts),
                              "devices_added": devices_changed["device_added"],
                              "devices_gone": devices_changed["device_gone"]}) + "\n")
        _sync(out)


def follow_control_file(path: str, side_lines: queue.Queue[dict[str, Any]],
                        stop: threading.Event) -> None:
    """Pass on each line the commands append to the control file, until one says to stop.

    :param path: the control file, made here when no command has made it yet.
    :param side_lines: where the lines to record go.
    :param stop: set when a line says to stop.
    """
    open(path, "a").close()
    with open(path, "rb") as control:
        while not stop.is_set():
            line = control.readline()
            if not line.endswith(b"\n"):       # nothing new, or a line still being written
                control.seek(control.tell() - len(line))
                stop.wait(CONTROL_POLL_S)
                continue
            asked = json.loads(line)
            if asked["type"] == "stop":
                stop.set()
                return
            side_lines.put(asked)


def watch_session_process(session_pid: int, side_lines: queue.Queue[dict[str, Any]],
                          stop: threading.Event) -> None:
    """A ``session_ended`` line, then ``stop``, once the process ``session_pid`` is not running:
    looked at every :data:`SESSION_POLL_S`, stamped when seen.

    :param session_pid: the session's own process.
    :param side_lines: where the line goes.
    :param stop: set when it has ended, or by another to end this.
    """
    while not stop.wait(SESSION_POLL_S):
        if is_process_running(session_pid):
            continue
        side_lines.put({"type": "session_ended", "perf_time": time.perf_counter(),
                        "wall_time": time.time(), "pid": session_pid})
        stop.set()
        return


def sample_clock(side_lines: queue.Queue[dict[str, Any]], stop: threading.Event) -> None:
    """A ``clock`` line every :data:`CLOCK_INTERVAL_S`: ``perf_counter`` with the wall time,
    so that a recording keeps both clocks however long it runs."""
    while not stop.wait(CLOCK_INTERVAL_S):
        side_lines.put({"type": "clock", "perf_time": time.perf_counter(),
                        "wall_time": time.time()})


def watch_stalls(side_lines: queue.Queue[dict[str, Any]], stop: threading.Event) -> None:
    """A ``stall`` line each time a sleep of :data:`HEARTBEAT_S` wakes more than
    :data:`STALL_THRESHOLD_S` late: from when it should have woken (``perf_time``), for how
    long (``duration_ms``)."""
    while not stop.is_set():
        asleep = time.perf_counter()
        time.sleep(HEARTBEAT_S)
        late = time.perf_counter() - asleep - HEARTBEAT_S
        if late > STALL_THRESHOLD_S:
            side_lines.put({"type": "stall", "perf_time": asleep + HEARTBEAT_S,
                            "duration_ms": round(1000 * late, 3)})


def watch_focus(front_application: Callable[[], FrontApplication | None],
                side_lines: queue.Queue[dict[str, Any]], stop: threading.Event) -> None:
    """A ``window_focus`` line each time another application has the focus: looked at every
    :data:`FOCUS_POLL_S`, stamped when seen.

    :param front_application: asks the OS which application has the focus.
    :param side_lines: where the lines go.
    :param stop: set to end.
    """
    last = None
    while not stop.is_set():
        front = front_application()
        if front is not None and front != last:
            side_lines.put({"type": "window_focus", "perf_time": time.perf_counter(),
                            "process": front.process, "pid": front.pid})
            last = front
        stop.wait(FOCUS_POLL_S)


def _sources(devices: OpenDevices,
             front_application: Callable[[], FrontApplication | None] | None) -> dict[str, str]:
    """What makes each type of line, and when it is stamped."""
    record = devices.record()
    return {
        "device_event": f"{record['reader']}, each event at its stamp ({record['clock']})",
        "device_added": f"{record['reader']}, when it said a device was plugged in",
        "window_focus": (f"looked at every {FOCUS_POLL_S:g} s, stamped when seen"
                         if front_application is not None else
                         "none: this OS does not say here which application has the focus"),
        "stall": f"a {1000 * HEARTBEAT_S:g} ms sleep woken over "
                 f"{1000 * STALL_THRESHOLD_S:g} ms late",
        "clock": f"every {CLOCK_INTERVAL_S:g} s",
        "mark": "session-logger mark, stamped when it ran",
        "session_ended": f"the session's own process seen gone, looked at every "
                         f"{SESSION_POLL_S:g} s"}


def _write_lines(out: IO[str], lines: list[dict[str, Any]], counts: Counter[str],
                 devices_changed: dict[str, list[dict[str, Any]]]) -> None:
    """Write ``lines``, one JSON object each, count them by type, and keep each device added
    (its line) and gone (its index and time) in ``devices_changed``."""
    out.writelines(json.dumps(line) + "\n" for line in lines)
    counts.update(line["type"] for line in lines)
    devices_changed["device_added"] += [
        {key: value for key, value in line.items() if key != "type"}
        for line in lines if line["type"] == "device_added"]
    devices_changed["device_gone"] += [{"device": line["device"], "perf_time": line["perf_time"]}
                                       for line in lines if line["type"] == "device_gone"]


def _drain(side_lines: queue.Queue[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every line ``side_lines`` holds now."""
    lines = []
    while True:
        try:
            lines.append(side_lines.get_nowait())
        except queue.Empty:
            return lines


def _sync(out: IO[str]) -> None:
    """Put what was written on disk."""
    out.flush()
    os.fsync(out.fileno())


if __name__ == "__main__":
    main()
