"""``session-logger``: a session's own recording, beside its runs and whatever else it runs: every
event of the rig's input devices, read from the OS with the time the OS stamped it, the
session's marks, and the clock.

``start`` leaves a recorder (:mod:`.recorder`) running in a process and a session of its own,
which no task imports or waits on, and prints the recording's folder once its devices are open;
``mark`` adds a line to it; ``stop`` ends it. A recording is a folder under the rig's data root,
beside the subjects' (:func:`recordings_folder`), and its events file is written as the lines
arrive, flushed to disk every second: first ``recording_start`` (what is read, with what, and a
``perf_counter`` time with its wall time), then one line per event, each with its ``type`` and
``perf_time`` (on the clock every process on the machine shares until it restarts), and last
``recording_end``, which a recording cut short lacks.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ...config.rig import RigConfig
from ...save import bids

#: The folder, under a data root, that holds every subject's session recordings.
SESSION_EVENTS_FOLDER = "session_events"
#: A recording's events, in its folder.
SESSION_EVENTS_FILENAME = "session_events.jsonl"
#: What the commands ask of a running recording, one line each, in its folder.
CONTROL_FILENAME = "control.jsonl"
#: The recorder's own messages, in the recording's folder.
LOG_FILENAME = "session_logger.log"
#: How a recording's folder is named: when it started.
FOLDER_TIME_FORMAT = "%Y%m%dT%H%M%S"
#: Seconds a device read waits for events before the recorder looks for anything else to write.
READ_TIMEOUT_S = 0.25
#: Seconds between the events file's flushes to disk.
FLUSH_INTERVAL_S = 1.0
#: Seconds between two ``clock`` lines; the longest a running recording goes without a line.
CLOCK_INTERVAL_S = 10.0
#: Seconds the recorder is given to open the devices and write its first line.
START_TIMEOUT_S = 30.0
#: Seconds a stopped recorder is given to write its last line.
STOP_TIMEOUT_S = 10.0
#: How the recorder is started: in a session of its own, so a terminal's Ctrl+C or hang-up does
#: not reach it; on Windows, in a process group of its own with no console. uv runs a command in
#: a Windows job that kills it when uv exits, but lets its children break away, as this one does.
_DETACHED: dict[str, Any] = (
    {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
    if sys.platform == "win32" else {"start_new_session": True})


class SessionLoggerError(RuntimeError):
    """The session cannot be recorded: a device is missing or unreadable, or the recorder
    stopped."""


def main(argv: list[str] | None = None) -> None:
    """``session-logger start|mark|stop``.

    :param argv: the command line; ``None`` reads the process's.
    :raises SessionLoggerError: when ``start`` cannot record.
    """
    parser = argparse.ArgumentParser(
        prog="session-logger", description="Record a session: the rig's input devices, read "
        "from the OS with the OS's own time, the session's marks and the clock.")
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="open the rig's devices and record until stopped; "
                                "prints the recording's folder")
    start.add_argument("--rig", required=True, help="the rig file: its device_events and data "
                       "root")
    start.add_argument("--subject", required=True, help="sub-<label>")
    start.add_argument("--ses", type=int, required=True, help="the session number, from 1")
    mark = commands.add_parser("mark", help="add a line to the recording, stamped now")
    mark.add_argument("recording", help="the recording's folder, as start printed it")
    mark.add_argument("text", help="what the line says")
    stop = commands.add_parser("stop", help="end the recording and wait for its last line")
    stop.add_argument("recording", help="the recording's folder, as start printed it")
    args = parser.parse_args(argv)
    if args.command == "start":
        print(start_recording(args.rig, args.subject, args.ses))
    elif args.command == "mark":
        mark_recording(args.recording, args.text)
    elif not stop_recording(args.recording):
        sys.exit(1)


def recordings_folder(data_root: str, subject: str, session: int) -> str:
    """Where a session's recordings go: beside the subjects' folders, so that a session that
    has played no run yet is not taken.

    :param data_root: the rig's data root.
    :param subject: ``sub-<label>``.
    :param session: the session number, from 1.
    :return: the folder, one recording per subfolder.
    """
    return bids.session_folder(os.path.join(data_root, SESSION_EVENTS_FOLDER), subject, session)


def recordings_folder_of_run(run_folder: str) -> str:
    """Where the recordings of a run's session are, from the run's folder:
    ``<data root>/<subject>/<session>/beh/<run>``.

    :param run_folder: the run's folder.
    :return: the folder :func:`recordings_folder` gives its session.
    """
    data_root, (subject, session) = Path(run_folder).parents[3], Path(run_folder).parts[-4:-2]
    return os.path.join(data_root, SESSION_EVENTS_FOLDER, subject, session)


def start_recording(rig_path: str, subject: str, session: int) -> str:
    """Leave a recorder writing a new recording of the session, its devices open.

    :param rig_path: the rig file: the devices it names in ``device_events``, its data root.
    :param subject: ``sub-<label>``.
    :param session: the session number, from 1.
    :return: the recording's folder, absolute, once its first line is on disk.
    :raises SessionLoggerError: if the recorder stops before its first line (a device missing,
        or one the OS will not let it read: its message says which) or does not write it in
        :data:`START_TIMEOUT_S`.
    """
    rig = RigConfig.open(rig_path)
    folder = os.path.abspath(os.path.join(
        recordings_folder(rig.data_root, bids.subject_label(subject), session),
        time.strftime(FOLDER_TIME_FORMAT)))
    os.makedirs(folder)
    with open(os.path.join(folder, LOG_FILENAME), "w") as log:
        recorder = subprocess.Popen(
            [sys.executable, "-m", f"{__name__}.recorder", "--folder", folder,
             "--rig", rig.path, "--subject", subject, "--ses", str(session)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, **_DETACHED)
    events_path = os.path.join(folder, SESSION_EVENTS_FILENAME)
    deadline = time.monotonic() + START_TIMEOUT_S
    while read_first_line(events_path) is None:
        if recorder.poll() is not None:
            raise SessionLoggerError(_recorder_log(folder) or "session-logger: the recorder "
                                     f"stopped before recording (exit {recorder.returncode})")
        if time.monotonic() > deadline:
            recorder.kill()
            raise SessionLoggerError(f"session-logger: the recorder wrote nothing in "
                                     f"{START_TIMEOUT_S:g} s; its log: {_recorder_log(folder)}")
        time.sleep(0.05)
    print(f"session-logger: recording {_devices_words(rig)} into {folder}", file=sys.stderr)
    return folder


def mark_recording(folder: str, text: str) -> None:
    """Add a ``mark`` line to a recording, stamped now; say on the console when its recorder
    is not running, since the mark then reaches no recording.

    :param folder: the recording's folder.
    :param text: what the line says.
    """
    append_line(os.path.join(folder, CONTROL_FILENAME),
                {"type": "mark", "perf_time": time.perf_counter(), "wall_time": time.time(),
                 "text": text})
    if not is_recording(folder):
        print(f"session-logger: WARNING: the recording in {folder} has stopped: "
              f"{_cut_short_words(folder)}", file=sys.stderr)


def stop_recording(folder: str) -> bool:
    """End a recording, and wait until its last line is on disk.

    :param folder: the recording's folder.
    :return: whether it ended when told to: ``False`` when its recorder had already stopped or
        did not end in :data:`STOP_TIMEOUT_S`, which the console is told.
    """
    events_path = os.path.join(folder, SESSION_EVENTS_FILENAME)
    was_recording = is_recording(folder)
    append_line(os.path.join(folder, CONTROL_FILENAME),
                {"type": "stop", "perf_time": time.perf_counter(), "wall_time": time.time()})
    deadline = time.monotonic() + (STOP_TIMEOUT_S if was_recording else 0.0)
    while True:
        last = read_last_line(events_path)
        if last is not None and last["type"] == "recording_end":
            print(f"session-logger: recording ended, {_counts_words(last['counts'])}: "
                  f"{events_path}", file=sys.stderr)
            return True
        if time.monotonic() > deadline:
            break
        time.sleep(0.1)
    print(f"session-logger: WARNING: the recording in {folder} did not end when told to: "
          f"{_cut_short_words(folder)}", file=sys.stderr)
    return False


def is_recording(folder: str) -> bool:
    """Whether the recording in ``folder`` has not written its last line and its recorder is
    running."""
    events_path = os.path.join(folder, SESSION_EVENTS_FILENAME)
    first = read_first_line(events_path)
    if first is None or read_last_line(events_path)["type"] == "recording_end":
        return False
    return _is_process_running(first["recorder_pid"])


def append_line(path: str, line: dict[str, Any]) -> None:
    """Append one JSON line to ``path``, on disk before this returns."""
    with open(path, "a") as f:
        f.write(json.dumps(line) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_first_line(path: str) -> dict[str, Any] | None:
    """A JSON-lines file's first line; ``None`` while it has no complete one."""
    try:
        with open(path) as f:
            line = f.readline()
    except FileNotFoundError:
        return None
    return json.loads(line) if line.endswith("\n") else None


def read_last_line(path: str) -> dict[str, Any] | None:
    """A JSON-lines file's last complete line; ``None`` when it has none."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 65536))
            tail = f.read().decode(errors="replace")
    except FileNotFoundError:
        return None
    complete = tail[:tail.rfind("\n") + 1].splitlines()
    return json.loads(complete[-1]) if complete else None


def _is_process_running(pid: int) -> bool:
    """Whether the process ``pid`` is running."""
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.OpenProcess.restype = ctypes.c_void_p
        handle = kernel32.OpenProcess(0x1000, False, pid)   # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        exit_code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(exit_code))
        kernel32.CloseHandle(ctypes.c_void_p(handle))
        return exit_code.value == 259                          # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _recorder_log(folder: str) -> str:
    """What the recorder wrote to its log, for a message that must say why it stopped."""
    with open(os.path.join(folder, LOG_FILENAME)) as f:
        return f.read().strip()


def _cut_short_words(folder: str) -> str:
    """When a recording's last line was written, and where its recorder said why it stopped."""
    last = read_last_line(os.path.join(folder, SESSION_EVENTS_FILENAME))
    when = ("it wrote nothing" if last is None else "its last line is from "
            + time.strftime("%H:%M:%S", time.localtime(_wall_time_of(last, folder))))
    return (f"{when}, and the session's events after it are not recorded (its log: "
            f"{os.path.join(folder, LOG_FILENAME)})")


def _wall_time_of(line: dict[str, Any], folder: str) -> float:
    """A line's wall time, from its own or from the recording's first line's clock pair."""
    if "wall_time" in line:
        return line["wall_time"]
    first = read_first_line(os.path.join(folder, SESSION_EVENTS_FILENAME))
    return line["perf_time"] + first["wall_time"] - first["perf_time"]


def _counts_words(counts: dict[str, int]) -> str:
    """A recording's line counts in words, by type."""
    return ", ".join(f"{count} {line_type}" for line_type, count in sorted(counts.items())) or \
        "no line"


def _devices_words(rig: RigConfig) -> str:
    """The devices a rig has recorded, or that it names none."""
    if rig.device_events is None:
        return f"no device ({rig.path} names no device_events)"
    return ", ".join(rig.device_events)
