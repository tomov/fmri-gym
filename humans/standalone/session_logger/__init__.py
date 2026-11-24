"""``session-logger``: a session's own recording, beside its runs and whatever else it runs: every
event of every input device, the ones plugged in during the session too, read from the OS with
the time the OS stamped it, which application has the focus, when the machine stalled, the
session's marks, and the clock.

``start`` leaves a recorder (:mod:`.recorder`) running in a process and a session of its own,
which no task imports or waits on, and prints the recording's events file once its devices are
open; ``mark`` adds a line to it; ``stop`` ends it. The recorder also ends it when the session's
own process (the script that started it) ends, however it ends: killed, crashed, its terminal
closed. A recording is a file in the session's
``beh/``, beside its runs, never written over, with two others while it runs
(:class:`RecordingFiles`); its events file is written as the lines arrive, flushed to disk every second: first ``recording_start`` (what is read, with what, and a
``perf_counter`` time with its wall time), then one line per event, each with its ``type`` and
``perf_time`` (on the clock every process on the machine shares until it restarts), and last
``recording_end``, which a recording cut short lacks.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from typing import Any, NamedTuple

from ...config.rig import RigConfig
from ...save import bids

#: How a recording is named, beside the session's runs: its subject, session and start.
RECORDING_NAME = "{subject}_ses-{session:03d}_session-{start}"
#: How a recording's start is written in its name.
RECORDING_TIME_FORMAT = "%Y%m%dT%H%M%S"
#: A recording's events: its name, then this.
SESSION_EVENTS_SUFFIX = "_events.jsonl"
#: What the commands ask of a running recording, one line each: its name, then this.
CONTROL_SUFFIX = "_control.jsonl"
#: What went wrong, as the recorder said it: its name, then this.
ERRORS_SUFFIX = "_errors.log"
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


class RecordingFiles(NamedTuple):
    """A recording's files, side by side in its session's ``beh/``: its events, and while it
    runs, the commands' and the recorder's errors, which a recording that ended leaves only
    when the recorder wrote one."""

    #: its lines, as :mod:`.recorder` writes them; the path that names the recording
    events: str
    #: what the commands ask of it, which its events hold once they are done
    control: str
    #: what went wrong, as the recorder said it
    errors: str

    @classmethod
    def named(cls, folder: str, name: str) -> RecordingFiles:
        """The files of the recording ``name`` in ``folder``."""
        stem = os.path.join(folder, name)
        return cls(stem + SESSION_EVENTS_SUFFIX, stem + CONTROL_SUFFIX, stem + ERRORS_SUFFIX)

    @classmethod
    def of_events_file(cls, events_path: str) -> RecordingFiles:
        """The files of the recording whose events file is ``events_path``.

        :raises SessionLoggerError: if ``events_path`` is not named like a recording's events.
        """
        if not events_path.endswith(SESSION_EVENTS_SUFFIX):
            raise SessionLoggerError(f"session-logger: {events_path} is not a recording's events "
                                     f"file (its name ends in {SESSION_EVENTS_SUFFIX})")
        stem = os.path.abspath(events_path.removesuffix(SESSION_EVENTS_SUFFIX))
        return cls.named(os.path.dirname(stem), os.path.basename(stem))

    @property
    def name(self) -> str:
        """The recording's name: its subject, session and start."""
        return os.path.basename(self.events).removesuffix(SESSION_EVENTS_SUFFIX)


class FrontApplication(NamedTuple):
    """The application that has the keyboard's focus, as the OS says."""

    #: its process's name
    process: str
    pid: int


def main(argv: list[str] | None = None) -> None:
    """``session-logger start|mark|stop``.

    :param argv: the command line; ``None`` reads the process's.
    :raises SessionLoggerError: when ``start`` cannot record.
    """
    parser = argparse.ArgumentParser(
        prog="session-logger", description="Record a session: every input device, read "
        "from the OS with the OS's own time, the session's marks and the clock.")
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="open every input device and record until stopped; "
                                "prints the recording's events file")
    start.add_argument("--rig", required=True, help="the rig file: its data root")
    start.add_argument("--subject", required=True, help="sub-<label>")
    start.add_argument("--ses", type=int, required=True, help="the session number, from 1")
    start.add_argument("--session-pid", type=int, required=True,
                       help="the session's own process, a script's $$: the recording ends "
                       "when it ends")
    mark = commands.add_parser("mark", help="add a line to the recording, stamped now")
    mark.add_argument("recording", help="the recording's events file, as start printed it")
    mark.add_argument("text", help="what the line says")
    stop = commands.add_parser("stop", help="end the recording and wait for its last line")
    stop.add_argument("recording", help="the recording's events file, as start printed it")
    args = parser.parse_args(argv)
    if args.command == "start":
        print(start_recording(args.rig, args.subject, args.ses, args.session_pid))
    elif args.command == "mark":
        mark_recording(args.recording, args.text)
    elif not stop_recording(args.recording):
        sys.exit(1)


def recordings_of_run(run_folder: str) -> list[RecordingFiles]:
    """The recordings of a run's session: the ones beside the run's folder, in its ``beh/``.

    :param run_folder: the run's folder.
    :return: them, in the order they started.
    """
    pattern = os.path.join(os.path.dirname(os.path.abspath(run_folder)),
                           "*" + SESSION_EVENTS_SUFFIX)
    return [RecordingFiles.of_events_file(path) for path in sorted(glob.glob(pattern))]


def start_recording(rig_path: str, subject: str, session: int, session_pid: int) -> str:
    """Leave a recorder writing a new recording of the session, every input device open, until
    it is stopped or the process ``session_pid`` ends.

    :param rig_path: the rig file: its data root.
    :param subject: ``sub-<label>``.
    :param session: the session number, from 1.
    :param session_pid: the session's own process.
    :return: the recording's events file, absolute, once its first line is on disk.
    :raises SessionLoggerError: if the process ``session_pid`` is not running, if a recording of
        that name is already there, if the recorder
        stops before its first line (a device the OS will not let it read: its message says
        which) or does not write it in :data:`START_TIMEOUT_S`.
    """
    if not is_process_running(session_pid):
        raise SessionLoggerError(f"session-logger: the session's process {session_pid} is not "
                                 "running, so its recording would end at once")
    rig = RigConfig.open(rig_path)
    subject_label = bids.subject_label(subject)
    folder = os.path.abspath(bids.beh_folder(rig.data_root, subject_label, session))
    files = RecordingFiles.named(folder, RECORDING_NAME.format(
        subject=subject_label, session=session, start=time.strftime(RECORDING_TIME_FORMAT)))
    os.makedirs(folder, exist_ok=True)
    if any(os.path.exists(path) for path in files):
        raise SessionLoggerError(f"session-logger: a recording named {files.name} is already in "
                                 f"{folder}, and a recording is never written over")
    with open(files.errors, "x") as errors:
        recorder = subprocess.Popen(
            [sys.executable, "-m", f"{__name__}.recorder", "--events", files.events,
             "--rig", rig.path, "--subject", subject, "--ses", str(session),
             "--session-pid", str(session_pid)],
            stdin=subprocess.DEVNULL, stdout=errors, stderr=errors, **_DETACHED)
    deadline = time.monotonic() + START_TIMEOUT_S
    while read_first_line(files.events) is None:
        if recorder.poll() is not None:
            raise SessionLoggerError(_recorder_errors(files) or "session-logger: the recorder "
                                     f"stopped before recording (exit {recorder.returncode})")
        if time.monotonic() > deadline:
            recorder.kill()
            raise SessionLoggerError(f"session-logger: the recorder wrote nothing in "
                                     f"{START_TIMEOUT_S:g} s; its errors: "
                                     f"{_recorder_errors(files)}")
        time.sleep(0.05)
    devices = read_first_line(files.events)["devices"]
    print(f"session-logger: recording every input device ({len(devices)}) into {files.events}",
          file=sys.stderr)
    return files.events


def mark_recording(recording: str, text: str) -> None:
    """Add a ``mark`` line to a recording, stamped now; when its recorder is not running, the
    mark would reach no recording: the console is told, and nothing is written.

    :param recording: the recording's events file.
    :param text: what the line says.
    """
    files = RecordingFiles.of_events_file(recording)
    if not is_recording(recording):
        print(f"session-logger: WARNING: the recording {files.events} has stopped, so this mark "
              f"is not in it: {_cut_short_words(files)}", file=sys.stderr)
        return
    append_line(files.control,
                {"type": "mark", "perf_time": time.perf_counter(), "wall_time": time.time(),
                 "text": text})


def stop_recording(recording: str) -> bool:
    """End a recording, wait until its last line is on disk and its recorder has exited, then
    remove its control file, which its events hold, and its errors file when it is empty.

    :param recording: the recording's events file.
    :return: whether it ended when told to, or had already: ``False`` when its recorder had
        stopped before its last line or did not end in :data:`STOP_TIMEOUT_S`, which the console
        is told, and its files are all kept.
    """
    files = RecordingFiles.of_events_file(recording)
    was_recording = is_recording(recording)
    if was_recording:
        append_line(files.control,
                    {"type": "stop", "perf_time": time.perf_counter(), "wall_time": time.time()})
    deadline = time.monotonic() + (STOP_TIMEOUT_S if was_recording else 0.0)
    while True:
        last = read_last_line(files.events)
        if last is not None and last["type"] == "recording_end":
            print(f"session-logger: recording ended, {_counts_words(last['counts'])}: "
                  f"{files.events}", file=sys.stderr)
            _warn_of_devices_gone(files, last)
            _remove_control_and_empty_errors(files, deadline)
            return True
        if time.monotonic() > deadline:
            break
        time.sleep(0.1)
    print(f"session-logger: WARNING: the recording {files.events} did not end when told to: "
          f"{_cut_short_words(files)}", file=sys.stderr)
    return False


def is_recording(recording: str) -> bool:
    """Whether the recording whose events file is ``recording`` has not written its last line
    and its recorder is running."""
    first = read_first_line(recording)
    if first is None or read_last_line(recording)["type"] == "recording_end":
        return False
    return is_process_running(first["recorder_pid"])


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


def is_process_running(pid: int) -> bool:
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


def _recorder_errors(files: RecordingFiles) -> str:
    """What the recorder wrote to its errors file, for a message that must say why it
    stopped."""
    with open(files.errors) as f:
        return f.read().strip()


def _remove_control_and_empty_errors(files: RecordingFiles, deadline: float) -> None:
    """Once the recorder has exited (waited for until ``deadline``), remove the control file
    and an empty errors file; the console is told of any it could not remove."""
    pid = read_first_line(files.events)["recorder_pid"]
    while is_process_running(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    removable = [files.control] + ([files.errors] if os.path.exists(files.errors)
                                   and os.path.getsize(files.errors) == 0 else [])
    for path in removable:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as error:
            print(f"session-logger: WARNING: {path} could not be removed: {error}",
                  file=sys.stderr)


def _cut_short_words(files: RecordingFiles) -> str:
    """When a recording's last line was written, and where its recorder said why it stopped."""
    last = read_last_line(files.events)
    when = ("it wrote nothing" if last is None else "its last line is from "
            + time.strftime("%H:%M:%S", time.localtime(_wall_time_of(last, files))))
    return (f"{when}, and the session's events after it are not recorded (its errors: "
            f"{files.errors})")


def _wall_time_of(line: dict[str, Any], files: RecordingFiles) -> float:
    """A line's wall time, from its own or from the recording's first line's clock pair."""
    if "wall_time" in line:
        return line["wall_time"]
    first = read_first_line(files.events)
    return line["perf_time"] + first["wall_time"] - first["perf_time"]


def _warn_of_devices_gone(files: RecordingFiles, last_line: dict[str, Any]) -> None:
    """Say on the console which device was unplugged during the recording, and when: the
    recording holds none of its events after that.

    :param files: the recording's files.
    :param last_line: its ``recording_end`` line, which lists the devices added and gone.
    """
    devices = read_first_line(files.events)["devices"] + last_line["devices_added"]
    for gone in last_line["devices_gone"]:
        when = time.strftime("%H:%M:%S", time.localtime(_wall_time_of(gone, files)))
        print(f"session-logger: WARNING: {devices[gone['device']]['name']!r} (device "
              f"{gone['device']}) was unplugged at {when}: the recording holds none of its "
              "events after that", file=sys.stderr)


def _counts_words(counts: dict[str, int]) -> str:
    """A recording's line counts in words, by type."""
    return ", ".join(f"{count} {line_type}" for line_type, count in sorted(counts.items())) or \
        "no line"
