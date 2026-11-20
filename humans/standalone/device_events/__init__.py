"""``device-events``: every event of a rig's input devices, read from the OS beside a run,
each stamped by the OS when it happened.

pygame collects a run's events only on its window's thread, between flips, so the time a run
has for a press is when it read it: with a vsync-locked display, up to a refresh late. The OS
keeps each event's own time. ``fmri-play`` starts this command before a run's window, on the
devices the rig names (its ``device_events``), in a process of its own, and stops it after the
run; it writes every event they send into the run's folder, beside the run's
``pygame_events.jsonl``, as ``device_events.jsonl``. Its first line, ``recording_start``, says
what the events are read with: the reader and its clock, each device, and when the devices
were opened (``perf_time`` and ``wall_time``). Then one line per event: ``type``
(``device_event``, or ``device_gone`` for a device unplugged), ``perf_time`` (the OS's stamp,
on the ``perf_counter`` clock every process on the machine shares until it restarts),
``device`` (its index in ``recording_start``'s ``devices``) and the OS's own fields. Its last,
``recording_end``, says how and when the recording ended: a file without one was cut short.

Linux reads each device's ``/dev/input/event*`` node (:mod:`.linux`); macOS, the IOKit HID
manager (:mod:`.macos`).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

from ...config.rig import RigConfig

#: The file, in a run's folder, that holds its devices' events.
DEVICE_EVENTS_FILENAME = "device_events.jsonl"
#: What ``--out -`` prints on its standard output once the devices are open.
READY = "ready"
#: Seconds a read waits for events before the recording looks whether it should stop.
READ_TIMEOUT_S = 0.25
#: Seconds between the events file's flushes to disk.
FLUSH_INTERVAL_S = 1.0


class DeviceEventsError(RuntimeError):
    """The named devices cannot be read: one is missing, or the OS refuses."""


class OpenDevices(Protocol):
    """An OS's reader of the named devices, open (:mod:`.linux`, :mod:`.macos`)."""

    def record(self) -> dict[str, Any]: ...
    def read(self, timeout: float) -> list[dict[str, Any]]: ...
    def close(self) -> None: ...


def main(argv: list[str] | None = None) -> None:
    """``device-events --rig <rig file> --out <file>``: record until stopped (Ctrl+C or a
    SIGTERM). With ``--out -``, as ``fmri-play`` runs it: print :data:`READY` once the devices
    are open, write into the file named by standard input's first line (the events read before
    it come first), and stop also when standard input closes.

    :param argv: the command line; ``None`` reads the process's.
    :raises SystemExit: when the rig names no device, or one cannot be read.
    """
    parser = argparse.ArgumentParser(description="Read the rig's input devices from the OS, "
                                     "each event with the OS's own time, until stopped.")
    parser.add_argument("--rig", required=True, help="the rig file: its device_events")
    parser.add_argument("--out", required=True,
                        help="the file to write; '-': the one named by standard input's first "
                        "line, once the devices are open, and stop when standard input closes")
    args = parser.parse_args(argv)
    rig = RigConfig.open(args.rig)
    if rig.device_events is None:
        sys.exit(f"device-events: {args.rig} names no device_events")
    try:
        devices = open_devices(list(rig.device_events))
    except DeviceEventsError as error:
        sys.exit(str(error))
    stop = threading.Event()
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signal_number, lambda number, frame: stop.set())
    out: list[str] = [] if args.out == "-" else [args.out]
    if not out:
        print(READY, flush=True)

        def follow_stdin() -> None:
            path = sys.stdin.readline().strip()
            if path:
                out.append(path)
                print(f"device-events: reading {', '.join(rig.device_events)} into {path}",
                      file=sys.stderr)
                sys.stdin.read()
            stop.set()
        threading.Thread(target=follow_stdin, daemon=True).start()
    else:
        print(f"device-events: reading {', '.join(rig.device_events)} into {args.out}; Ctrl+C "
              "stops", file=sys.stderr)
    record_device_events(devices, lambda: out[0] if out else None, stop)


def open_devices(device_names: list[str]) -> OpenDevices:
    """This OS's reader of ``device_names``, its devices open.

    :param device_names: the devices, by the names the OS gives them.
    :return: the reader.
    :raises DeviceEventsError: on an OS with no reader, a name no device has, or a device the
        OS will not let this process read.
    """
    if sys.platform.startswith("linux"):
        from .linux import LinuxDevices
        return LinuxDevices(device_names)
    if sys.platform == "darwin":
        from .macos import MacDevices
        return MacDevices(device_names)
    raise DeviceEventsError(f"device_events: no reader for {sys.platform}")


def record_device_events(devices: OpenDevices, path: Callable[[], str | None],
                         stop: threading.Event) -> None:
    """Write every event ``devices`` read until ``stop``, flushed to disk every
    :data:`FLUSH_INTERVAL_S`, between a ``recording_start`` line and a ``recording_end`` one.

    :param devices: the open devices; closed at the end.
    :param path: the file to write, ``None`` while it is not known: the events read until it is
        are kept, and written first. A recording stopped before it is known writes nothing.
    :param stop: set to end the recording.
    """
    pending = [{"type": "recording_start", **devices.record(), "wall_time": time.time(),
                "perf_time": time.perf_counter()}]
    out = None
    n_events = 0
    last_flush = time.monotonic()
    try:
        while not stop.is_set():
            lines = devices.read(READ_TIMEOUT_S)
            pending += lines
            n_events += len(lines)
            if out is None:
                if path() is None:
                    continue
                out = open(path(), "a")
            out.writelines(json.dumps(line) + "\n" for line in pending)
            pending = []
            if time.monotonic() - last_flush >= FLUSH_INTERVAL_S:
                out.flush()
                os.fsync(out.fileno())
                last_flush = time.monotonic()
        lines = devices.read(0.0)
        pending += lines
        n_events += len(lines)
    finally:
        devices.close()
    if out is None and path() is not None:          # told where, and to stop, at once
        out = open(path(), "a")
    if out is None:
        print("device-events: stopped before it was told where to write: nothing recorded",
              file=sys.stderr)
        return
    pending.append({"type": "recording_end", "how": "stopped", "wall_time": time.time(),
                    "perf_time": time.perf_counter()})
    with out:
        out.writelines(json.dumps(line) + "\n" for line in pending)
        out.flush()
        os.fsync(out.fileno())
    print(f"device-events: {n_events} events in {path()}", file=sys.stderr)
