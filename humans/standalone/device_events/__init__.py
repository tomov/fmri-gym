"""``device-events``: every event of a rig's input devices, read from the OS beside a run,
each stamped by the OS when it happened.

pygame collects a run's events only on its window's thread, between flips, so the time a run
has for a press is when it read it: with a vsync-locked display, up to a refresh late. The OS
keeps each event's own time. Started beside a run and stopped after it, this command
reads the devices the rig names (its ``device_events``) beneath pygame, in a process of its
own, and writes every event they send to
``<subject>/ses-<NNN>/device_events/<start>/events.jsonl`` under the rig's data root:
``type`` (``device_event``, or ``device_gone`` for a device unplugged), ``perf_time`` (the
OS's stamp, on the ``perf_counter`` clock every process on the machine shares until it
restarts), ``device`` (its index in the folder's ``manifest.json``) and the OS's own fields.
The manifest says what the events are read with: the reader and its clock, each device, a
``perf_counter`` time with its wall time (what matches the recording to a run's clock), the
code and the machine, and how the recording ended.

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
from typing import Any, Protocol

from ...config.rig import RigConfig
from ...save import bids
from ...save.provenance import platform_info, versions

#: The folder, in a session's, that holds its recordings.
DEVICE_EVENTS_FOLDER = "device_events"
EVENTS_FILENAME = "events.jsonl"
MANIFEST_FILENAME = "manifest.json"
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
    """``device-events --rig <rig file> --subject sub-.. --ses N``: record until stopped (Ctrl+C,
    a SIGTERM, or with ``--until-stdin-closes``, the end of its standard input).

    :param argv: the command line; ``None`` reads the process's.
    :raises SystemExit: when the rig names no device, or one cannot be read.
    """
    parser = argparse.ArgumentParser(description="Read the rig's input devices from the OS, "
                                     "each event with the OS's own time, until stopped.")
    parser.add_argument("--rig", required=True, help="the rig file: its device_events and data "
                        "root")
    parser.add_argument("--subject", required=True, help="sub-<label>")
    parser.add_argument("--ses", type=int, required=True, help="the session number, from 1")
    parser.add_argument("--until-stdin-closes", action="store_true",
                        help="also stop when standard input closes: the process that started "
                        "it ends the recording by closing the pipe, or by exiting")
    args = parser.parse_args(argv)
    rig = RigConfig.open(args.rig)
    if rig.device_events is None:
        sys.exit(f"device-events: {args.rig} names no device_events")
    folder = os.path.join(bids.session_folder(rig.data_root, bids.subject_label(args.subject),
                                              args.ses),
                          DEVICE_EVENTS_FOLDER, time.strftime("%Y%m%dT%H%M%S"))
    try:
        devices = open_devices(list(rig.device_events))
    except DeviceEventsError as error:
        sys.exit(str(error))
    stop = threading.Event()
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signal_number, lambda number, frame: stop.set())
    if args.until_stdin_closes:
        threading.Thread(target=lambda: (sys.stdin.buffer.read(), stop.set()),
                         daemon=True).start()
    print(f"device-events: reading {', '.join(rig.device_events)} into {folder}; Ctrl+C stops",
          file=sys.stderr)
    record_device_events(devices, folder, rig, stop)


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


def record_device_events(devices: OpenDevices, folder: str, rig: RigConfig,
                         stop: threading.Event) -> None:
    """Write every event ``devices`` read until ``stop``, flushed to disk every
    :data:`FLUSH_INTERVAL_S`, and the manifest that says what they are read with: first when the
    recording starts (then the folder is printed on stdout), again when it ends.

    :param devices: the open devices; closed at the end.
    :param folder: the recording's folder, made here.
    :param rig: the rig, recorded in the manifest.
    :param stop: set to end the recording.
    """
    os.makedirs(folder)
    manifest = {**devices.record(), "rig": rig.record(), "versions": versions(),
                "platform": platform_info(),
                "started": {"wall_time": time.time(), "perf_time": time.perf_counter()},
                "ended": None}
    _write_manifest(folder, manifest)
    print(folder, flush=True)       # on stdout: the recording has started, here
    n_events = 0
    with open(os.path.join(folder, EVENTS_FILENAME), "a") as out:
        last_flush = time.monotonic()
        while not stop.is_set():
            lines = devices.read(READ_TIMEOUT_S)
            out.writelines(json.dumps(line) + "\n" for line in lines)
            n_events += len(lines)
            if time.monotonic() - last_flush >= FLUSH_INTERVAL_S:
                out.flush()
                os.fsync(out.fileno())
                last_flush = time.monotonic()
        last_lines = devices.read(0.0)
        out.writelines(json.dumps(line) + "\n" for line in last_lines)
        n_events += len(last_lines)
        out.flush()
        os.fsync(out.fileno())
    devices.close()
    manifest["ended"] = {"how": "stopped", "wall_time": time.time(),
                         "perf_time": time.perf_counter()}
    _write_manifest(folder, manifest)
    print(f"device-events: {n_events} events in {folder}", file=sys.stderr)


def _write_manifest(folder: str, manifest: dict[str, Any]) -> None:
    """The recording's manifest, written whole."""
    path = os.path.join(folder, MANIFEST_FILENAME)
    with open(f"{path}.tmp", "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(f"{path}.tmp", path)
