"""Linux: every ``/dev/input/event*`` node, the ones plugged in later too, its events stamped by
the kernel, and the application whose window the X window manager says is active.

The kernel stamps each event when the device's report arrives, on ``CLOCK_MONOTONIC`` once
asked to (``EVIOCSCLOCKID``): the clock ``perf_counter`` reads on Linux. The devices are
described by sysfs, which anyone may read; reading a node takes the ``input`` group. The active
window is read with ``xprop``, so a Wayland session without X says none.
"""

from __future__ import annotations

import fcntl
import glob
import os
import re
import select
import struct
import subprocess
import sys
import time
from typing import Any

from . import FrontApplication, SessionLoggerError

#: The kernel's ``struct input_event``: a ``timeval``, then type, code and value.
_EVENT = struct.Struct("@llHHi")
_CLOCK_MONOTONIC = 1
#: ``EVIOCSCLOCKID``: ``_IOW('E', 0xa0, int)``.
_EVIOCSCLOCKID = (1 << 30) | (struct.calcsize("i") << 16) | (ord("E") << 8) | 0xa0
#: Seconds between two looks for a device plugged in.
DEVICE_LOOK_S = 1.0
#: Seconds a node plugged in is given to become readable: udev sets its group after making it.
NEW_NODE_GRACE_S = 3.0
#: A window id as ``xprop`` prints it.
_X_WINDOW = re.compile(r"window id # (0x[0-9a-fA-F]+)")
#: The last application :func:`front_application` named, by its window's id.
_last_front_application: tuple[str, FrontApplication | None] = ("", None)


class LinuxDevices:
    """Every event node, the ones plugged in later too, open; their events as the kernel
    stamped them."""

    def __init__(self) -> None:
        """Open every event node.

        :raises SessionLoggerError: if ``perf_counter`` is not on ``CLOCK_MONOTONIC``, or a node
            cannot be read.
        """
        implementation = time.get_clock_info("perf_counter").implementation
        if implementation != "clock_gettime(CLOCK_MONOTONIC)":
            raise SessionLoggerError(f"session-logger: perf_counter reads {implementation}, "
                                    "not the kernel's CLOCK_MONOTONIC")
        #: each device's identity, by its index
        self._devices: list[dict[str, Any]] = []
        #: file descriptor -> the device's index
        self._fds: dict[int, int] = {}
        #: the node of each device read now -> its file descriptor
        self._fd_of_node: dict[str, int] = {}
        #: a node plugged in that could not be read yet -> when it was first seen
        self._unreadable_since: dict[str, float] = {}
        #: nodes that stayed unreadable, each a line already
        self._refused: set[str] = set()
        self._next_look = time.monotonic() + DEVICE_LOOK_S
        for device in _present_devices():
            try:
                self._open(device)
            except PermissionError:
                self.close()
                raise SessionLoggerError(f"session-logger: {device['path']} ({device['name']}) "
                                         "cannot be read by this user") from None

    def record(self) -> dict[str, Any]:
        """The reader, the clock of its stamps, and each device open now by its index."""
        return {"reader": "linux evdev", "clock": "CLOCK_MONOTONIC",
                "devices": list(self._devices)}

    def read(self, timeout: float) -> list[dict[str, Any]]:
        """The events that arrived, waiting up to ``timeout`` seconds for the first.

        :param timeout: seconds to wait when none is there yet.
        :return: one line per event (``event_type``, ``code``, ``value``, as the kernel gives
            them), a ``device_added`` line for a device plugged in, and a ``device_gone`` line
            for one unplugged, whose node is dropped.
        """
        lines = self._look_for_devices() if time.monotonic() >= self._next_look else []
        if not self._fds:
            time.sleep(timeout)
            return lines
        ready, _, _ = select.select(list(self._fds), [], [], timeout)
        for fd in ready:
            try:
                data = os.read(fd, _EVENT.size * 64)
            except BlockingIOError:
                continue
            except OSError:
                lines.append({"type": "device_gone", "perf_time": time.perf_counter(),
                              "device": self._close(fd)})
                continue
            lines += _event_lines(data, self._fds[fd])
        return lines

    def close(self) -> None:
        """Close every node still open."""
        for fd in list(self._fds):
            self._close(fd)

    def _open(self, device: dict[str, Any]) -> int:
        """Open a device's node, stamped on ``CLOCK_MONOTONIC``, and give it the next index.

        :raises PermissionError: if this user cannot read it.
        """
        fd = os.open(device["path"], os.O_RDONLY | os.O_NONBLOCK)
        fcntl.ioctl(fd, _EVIOCSCLOCKID, struct.pack("i", _CLOCK_MONOTONIC))
        self._fds[fd] = len(self._devices)
        self._fd_of_node[device["path"]] = fd
        self._devices.append(device)
        return self._fds[fd]

    def _close(self, fd: int) -> int:
        """Close a node; its device's index."""
        index = self._fds.pop(fd)
        del self._fd_of_node[self._devices[index]["path"]]
        os.close(fd)
        return index

    def _look_for_devices(self) -> list[dict[str, Any]]:
        """Open each node plugged in since the last look: a ``device_added`` line each, with
        its identity; one still unreadable after :data:`NEW_NODE_GRACE_S` is a line too, with
        why, and the console is told."""
        now = time.monotonic()
        self._next_look = now + DEVICE_LOOK_S
        present = _present_devices()
        nodes = {device["path"] for device in present}
        self._refused &= nodes
        self._unreadable_since = {node: since for node, since in self._unreadable_since.items()
                                  if node in nodes}
        lines = []
        for device in present:
            node = device["path"]
            if node in self._fd_of_node or node in self._refused:
                continue
            found = time.perf_counter()
            try:
                index = self._open(device)
            except PermissionError:
                since = self._unreadable_since.setdefault(node, now)
                if now - since < NEW_NODE_GRACE_S:
                    continue
                self._refused.add(node)
                index, error = len(self._devices), (f"session-logger: {node} ({device['name']}) "
                                                    "cannot be read by this user")
                self._devices.append({**device, "error": error})
                print(f"{error} (plugged in during the recording, as device {index})",
                      file=sys.stderr, flush=True)
            self._unreadable_since.pop(node, None)
            lines.append({"type": "device_added", "perf_time": found, "device": index,
                          **self._devices[index]})
        return lines


def front_application() -> FrontApplication | None:
    """The application whose window the window manager says is active
    (``_NET_ACTIVE_WINDOW``); ``None`` without an X display or ``xprop``, or when no window is.
    """
    global _last_front_application
    found = _X_WINDOW.search(_xprop("-root", "_NET_ACTIVE_WINDOW"))
    if found is None or int(found[1], 16) == 0:
        return None
    if found[1] == _last_front_application[0]:
        return _last_front_application[1]
    pid = re.search(r"= (\d+)", _xprop("-id", found[1], "_NET_WM_PID"))
    front = None if pid is None else FrontApplication(_process_name(int(pid[1])), int(pid[1]))
    _last_front_application = (found[1], front)
    return front


def _xprop(*arguments: str) -> str:
    """What ``xprop`` prints; nothing without an X display, or when it cannot run."""
    if not os.environ.get("DISPLAY"):
        return ""
    try:
        return subprocess.run(["xprop", *arguments], capture_output=True, text=True,
                              timeout=1).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _process_name(pid: int) -> str:
    """A process's name, as the kernel gives it; ``""`` once it has ended."""
    try:
        with open(f"/proc/{pid}/comm") as f:
            return f.read().strip()
    except OSError:
        return ""

def _event_lines(data: bytes, device: int) -> list[dict[str, Any]]:
    """The kernel's ``input_event`` records in ``data`` as lines, each at the kernel's stamp.

    :param data: whole records, as read from a node.
    :param device: the device's index.
    :return: one line per record.
    """
    lines = []
    for offset in range(0, len(data) - len(data) % _EVENT.size, _EVENT.size):
        seconds, microseconds, event_type, code, value = _EVENT.unpack_from(data, offset)
        lines.append({"type": "device_event", "perf_time": seconds + microseconds / 1e6,
                      "device": device, "event_type": event_type, "code": code,
                      "value": value})
    return lines


def _present_devices() -> list[dict[str, Any]]:
    """Each event node's device, with its identity (:func:`_sysfs_device`), in node order."""
    return [_sysfs_device(node) for node in sorted(glob.glob("/sys/class/input/event*"))]


def _sysfs_device(node: str) -> dict[str, Any]:
    """An event node's device as sysfs describes it: name, bus, vendor, product, version,
    physical path, and its ``/dev/input`` node."""
    def read(name: str) -> str:
        try:
            with open(os.path.join(node, "device", name)) as f:
                return f.read().strip()
        except OSError:
            return ""
    return {"name": read("name"), "path": os.path.join("/dev/input", os.path.basename(node)),
            "bustype": read("id/bustype"), "vendor": read("id/vendor"),
            "product": read("id/product"), "version": read("id/version"),
            "phys": read("phys")}
