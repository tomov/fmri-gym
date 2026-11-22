"""Linux: each device's ``/dev/input/event*`` node, its events stamped by the kernel.

The kernel stamps each event when the device's report arrives, on ``CLOCK_MONOTONIC`` once
asked to (``EVIOCSCLOCKID``): the clock ``perf_counter`` reads on Linux. The devices are found
by name in sysfs, which anyone may read; reading a node takes the ``input`` group.
"""

from __future__ import annotations

import fcntl
import glob
import os
import select
import struct
import time
from typing import Any

from . import SessionLoggerError

#: The kernel's ``struct input_event``: a ``timeval``, then type, code and value.
_EVENT = struct.Struct("@llHHi")
_CLOCK_MONOTONIC = 1
#: ``EVIOCSCLOCKID``: ``_IOW('E', 0xa0, int)``.
_EVIOCSCLOCKID = (1 << 30) | (struct.calcsize("i") << 16) | (ord("E") << 8) | 0xa0


class LinuxDevices:
    """The named devices' event nodes, open; their events as the kernel stamped them."""

    def __init__(self, device_names: list[str]) -> None:
        """Open every event node whose device has one of ``device_names``.

        :param device_names: device names as the kernel gives them (``/sys/class/input/
            event*/device/name``).
        :raises SessionLoggerError: if ``perf_counter`` is not on ``CLOCK_MONOTONIC``, a name
            matches no device, or a node cannot be read.
        """
        implementation = time.get_clock_info("perf_counter").implementation
        if implementation != "clock_gettime(CLOCK_MONOTONIC)":
            raise SessionLoggerError(f"session-logger: perf_counter reads {implementation}, "
                                    "not the kernel's CLOCK_MONOTONIC")
        self._devices = _devices_named(device_names)
        #: file descriptor -> the device's index
        self._fds: dict[int, int] = {}
        for index, device in enumerate(self._devices):
            try:
                fd = os.open(device["path"], os.O_RDONLY | os.O_NONBLOCK)
            except PermissionError:
                self.close()
                raise SessionLoggerError(f"session-logger: {device['path']} ({device['name']}) "
                                        "cannot be read by this user") from None
            fcntl.ioctl(fd, _EVIOCSCLOCKID, struct.pack("i", _CLOCK_MONOTONIC))
            self._fds[fd] = index

    def record(self) -> dict[str, Any]:
        """The reader, the clock of its stamps, and each device by its index."""
        return {"reader": "linux evdev", "clock": "CLOCK_MONOTONIC", "devices": self._devices}

    def read(self, timeout: float) -> list[dict[str, Any]]:
        """The events that arrived, waiting up to ``timeout`` seconds for the first.

        :param timeout: seconds to wait when none is there yet.
        :return: one line per event (``event_type``, ``code``, ``value``, as the kernel gives
            them), and a ``device_gone`` line for a device unplugged, whose node is dropped.
        """
        if not self._fds:
            time.sleep(timeout)
            return []
        ready, _, _ = select.select(list(self._fds), [], [], timeout)
        lines: list[dict[str, Any]] = []
        for fd in ready:
            try:
                data = os.read(fd, _EVENT.size * 64)
            except BlockingIOError:
                continue
            except OSError:
                lines.append({"type": "device_gone", "perf_time": time.perf_counter(),
                              "device": self._fds.pop(fd)})
                os.close(fd)
                continue
            lines += _event_lines(data, self._fds[fd])
        return lines

    def close(self) -> None:
        """Close every node still open."""
        for fd in list(self._fds):
            os.close(fd)
        self._fds.clear()


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


def _devices_named(device_names: list[str]) -> list[dict[str, Any]]:
    """Each event node whose device has one of ``device_names``, with its identity.

    :param device_names: device names as the kernel gives them.
    :return: each node's identity (:func:`_sysfs_device`), in node order.
    :raises SessionLoggerError: if a name matches no device.
    """
    present = [_sysfs_device(node) for node in sorted(glob.glob("/sys/class/input/event*"))]
    missing = sorted(set(device_names) - {device["name"] for device in present})
    if missing:
        raise SessionLoggerError(
            f"session-logger: no input device is named {', '.join(map(repr, missing))}; "
            f"these are: {', '.join(sorted({repr(device['name']) for device in present}))}")
    return [device for device in present if device["name"] in device_names]


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
