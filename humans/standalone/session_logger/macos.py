"""macOS: every HID device's input values, each with the time IOKit stamped it, and the
application LaunchServices puts in front.

IOKit stamps each value in mach absolute time, the clock ``perf_counter`` reads on macOS
(checked when the reader opens). A keyboard-type device -- the Mac's own keyboard, a trigger box
that types keys -- is read only with the Input Monitoring permission (System Settings, Privacy &
Security), given to the app the recording is started from.
"""

from __future__ import annotations

import ctypes
import re
import subprocess
import sys
import time
from typing import Any

from . import FrontApplication, SessionLoggerError

_CF = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
_IOKIT = ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
_SYSTEM = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
_POINTER = ctypes.c_void_p
_UTF8 = 0x08000100                  # kCFStringEncodingUTF8
_SINT32 = 3                         # kCFNumberSInt32Type
_NOT_PERMITTED = -536870174         # kIOReturnNotPermitted, 0xE00002E2, as an IOReturn
_VALUE_CALLBACK = ctypes.CFUNCTYPE(None, _POINTER, ctypes.c_int32, _POINTER, _POINTER)
_REMOVAL_CALLBACK = ctypes.CFUNCTYPE(None, _POINTER, ctypes.c_int32, _POINTER)
_MATCHING_CALLBACK = ctypes.CFUNCTYPE(None, _POINTER, ctypes.c_int32, _POINTER, _POINTER)


class _Timebase(ctypes.Structure):
    _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]


def _function(library: ctypes.CDLL, name: str, result: Any, *arguments: Any) -> Any:
    """``name`` in ``library``, typed."""
    function = getattr(library, name)
    function.restype = result
    function.argtypes = list(arguments)
    return function


_CFStringCreateWithCString = _function(_CF, "CFStringCreateWithCString", _POINTER, _POINTER,
                                       ctypes.c_char_p, ctypes.c_uint32)
_CFStringGetCString = _function(_CF, "CFStringGetCString", ctypes.c_bool, _POINTER,
                                ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32)
_CFNumberGetValue = _function(_CF, "CFNumberGetValue", ctypes.c_bool, _POINTER, ctypes.c_long,
                              _POINTER)
_CFGetTypeID = _function(_CF, "CFGetTypeID", ctypes.c_ulong, _POINTER)
_CFStringGetTypeID = _function(_CF, "CFStringGetTypeID", ctypes.c_ulong)
_CFSetGetCount = _function(_CF, "CFSetGetCount", ctypes.c_long, _POINTER)
_CFSetGetValues = _function(_CF, "CFSetGetValues", None, _POINTER, ctypes.POINTER(_POINTER))
_CFRetain = _function(_CF, "CFRetain", _POINTER, _POINTER)
_CFRunLoopGetCurrent = _function(_CF, "CFRunLoopGetCurrent", _POINTER)
_CFRunLoopRunInMode = _function(_CF, "CFRunLoopRunInMode", ctypes.c_int32, _POINTER,
                                ctypes.c_double, ctypes.c_bool)
_DEFAULT_MODE = _POINTER.in_dll(_CF, "kCFRunLoopDefaultMode")

_IOHIDManagerCreate = _function(_IOKIT, "IOHIDManagerCreate", _POINTER, _POINTER,
                                ctypes.c_uint32)
_IOHIDManagerSetDeviceMatching = _function(_IOKIT, "IOHIDManagerSetDeviceMatching", None,
                                           _POINTER, _POINTER)
_IOHIDManagerCopyDevices = _function(_IOKIT, "IOHIDManagerCopyDevices", _POINTER, _POINTER)
_IOHIDManagerRegisterDeviceMatchingCallback = _function(
    _IOKIT, "IOHIDManagerRegisterDeviceMatchingCallback", None, _POINTER, _MATCHING_CALLBACK,
    _POINTER)
_IOHIDManagerScheduleWithRunLoop = _function(_IOKIT, "IOHIDManagerScheduleWithRunLoop", None,
                                             _POINTER, _POINTER, _POINTER)
_IOHIDDeviceOpen = _function(_IOKIT, "IOHIDDeviceOpen", ctypes.c_int32, _POINTER,
                             ctypes.c_uint32)
_IOHIDDeviceClose = _function(_IOKIT, "IOHIDDeviceClose", ctypes.c_int32, _POINTER,
                              ctypes.c_uint32)
_IOHIDDeviceScheduleWithRunLoop = _function(_IOKIT, "IOHIDDeviceScheduleWithRunLoop", None,
                                            _POINTER, _POINTER, _POINTER)
_IOHIDDeviceUnscheduleFromRunLoop = _function(_IOKIT, "IOHIDDeviceUnscheduleFromRunLoop",
                                              None, _POINTER, _POINTER, _POINTER)
_IOHIDDeviceGetProperty = _function(_IOKIT, "IOHIDDeviceGetProperty", _POINTER, _POINTER,
                                    _POINTER)
_IOHIDDeviceRegisterInputValueCallback = _function(
    _IOKIT, "IOHIDDeviceRegisterInputValueCallback", None, _POINTER, _VALUE_CALLBACK, _POINTER)
_IOHIDDeviceRegisterRemovalCallback = _function(
    _IOKIT, "IOHIDDeviceRegisterRemovalCallback", None, _POINTER, _REMOVAL_CALLBACK, _POINTER)
_IOHIDValueGetElement = _function(_IOKIT, "IOHIDValueGetElement", _POINTER, _POINTER)
_IOHIDValueGetIntegerValue = _function(_IOKIT, "IOHIDValueGetIntegerValue", ctypes.c_long,
                                       _POINTER)
_IOHIDValueGetTimeStamp = _function(_IOKIT, "IOHIDValueGetTimeStamp", ctypes.c_uint64,
                                    _POINTER)
_IOHIDElementGetUsagePage = _function(_IOKIT, "IOHIDElementGetUsagePage", ctypes.c_uint32,
                                      _POINTER)
_IOHIDElementGetUsage = _function(_IOKIT, "IOHIDElementGetUsage", ctypes.c_uint32, _POINTER)
_mach_absolute_time = _function(_SYSTEM, "mach_absolute_time", ctypes.c_uint64)
_mach_timebase_info = _function(_SYSTEM, "mach_timebase_info", ctypes.c_int,
                                ctypes.POINTER(_Timebase))


class MacDevices:
    """Every HID device, the ones plugged in later too, each open on this thread's run loop;
    their input values as IOKit stamped them."""

    def __init__(self) -> None:
        """Open every HID device, and watch for more.

        :raises SessionLoggerError: if ``perf_counter`` is not on mach absolute time, or IOKit
            will not open a device (a keyboard-type device without the Input Monitoring
            permission).
        """
        self._seconds_per_tick = _seconds_per_mach_tick()
        self._manager = _IOHIDManagerCreate(None, 0)
        _IOHIDManagerSetDeviceMatching(self._manager, None)        # every device
        self._run_loop = _CFRunLoopGetCurrent()
        #: each device's identity, by its index
        self._devices: list[dict[str, Any]] = []
        #: each device ever opened -> its index; kept retained, so no new device takes its address
        self._index_of_handle: dict[int, int] = {}
        #: the devices still plugged in, by index
        self._open_handles: dict[int, int] = {}
        #: the lines of what IOKit delivered since the last :meth:`read`
        self._pending: list[dict[str, Any]] = []
        # Kept alive here: IOKit holds only the pointers.
        self._callback = _VALUE_CALLBACK(self._on_value)
        self._removal_callback = _REMOVAL_CALLBACK(self._on_removal)
        self._matching_callback = _MATCHING_CALLBACK(self._on_matching)
        try:
            for handle in _devices(self._manager):
                self._open(handle)
        except SessionLoggerError:
            self.close()
            raise
        _IOHIDManagerRegisterDeviceMatchingCallback(self._manager, self._matching_callback, None)
        _IOHIDManagerScheduleWithRunLoop(self._manager, self._run_loop, _DEFAULT_MODE)

    def record(self) -> dict[str, Any]:
        """The reader, the clock of its stamps, and each device open now by its index."""
        return {"reader": "macos iokit", "clock": "mach_absolute_time",
                "devices": list(self._devices)}

    def read(self, timeout: float) -> list[dict[str, Any]]:
        """What IOKit delivers while this thread's run loop runs for ``timeout`` seconds.

        :param timeout: seconds to run the run loop.
        :return: one line per value (``usage_page``, ``usage``, ``value``), a ``device_added``
            line for a device plugged in and a ``device_gone`` line for one unplugged.
        """
        _CFRunLoopRunInMode(_DEFAULT_MODE, timeout, False)
        lines, self._pending = self._pending, []
        return lines

    def close(self) -> None:
        """Take the devices still plugged in off the run loop and close them."""
        for handle in self._open_handles.values():
            _IOHIDDeviceUnscheduleFromRunLoop(handle, self._run_loop, _DEFAULT_MODE)
            _IOHIDDeviceClose(handle, 0)
        self._open_handles.clear()

    def _open(self, handle: int) -> int:
        """Open a device, read its values from now on, and give it the next index.

        :raises SessionLoggerError: if IOKit will not open it.
        """
        _open_device(handle)
        index = len(self._devices)
        _CFRetain(handle)
        self._devices.append(_identity(handle))
        self._index_of_handle[handle] = index
        self._open_handles[index] = handle
        _IOHIDDeviceRegisterInputValueCallback(handle, self._callback, index + 1)
        _IOHIDDeviceRegisterRemovalCallback(handle, self._removal_callback, index + 1)
        _IOHIDDeviceScheduleWithRunLoop(handle, self._run_loop, _DEFAULT_MODE)
        return index

    def _on_value(self, context: int, result: int, sender: int, value: int) -> None:
        """Keep one input value: its device (``context`` is its index + 1), element and stamp."""
        element = _IOHIDValueGetElement(value)
        self._pending.append({
            "type": "device_event",
            "perf_time": _IOHIDValueGetTimeStamp(value) * self._seconds_per_tick,
            "device": context - 1, "usage_page": _IOHIDElementGetUsagePage(element),
            "usage": _IOHIDElementGetUsage(element),
            "value": _IOHIDValueGetIntegerValue(value)})

    def _on_removal(self, context: int, result: int, sender: int) -> None:
        """Keep a ``device_gone`` line for a device unplugged (``context`` is its index + 1),
        stamped when IOKit said so, and close it."""
        handle = self._open_handles.pop(context - 1, None)
        if handle is not None:
            _IOHIDDeviceUnscheduleFromRunLoop(handle, self._run_loop, _DEFAULT_MODE)
            _IOHIDDeviceClose(handle, 0)
        self._pending.append({"type": "device_gone", "perf_time": time.perf_counter(),
                              "device": context - 1})

    def _on_matching(self, context: int, result: int, sender: int, device: int) -> None:
        """Open a device IOKit has just found, unless it is one already open, and keep a
        ``device_added`` line with its identity, stamped when IOKit said so; one IOKit will not
        open is a line too, with why, and the console is told."""
        if device in self._index_of_handle:
            return
        found = time.perf_counter()
        try:
            index = self._open(device)
        except SessionLoggerError as error:
            index = len(self._devices)
            _CFRetain(device)
            self._devices.append({**_identity(device), "error": str(error)})
            self._index_of_handle[device] = index
            print(f"{error} (plugged in during the recording, as device {index})",
                  file=sys.stderr, flush=True)
        self._pending.append({"type": "device_added", "perf_time": found, "device": index,
                              **self._devices[index]})


#: The last application :func:`front_application` named, by its LaunchServices ASN.
_last_front_application: tuple[str, FrontApplication | None] = ("", None)


def front_application() -> FrontApplication | None:
    """The application LaunchServices puts in front, as ``lsappinfo`` says; ``None`` when it
    names none."""
    global _last_front_application
    asn = _lsappinfo("front").strip()
    if asn == _last_front_application[0]:
        return _last_front_application[1]
    info = _lsappinfo("info", "-only", "name,pid", asn) if asn else ""
    name, pid = re.search(r'"LSDisplayName"="(.*)"', info), re.search(r'"pid"=(\d+)', info)
    front = FrontApplication(name[1], int(pid[1])) if name and pid else None
    _last_front_application = (asn, front)
    return front


def _lsappinfo(*arguments: str) -> str:
    """What ``lsappinfo`` prints; nothing when it cannot run."""
    try:
        return subprocess.run(["lsappinfo", *arguments], capture_output=True, text=True,
                              timeout=1).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""

def _seconds_per_mach_tick() -> float:
    """Seconds per mach absolute time unit, checked to be ``perf_counter``'s clock.

    :raises SessionLoggerError: if ``perf_counter`` is more than a millisecond off mach time.
    """
    timebase = _Timebase()
    _mach_timebase_info(ctypes.byref(timebase))
    seconds_per_tick = timebase.numer / timebase.denom / 1e9
    gap = time.perf_counter() - _mach_absolute_time() * seconds_per_tick
    if abs(gap) > 1e-3:
        raise SessionLoggerError(f"session-logger: perf_counter is {gap:+.3f} s off mach "
                                "absolute time, the clock of IOKit's stamps")
    return seconds_per_tick


def _cfstring(text: str) -> int:
    """A CFString of ``text`` (never released: a handful, once per run)."""
    return _CFStringCreateWithCString(None, text.encode(), _UTF8)


def _open_device(device: int) -> None:
    """Open ``device`` for reading.

    :raises SessionLoggerError: if IOKit will not open it (a keyboard-type device without the
        Input Monitoring permission).
    """
    opened = _IOHIDDeviceOpen(device, 0)
    if opened == _NOT_PERMITTED:
        raise SessionLoggerError("session-logger: this app lacks the Input Monitoring "
                                "permission, which reading a keyboard-type device takes")
    if opened != 0:
        raise SessionLoggerError(f"session-logger: IOKit would not open "
                                f"{_string_property(device, 'Product')!r} (IOReturn {opened:#x})")


def _devices(manager: int) -> list[int]:
    """The devices ``manager`` matches."""
    devices = _IOHIDManagerCopyDevices(manager)
    if not devices:
        return []
    count = _CFSetGetCount(devices)
    handles = (_POINTER * count)()
    _CFSetGetValues(devices, handles)
    return list(handles)


def _identity(device: int) -> dict[str, Any]:
    """A device's product name, vendor and product ids, USB location, and the HID page and
    usage of what it is (one physical device can be several, each a device here)."""
    return {"name": _string_property(device, "Product"),
            "vendor": _number_property(device, "VendorID"),
            "product": _number_property(device, "ProductID"),
            "location": _number_property(device, "LocationID"),
            "usage_page": _number_property(device, "PrimaryUsagePage"),
            "usage": _number_property(device, "PrimaryUsage")}


def _string_property(device: int, key: str) -> str:
    """A device's string property, ``""`` when it has none."""
    value = _IOHIDDeviceGetProperty(device, _cfstring(key))
    if not value or _CFGetTypeID(value) != _CFStringGetTypeID():
        return ""
    buffer = ctypes.create_string_buffer(512)
    return buffer.value.decode() if _CFStringGetCString(value, buffer, 512, _UTF8) else ""


def _number_property(device: int, key: str) -> int | None:
    """A device's integer property, ``None`` when it has none."""
    value = _IOHIDDeviceGetProperty(device, _cfstring(key))
    number = ctypes.c_int32()
    if not value or not _CFNumberGetValue(value, _SINT32, ctypes.byref(number)):
        return None
    return number.value
