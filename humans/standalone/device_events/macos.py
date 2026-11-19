"""macOS: the IOKit HID manager's input values, each with the time IOKit stamped it.

IOKit stamps each value in mach absolute time, the clock ``perf_counter`` reads on macOS
(checked when the reader opens). A keyboard-type device -- a trigger box that types keys
included -- is read only with the Input Monitoring permission (System Settings, Privacy &
Security), given to the app the recording is started from; a gamepad needs none.
"""

from __future__ import annotations

import ctypes
import time
from typing import Any

from . import DeviceEventsError

_CF = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
_IOKIT = ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
_SYSTEM = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
_POINTER = ctypes.c_void_p
_UTF8 = 0x08000100                  # kCFStringEncodingUTF8
_SINT32 = 3                         # kCFNumberSInt32Type
_NOT_PERMITTED = -536870174         # kIOReturnNotPermitted, 0xE00002E2, as an IOReturn
_VALUE_CALLBACK = ctypes.CFUNCTYPE(None, _POINTER, ctypes.c_int32, _POINTER, _POINTER)


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
_CFRunLoopGetCurrent = _function(_CF, "CFRunLoopGetCurrent", _POINTER)
_CFRunLoopRunInMode = _function(_CF, "CFRunLoopRunInMode", ctypes.c_int32, _POINTER,
                                ctypes.c_double, ctypes.c_bool)
_DEFAULT_MODE = _POINTER.in_dll(_CF, "kCFRunLoopDefaultMode")

_IOHIDManagerCreate = _function(_IOKIT, "IOHIDManagerCreate", _POINTER, _POINTER,
                                ctypes.c_uint32)
_IOHIDManagerSetDeviceMatching = _function(_IOKIT, "IOHIDManagerSetDeviceMatching", None,
                                           _POINTER, _POINTER)
_IOHIDManagerCopyDevices = _function(_IOKIT, "IOHIDManagerCopyDevices", _POINTER, _POINTER)
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
    """The named HID devices, each open on this thread's run loop;
    their input values as IOKit stamped them."""

    def __init__(self, device_names: list[str]) -> None:
        """Open every HID device whose product name is one of ``device_names``.

        :param device_names: product names as IOKit gives them.
        :raises DeviceEventsError: if ``perf_counter`` is not on mach absolute time, a name
            matches no device, or IOKit will not open them (a keyboard-type device without the
            Input Monitoring permission).
        """
        self._seconds_per_tick = _seconds_per_mach_tick()
        # Every device, kept by name here: IOKit ignores a product name in a matching dictionary.
        self._manager = _IOHIDManagerCreate(None, 0)
        _IOHIDManagerSetDeviceMatching(self._manager, None)
        all_handles = _devices(self._manager)
        handles = [handle for handle in all_handles
                   if _string_property(handle, "Product") in device_names]
        self._devices = [_identity(handle) for handle in handles]
        missing = sorted(set(device_names) - {device["name"] for device in self._devices})
        if missing:
            raise DeviceEventsError(
                f"device_events: no HID device is named {', '.join(map(repr, missing))}; "
                f"these are: {', '.join(sorted(map(repr, _product_names(all_handles))))}")
        for handle in handles:
            _open_device(handle)
        self._handles = handles
        #: the lines of the values IOKit delivered since the last :meth:`read`
        self._pending: list[dict[str, Any]] = []
        # Kept alive here: IOKit holds only the pointer.
        self._callback = _VALUE_CALLBACK(self._on_value)
        self._run_loop = _CFRunLoopGetCurrent()
        for index, handle in enumerate(handles):
            _IOHIDDeviceRegisterInputValueCallback(handle, self._callback, index + 1)
            _IOHIDDeviceScheduleWithRunLoop(handle, self._run_loop, _DEFAULT_MODE)

    def record(self) -> dict[str, Any]:
        """The reader, the clock of its stamps, and each device by its index."""
        return {"reader": "macos iokit", "clock": "mach_absolute_time",
                "devices": self._devices}

    def read(self, timeout: float) -> list[dict[str, Any]]:
        """The values IOKit delivers while this thread's run loop runs for ``timeout`` seconds.

        :param timeout: seconds to run the run loop.
        :return: one line per value (``usage_page``, ``usage``, ``value``).
        """
        _CFRunLoopRunInMode(_DEFAULT_MODE, timeout, False)
        lines, self._pending = self._pending, []
        return lines

    def close(self) -> None:
        """Take the devices off the run loop and close them."""
        for handle in self._handles:
            _IOHIDDeviceUnscheduleFromRunLoop(handle, self._run_loop, _DEFAULT_MODE)
            _IOHIDDeviceClose(handle, 0)

    def _on_value(self, context: int, result: int, sender: int, value: int) -> None:
        """Keep one input value: its device (``context`` is its index + 1), element and stamp."""
        element = _IOHIDValueGetElement(value)
        self._pending.append({
            "type": "device_event",
            "perf_time": _IOHIDValueGetTimeStamp(value) * self._seconds_per_tick,
            "device": context - 1, "usage_page": _IOHIDElementGetUsagePage(element),
            "usage": _IOHIDElementGetUsage(element),
            "value": _IOHIDValueGetIntegerValue(value)})


def _seconds_per_mach_tick() -> float:
    """Seconds per mach absolute time unit, checked to be ``perf_counter``'s clock.

    :raises DeviceEventsError: if ``perf_counter`` is more than a millisecond off mach time.
    """
    timebase = _Timebase()
    _mach_timebase_info(ctypes.byref(timebase))
    seconds_per_tick = timebase.numer / timebase.denom / 1e9
    gap = time.perf_counter() - _mach_absolute_time() * seconds_per_tick
    if abs(gap) > 1e-3:
        raise DeviceEventsError(f"device_events: perf_counter is {gap:+.3f} s off mach "
                                "absolute time, the clock of IOKit's stamps")
    return seconds_per_tick


def _cfstring(text: str) -> int:
    """A CFString of ``text`` (never released: a handful, once per run)."""
    return _CFStringCreateWithCString(None, text.encode(), _UTF8)


def _open_device(device: int) -> None:
    """Open ``device`` for reading.

    :raises DeviceEventsError: if IOKit will not open it (a keyboard-type device without the
        Input Monitoring permission).
    """
    opened = _IOHIDDeviceOpen(device, 0)
    if opened == _NOT_PERMITTED:
        raise DeviceEventsError("device_events: this app lacks the Input Monitoring "
                                "permission, which reading a keyboard-type device takes")
    if opened != 0:
        raise DeviceEventsError(f"device_events: IOKit would not open "
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


def _product_names(devices: list[int]) -> list[str]:
    """The devices' product names, for an error that has to name them."""
    return sorted({name for name in map(lambda device: _string_property(device, "Product"),
                                        devices) if name})


def _identity(device: int) -> dict[str, Any]:
    """A device's product name, vendor and product ids and USB location."""
    return {"name": _string_property(device, "Product"),
            "vendor": _number_property(device, "VendorID"),
            "product": _number_property(device, "ProductID"),
            "location": _number_property(device, "LocationID")}


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
