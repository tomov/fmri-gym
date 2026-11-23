"""Windows: every device's Raw Input, the ones plugged in later too, stamped by ``perf_counter``
(QueryPerformanceCounter) as it arrives, and the application of the foreground window.

Windows gives raw input no time of its own, so a thread that does nothing else, at the highest
priority a thread may have without administrator rights, stamps each report as it receives it:
the stamp is late by the time Windows took to hand it over, and by any stall the recording
shows. Raw input is asked for by kind of device (a HID page and usage): the kinds of the devices
present and :data:`_KINDS_ALWAYS_ASKED`, so a device plugged in later is read when it is of one
of them. A HID device's report is read into buttons and values by the HID parser Windows keeps
for the device, and each one that changed is a line, as IOKit gives them on macOS; a report that
parser refuses is kept whole.
"""

from __future__ import annotations

import ctypes
import functools
import ntpath
import queue
import re
import threading
import time
from types import SimpleNamespace
from typing import Any

from . import FrontApplication, SessionLoggerError

_HANDLE = ctypes.c_void_p
_USHORT, _ULONG, _LONG, _UINT = ctypes.c_uint16, ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32
_RIM_TYPEMOUSE, _RIM_TYPEKEYBOARD, _RIM_TYPEHID = 0, 1, 2
_RID_INPUT = 0x10000003
_RIDI_PREPARSEDDATA, _RIDI_DEVICENAME, _RIDI_DEVICEINFO = 0x20000005, 0x20000007, 0x2000000B
_RIDEV_INPUTSINK, _RIDEV_DEVNOTIFY, _RIDEV_REMOVE = 0x00000100, 0x00002000, 0x00000001
_WM_INPUT_DEVICE_CHANGE, _WM_INPUT, _WM_QUIT = 0x00FE, 0x00FF, 0x0012
_GIDC_ARRIVAL, _GIDC_REMOVAL = 1, 2
_RI_KEY_BREAK = 1
_HWND_MESSAGE = -3
_THREAD_PRIORITY_TIME_CRITICAL = 15
_HIDP_INPUT = 0
_HIDP_STATUS_SUCCESS = 0x00110000
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_FILE_SHARE_READ_WRITE, _OPEN_EXISTING = 3, 3
_INVALID_HANDLE = ctypes.c_void_p(-1).value
#: The kind of device each usage of the generic desktop page (1) is asked for as.
_KIND_OF_TYPE = {_RIM_TYPEMOUSE: (1, 2), _RIM_TYPEKEYBOARD: (1, 6)}
#: Kinds of device asked for even when none is plugged in at the start: mice, joysticks,
#: gamepads, keyboards, multi-axis controllers (generic desktop page) and consumer controls.
_KINDS_ALWAYS_ASKED = {(1, 2), (1, 4), (1, 5), (1, 6), (1, 8), (0x0C, 1)}
_VENDOR_PRODUCT = re.compile(r"VID_([0-9A-Fa-f]{4})&PID_([0-9A-Fa-f]{4})")


class _RawInputHeader(ctypes.Structure):
    _fields_ = [("type", _ULONG), ("size", _ULONG), ("device", _HANDLE),
                ("wparam", ctypes.c_size_t)]


class _RawKeyboard(ctypes.Structure):
    _fields_ = [("make_code", _USHORT), ("flags", _USHORT), ("reserved", _USHORT),
                ("vkey", _USHORT), ("message", _UINT), ("extra_information", _ULONG)]


class _RawMouse(ctypes.Structure):
    # The buttons' union is a ULONG, so it starts on the next 4 bytes.
    _fields_ = [("flags", _USHORT), ("padding", _USHORT), ("button_flags", _USHORT),
                ("button_data", _USHORT), ("raw_buttons", _ULONG), ("last_x", _LONG),
                ("last_y", _LONG), ("extra_information", _ULONG)]


class _RawHid(ctypes.Structure):
    _fields_ = [("report_size", _ULONG), ("report_count", _ULONG)]


class _RawInputDevice(ctypes.Structure):
    _fields_ = [("usage_page", _USHORT), ("usage", _USHORT), ("flags", _ULONG),
                ("window", _HANDLE)]


class _RawInputDeviceList(ctypes.Structure):
    _fields_ = [("device", _HANDLE), ("type", _ULONG)]


class _DeviceInfoHid(ctypes.Structure):
    _fields_ = [("vendor_id", _ULONG), ("product_id", _ULONG), ("version", _ULONG),
                ("usage_page", _USHORT), ("usage", _USHORT)]


class _DeviceInfoOfType(ctypes.Union):
    _fields_ = [("hid", _DeviceInfoHid), ("keyboard", _ULONG * 6), ("mouse", _ULONG * 4)]


class _DeviceInfo(ctypes.Structure):
    _fields_ = [("size", _ULONG), ("type", _ULONG), ("of_type", _DeviceInfoOfType)]


class _HidpCaps(ctypes.Structure):
    _fields_ = [("usage", _USHORT), ("usage_page", _USHORT), ("input_report_length", _USHORT),
                ("output_report_length", _USHORT), ("feature_report_length", _USHORT),
                ("reserved", _USHORT * 17), ("link_collection_nodes", _USHORT),
                ("input_button_caps", _USHORT), ("input_value_caps", _USHORT),
                ("input_data_indices", _USHORT), ("output_button_caps", _USHORT),
                ("output_value_caps", _USHORT), ("output_data_indices", _USHORT),
                ("feature_button_caps", _USHORT), ("feature_value_caps", _USHORT),
                ("feature_data_indices", _USHORT)]


class _HidpButtonCaps(ctypes.Structure):
    # The last 8 USHORTs are a union: a range (UsageMin, UsageMax, ...) or one usage (Usage, ...).
    _fields_ = [("usage_page", _USHORT), ("report_id", ctypes.c_uint8),
                ("is_alias", ctypes.c_uint8), ("bit_field", _USHORT),
                ("link_collection", _USHORT), ("link_usage", _USHORT),
                ("link_usage_page", _USHORT), ("is_range", ctypes.c_uint8),
                ("is_string_range", ctypes.c_uint8), ("is_designator_range", ctypes.c_uint8),
                ("is_absolute", ctypes.c_uint8), ("report_count", _USHORT),
                ("reserved2", _USHORT), ("reserved", _ULONG * 9), ("usages", _USHORT * 8)]


class _HidpValueCaps(ctypes.Structure):
    _fields_ = [("usage_page", _USHORT), ("report_id", ctypes.c_uint8),
                ("is_alias", ctypes.c_uint8), ("bit_field", _USHORT),
                ("link_collection", _USHORT), ("link_usage", _USHORT),
                ("link_usage_page", _USHORT), ("is_range", ctypes.c_uint8),
                ("is_string_range", ctypes.c_uint8), ("is_designator_range", ctypes.c_uint8),
                ("is_absolute", ctypes.c_uint8), ("has_null", ctypes.c_uint8),
                ("reserved", ctypes.c_uint8), ("bit_size", _USHORT), ("report_count", _USHORT),
                ("reserved2", _USHORT * 5), ("units_exponent", _ULONG), ("units", _ULONG),
                ("logical_min", _LONG), ("logical_max", _LONG), ("physical_min", _LONG),
                ("physical_max", _LONG), ("usages", _USHORT * 8)]


class _Message(ctypes.Structure):
    _fields_ = [("window", _HANDLE), ("message", _UINT), ("wparam", ctypes.c_size_t),
                ("lparam", ctypes.c_ssize_t), ("time", _ULONG), ("x", _LONG), ("y", _LONG),
                ("private", _ULONG)]


class _WindowClass(ctypes.Structure):
    _fields_ = [("size", _UINT), ("style", _UINT), ("window_procedure", ctypes.c_void_p),
                ("class_extra", ctypes.c_int), ("window_extra", ctypes.c_int),
                ("instance", _HANDLE), ("icon", _HANDLE), ("cursor", _HANDLE),
                ("background", _HANDLE), ("menu_name", ctypes.c_wchar_p),
                ("class_name", ctypes.c_wchar_p), ("small_icon", _HANDLE)]


class WindowsDevices:
    """Every device's raw input, the ones plugged in later too, received by a thread of its own;
    their reports as it stamped them."""

    def __init__(self) -> None:
        """Find every raw input device, and start receiving their input.

        :raises SessionLoggerError: if Windows will not send this process their input.
        """
        win32 = _win32()
        present = _raw_input_devices(win32)
        #: each device's identity, by its index
        self._devices: list[dict[str, Any]] = []
        #: each device plugged in -> its index
        self._index_of_handle: dict[int, int] = {}
        self._hid_reports: dict[int, HidReports] = {}
        for device in present:
            self._add(win32, device)
        #: (device, make code, flags but the break bit) of each key held down
        self._keys_down: set[tuple[int, int, int]] = set()
        self._lines: queue.Queue[dict[str, Any]] = queue.Queue()
        self._receiver_started = threading.Event()
        self._receiver_failure: str | None = None
        self._receiver_thread_id = 0
        kinds = sorted({(device["usage_page"], device["usage"]) for device in present}
                       | _KINDS_ALWAYS_ASKED)
        self._receiver = threading.Thread(target=self._receive, args=(win32, kinds),
                                          daemon=True)
        self._receiver.start()
        self._receiver_started.wait()
        if self._receiver_failure is not None:
            raise SessionLoggerError(self._receiver_failure)

    def record(self) -> dict[str, Any]:
        """The reader, the clock of its stamps, and each device plugged in now by its index."""
        return {"reader": "windows raw input", "clock": "QueryPerformanceCounter, at arrival",
                "devices": list(self._devices)}

    def read(self, timeout: float) -> list[dict[str, Any]]:
        """The lines received, waiting up to ``timeout`` seconds for the first.

        :param timeout: seconds to wait when none is there yet.
        :return: one line per key, button, value or mouse report (:func:`lines_of_raw_input`),
            a ``device_added`` line for a device plugged in and a ``device_gone`` line for one
            unplugged.
        """
        lines = []
        try:
            lines.append(self._lines.get(timeout=timeout) if timeout > 0
                         else self._lines.get_nowait())
        except queue.Empty:
            return lines
        while True:
            try:
                lines.append(self._lines.get_nowait())
            except queue.Empty:
                return lines

    def close(self) -> None:
        """Stop receiving: the thread ends its message loop and unregisters."""
        _win32().user32.PostThreadMessageW(self._receiver_thread_id, _WM_QUIT, 0, 0)
        self._receiver.join(timeout=2.0)

    def _add(self, win32: SimpleNamespace, device: dict[str, Any]) -> int:
        """Give a device the next index, and its HID parser to a HID device."""
        index = len(self._devices)
        self._devices.append({key: value for key, value in device.items() if key != "handle"})
        self._index_of_handle[device["handle"]] = index
        if device["kind"] == "hid":
            self._hid_reports[device["handle"]] = HidReports(win32, device["handle"])
        return index

    def _index_of(self, win32: SimpleNamespace, handle: int, arrived: float) -> int:
        """A device's index; a device not seen yet is added, with a ``device_added`` line."""
        index = self._index_of_handle.get(handle)
        if index is not None:
            return index
        index = self._add(win32, _raw_input_device(win32, handle))
        self._lines.put({"type": "device_added", "perf_time": arrived, "device": index,
                         **self._devices[index]})
        return index

    def _receive(self, win32: SimpleNamespace, kinds: list[tuple[int, int]]) -> None:
        """The receiving thread: a message-only window that asks for ``kinds``' raw input, and
        its message loop, until a quit message."""
        self._receiver_thread_id = win32.kernel32.GetCurrentThreadId()
        win32.kernel32.SetThreadPriority(win32.kernel32.GetCurrentThread(),
                                         _THREAD_PRIORITY_TIME_CRITICAL)
        window = _message_window(win32)
        requests = (_RawInputDevice * len(kinds))(*[
            _RawInputDevice(page, usage, _RIDEV_INPUTSINK | _RIDEV_DEVNOTIFY, window)
            for page, usage in kinds])
        if not window or not win32.user32.RegisterRawInputDevices(
                requests, len(kinds), ctypes.sizeof(_RawInputDevice)):
            self._receiver_failure = (f"session-logger: Windows would not send this process the "
                                      f"devices' raw input (error {ctypes.get_last_error()})")
            self._receiver_started.set()
            return
        self._receiver_started.set()
        message = _Message()
        while win32.user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
            arrived = time.perf_counter()
            if message.message == _WM_INPUT:
                self._keep_raw_input(win32, message.lparam, arrived)
            elif message.message == _WM_INPUT_DEVICE_CHANGE:
                self._keep_device_change(win32, message.wparam, message.lparam, arrived)
            win32.user32.DispatchMessageW(ctypes.byref(message))
        for request in requests:
            request.flags, request.window = _RIDEV_REMOVE, None
        win32.user32.RegisterRawInputDevices(requests, len(kinds),
                                             ctypes.sizeof(_RawInputDevice))
        win32.user32.DestroyWindow(window)

    def _keep_device_change(self, win32: SimpleNamespace, change: int, handle: int,
                            arrived: float) -> None:
        """A device plugged in (Windows names the ones already there too, which are known) or
        unplugged: a ``device_added`` or ``device_gone`` line."""
        if change == _GIDC_ARRIVAL:
            self._index_of(win32, handle, arrived)
        elif change == _GIDC_REMOVAL and handle in self._index_of_handle:
            self._lines.put({"type": "device_gone", "perf_time": arrived,
                             "device": self._index_of_handle.pop(handle)})
            self._hid_reports.pop(handle, None)

    def _keep_raw_input(self, win32: SimpleNamespace, raw_input: int, arrived: float) -> None:
        """Keep the lines of one raw input message."""
        size = _UINT(0)
        header_size = ctypes.sizeof(_RawInputHeader)
        win32.user32.GetRawInputData(raw_input, _RID_INPUT, None, ctypes.byref(size),
                                     header_size)
        data = ctypes.create_string_buffer(size.value)
        if win32.user32.GetRawInputData(raw_input, _RID_INPUT, data, ctypes.byref(size),
                                        header_size) != size.value:
            return
        handle = _RawInputHeader.from_buffer(data).device
        index = self._index_of(win32, handle, arrived)
        for line in lines_of_raw_input(data.raw, index, arrived, self._hid_reports.get(handle),
                                       self._keys_down):
            self._lines.put(line)


class HidReports:
    """A HID device's input reports read into its buttons and values, with the parser Windows
    keeps for it; what changed since its last report is a line each."""

    def __init__(self, win32: SimpleNamespace, handle: int) -> None:
        """Read the device's report layout from Windows.

        :param win32: the loaded Windows functions.
        :param handle: the device's raw input handle.
        """
        self._win32 = win32
        size = _UINT(0)
        win32.user32.GetRawInputDeviceInfoW(handle, _RIDI_PREPARSEDDATA, None, ctypes.byref(size))
        self._parser = ctypes.create_string_buffer(size.value)
        win32.user32.GetRawInputDeviceInfoW(handle, _RIDI_PREPARSEDDATA, self._parser,
                                            ctypes.byref(size))
        caps = _HidpCaps()
        win32.hid.HidP_GetCaps(self._parser, ctypes.byref(caps))
        #: report id -> the usage pages of its buttons
        self._button_pages: dict[int, set[int]] = {}
        for button in _caps(win32.hid.HidP_GetButtonCaps, _HidpButtonCaps,
                            caps.input_button_caps, self._parser):
            self._button_pages.setdefault(button.report_id, set()).add(button.usage_page)
        #: report id -> its values' (usage page, usage), one value each
        self._values: dict[int, list[tuple[int, int]]] = {}
        for value in _caps(win32.hid.HidP_GetValueCaps, _HidpValueCaps, caps.input_value_caps,
                           self._parser):
            if value.report_count != 1:        # an array of values: not read one by one
                continue
            first, last = ((value.usages[0], value.usages[1]) if value.is_range
                           else (value.usages[0], value.usages[0]))
            self._values.setdefault(value.report_id, []).extend(
                (value.usage_page, usage) for usage in range(first, last + 1))
        self._pressed: dict[tuple[int, int], set[int]] = {}
        self._last_values: dict[tuple[int, int], int] = {}

    def lines(self, report: bytes, device: int, arrived: float) -> list[dict[str, Any]] | None:
        """One line per button pressed or let go and per value changed since the last report
        with the same id; ``None`` when the parser refuses the report."""
        report_id = report[0]
        buffer = ctypes.create_string_buffer(report, len(report))
        lines, pressed_now = [], {}
        for page in sorted(self._button_pages.get(report_id, ())):
            length = _ULONG(self._win32.hid.HidP_MaxUsageListLength(_HIDP_INPUT, page,
                                                                    self._parser))
            usages = (_USHORT * max(1, length.value))()
            if self._win32.hid.HidP_GetUsages(_HIDP_INPUT, page, 0, usages, ctypes.byref(length),
                                              self._parser, buffer, len(report)) \
                    != _HIDP_STATUS_SUCCESS:
                return None
            pressed = pressed_now[(report_id, page)] = set(usages[:length.value])
            before = self._pressed.get((report_id, page), set())
            lines += [_hid_line(arrived, device, page, usage, 1)
                      for usage in sorted(pressed - before)]
            lines += [_hid_line(arrived, device, page, usage, 0)
                      for usage in sorted(before - pressed)]
        self._pressed.update(pressed_now)
        for page, usage in self._values.get(report_id, ()):
            value = _ULONG()
            if self._win32.hid.HidP_GetUsageValue(_HIDP_INPUT, page, 0, usage,
                                                  ctypes.byref(value), self._parser, buffer,
                                                  len(report)) != _HIDP_STATUS_SUCCESS:
                continue
            if self._last_values.get((page, usage)) != value.value:
                self._last_values[(page, usage)] = value.value
                lines.append(_hid_line(arrived, device, page, usage, value.value))
        return lines


def lines_of_raw_input(data: bytes, device: int, arrived: float,
                       hid_reports: HidReports | None,
                       keys_down: set[tuple[int, int, int]]) -> list[dict[str, Any]]:
    """The lines of one raw input: a keyboard's key (its make code, flags and virtual key, and
    ``value`` 1 down, 0 up, or 2 for Windows repeating a key held down, as evdev says), a
    mouse's report as Windows gives it, or a HID device's buttons and values that changed
    (``device_report`` with the report whole when its parser refuses it).

    :param data: the ``RAWINPUT``, its header first.
    :param device: the device's index.
    :param arrived: when it arrived, on ``perf_counter``.
    :param hid_reports: the device's report reader, for a HID device.
    :param keys_down: the keys held down, by device, make code and flags but the break bit;
        kept up to date here.
    :return: its lines.
    """
    header = _RawInputHeader.from_buffer_copy(data)
    body = ctypes.sizeof(_RawInputHeader)
    line = {"type": "device_event", "perf_time": arrived, "device": device}
    if header.type == _RIM_TYPEKEYBOARD:
        key = _RawKeyboard.from_buffer_copy(data, body)
        held = (device, key.make_code, key.flags & ~_RI_KEY_BREAK)
        if key.flags & _RI_KEY_BREAK:
            keys_down.discard(held)
            value = 0
        else:
            value = 2 if held in keys_down else 1
            keys_down.add(held)
        return [{**line, "make_code": key.make_code, "flags": key.flags, "vkey": key.vkey,
                 "value": value}]
    if header.type == _RIM_TYPEMOUSE:
        mouse = _RawMouse.from_buffer_copy(data, body)
        return [{**line, "mouse_flags": mouse.flags, "button_flags": mouse.button_flags,
                 "button_data": mouse.button_data, "last_x": mouse.last_x,
                 "last_y": mouse.last_y}]
    hid = _RawHid.from_buffer_copy(data, body)
    first = body + ctypes.sizeof(_RawHid)
    lines = []
    for index in range(hid.report_count):
        report = data[first + index * hid.report_size:first + (index + 1) * hid.report_size]
        read = None if hid_reports is None else hid_reports.lines(report, device, arrived)
        lines += read if read is not None else [{"type": "device_report", "perf_time": arrived,
                                                 "device": device, "report": report.hex()}]
    return lines


def front_application() -> FrontApplication | None:
    """The application of the foreground window: its process's executable and pid; ``None``
    when no window is in the foreground."""
    win32 = _win32()
    window = win32.user32.GetForegroundWindow()
    if not window:
        return None
    pid = _ULONG()
    win32.user32.GetWindowThreadProcessId(window, ctypes.byref(pid))
    process = win32.kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    name = ""
    if process:
        path, size = ctypes.create_unicode_buffer(1024), _ULONG(1024)
        if win32.kernel32.QueryFullProcessImageNameW(process, 0, path, ctypes.byref(size)):
            name = ntpath.basename(path.value)
        win32.kernel32.CloseHandle(process)
    return FrontApplication(name, pid.value)


def _hid_line(arrived: float, device: int, usage_page: int, usage: int,
              value: int) -> dict[str, Any]:
    """One HID button's or value's line."""
    return {"type": "device_event", "perf_time": arrived, "device": device,
            "usage_page": usage_page, "usage": usage, "value": value}


def _caps(get_caps: Any, caps_type: type, count: int, parser: Any) -> list[Any]:
    """A HID device's input button or value capabilities."""
    if count == 0:
        return []
    caps, length = (caps_type * count)(), _USHORT(count)
    if get_caps(_HIDP_INPUT, caps, ctypes.byref(length), parser) != _HIDP_STATUS_SUCCESS:
        return []
    return list(caps[:length.value])


def _raw_input_devices(win32: SimpleNamespace) -> list[dict[str, Any]]:
    """Every raw input device (:func:`_raw_input_device`)."""
    count = _UINT(0)
    size = ctypes.sizeof(_RawInputDeviceList)
    win32.user32.GetRawInputDeviceList(None, ctypes.byref(count), size)
    listed = (_RawInputDeviceList * count.value)()
    found = win32.user32.GetRawInputDeviceList(listed, ctypes.byref(count), size)
    return [_raw_input_device(win32, entry.device)
            for entry in listed[:found if found != 0xFFFFFFFF else 0]]


def _raw_input_device(win32: SimpleNamespace, handle: int) -> dict[str, Any]:
    """A raw input device: its handle, kind, product string, path, vendor and product ids,
    and the HID page and usage it is asked for by."""
    info = _DeviceInfo(size=ctypes.sizeof(_DeviceInfo))
    info_size = _UINT(ctypes.sizeof(_DeviceInfo))
    win32.user32.GetRawInputDeviceInfoW(handle, _RIDI_DEVICEINFO, ctypes.byref(info),
                                        ctypes.byref(info_size))
    path = _device_path(win32, handle)
    ids = _VENDOR_PRODUCT.search(path)
    usage_page, usage = (_KIND_OF_TYPE[info.type] if info.type in _KIND_OF_TYPE else
                         (info.of_type.hid.usage_page, info.of_type.hid.usage))
    return {"handle": handle, "name": _product_string(win32, path),
            "kind": {_RIM_TYPEMOUSE: "mouse", _RIM_TYPEKEYBOARD: "keyboard"}.get(info.type, "hid"),
            "vendor": int(ids[1], 16) if ids else None,
            "product": int(ids[2], 16) if ids else None,
            "usage_page": usage_page, "usage": usage, "path": path}


def _device_path(win32: SimpleNamespace, handle: int) -> str:
    """A raw input device's path, which names its USB vendor and product."""
    size = _UINT(0)
    win32.user32.GetRawInputDeviceInfoW(handle, _RIDI_DEVICENAME, None, ctypes.byref(size))
    path = ctypes.create_unicode_buffer(size.value + 1)
    win32.user32.GetRawInputDeviceInfoW(handle, _RIDI_DEVICENAME, path, ctypes.byref(size))
    return path.value


def _product_string(win32: SimpleNamespace, path: str) -> str:
    """A device's USB product string, opened for no access (which Windows allows even for a
    keyboard it keeps for itself); ``""`` when it has none."""
    device = win32.kernel32.CreateFileW(path, 0, _FILE_SHARE_READ_WRITE, None, _OPEN_EXISTING, 0,
                                        None)
    if device in (None, _INVALID_HANDLE):
        return ""
    product = ctypes.create_unicode_buffer(127)
    has_product = win32.hid.HidD_GetProductString(device, product, ctypes.sizeof(product))
    win32.kernel32.CloseHandle(device)
    return product.value.strip() if has_product else ""


def _message_window(win32: SimpleNamespace) -> int:
    """A message-only window of this thread, whose messages Windows' default procedure answers;
    ``0`` when it cannot be made."""
    instance = win32.kernel32.GetModuleHandleW(None)
    window_class = _WindowClass(
        size=ctypes.sizeof(_WindowClass),
        window_procedure=ctypes.cast(win32.user32.DefWindowProcW, ctypes.c_void_p).value,
        instance=instance, class_name="fmri_gym_session_logger")
    win32.user32.RegisterClassExW(ctypes.byref(window_class))
    return win32.user32.CreateWindowExW(0, "fmri_gym_session_logger", "session-logger", 0, 0,
                                        0, 0, 0, _HWND_MESSAGE, None, instance, None) or 0


@functools.cache
def _win32() -> SimpleNamespace:
    """user32's, kernel32's and hid's functions, typed: loaded once, on Windows only."""
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    hid = ctypes.WinDLL("hid")
    pointer, uint_pointer = ctypes.c_void_p, ctypes.POINTER(_UINT)
    for library, name, result, arguments in (
            (user32, "GetRawInputDeviceList", _UINT, [pointer, uint_pointer, _UINT]),
            (user32, "GetRawInputDeviceInfoW", _UINT, [_HANDLE, _UINT, pointer, uint_pointer]),
            (user32, "GetRawInputData", _UINT, [_HANDLE, _UINT, pointer, uint_pointer, _UINT]),
            (user32, "RegisterRawInputDevices", ctypes.c_int, [pointer, _UINT, _UINT]),
            (user32, "RegisterClassExW", _USHORT, [pointer]),
            (user32, "CreateWindowExW", _HANDLE, [_ULONG, ctypes.c_wchar_p, ctypes.c_wchar_p,
                                                  _ULONG, ctypes.c_int, ctypes.c_int,
                                                  ctypes.c_int, ctypes.c_int, ctypes.c_ssize_t,
                                                  _HANDLE, _HANDLE, pointer]),
            (user32, "DestroyWindow", ctypes.c_int, [_HANDLE]),
            (user32, "GetMessageW", ctypes.c_int, [pointer, _HANDLE, _UINT, _UINT]),
            (user32, "DispatchMessageW", ctypes.c_ssize_t, [pointer]),
            (user32, "PostThreadMessageW", ctypes.c_int, [_ULONG, _UINT, ctypes.c_size_t,
                                                          ctypes.c_ssize_t]),
            (user32, "GetForegroundWindow", _HANDLE, []),
            (user32, "GetWindowThreadProcessId", _ULONG, [_HANDLE, ctypes.POINTER(_ULONG)]),
            (kernel32, "GetCurrentThreadId", _ULONG, []),
            (kernel32, "GetCurrentThread", _HANDLE, []),
            (kernel32, "SetThreadPriority", ctypes.c_int, [_HANDLE, ctypes.c_int]),
            (kernel32, "GetModuleHandleW", _HANDLE, [ctypes.c_wchar_p]),
            (kernel32, "CreateFileW", _HANDLE, [ctypes.c_wchar_p, _ULONG, _ULONG, pointer,
                                                _ULONG, _ULONG, _HANDLE]),
            (kernel32, "CloseHandle", ctypes.c_int, [_HANDLE]),
            (kernel32, "OpenProcess", _HANDLE, [_ULONG, ctypes.c_int, _ULONG]),
            (kernel32, "QueryFullProcessImageNameW", ctypes.c_int,
             [_HANDLE, _ULONG, ctypes.c_wchar_p, ctypes.POINTER(_ULONG)]),
            (hid, "HidD_GetProductString", ctypes.c_uint8, [_HANDLE, pointer, _ULONG]),
            (hid, "HidP_GetCaps", _LONG, [pointer, pointer]),
            (hid, "HidP_GetButtonCaps", _LONG, [ctypes.c_int, pointer,
                                                ctypes.POINTER(_USHORT), pointer]),
            (hid, "HidP_GetValueCaps", _LONG, [ctypes.c_int, pointer, ctypes.POINTER(_USHORT),
                                               pointer]),
            (hid, "HidP_MaxUsageListLength", _ULONG, [ctypes.c_int, _USHORT, pointer]),
            (hid, "HidP_GetUsages", _LONG, [ctypes.c_int, _USHORT, _USHORT, pointer,
                                            ctypes.POINTER(_ULONG), pointer, pointer, _ULONG]),
            (hid, "HidP_GetUsageValue", _LONG, [ctypes.c_int, _USHORT, _USHORT, _USHORT,
                                                ctypes.POINTER(_ULONG), pointer, pointer,
                                                _ULONG])):
        function = getattr(library, name)
        function.restype, function.argtypes = result, arguments
    return SimpleNamespace(user32=user32, kernel32=kernel32, hid=hid)
