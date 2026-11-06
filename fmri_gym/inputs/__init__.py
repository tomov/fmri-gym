"""The input devices a participant answers with, one class per kind, and a run's set of them.

A rig file lists its devices under ``"inputs"``, the keyboard among them, by
the names in :data:`INPUT_DEVICE_CLASSES_BY_NAME`; a curriculum names the
controls it maps as each device's file writes them. What a rig can press is
known before anything is plugged in (:func:`controls_of_input_devices`), and
a run opens its devices once (:func:`open_input_devices`), before its window.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .base import QUIT, Edge, InputDevice, InputDevices, InputEvent
from .calibration import CalibrationError, newest
from .controller import Controller, sdl_joystick_names, start_sdl_joysticks_and_wait_for
from .current_designs_gamepad3 import CurrentDesignsGamepad3
from .keyboard import ESCAPE, Keyboard
from .xbox_controller import XBoxController

__all__ = ["ESCAPE", "INPUT_DEVICE_CLASSES_BY_NAME", "QUIT", "CalibrationError", "Controller",
           "CurrentDesignsGamepad3", "Edge", "InputDevice", "InputDevices", "InputEvent",
           "Keyboard", "XBoxController", "controls_of_input_devices", "detect_controllers",
           "first_controller_class_plugged_in", "is_control_of_any_input_device",
           "open_input_devices"]

#: Each kind of input device, by the name a rig file lists it by.
INPUT_DEVICE_CLASSES_BY_NAME: dict[str, type[InputDevice]] = {
    cls.name: cls for cls in (Keyboard, XBoxController, CurrentDesignsGamepad3)}
#: The kinds that are controllers, by the same names.
CONTROLLER_CLASSES_BY_NAME: dict[str, type[Controller]] = {
    name: cls for name, cls in INPUT_DEVICE_CLASSES_BY_NAME.items()
    if issubclass(cls, Controller)}


def controls_of_input_devices(input_device_names: Iterable[str]) -> frozenset[str]:
    """Every control a rig with the devices ``input_device_names`` can have."""
    return frozenset().union(*(INPUT_DEVICE_CLASSES_BY_NAME[n].CONTROLS
                               for n in input_device_names))


_CONTROLS_OF_ALL_INPUT_DEVICES = controls_of_input_devices(INPUT_DEVICE_CLASSES_BY_NAME)


def is_control_of_any_input_device(name: str) -> bool:
    return name in _CONTROLS_OF_ALL_INPUT_DEVICES


def first_controller_class_plugged_in(
        controller_names: Iterable[str]) -> type[Controller] | None:
    """The first of the kinds ``controller_names`` that a controller plugged in is."""
    return next((cls for name in controller_names
                 if (cls := CONTROLLER_CLASSES_BY_NAME[name]).plugged_in_indices()), None)


def detect_controllers() -> list[dict]:
    """The joysticks SDL sees: each one's index, name, kind (or None) and newest calibration.

    :raises CalibrationError: when a calibration file of a device is malformed.
    """
    start_sdl_joysticks_and_wait_for(1)
    out = []
    for index, device_name in enumerate(sdl_joystick_names()):
        cls = next((c for c in CONTROLLER_CLASSES_BY_NAME.values()
                    if index in c.plugged_in_indices()), None)
        calibration_file = newest(device_name) if cls and cls.AXIS_NAMES else None
        out.append({"index": index, "device_name": device_name, "input": cls and cls.name,
                    "calibration": str(calibration_file[0]) if calibration_file else None})
    return out


def open_input_devices(input_device_names: Sequence[str],
                       required_controls: frozenset[str] = frozenset()) -> InputDevices:
    """A device of each kind ``input_device_names`` lists, each controller on one plugged in.

    :param input_device_names: a rig's ``"inputs"``.
    :param required_controls: the controls the curriculum names; a stick direction
        among them needs its controller calibrated. Empty for a run that calibrates.
    :raises ValueError: for a kind that is not one, listed twice, or not plugged in.
    :raises CalibrationError: for a malformed calibration, or a needed axis without one.
    """
    unknown = [n for n in input_device_names if n not in INPUT_DEVICE_CLASSES_BY_NAME]
    if unknown or len(set(input_device_names)) < len(input_device_names):
        raise ValueError(f'"inputs": {list(input_device_names)}: expected distinct names from '
                         f"{sorted(INPUT_DEVICE_CLASSES_BY_NAME)}")
    classes = [INPUT_DEVICE_CLASSES_BY_NAME[n] for n in input_device_names]
    start_sdl_joysticks_and_wait_for(sum(issubclass(c, Controller) for c in classes))
    devices = [cls.open() for cls in classes]
    opened = {d.device_name for d in devices if isinstance(d, Controller)}
    inputs = InputDevices(devices, [n for n in sdl_joystick_names() if n not in opened])
    inputs.require_calibrated_axes_for(required_controls)
    return inputs
