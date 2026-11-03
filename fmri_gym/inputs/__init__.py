"""The input devices a participant answers with, one class per kind, each naming its controls.

The keyboard is always on. A rig file lists the other devices it has under
``"inputs"``, by the names in :data:`ALL_INPUT_DEVICE_NAMES`, and a curriculum names the
controls it maps (``"keys"``) as each device's file writes them. What a rig can
press is known before anything is plugged in (:func:`controls_of`), and a run
opens its devices once (:func:`open_inputs`), before its window.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence

import pygame

from .base import QUIT, Edge, InputDevice, InputDevices, InputEvent
from .calibration import CalibrationError, newest
from .gamepad import Gamepad
from .gmpd3 import Gmpd3
from .joystick import Joystick, start_subsystems
from .keyboard import ESCAPE, Keyboard

__all__ = ["ALL_INPUT_DEVICE_NAMES", "ESCAPE", "OPTIONAL_INPUT_DEVICE_NAMES", "QUIT",
           "CalibrationError", "Edge", "InputDevice", "InputDevices", "InputEvent", "Joystick",
           "Keyboard", "controls_of", "detect", "find_device", "is_control", "open_device",
           "open_inputs"]

#: Each kind of input device, by the name a rig file lists it by.
ALL_INPUT_DEVICE_NAMES: dict[str, type[InputDevice]] = {
    cls.name: cls for cls in (Keyboard, Gamepad, Gmpd3)}
#: The devices a rig lists: every kind but the keyboard, which is always on.
OPTIONAL_INPUT_DEVICE_NAMES: dict[str, type[Joystick]] = {
    cls.name: cls for cls in (Gamepad, Gmpd3)}
#: How long SDL is given to announce a device it sees late (macOS: about 0.1 s).
_SDL_DEVICE_ARRIVAL_S = 0.5


def controls_of(names: Iterable[str]) -> frozenset[str]:
    """Every control a rig with the devices ``names`` can have, the keyboard's included."""
    return Keyboard.CONTROLS.union(*(ALL_INPUT_DEVICE_NAMES[n].CONTROLS for n in names))


#: Every control of every kind of device.
_EVERY = controls_of(ALL_INPUT_DEVICE_NAMES)


def is_control(name: str) -> bool:
    """Whether ``name`` is a control of any kind of device."""
    return name in _EVERY


def _class_of(index: int) -> type[Joystick] | None:
    """The kind of device SDL's joystick ``index`` is, or None for none of them."""
    return next((cls for cls in OPTIONAL_INPUT_DEVICE_NAMES.values() if cls.accepts(index)), None)


def _start(listed: int) -> None:
    """Start SDL's controller subsystems, giving a device SDL announces late time to arrive.

    On macOS SDL sees a gamepad only with its display subsystem up (which opens
    no window), and the pad arrives as an event about 0.1 s after that.

    :param listed: how many devices are expected; with none, no time is given.
    """
    start_subsystems()
    if not listed or pygame.joystick.get_count() >= listed:
        return
    pygame.display.init()
    end = time.monotonic() + _SDL_DEVICE_ARRIVAL_S
    while time.monotonic() < end and pygame.joystick.get_count() < listed:
        pygame.event.pump()
        time.sleep(0.05)


def detect() -> list[dict]:
    """The joysticks SDL sees: each one's index, name, kind (or None) and newest calibration.

    :raises CalibrationError: when a device's newest calibration file is malformed.
    """
    _start(1)
    out = []
    for i in range(pygame.joystick.get_count()):
        name, cls = pygame.joystick.Joystick(i).get_name(), _class_of(i)
        found = newest(name)
        out.append({"index": i, "device_name": name, "input": cls and cls.name,
                    "calibration": None if found is None else str(found[0])})
    return out


def find_device(kinds: Iterable[str]) -> tuple[type[Joystick], int] | None:
    """The first joystick SDL sees that is a device of one of ``kinds``: its class and index."""
    classes = [OPTIONAL_INPUT_DEVICE_NAMES[k] for k in kinds]
    _start(1)
    return next(((cls, i) for i in range(pygame.joystick.get_count()) for cls in classes
                 if cls.accepts(i)), None)


def open_device(cls: type[Joystick], index: int) -> Joystick:
    """A device of kind ``cls`` on SDL's joystick ``index``, read through its newest calibration.

    :raises CalibrationError: when that calibration file is malformed.
    """
    return cls(index, newest(pygame.joystick.Joystick(index).get_name()))


def open_inputs(names: Sequence[str], needed: frozenset[str] = frozenset()) -> InputDevices:
    """The keyboard, and a device of each kind ``names`` lists, each on a joystick plugged in.

    :param names: a rig's ``"inputs"``.
    :param needed: the controls the curriculum names; a stick direction among
        them needs its stick calibrated. Empty for a run that calibrates.
    :raises ValueError: for a kind that is not one, listed twice, or not plugged in.
    :raises CalibrationError: for a malformed calibration, or a needed stick without one.
    """
    unknown = [n for n in names if n not in OPTIONAL_INPUT_DEVICE_NAMES]
    if unknown or len(set(names)) < len(names):
        raise ValueError(f'"inputs": {list(names)}: expected distinct names from '
                         f"{sorted(OPTIONAL_INPUT_DEVICE_NAMES)} (the keyboard is always on)")
    _start(len(names))
    seen = [pygame.joystick.Joystick(i).get_name() for i in range(pygame.joystick.get_count())]
    devices: list[InputDevice] = [Keyboard()]
    claimed: set[int] = set()
    for name in names:
        cls = OPTIONAL_INPUT_DEVICE_NAMES[name]
        index = next((i for i in range(len(seen)) if i not in claimed and cls.accepts(i)), None)
        if index is None:
            raise ValueError(f'the rig lists "{name}" under "inputs", but none is plugged in '
                             f"(joysticks SDL sees: {seen or 'none'}): plug it in, or take it "
                             'out of the rig\'s "inputs"')
        claimed.add(index)
        devices.append(open_device(cls, index))
    inputs = InputDevices(devices, [n for i, n in enumerate(seen) if i not in claimed])
    inputs.require(needed)
    return inputs
