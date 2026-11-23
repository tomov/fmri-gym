"""What every input device is: named controls, their input events, and the devices as one.

Every key, button and stick direction a participant can touch has one name,
unique across devices: its value in
:class:`~humans.inputs.control_names.ControlNames`. A device lists its names at
the top of its own file, and its :meth:`InputDevice.handle` turns the pygame
events that are its own into input events (:class:`InputEvent`); it never reads
pygame's queue itself. As in a game engine, a control is read three ways: an input
event the frame it happens, held state (:meth:`InputDevice.held`) at the tick, and
its value at the tick (:meth:`InputDevice.values`). Several devices together are one :class:`InputDevices`, which is itself a device.
"""

from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from functools import reduce
from enum import Enum
from typing import Any, ClassVar, NamedTuple

import pygame

#: The name the run's inputs give a request to stop the run: the window's close button, or
#: :data:`QUIT_KEYS`.
QUIT = "QUIT"
#: The keys that stop a run, held together, as the experimenter is told them: ESCAPE with
#: CTRL held, so that no single stray press stops it.
QUIT_KEYS = "CTRL+ESCAPE"


class Edge(Enum):
    """Which way a control went: pressed down, or let up."""

    DOWN = "down"
    UP = "up"


class InputEvent(NamedTuple):
    """A control that went down or up: its name, and which way."""

    name: str
    edge: Edge


class InputDevice(ABC):
    """One input device: its controls, and which of them it holds.

    A subclass names itself (:attr:`name`), lists its controls (:attr:`CONTROLS`)
    and presses and releases them with :meth:`_press` and :meth:`_release` from its
    :meth:`handle`.
    """

    #: the name the kind of device goes by
    name: ClassVar[str]
    #: every control the class has, known before any device is opened
    CONTROLS: ClassVar[frozenset[str]]

    def __init__(self) -> None:
        self._held: set[str] = set()

    @classmethod
    @abstractmethod
    def open(cls) -> InputDevice:
        """The device, ready to read.

        :raises ValueError: when it is not plugged in.
        """

    @abstractmethod
    def handle(self, events: list[pygame.event.Event], now: float) -> list[InputEvent]:
        """The input events ``events`` make on this device, in order; other devices' it ignores.

        :param now: ``perf_counter`` when the events were taken off the queue.
        """

    def controls(self) -> frozenset[str]:
        """Every control this device has."""
        return self.CONTROLS

    def held(self) -> frozenset[str]:
        """The controls down as of the last :meth:`handle`."""
        return frozenset(self._held)

    def values(self) -> dict[str, float]:
        """Each control with a range of values (a stick, a trigger), by its value now.

        :return: the value of each such control not at rest; a control left out is at rest,
            or has no range (a key or button, whose value is whether it is held).
        """
        return {}

    def require_calibrated_axes_for(self, required_controls: frozenset[str]) -> None:
        """Refuse to go on when a stick direction in ``required_controls`` has no calibration.

        A device without stick axes has nothing to refuse.

        :raises CalibrationError: naming the device, and how to calibrate it.
        """

    def status(self) -> str:
        """One line naming the device and how it is read."""
        return self.name

    def record(self) -> dict[str, Any]:
        """The device, and everything that sets how it is read."""
        return {"input": self.name}

    def close(self) -> None:
        """Let go of the device."""

    def _press(self, name: str) -> list[InputEvent]:
        """Press ``name``: its down input event, or nothing when it is already down."""
        if name in self._held:
            return []
        self._held.add(name)
        return [InputEvent(name, Edge.DOWN)]

    def _release(self, name: str) -> list[InputEvent]:
        """Release ``name``: its up input event, or nothing when it is already up."""
        if name not in self._held:
            return []
        self._held.discard(name)
        return [InputEvent(name, Edge.UP)]


class InputDevices(InputDevice):
    """Several input devices read as one: a run's keyboard and controllers together.

    Its input events are each device's, in the devices' order, after the window's
    close button and each press of :data:`QUIT_KEYS` as :data:`QUIT`.
    """

    name = "inputs"
    CONTROLS = frozenset()

    def __init__(self, devices: list[InputDevice], ignored_joysticks: list[str] = ()) -> None:
        """Read ``devices`` together.

        :param ignored_joysticks: joysticks plugged in that no device was opened on, by
            name, for :meth:`status`.
        """
        super().__init__()
        self.devices = tuple(devices)
        self.ignored_joysticks = tuple(ignored_joysticks)

    @classmethod
    def open(cls) -> InputDevices:
        """Nothing: the devices are opened one by one (:func:`humans.inputs.open_input_devices`)."""
        return cls([])

    def handle(self, events: list[pygame.event.Event], now: float) -> list[InputEvent]:
        out = [InputEvent(QUIT, Edge.DOWN) for e in events if is_quit_request(e)]
        for device in self.devices:
            out += device.handle(events, now)
        return out

    def controls(self) -> frozenset[str]:
        return frozenset().union(*(d.controls() for d in self.devices))

    def held(self) -> frozenset[str]:
        return frozenset().union(*(d.held() for d in self.devices))

    def values(self) -> dict[str, float]:
        return reduce(operator.or_, (d.values() for d in self.devices), {})

    def require_calibrated_axes_for(self, required_controls: frozenset[str]) -> None:
        for device in self.devices:
            device.require_calibrated_axes_for(required_controls)

    def status(self) -> str:
        """Each device's line, and the joysticks ignored."""
        lines = [d.status() for d in self.devices]
        lines += [f"IGNORED {name!r}: not among the devices opened"
                  for name in self.ignored_joysticks]
        return "; ".join(lines)

    def record(self) -> dict[str, Any]:
        """Each device's record, and the joysticks ignored."""
        return {"devices": [d.record() for d in self.devices],
                "ignored_joysticks": list(self.ignored_joysticks)}

    def close(self) -> None:
        for device in self.devices:
            device.close()


def is_quit_request(event: pygame.event.Event) -> bool:
    """Whether ``event`` is the window's close button, or ESCAPE pressed with CTRL held."""
    return event.type == pygame.QUIT or (
        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
        and bool(getattr(event, "mod", 0) & pygame.KMOD_CTRL))
