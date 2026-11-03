"""The seam between the input devices and the run: named controls, their events, the devices.

Every button, key, axis and stick direction a participant can touch has one
name, unique across devices, and a curriculum names it as it is written here
(``"SPACE"``, ``"GMPD3_A"``, ``"CONTROLLER_LEFTSTICK_UP"``). A device class
lists its names at the top of its own file and turns the pygame events that
are its own into :class:`InputEvent` edges; :class:`InputDevices` is a run's
devices together, itself a device.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import ClassVar, NamedTuple

import pygame

#: The name :class:`InputDevices` gives the window's close button.
QUIT = "QUIT"


class Edge(Enum):
    """Which way a control went: pressed down, or let up."""

    DOWN = "down"
    UP = "up"


class InputEvent(NamedTuple):
    """One edge of a control."""

    name: str
    edge: Edge


class InputDevice(ABC):
    """One input device: the names of its controls, and what it held at its last events."""

    #: the name a rig file lists it by (``"inputs"``)
    name: ClassVar[str]
    #: every control the class can have, known before any device is opened
    CONTROLS: ClassVar[frozenset[str]]

    def controls(self) -> frozenset[str]:
        """The controls this device has: :attr:`CONTROLS`, or fewer."""
        return self.CONTROLS

    def devices(self) -> tuple[InputDevice, ...]:
        """The single devices this one is: itself."""
        return (self,)

    def poll(self) -> list[InputEvent]:
        """Take every event off pygame's queue and :meth:`handle` it.

        A queue left to fill drops new events, key presses included, so
        nothing is left there.
        """
        events = pygame.event.get()
        return self.handle(events, time.perf_counter())

    @abstractmethod
    def handle(self, events: list[pygame.event.Event], now: float) -> list[InputEvent]:
        """The edges in ``events`` that are this device's, in order; the rest it ignores.

        :param now: ``perf_counter`` when the events were taken off the queue.
        """

    @abstractmethod
    def last_held(self) -> frozenset[str]:
        """The buttons, keys and stick directions down as of the last :meth:`handle`."""

    def axes(self) -> dict[str, float]:
        """Its axes now, by name, raw, -1..1; none for a device without."""
        return {}

    def require(self, needed: frozenset[str]) -> None:
        """Refuse a run that needs, of this device, what it cannot give.

        :param needed: control names a curriculum asks for, of every device.
        :raises ValueError: by default, never.
        """

    @abstractmethod
    def status(self) -> str:
        """One line: which device, and how it is read."""

    def close(self) -> None:
        """Let go of the device."""


class InputDevices(InputDevice):
    """A run's input devices together, as one: the keyboard, and those its rig lists."""

    name = "inputs"

    def __init__(self, devices: list[InputDevice], ignored: list[str] = ()) -> None:
        """Hold the devices a run reads.

        :param devices: the devices, the keyboard first.
        :param ignored: joysticks plugged in that no device was opened on, by name.
        """
        self._devices = tuple(devices)
        self.ignored = tuple(ignored)

    def devices(self) -> tuple[InputDevice, ...]:
        return self._devices

    def handle(self, events: list[pygame.event.Event], now: float) -> list[InputEvent]:
        """Every device's edges, device by device; the window closing as :data:`QUIT`."""
        out = [InputEvent(QUIT, Edge.DOWN) for e in events if e.type == pygame.QUIT]
        for device in self._devices:
            out += device.handle(events, now)
        return out

    def last_held(self) -> frozenset[str]:
        return frozenset().union(*(d.last_held() for d in self._devices))

    def axes(self) -> dict[str, float]:
        return {k: v for d in self._devices for k, v in d.axes().items()}

    def controls(self) -> frozenset[str]:
        return frozenset().union(*(d.controls() for d in self._devices))

    def require(self, needed: frozenset[str]) -> None:
        for device in self._devices:
            device.require(needed)

    def status(self) -> str:
        """The devices' lines, and the joysticks ignored."""
        lines = [d.status() for d in self._devices]
        lines += [f"IGNORED {name!r}: not in the rig's \"inputs\"" for name in self.ignored]
        return "; ".join(lines)

    def close(self) -> None:
        for device in self._devices:
            device.close()
