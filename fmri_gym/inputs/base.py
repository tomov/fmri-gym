"""The seam between the input devices and the run: named controls, their presses, the devices.

Every button, key, axis and stick direction a participant can touch has one
name, unique across devices, and a curriculum names it as it is written here
(``"SPACE"``, ``"GMPD3_A"``, ``"CONTROLLER_LEFTSTICK_UP"``). A device class
lists its names at the top of its own file and turns the pygame events that
are its own into :class:`Press` edges; :class:`Inputs` is a run's devices
together, and the one reader of pygame's event queue during a run.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import ClassVar, NamedTuple

import pygame

#: The name :meth:`Inputs.poll` gives the window's close button.
QUIT = "QUIT"


class Press(NamedTuple):
    """One edge of a control: it went down, or came up."""

    name: str
    down: bool
    #: what a keyboard key types ("=" for a scanner's pulse); "" for anything else
    char: str = ""


class Input(ABC):
    """One input device: the names of its controls, and what it holds now."""

    #: the name a rig file lists it by (``"inputs"``)
    name: ClassVar[str]
    #: every control the class can have, known before any device is opened
    CONTROLS: ClassVar[frozenset[str]]

    def controls(self) -> frozenset[str]:
        """The controls this device has: :attr:`CONTROLS`, or fewer."""
        return self.CONTROLS

    @abstractmethod
    def poll(self, events: list[pygame.event.Event], now: float) -> list[Press]:
        """The presses in ``events`` that are this device's, in order; the rest it ignores.

        :param now: ``perf_counter`` when the events were taken off the queue.
        """

    @abstractmethod
    def held(self) -> frozenset[str]:
        """The buttons, keys and stick directions it holds down now."""

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


class Inputs:
    """A run's input devices together: the keyboard, and those its rig lists."""

    def __init__(self, devices: list[Input], ignored: list[str] = ()) -> None:
        """Hold the devices a run reads.

        :param devices: the devices, the keyboard first.
        :param ignored: joysticks plugged in that no device was opened on, by name.
        """
        self.devices = tuple(devices)
        self.ignored = tuple(ignored)

    def poll(self) -> list[Press]:
        """Every press since the last poll, device by device; the window closing as :data:`QUIT`.

        Takes every event off the queue: a queue left to fill drops new events,
        key presses included, so nothing else is left there.
        """
        events = pygame.event.get()
        now = time.perf_counter()
        presses = [Press(QUIT, True) for e in events if e.type == pygame.QUIT]
        for device in self.devices:
            presses += device.poll(events, now)
        return presses

    def held(self) -> frozenset[str]:
        """What every device holds down now."""
        return frozenset().union(*(d.held() for d in self.devices))

    def axes(self) -> dict[str, float]:
        """Every device's axes now."""
        return {k: v for d in self.devices for k, v in d.axes().items()}

    def controls(self) -> frozenset[str]:
        """Every control of every device."""
        return frozenset().union(*(d.controls() for d in self.devices))

    def status(self) -> str:
        """The devices' lines, and the joysticks ignored."""
        lines = [d.status() for d in self.devices]
        lines += [f"IGNORED {name!r}: not in the rig's \"inputs\"" for name in self.ignored]
        return "; ".join(lines)

    def close(self) -> None:
        for device in self.devices:
            device.close()
