"""What every input device is: named controls, their edges, and the devices as one.

Every key, button and stick direction a participant can touch has one name,
unique across devices: its value in
:class:`~humans.inputs.control_names.ControlNames`. A device lists its names at
the top of its own file, and its :meth:`InputDevice.handle` turns the pygame
events that are its own into :class:`InputEvent` edges; it never reads pygame's
queue itself. As in a game engine, a control is read two ways: an edge the frame
it happens, and held state (:meth:`InputDevice.held`) at the tick. Several
devices together are one :class:`InputDevices`, which is itself a device.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, ClassVar, NamedTuple

import pygame

#: The name the run's inputs give the window's close button.
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
        #: ``(axis, raw value)`` of each stick reading since they were last taken; ``None``:
        #: not kept. A device without sticks never adds to it.
        self.axis_readings: list[tuple[str, float]] | None = None

    @classmethod
    @abstractmethod
    def open(cls) -> InputDevice:
        """The device, ready to read.

        :raises ValueError: when it is not plugged in.
        """

    @abstractmethod
    def handle(self, events: list[pygame.event.Event], now: float) -> list[InputEvent]:
        """The edges ``events`` make on this device, in order; the events of others it ignores.

        :param now: ``perf_counter`` when the events were taken off the queue.
        """

    def controls(self) -> frozenset[str]:
        """Every control this device has."""
        return self.CONTROLS

    def held(self) -> frozenset[str]:
        """The controls down as of the last :meth:`handle`."""
        return frozenset(self._held)

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
        """Press ``name``: its down edge, or nothing when it is already down."""
        if name in self._held:
            return []
        self._held.add(name)
        return [InputEvent(name, Edge.DOWN)]

    def _release(self, name: str) -> list[InputEvent]:
        """Release ``name``: its up edge, or nothing when it is already up."""
        if name not in self._held:
            return []
        self._held.discard(name)
        return [InputEvent(name, Edge.UP)]


class InputDevices(InputDevice):
    """Several input devices read as one: a run's keyboard and controllers together.

    Its edges are each device's, in the devices' order, after the window's
    close button as :data:`QUIT`.
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
        self._recorder: Callable[[float, dict], None] | None = None

    def record_with(self, recorder: Callable[[float, dict], None]) -> None:
        """Hand ``recorder`` each edge and each raw stick reading as it is read, with the time
        it was read (``perf_counter``): an ``input_event`` (``name``, ``down``) or an ``axis``
        (``input``, ``axis``, ``value``) record."""
        self._recorder = recorder
        for device in self.devices:
            device.axis_readings = []

    @classmethod
    def open(cls) -> InputDevices:
        """Nothing: the devices are opened one by one (:func:`humans.inputs.open_input_devices`)."""
        return cls([])

    def handle(self, events: list[pygame.event.Event], now: float) -> list[InputEvent]:
        out = [InputEvent(QUIT, Edge.DOWN) for e in events if e.type == pygame.QUIT]
        for device in self.devices:
            out += device.handle(events, now)
        if self._recorder is not None:
            self._record(out, now)
        return out

    def _record(self, edges: list[InputEvent], now: float) -> None:
        """Hand the recorder each device's stick readings, then ``edges``."""
        for device in self.devices:
            for axis, value in device.axis_readings:
                self._recorder(now, {"type": "axis", "input": device.name, "axis": axis,
                                     "value": value})
            device.axis_readings.clear()
        for name, edge in edges:
            self._recorder(now, {"type": "input_event", "name": name, "down": edge is Edge.DOWN})

    def controls(self) -> frozenset[str]:
        return frozenset().union(*(d.controls() for d in self.devices))

    def held(self) -> frozenset[str]:
        return frozenset().union(*(d.held() for d in self.devices))

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
