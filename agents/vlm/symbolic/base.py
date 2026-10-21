"""A game's state as text beside the frame, read from the variables ``capture()`` logged.

A reader sees every logged frame's ``variables`` in order (``see``) and says what the
player knows now (``observation``): what the screen and HUD show and what just happened.
It never touches a game: the same reader serves a block's log and a live ``capture()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SymbolicObject:
    """One thing the player can see: where it is, or which way and how far."""

    name: str
    position: tuple[float, ...]
    bearing: float | None = None
    distance: float | None = None

    def text(self) -> str:
        """``"Demon: 12 deg left, 410 away"``, or ``"Demon at (-587, -425)"``."""
        if self.bearing is None:
            return f"{self.name} at ({', '.join(str(round(c)) for c in self.position)})"
        b = round(self.bearing)
        side = "ahead" if b == 0 else f"{abs(b)} deg {'left' if b > 0 else 'right'}"
        return f"{self.name}: {side}, {round(self.distance)} away"


@dataclass(frozen=True)
class Observation:
    """What the player knows at one frame: status, things in view, what just happened."""

    inventory: tuple[tuple[str, float | str], ...] = ()
    objects: tuple[SymbolicObject, ...] = ()
    events: tuple[str, ...] = ()

    def text(self) -> str:
        """One line per field present; ``""`` when the reader knows nothing to say."""
        lines = []
        if self.inventory:
            lines.append("Inventory: [" + ", ".join(f"{k}: {_value(v)}"
                                                    for k, v in self.inventory) + "]")
        if self.objects:
            lines.append("Objects:")
            lines.extend(f"- {o.text()}" for o in self.objects)
        if self.events:
            lines.append("Events: " + "; ".join(self.events))
        return "\n".join(lines)


class SymbolicReader:
    """The reader for a game with none of its own: its logged scalars as the inventory."""

    #: True for a reader whose fields cost real per-frame engine work, so a block's
    #: logged ``variables`` alone are never enough and a caller must replay the block
    #: through a fresh adapter, reading its game object each frame, to feed ``see``
    #: (see ``agents.vlm.extract_features._replay_adapter`` and ``._engine_state``).
    #: False needs only the log.
    wants_engine_state: bool = False

    def __init__(self) -> None:
        self.variables: dict = {}

    def reset(self) -> None:
        """Start an episode: forget the last one's frames."""
        self.variables = {}

    def see(self, variables: dict) -> None:
        """Take one logged frame's ``variables``; every frame, in order."""
        self.variables = variables

    def observation(self) -> Observation:
        """What the player knows after the frames seen so far."""
        return Observation(inventory=tuple(
            (k, float(v)) for k, v in self.variables.items()
            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)))


def text_of(raw: Any) -> str:
    """NetHack-style bytes (a list of ints, NUL-padded) or a string, as text."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    a = np.asarray(raw)
    if a.dtype.kind in "US":
        return str(a)
    return bytes(a.astype(np.uint8)).split(b"\x00", 1)[0].decode("ascii", errors="replace")


def _value(v: float | str) -> str:
    """``26`` for 26.0, ``3.5`` for 3.5156, a string as it is."""
    if isinstance(v, str):
        return v
    return str(int(v)) if float(v).is_integer() else f"{v:.1f}"
