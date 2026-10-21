"""ViZDoom and COOM: the status bar, the actors on screen, and what the player felt happen.

From ``player_variables`` (status bar, position, the kill/hit counters), ``labels`` (the
actors with pixels on screen) and, for vizdoom, ``notifications`` (Doom's pickup lines).
``player_variables``, ``objects`` and ``labels`` cost real per-frame engine work nothing
on the live path needs, so no adapter logs them: :func:`player_variables`,
:func:`objects` and :func:`labels` read them from the engine themselves, for a replay
pass to call on the game object underneath a fresh adapter
(``extract_features._replay_adapter``), before feeding this reader's
:meth:`~.base.SymbolicReader.see`.
"""

from __future__ import annotations

import math
from typing import Any

from .base import Observation, SymbolicObject, SymbolicReader, text_of

#: The ``env_kwargs`` :func:`objects` and :func:`labels` need, set before init. Not
#: ``sectors_info_enabled``: it hangs or crashes ViZDoom 1.3 starting some Freedoom maps (E1M4).
ENGINE_OUTPUTS = {"objects_info_enabled": True, "labels_buffer_enabled": True}

#: The player's state: the HUD, the event counters, and where the player is and moves.
PLAYER_VARIABLES = (
    "HEALTH", "ARMOR", "DEAD", "ON_GROUND", "ATTACK_READY", "ALTATTACK_READY",
    "SELECTED_WEAPON", "SELECTED_WEAPON_AMMO",
    *(f"AMMO{i}" for i in range(10)), *(f"WEAPON{i}" for i in range(10)),
    "KILLCOUNT", "ITEMCOUNT", "SECRETCOUNT", "HITCOUNT", "HITS_TAKEN",
    "DAMAGECOUNT", "DAMAGE_TAKEN",
    "POSITION_X", "POSITION_Y", "POSITION_Z", "ANGLE", "PITCH",
    "VELOCITY_X", "VELOCITY_Y", "VELOCITY_Z",
)

OBJECT_FIELDS = ("id", "name", "position_x", "position_y", "position_z",
                 "angle", "pitch", "roll", "velocity_x", "velocity_y", "velocity_z")
LABEL_FIELDS = ("object_id", "object_name", "object_category", "x", "y", "width", "height",
                "value", "object_position_x", "object_position_y", "object_position_z")

#: Doom's weapon slots, by ``SELECTED_WEAPON``.
WEAPONS = {1: "Fist/Chainsaw", 2: "Pistol", 3: "Shotgun", 4: "Chaingun",
           5: "Rocket launcher", 6: "Plasma rifle", 7: "BFG9000"}
#: The status bar's ammo table: ammo type and the weapon slot it feeds.
AMMO_TABLE = (("Bullets", 2), ("Shells", 3), ("Rockets", 5), ("Cells", 6))
#: Counters whose rise since the last observation is something the player noticed.
EVENTS = (("KILLCOUNT", "You killed {}"), ("HITCOUNT", "Your shots hit {} time(s)"),
          ("HITS_TAKEN", "You were hit {} time(s)"))
#: The actor that is the player; its label is the player's own weapon sprite.
PLAYER = "DoomPlayer"


def objects(state: Any) -> list[dict]:
    """Every actor in the level, the player included."""
    if state is None or state.objects is None:
        return []
    return [{f: getattr(o, f) for f in OBJECT_FIELDS} for o in state.objects]


def labels(state: Any) -> list[dict]:
    """The actors with pixels on the screen this frame, and their boxes there."""
    if state is None or state.labels is None:
        return []
    return [{f: getattr(lb, f) for f in LABEL_FIELDS} for lb in state.labels]


def player_variables(game: Any) -> dict[str, float]:
    """The player's state now, by name -- fixed names, not the scenario's own variables."""
    import vizdoom as vzd
    return {n: game.get_game_variable(getattr(vzd.GameVariable, n)) for n in PLAYER_VARIABLES}


class DoomReader(SymbolicReader):
    wants_engine_state = True

    def reset(self) -> None:
        super().reset()
        self._counted: dict[str, float] | None = None
        self._messages: list[str] = []

    def see(self, variables: dict) -> None:
        super().see(variables)
        message = text_of(variables.get("notifications"))
        self._messages += [m for m in message.splitlines() if m]

    def observation(self) -> Observation:
        pv = self.variables.get("player_variables")
        if not pv:
            return Observation()
        events = self._counter_events(pv) + tuple(self._messages)
        self._messages = []
        return Observation(inventory=_status_bar(pv),
                           objects=_in_view(self.variables.get("labels") or [], pv),
                           events=events)

    def _counter_events(self, pv: dict) -> tuple[str, ...]:
        """What the :data:`EVENTS` counters rose by since the last observation."""
        now = {name: float(pv[name]) for name, _ in EVENTS}
        before, self._counted = self._counted or now, now
        return tuple(text.format(int(now[n] - before[n])) for n, text in EVENTS
                     if now[n] > before[n])


def _status_bar(pv: dict) -> tuple[tuple[str, float | str], ...]:
    """Doom's status bar, left to right, then the held weapon and the ammo table."""
    arms = " ".join(str(s) for s in range(2, 8) if pv[f"WEAPON{s}"] > 0)
    weapon = int(pv["SELECTED_WEAPON"])
    return (("Ammo", pv["SELECTED_WEAPON_AMMO"]), ("Health", f"{int(pv['HEALTH'])}%"),
            ("Arms", arms or "none"), ("Armor", f"{int(pv['ARMOR'])}%"),
            ("Weapon", WEAPONS.get(weapon, f"slot {weapon}")),
            *((ammo, pv[f"AMMO{slot}"]) for ammo, slot in AMMO_TABLE))


def _in_view(labels: list[dict], pv: dict) -> tuple[SymbolicObject, ...]:
    """The actors on screen by bearing (positive left) and distance from the player."""
    x, y, facing = pv["POSITION_X"], pv["POSITION_Y"], pv["ANGLE"]
    seen = []
    for label in sorted(labels, key=lambda lb: lb["object_id"]):
        if label["object_name"] == PLAYER:
            continue
        dx, dy = label["object_position_x"] - x, label["object_position_y"] - y
        bearing = (math.degrees(math.atan2(dy, dx)) - facing + 180) % 360 - 180
        seen.append(SymbolicObject(label["object_name"], (label["object_position_x"],
                                   label["object_position_y"]), bearing, math.hypot(dx, dy)))
    return tuple(seen)
