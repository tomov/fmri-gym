"""Readers for the grid and puzzle games, each from the variables its adapter logs."""

from __future__ import annotations

import numpy as np

from .base import Observation, SymbolicObject, SymbolicReader, text_of

#: Crafter's own view is 9x9 tiles centred on the player: a Chebyshev distance of 4.
CRAFTER_VIEW = 4
#: Crafter's facing vector -> the direction the player sprite points.
FACING = {(-1, 0): "left", (1, 0): "right", (0, -1): "up", (0, 1): "down"}
#: blstats field (``nle.nethack.NLE_BL_*``) -> its label.
NLE_STATUS = {"HP": "NLE_BL_HP", "Max HP": "NLE_BL_HPMAX", "Gold": "NLE_BL_GOLD",
              "Hunger": "NLE_BL_HUNGER", "Dungeon Level": "NLE_BL_DEPTH"}


class CrafterReader(SymbolicReader):
    """Inventory and vitals, daylight, facing, and the creatures within the view."""

    def __init__(self) -> None:
        super().__init__()
        from crafter_gym.env import import_crafter
        # The adapter logs the inventory's values; crafter's item order names them.
        self._names = list(import_crafter().constants.items)

    def observation(self) -> Observation:
        v = self.variables
        if "inventory" not in v:
            return Observation()
        carried = tuple((n, c) for n, c in zip(self._names, v["inventory"]) if c)
        light = (("Daylight", round(float(v["daylight"]), 2)),
                 ("Facing", FACING.get(tuple(v["facing"]), str(v["facing"]))))
        px, py = v["player_pos"]
        seen = tuple(SymbolicObject(o["name"], tuple(o["pos"])) for o in v["objects"]
                     if max(abs(o["pos"][0] - px), abs(o["pos"][1] - py)) <= CRAFTER_VIEW)
        return Observation(inventory=carried + light, objects=seen)


class NLEReader(SymbolicReader):
    """MiniHack and NetHack: the status line, the inventory, and the message."""

    def observation(self) -> Observation:
        v = self.variables
        if "blstats" not in v:
            return Observation()
        import nle.nethack as nh
        status = tuple((name, float(v["blstats"][getattr(nh, i)])) for name, i in NLE_STATUS.items())
        carried = tuple((chr(letter), text_of(raw)) for letter, raw
                        in zip(v.get("inv_letters", []), v.get("inv_strs", [])) if letter)
        message = text_of(v.get("message"))
        return Observation(inventory=status + carried, events=(message,) if message else ())


class RushHourReader(SymbolicReader):
    """Every car on the board: slot 0 the red one, a length-0 slot padding."""

    def observation(self) -> Observation:
        cars = self.variables.get("cars")
        if cars is None:
            return Observation()
        seen = tuple(SymbolicObject(("Red car" if slot == 0 else f"Car {slot}")
                                    + f", length {int(length)}, "
                                    + ("horizontal" if horizontal else "vertical"), (col, row))
                     for slot, (row, col, length, horizontal)
                     in enumerate(np.asarray(cars).tolist()) if length)
        return Observation(objects=seen)


class BabaReader(SymbolicReader):
    """The rules the word tiles spell out now."""

    def observation(self) -> Observation:
        rules = (self.variables.get("ruleset") or {}).get("_rule_", [])
        return Observation(inventory=tuple((r["object"], r["property"].removeprefix("is_"))
                                           for r in rules))
