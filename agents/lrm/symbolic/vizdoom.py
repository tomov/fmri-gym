"""ViZDoom as text: the status bar, what is in view, and what the player felt happen.

What a subject has in front of them in every vizdoom curriculum here (full
HUD, messages on) and where each piece is read from:

- the session's strip above the frame, time left and score: ``hud``, passed in;
- Doom's status bar: AMMO (the held weapon's), HEALTH %, ARMS (weapon slots
  2-7 held), ARMOR %, and the ammo table, all from ``get_game_variable``,
  which answers for every variable whatever the scenario's .cfg lists; the
  held weapon is on screen as its sprite;
- what is in view: the labels buffer (``state.labels``), which lists exactly
  the actors with pixels on the screen this frame, each placed by its bearing
  from straight ahead and its distance, as a player judges them;
- the pickup messages Doom prints at the top of the screen
  (``notifications_buffer``).

What a player knows without it being written anywhere is said as events since
the last turn: a kill, a shot that landed, being hit (``KILLCOUNT``,
``HITCOUNT``, ``HITS_TAKEN``).

Left out, because the subject could not know it: every actor out of view
(``state.objects`` lists the whole level), the map's walls (``state.sectors``),
the player's map coordinates and compass heading (used here only to turn an
actor's position into a bearing and distance), velocities, the scenario's
``USER`` script variables, and secret and item counts. The status bar's face
(which glances toward damage) and key slots have no variable to read, and the
ammo table's maximums (200/50/50/300, doubled by a backpack) none either.

The labels and messages are buffers a subject's block does not switch on; the
replay turns them on through the scenario's ``env_kwargs``, which ViZDoom
passes to ``DoomGame.set_config``. They fill buffers and leave the simulation
alone, and the replay's rewards checking against the log holds them to it.

Actions are the scenario's own button names (``TURN_LEFT + ATTACK``), not the
keys the subject pressed, so the model is told what an action does.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from .base import NOOP, Observation, SymbolicObject, SymbolicReader

if TYPE_CHECKING:
    from fmri_gym.adapters.base import EnvAdapter

#: The actor that is the player; its label is the player's own weapon sprite.
PLAYER = "DoomPlayer"

#: Doom's weapon slots, by ``SELECTED_WEAPON``.
WEAPONS = {1: "Fist/Chainsaw", 2: "Pistol", 3: "Shotgun", 4: "Chaingun",
           5: "Rocket launcher", 6: "Plasma rifle", 7: "BFG9000"}

#: The status bar's ammo table, top to bottom: ammo type and the weapon slot it feeds.
AMMO_TABLE = (("Bullets", 2), ("Shells", 3), ("Rockets", 5), ("Cells", 6))

#: Counters whose rise since the last turn is something the player noticed.
EVENTS = (("KILLCOUNT", "You killed {}"), ("HITCOUNT", "Your shots hit {} time(s)"),
          ("HITS_TAKEN", "You were hit {} time(s)"))


class VizDoomReader(SymbolicReader):
    name = "vizdoom"

    def __init__(self) -> None:
        self._counters: dict[str, float] = {}
        self._messages: list[str] = []

    def replay_spec(self, spec: dict) -> dict:
        return {**spec, "env_kwargs": {**spec.get("env_kwargs", {}),
                                       "labels_buffer_enabled": True,
                                       "notifications_buffer_enabled": True}}

    def reset(self, adapter: EnvAdapter) -> None:
        self._counters = _counters(adapter.env.unwrapped.game)
        self._messages = []

    def after_step(self, adapter: EnvAdapter) -> None:
        state = adapter.env.unwrapped.game.get_state()
        if state is not None and state.notifications_buffer:
            self._messages += [m for m in state.notifications_buffer.splitlines() if m]

    def observe(self, adapter: EnvAdapter, variables: dict, score: float,
                hud: tuple[str, ...]) -> Observation:
        game = adapter.env.unwrapped.game
        counters = _counters(game)
        events = [text.format(int(counters[name] - self._counters[name]))
                  for name, text in EVENTS if counters[name] > self._counters[name]]
        events += self._messages
        self._counters, self._messages = counters, []
        return Observation(score=score, hud=hud, inventory=_status_bar(game),
                           objects=_in_view(game), events=tuple(events))

    def action_names(self, adapter: EnvAdapter) -> tuple[str, ...]:
        return tuple(_buttons(adapter)) + (NOOP,)

    def action_name(self, adapter: EnvAdapter, action: Any) -> str:
        # The logged action is the MultiBinary button vector every vizdoom
        # curriculum here uses (env_kwargs.max_buttons_pressed 0).
        buttons = _buttons(adapter)
        if len(action) != len(buttons):
            raise ValueError(f"action {action!r} is not a vector over the scenario's "
                             f"buttons {buttons}; set max_buttons_pressed to 0")
        return " + ".join(b for b, down in zip(buttons, action) if down) or NOOP


def _var(game: Any, name: str) -> float:
    """One ``GameVariable`` by name; ViZDoom answers for any, listed or not."""
    import vizdoom as vzd
    return float(game.get_game_variable(vzd.GameVariable.__members__[name]))


def _counters(game: Any) -> dict[str, float]:
    """The :data:`EVENTS` counters now."""
    return {name: _var(game, name) for name, _ in EVENTS}


def _status_bar(game: Any) -> tuple[tuple[str, float | str], ...]:
    """Doom's status bar, left to right, then the held weapon and the ammo table."""
    arms = " ".join(str(s) for s in range(2, 8) if _var(game, f"WEAPON{s}") > 0)
    weapon = int(_var(game, "SELECTED_WEAPON"))
    return (("Ammo", _var(game, "SELECTED_WEAPON_AMMO")),
            ("Health", f"{int(_var(game, 'HEALTH'))}%"),
            ("Arms", arms or "none"),
            ("Armor", f"{int(_var(game, 'ARMOR'))}%"),
            ("Weapon", WEAPONS.get(weapon, f"slot {weapon}")),
            *((ammo, _var(game, f"AMMO{slot}")) for ammo, slot in AMMO_TABLE))


def _in_view(game: Any) -> tuple[SymbolicObject, ...]:
    """The actors on screen, by bearing and distance from the player.

    The engine lists labels in no fixed order from one tic to the next; by id,
    each keeps its line, and a change in the text is a change in the view.
    """
    state = game.get_state()
    if state is None:
        return ()
    x, y = _var(game, "POSITION_X"), _var(game, "POSITION_Y")
    facing = _var(game, "ANGLE")
    seen = []
    for label in sorted(state.labels, key=lambda lb: lb.object_id):
        if label.object_name == PLAYER:
            continue
        dx, dy = label.object_position_x - x, label.object_position_y - y
        # Doom's angles run counter-clockwise from +x, so positive is to the left.
        bearing = (math.degrees(math.atan2(dy, dx)) - facing + 180) % 360 - 180
        seen.append(SymbolicObject(label.object_name,
                                   (label.object_position_x, label.object_position_y),
                                   bearing, math.hypot(dx, dy)))
    return tuple(seen)


def _buttons(adapter: EnvAdapter) -> list[str]:
    """The scenario's buttons in action-vector order: ``["TURN_LEFT", ...]``."""
    return [_name(b) for b in adapter.env.unwrapped.game.get_available_buttons()]


def _name(enum_value: Any) -> str:
    """``Button.TURN_LEFT`` -> ``"TURN_LEFT"``."""
    return str(enum_value).rsplit(".", 1)[-1]
