"""A game's state at one frame, as the symbolic observation a language model reads.

An LRM cannot look at the frame the subject saw, so it is handed the same
moment as text. That text is read off the live env during a replay of the
subject's block (:mod:`agents.lrm.transcript`), at the instant before each
logged action, and it holds to one rule: what the subject could know, and
nothing else. That is what was drawn -- the session's strip above the frame
(:meth:`~fmri_gym.adapters.base.EnvAdapter.hud`, the same call the run makes)
and what the game drew in its picture -- and what a player knows without
reading it off the screen, such as that their last shot hit. An engine knows
much more (where everything in the level is, the player's map coordinates),
and a reader leaves it out: the model is to read the subject's game, not a
better one.

A reader only looks: it never steps, resets or seeds the env, so the replay it
rides on stays the subject's game. :class:`SymbolicReader` is the fallback for
a backend with no reader of its own: its inventory is the scalar ``variables``
the adapter already logs, whether or not the game shows them, and its actions
are the phase's key names. A game gets a subclass in this package that reads
the engine (``vizdoom.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fmri_gym.adapters.base import EnvAdapter

#: The name of the action that presses nothing.
NOOP = "NOOP"


@dataclass(frozen=True)
class SymbolicObject:
    """One thing the player can see, and where it is.

    :param name: the engine's name for it.
    :param position: where it is in the engine's own coordinates; kept for
        analysis, and shown only when there is no ``bearing``.
    :param bearing: degrees from straight ahead, positive to the left, for a
        first-person game; ``None`` otherwise.
    :param distance: how far away, in the engine's units, with ``bearing``.
    """

    name: str
    position: tuple[float, ...]
    bearing: float | None = None
    distance: float | None = None

    def text(self) -> str:
        """``"Demon: 12 deg left, 410 away"``, or ``"Demon at (-587, -425)"``."""
        if self.bearing is None:
            return f"{self.name} at {_point(self.position)}"
        b = round(self.bearing)
        side = "ahead" if b == 0 else f"{abs(b)} deg {'left' if b > 0 else 'right'}"
        return f"{self.name}: {side}, {round(self.distance)} away"

    def record(self) -> dict:
        return {"name": self.name, "position": list(self.position),
                "bearing": self.bearing, "distance": self.distance}

    @classmethod
    def from_dict(cls, d: dict) -> SymbolicObject:
        return cls(d["name"], tuple(d["position"]), d["bearing"], d["distance"])


@dataclass(frozen=True)
class Observation:
    """What the player could know at one frame, as named fields.

    :param score: the episode's reward summed up to (not including) this frame.
    :param hud: the session's lines above the frame, as the subject read them
        (``("287 s", "Score: 0")``).
    :param inventory: ``(name, value)`` pairs the game shows of the player
        (health, ammo, weapon), in the order it shows them.
    :param objects: the things the player can see.
    :param events: what happened since the previous turn that the player
        noticed (``"You killed 1"``, a pickup message), oldest first.
    """

    score: float
    hud: tuple[str, ...] = ()
    inventory: tuple[tuple[str, float | str], ...] = ()
    objects: tuple[SymbolicObject, ...] = ()
    events: tuple[str, ...] = ()

    def text(self) -> str:
        """The observation as the model reads it, one field per line."""
        lines = [f"HUD: {', '.join(self.hud)}"] if self.hud else []
        if self.inventory:
            items = ", ".join(f"{name}: {_value(v)}" for name, v in self.inventory)
            lines.append(f"Inventory: [{items}]")
        lines.append("Objects:")
        lines.extend(f"- {o.text()}" for o in self.objects)
        if not self.objects:
            lines.append("- none")
        if self.events:
            lines.append(f"Events: {'; '.join(self.events)}")
        return "\n".join(lines)

    def record(self) -> dict:
        """The observation as a JSON object (:meth:`from_dict` reads it back)."""
        return {"score": self.score, "hud": list(self.hud),
                "inventory": [list(i) for i in self.inventory],
                "objects": [o.record() for o in self.objects], "events": list(self.events)}

    @classmethod
    def from_dict(cls, d: dict) -> Observation:
        """Inverse of :meth:`record`."""
        return cls(score=d["score"], hud=tuple(d["hud"]),
                   inventory=tuple((name, v) for name, v in d["inventory"]),
                   objects=tuple(SymbolicObject.from_dict(o) for o in d["objects"]),
                   events=tuple(d["events"]))


class SymbolicReader:
    """Reads :class:`Observation` s and action names off a live adapter.

    The base class knows no game: its inventory is the adapter's scalar
    ``variables`` and its action names are the phase's key names, which says
    what was pressed but not what it did. Subclass it per backend to read the
    engine.
    """

    #: names the reader in a transcript's header
    name: str = "generic"

    def replay_spec(self, spec: dict) -> dict:
        """The phase config to build the replay's adapter from.

        A subclass may switch on engine outputs the subject's block did not
        need (object lists, labels), but nothing that changes the game: the
        replay is checked against the log's rewards frame by frame.

        :param spec: the block's logged game phase.
        :return: the phase to build the adapter from; the default is ``spec``.
        """
        return spec

    def reset(self, adapter: EnvAdapter) -> None:
        """Start an episode: the env has just been reset (and restored).

        A reader that reports what changed between turns takes its baseline here.
        """

    def after_step(self, adapter: EnvAdapter) -> None:
        """See one replayed step, observed or not.

        A reader whose events last a single step (a message) collects them here,
        so a turn reports those of the frames between turns too.
        """

    def observe(self, adapter: EnvAdapter, variables: dict, score: float,
                hud: tuple[str, ...]) -> Observation:
        """The state the player acts on now.

        :param adapter: the replay's adapter, at the frame before the action.
        :param variables: the adapter's :meth:`capture` variables for that frame.
        :param score: the episode's reward so far.
        :param hud: the session's lines above the frame at this moment.
        :return: the frame as an :class:`Observation`.
        """
        scalars = tuple((k, float(v)) for k, v in variables.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool))
        return Observation(score=score, hud=hud, inventory=scalars)

    def action_names(self, adapter: EnvAdapter) -> tuple[str, ...]:
        """Every name :meth:`action_name` can return, for the system prompt."""
        names = sorted({_combo(c) for c in adapter.keymap.combos})
        if _is_multibinary(adapter.keymap):
            names.append(NOOP)
        return tuple(names)

    def action_name(self, adapter: EnvAdapter, action: Any) -> str:
        """The logged ``action`` as the key names that produce it.

        :param adapter: the replay's adapter (its keymap).
        :param action: one frame's logged ``action``.
        :return: ``"LEFT"``, ``"LEFT + SPACE"``, or :data:`NOOP`.
        :raises ValueError: an action no key of the phase produces.
        """
        keymap = adapter.keymap
        if _is_multibinary(keymap):
            held = [c for c, i in keymap.combos.items() if len(c) == 1 and action[i]]
            return " + ".join(sorted(_combo(c) for c in held)) or NOOP
        matches = [c for c, value in keymap.combos.items() if _same(value, action)]
        if not matches:
            raise ValueError(f"action {action!r} is not one of the phase's keys")
        return _combo(min(matches, key=len))


def _is_multibinary(keymap: Any) -> bool:
    """Whether held keys combine: a MultiBinary keymap (imported here, see transcript.py)."""
    from fmri_gym.adapters.keymap import MultiBinaryKeymap
    return isinstance(keymap, MultiBinaryKeymap)


def _combo(combo: frozenset[str]) -> str:
    """A keymap combo as a name: ``"LEFT + SPACE"``, or :data:`NOOP` for none."""
    return " + ".join(sorted(combo)) or NOOP


def _same(value: Any, action: Any) -> bool:
    """Whether a keymap value is the logged action (an int, or a list for a Box)."""
    try:
        return list(value) == list(action)
    except TypeError:
        return value == action


def _value(v: float | str) -> str:
    """``26`` for 26.0, ``3.5`` for 3.5156, a string as it is."""
    if isinstance(v, str):
        return v
    return str(int(v)) if float(v).is_integer() else f"{v:.1f}"


def _point(p: tuple[float, ...]) -> str:
    """``(-587, -425)``: positions rounded to the engine's unit."""
    return "(" + ", ".join(str(round(c)) for c in p) + ")"
