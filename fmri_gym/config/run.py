"""The run config: a curriculum on a rig, and the few things only this run says.

A curriculum (:class:`~fmri_gym.config.curriculum.Curriculum`) knows nothing of
any rig, and a rig (:class:`~fmri_gym.config.rig.RigConfig`) nothing of any
curriculum; a run is one of each, played for one subject, so what needs both
-- a game key the rig's scanner types at every volume, or one its controller
cannot press -- is checked here
(:meth:`RunConfig.problems`). ``fmri_play`` owns the flags and their
spellings; :meth:`RunConfig.from_args` is the one place the command line is read.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import NamedTuple

from ..menu import MenuSettings
from .curriculum import Curriculum
from .phases import GamePhase, Message, Phase, Survey
from .rig import RigConfig


class KeyNeed(NamedTuple):
    """One thing a phase asks of the participant, and the key combos that do it."""

    #: what it is, as a problem names it: "action 3", "its menu's X", "dismissing it"
    what: str
    #: the combos that do it, each the key names held together; any one will do
    combos: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class RunConfig:
    """One run: its curriculum, its rig, and who, which run, for how long.

    Every field is required, so a flag a caller forgets is a ``TypeError`` at
    the desk and never a default in the scanner.
    """

    #: what is played
    curriculum: Curriculum
    #: where: window, pad, audio, triggers, data root
    rig: RigConfig
    #: ``sub-<label>``: the BIDS subject.
    subject: str
    #: the BIDS session number, from 1.
    ses: int
    #: this task's run number in the session, from 1.
    run: int
    #: the run's seconds from the trigger, the rig's fixations included;
    #: ``None`` plays the curriculum to its end.
    duration: float | None
    #: carry on from where the subject's latest run of this curriculum stopped.
    resume: bool
    #: play the curriculum again from its start when it ends, until ``duration``.
    loop: bool
    #: skip the experimenter and scanner waits: a test run, not a session.
    dummy_trigger: bool

    @classmethod
    def from_args(cls, args: argparse.Namespace, curriculum: Curriculum,
                  rig: RigConfig) -> RunConfig:
        """Read ``fmri_play``'s parsed flags.

        :param args: the namespace of ``fmri_play``'s parser, one attribute per flag.
        :param curriculum: the curriculum ``--curriculum`` names, loaded.
        :param rig: the rig ``--rig`` picked.
        """
        return cls(curriculum=curriculum, rig=rig, subject=args.subject, ses=args.ses,
                   run=args.run, duration=args.duration, resume=args.resume, loop=args.loop,
                   dummy_trigger=args.dummy_trigger)

    @property
    def start_fixation(self) -> float | None:
        """Seconds of fixation the run opens on: the rig's; ``None`` for none, and for a
        rig check, which measures rather than scans."""
        return None if self.curriculum.has_checks else self.rig.start_fixation

    @property
    def end_fixation(self) -> float | None:
        """Seconds of fixation the run closes on: the rig's; ``None`` as for the start."""
        return None if self.curriculum.has_checks else self.rig.end_fixation

    def problems(self) -> list[str]:
        """What keeps this curriculum from being played on this rig, which neither shows alone.

        A button box in a digit mode sends its trigger as a digit too (Current
        Designs: ``5``). With the rig waiting for that key (``sync.mode``
        ``wait``), a game bound to it would act at every volume. A rig played
        with a controller presses only the keys its ``"bindings"`` name, so a
        phase needing another would leave the participant a dead button. A
        ``duration`` must leave time for something besides the rig's fixations.

        :return: one problem per clashing binding, per thing no control can do, and
            a duration the fixations fill.
        """
        return (self._scanner_key_problems() + self._unpressable_problems()
                + self._duration_problems())

    def _duration_problems(self) -> list[str]:
        """A ``duration`` no longer than the rig's fixations together."""
        fixed = (self.start_fixation or 0) + (self.end_fixation or 0)
        if self.duration is None or self.duration > fixed:
            return []
        return [f"--duration {self.duration:g}: {self.rig.path}'s fixations take {fixed:g} s "
                "(its start_fixation and end_fixation), so the run must be longer than that"]

    def _scanner_key_problems(self) -> list[str]:
        """The game keys the rig's scanner types at every volume."""
        sync = self.rig.triggers.sync
        name = sync.key.upper()
        if sync.mode != "wait" or not (len(name) == 1 and name.isalnum()):
            return []  # "=" is no game key: nothing can clash
        return [f"phase {i}: keys: {combo} uses {sync.key!r}, the key {self.rig.path}'s scanner "
                "types at every volume (its triggers.sync.key): bind another key"
                for i, phase in enumerate(self.curriculum.phases) if isinstance(phase, GamePhase)
                for combo in phase.spec["keys"] if name in combo.upper().split("+")]

    def _unpressable_problems(self) -> list[str]:
        """What a phase asks of the participant that no control of the rig's controller presses."""
        if not (self.rig.has_pad and self.rig.bindings):
            return []  # a keyboard, or a controller that types its keys itself
        pressed = set(self.rig.bindings.values())
        return [f"phase {i}: {what} needs {' or '.join('+'.join(c) for c in combos)}, which no "
                f'control of {self.rig.path}\'s controller presses: bind one in its "bindings", '
                "or play this curriculum on a rig without a controller"
                for i, phase in enumerate(self.curriculum.phases)
                for what, combos in key_needs(phase)
                if not any(set(c) <= pressed for c in combos)]


def key_needs(phase: Phase) -> list[KeyNeed]:
    """What the participant must be able to do in ``phase``, by key.

    :param phase: any phase; one that takes no key needs nothing.
    :return: one :class:`KeyNeed` per thing to do: each game action (any combo
        bound to it does it, so a key that only doubles another is never needed
        by itself), each key of the menu, and dismissing a screen.
    """
    if isinstance(phase, Survey):
        return [KeyNeed(f"the survey's {k}", ((k,),)) for k in ("LEFT", "RIGHT", "RETURN")]
    if isinstance(phase, Message):
        return _message_needs(phase)
    if isinstance(phase, GamePhase):
        return _game_needs(phase)
    return []


def _message_needs(message: Message) -> list[KeyNeed]:
    """Dismissing a message: one of its ``keys_to_dismiss``, or its ``key`` when it is untimed."""
    if message.keys_to_dismiss is not None:
        return [KeyNeed("dismissing it", tuple((k,) for k in message.keys_to_dismiss))]
    if message.duration is not None or message.key == "any":
        return []
    return [KeyNeed("dismissing it", (({" ": "SPACE"}.get(message.key, message.key.upper()),),))]


def _game_needs(game: GamePhase) -> list[KeyNeed]:
    """A game's actions, the keys of its menu, and dismissing its instructions."""
    combos_by_action: dict[str, list[tuple[str, ...]]] = {}
    for combo, action in game.spec["keys"].items():
        if combo:
            combos_by_action.setdefault(json.dumps(action), []).append(tuple(combo.split("+")))
    needs = [KeyNeed(f"action {a}", tuple(c)) for a, c in combos_by_action.items()]
    if game.instructions is not None and game.instructions.keys_to_dismiss is not None:
        dismiss = tuple((k,) for k in game.instructions.keys_to_dismiss)
        needs.append(KeyNeed("its instructions", dismiss))
    if "menu" in game.spec:
        menu = MenuSettings.from_dict(game.spec["menu"])
        needs += [KeyNeed(f"its menu's {k}", ((k,),)) for k in (menu.key, *menu.move, menu.confirm)]
    return needs
