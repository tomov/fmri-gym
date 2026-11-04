"""The run config: a curriculum on a rig, and the few things only this run says.

A curriculum (:class:`~fmri_gym.config.curriculum.Curriculum`) knows nothing of
any rig, and a rig (:class:`~fmri_gym.config.rig.RigConfig`) nothing of any
curriculum; a run is one of each, played for one subject, so what needs both
-- a game key the rig's scanner types at every volume, or a control none of
its input devices has -- is checked here
(:meth:`RunConfig.problems`), and so, on request, is whether each game starts on
this machine. ``fmri_play`` owns the flags and their
spellings; :meth:`RunConfig.from_args` is the one place the command line is read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from importlib import metadata

from ..inputs import controls_of_input_devices
from ..menu import MenuSettings
from .curriculum import Curriculum
from .phases import GamePhase, Message, Phase, Survey
from .rig import RigConfig


@dataclass(frozen=True, slots=True)
class RunConfig:
    """One run: its curriculum, its rig, and who, which run, for how long.

    Every field is required, so a flag a caller forgets is a ``TypeError`` at
    the desk and never a default in the scanner.
    """

    #: what is played
    curriculum: Curriculum
    #: where: window, input devices, audio, triggers, data root
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

    def problems(self, *, load_games: bool = False) -> list[str]:
        """What keeps this curriculum from being played on this rig, which neither shows alone.

        A button box in a digit mode sends its trigger as a digit too (Current
        Designs: ``5``). With the rig waiting for that key (``sync.mode``
        ``wait``), a game bound to it would act at every volume. A rig provides
        the controls of the keyboard and of its ``"inputs"``, so a phase needing
        another would leave the participant a dead button. A
        ``duration`` must leave time for something besides the rig's fixations.

        :param load_games: also build, reset and close each distinct game once, in
            this process: what this machine has installed, and seconds per game.
        :return: one problem per clashing binding, per thing no control can do, for
            a duration the fixations fill, and, with ``load_games``, per game that
            cannot start.
        """
        return (self._scanner_key_problems() + self._curriculum_requires_controls_missing_from_rig()
                + self._duration_problems() + (self._game_problems() if load_games else []))

    def _game_problems(self) -> list[str]:
        """Build, reset and close each distinct game once; the ones that cannot start.

        Two game phases whose backend and spec differ only in their seed are one
        game, reset with the first one's seed; a phase with no seed of its own is
        reset with its index, since what is checked is that it starts, not which
        episode. Each load is said on stderr.
        """
        from ..adapters import get_adapter  # imports no engine until a game is built
        problems: list[str] = []
        seen: set[str] = set()
        seeded, _ = self.curriculum.with_seeds(lambda i: i)
        for i, phase in enumerate(seeded.phases):
            if not isinstance(phase, GamePhase):
                continue
            unseeded = {k: v for k, v in phase.spec.items() if k != "seed"}
            key = json.dumps([phase.backend, unseeded], sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            start = time.perf_counter()
            try:
                adapter = get_adapter(phase.backend, phase.spec)
                adapter.reset(phase.seed)
                adapter.close()
            except Exception as e:
                problems.append(f"phase {i}: {phase.backend} game {phase.game!r} cannot start "
                                f"({type(e).__name__}: {e}){_install_hint(e, phase.backend)}")
                continue
            print(f"games: phase {i}, {phase.backend} {phase.game}: built, reset and closed in "
                  f"{time.perf_counter() - start:.1f} s", file=sys.stderr)
        return problems

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
        if sync.mode != "wait":
            return []
        return [f"phase {i}: controls: {combo} uses {sync.key!r}, the key {self.rig.path}'s "
                "scanner types at every volume (its triggers.sync.key): bind another control"
                for i, phase in enumerate(self.curriculum.phases) if isinstance(phase, GamePhase)
                for combo in phase.spec["controls"] if sync.key in combo.split("+")]

    def _curriculum_requires_controls_missing_from_rig(self) -> list[str]:
        """The controls a phase names that none of the rig's input devices has."""
        provided = controls_of_input_devices(self.rig.inputs)
        return [f"phase {i}: needs {sorted(missing)}, which none of {self.rig.path}'s inputs "
                f"has ({', '.join(self.rig.inputs)}): add the device to its \"inputs\", or play "
                "this curriculum on a rig that has it"
                for i, phase in enumerate(self.curriculum.phases)
                if (missing := get_all_required_controls_for(phase) - provided)]

    def required_controls(self) -> frozenset[str]:
        """Every control a phase of the curriculum names."""
        return frozenset().union(*(get_all_required_controls_for(p)
                                   for p in self.curriculum.phases))


def _install_hint(error: Exception, backend: str) -> str:
    """How to install what ``error`` found missing: an extra, by name when ``backend`` is one."""
    if not isinstance(error, ModuleNotFoundError):
        return ""
    extras = metadata.metadata("fmri-gym").get_all("Provides-Extra")
    extra = backend if backend in extras else f"<one of: {', '.join(extras)}>"
    return (f". Sync its extra together with every other game's, since a sync uninstalls the "
            f"extras it does not name: uv sync --extra {extra} --extra ...")


def get_all_required_controls_for(phase: Phase) -> frozenset[str]:
    """Every control the participant must be able to press in ``phase``.

    :param phase: any phase; one that takes no control needs none.
    :return: a game's bound controls, its menu's and its instructions'; a
        message's dismissing controls; a survey's points.
    """
    if isinstance(phase, Survey):
        return frozenset(phase.controls)
    if isinstance(phase, Message):
        return _controls_that_dismiss(phase)
    if isinstance(phase, GamePhase):
        return _controls_of_game(phase)
    return frozenset()


def _controls_that_dismiss(message: Message) -> frozenset[str]:
    """One of its ``controls_to_dismiss``, or its ``control`` when it is untimed."""
    if message.controls_to_dismiss is not None:
        return frozenset(message.controls_to_dismiss)
    if message.duration is not None or message.control == "any":
        return frozenset()
    return frozenset({message.control})


def _controls_of_game(game: GamePhase) -> frozenset[str]:
    """Every control in its ``controls`` combos, its menu's, and its instructions'."""
    out = {name for combo in game.spec["controls"] for name in combo.split("+") if name}
    if game.instructions is not None and game.instructions.controls_to_dismiss is not None:
        out |= set(game.instructions.controls_to_dismiss)
    if "menu" in game.spec:
        menu = MenuSettings.from_dict(game.spec["menu"])
        out |= {menu.control, *menu.move, menu.confirm}
    return frozenset(out)
