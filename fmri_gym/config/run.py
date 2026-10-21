"""The run config: a curriculum on a rig, and the few things only this run says.

A curriculum (:class:`~fmri_gym.config.curriculum.Curriculum`) knows nothing of
any rig, and a rig (:class:`~fmri_gym.config.rig.RigConfig`) nothing of any
curriculum; a run is one of each, played for one subject, so what needs both
-- a game key the rig's scanner types at every volume -- is checked here
(:meth:`RunConfig.problems`). ``fmri_play`` owns the flags and their
spellings; :meth:`RunConfig.from_args` is the one place the command line is read.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from .curriculum import Curriculum
from .phases import GamePhase
from .rig import RigConfig


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
    #: the run's seconds from the trigger, closing fixation included; ``None``
    #: only for a rig check, which plays its curriculum to its end.
    duration: float | None
    #: seconds of the fixation a run with a ``duration`` closes on.
    end_fixation: float
    #: carry on from where the subject's latest run of this curriculum stopped.
    resume: bool
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
                   run=args.run, duration=args.duration, end_fixation=args.end_fixation,
                   resume=args.resume, dummy_trigger=args.dummy_trigger)

    def problems(self) -> list[str]:
        """What keeps this curriculum from being played on this rig, which neither shows alone.

        A button box in a digit mode sends its trigger as a digit too (Current
        Designs: ``5``). With the rig waiting for that key (``sync.mode``
        ``wait``), a game bound to it would act at every volume.

        :return: one problem per clashing binding.
        """
        sync = self.rig.triggers.sync
        name = sync.key.upper()
        if sync.mode != "wait" or not (len(name) == 1 and name.isalnum()):
            return []  # "=" is no game key: nothing can clash
        return [f"phase {i}: keys: {combo} uses {sync.key!r}, the key {self.rig.path}'s scanner "
                "types at every volume (its triggers.sync.key): bind another key"
                for i, phase in enumerate(self.curriculum.phases) if isinstance(phase, GamePhase)
                for combo in phase.spec["keys"] if name in combo.upper().split("+")]
