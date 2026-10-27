"""fmri_play.py -- run any Gymnasium-compatible game as an fMRI task.

One experiment framework across backends: Atari (ALE), stable-retro consoles
(NES/SNES/Genesis/...), and any plain Gymnasium env. The backend is chosen
per game block in the curriculum; the experiment loop is identical for all.

One config file is one run, and this plays it: nothing here loops over runs,
opens an editor or writes a config. A session of several runs is a shell
script with one of these commands per line (README, "Runs and sessions"),
which ``session-editor`` writes and any shell plays.

How it is played -- the window, the controller, the audio, where the data
goes -- is the rig's, not the command's: each rig has a file
(:mod:`fmri_gym.config.rig`), and ``--rig`` names it.

Usage (--ses and --run say which run of which session this is; a session
script passes the same --ses to all of its runs, each with its own --run):
    python fmri_play.py --subject sub-01 --ses 1 --run 1 \
        --rig configs/examples/rigs/laptop.json --curriculum my.json
    python fmri_play.py ... --dummy-trigger    # testing: no experimenter/scanner wait
    python fmri_play.py ... --duration 600 --resume  # a 10-min run of a longer curriculum

See configs/curricula/demos/mixed.json for a curriculum that mixes all three backends,
and README.md for the config schema.
"""

from __future__ import annotations

import argparse
import signal
import sys

from fmri_gym import Run, checks
from fmri_gym.config.curriculum import EXIT_QUIT, Curriculum
from fmri_gym.display import quit_like_esc
from fmri_gym.config.rig import RigConfig
from fmri_gym.config.run import RunConfig
from fmri_gym.run import END_FIXATION_S



class _Parser(argparse.ArgumentParser):
    """argparse, plus where the numbers it insists on come from."""

    def error(self, message: str) -> None:
        if "--ses" in message or "--run" in message:
            message += ("\n  a run says which run of which session it is: a session script "
                        "passes one --ses to all its runs and gives each its own --run. At the "
                        "desk: `fmri-ses --subject <sub>` prints the next free session, and "
                        "--run 1 is the first run of this task in it")
        super().error(message)


def _parser() -> argparse.ArgumentParser:
    """The command line: one run, and the rig that plays it."""
    p = _Parser(description="Run any gym game as an fMRI task.")
    p.add_argument("--subject", default="sub-test", help="BIDS subject: sub-<letters/digits>")
    p.add_argument("--curriculum", required=True, help="the run's curriculum JSON (see "
                   "README); session-editor writes one without the JSON")
    p.add_argument("--ses", type=int, required=True,
                   help="BIDS session number, from 1: which scanning session this run belongs "
                   "to. A session script takes it once and passes it to every run (fmri-ses)")
    p.add_argument("--run", type=int, required=True,
                   help="this task's run number in the session, from 1: which run of the "
                   "design this is, so it stays the same however the session went. A run that "
                   "already has data is re-acquired beside it, never overwritten")
    p.add_argument("--duration", type=float, default=None,
                   help="the run's length in seconds from the scanner trigger: the phases stop "
                   "--end-fixation seconds before it, whatever is playing, and the run closes "
                   "on a fixation cross. Default: play the curriculum to its end")
    p.add_argument("--end-fixation", type=float, default=END_FIXATION_S, metavar="SECONDS",
                   help="seconds of the fixation a run with --duration closes on, for the "
                   f"haemodynamic response to the last thing played (default {END_FIXATION_S:g})")
    p.add_argument("--resume", action="store_true",
                   help="carry on from where this subject's latest run of this config (any "
                   "session) stopped, mid-block included. Default: start at the beginning")
    p.add_argument("--loop", action="store_true",
                   help="when the curriculum ends before --duration, play it again from its "
                   "first phase, until --duration (which it needs)")
    p.add_argument("--rig", required=True,
                   help="the rig file it is played on, e.g. configs/examples/rigs/laptop.json: "
                   "it says the window, controller, audio and data root. A rig check makes "
                   "one when the file is missing")
    p.add_argument("--dummy-trigger", action="store_true")
    return p


def main() -> None:
    args = _parser().parse_args()
    # The editor's Check runs on load, so a file edited by hand gets it too.
    curriculum = Curriculum.load(args.curriculum)
    do_rig_check = curriculum.has_checks
    # A rig check is how a rig gets its file: it opens the form when there is none.
    rig = checks.ensure_rig(args.rig) if do_rig_check else RigConfig.open(args.rig)
    trigger_error = None
    if do_rig_check:  # its triggers, and a line that opens, first
        rig, trigger_error = checks.prepare(rig)
    print(f"rig: {rig.status()}", file=sys.stderr)
    run_config = RunConfig.from_args(args, curriculum, rig)
    problems = run_config.problems()
    if problems:
        raise ValueError(f"{args.curriculum} on {rig.path}: " + "; ".join(problems))
    if not do_rig_check:
        checks.offer_check(run_config)
    run = Run.from_config(run_config)
    if do_rig_check:
        run.logger.set_extra("trigger_error", trigger_error)
    previous = signal.signal(signal.SIGINT, quit_like_esc)
    try:
        completed = run.play()
    finally:
        run.close()
        # Last: a terminal's Ctrl+C can come twice (to uv and to us), the second one late.
        signal.signal(signal.SIGINT, previous)
    if do_rig_check:
        # A failed test is a finding, listed in the report: the session goes on.
        checks.finish(run, run_config.ses)
    if not completed:
        # A session script (set -e) must not start the next run after an ESC.
        sys.exit(EXIT_QUIT)


if __name__ == "__main__":
    main()
