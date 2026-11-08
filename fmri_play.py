"""fmri_play.py -- run any Gymnasium-compatible game as an fMRI task.

One experiment framework across backends: Atari (ALE), stable-retro consoles
(NES/SNES/Genesis/...), and any plain Gymnasium env. The backend is chosen
per game block in the curriculum; the experiment loop is identical for all.

One config file is one run, and this plays it: nothing here loops over runs,
opens an editor or writes a config. A session of several runs is a shell
script with one of these commands per line (README, "Runs and sessions"),
which ``session-editor`` writes and any shell plays.

How it is played -- the window, the input devices, the audio, where the data
goes -- is the rig's, not the command's: each rig has a file
(:mod:`fmri_gym.config.rig`), and ``--rig`` names it.

Usage (--ses and --run say which run of which session this is; a session
script passes the same --ses to all of its runs, each with its own --run):
    python fmri_play.py --subject sub-01 --ses 1 --run 1 \
        --rig configs/examples/rigs/laptop.json --curriculum my.json
    python fmri_play.py ... --dummy-trigger    # testing: no experimenter/scanner wait
    python fmri_play.py ... --skip-load-games   # do not build each game before the window
    python fmri_play.py ... --duration 600 --resume  # a 10-min run of a longer curriculum

See configs/curricula/demos/mixed.json for a curriculum that mixes all three backends,
and README.md for the config schema.
"""

from __future__ import annotations

import argparse
import signal
import sys

from fmri_gym import Run
from fmri_gym.config.curriculum import EXIT_QUIT, Curriculum
from fmri_gym.display import quit_like_esc
from fmri_gym.config.rig import RigConfig
from fmri_gym.config.run import RunConfig



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
                   help="the run's length in seconds from the scanner trigger, the rig's "
                   "fixations included: the phases stop in time for the rig's end_fixation, "
                   "whatever is playing. Default: play the curriculum to its end")
    p.add_argument("--resume", action="store_true",
                   help="carry on from where this subject's latest run of this config (any "
                   "session) stopped, mid-block included. Default: start at the beginning")
    p.add_argument("--loop", action="store_true",
                   help="when the curriculum ends before --duration, play it again from its "
                   "first phase, until --duration (which it needs)")
    p.add_argument("--rig", required=True,
                   help="the rig file it is played on, e.g. configs/examples/rigs/laptop.json: "
                   "it says the window, input devices, audio and data root")
    p.add_argument("--skip-load-games", action="store_true",
                   help="do not build, reset and close each game once before the window. "
                   "Default: do, so a game that cannot start fails before the participant screen")
    p.add_argument("--dummy-trigger", action="store_true")
    return p


def main() -> None:
    args = _parser().parse_args()
    # The editor's Check runs on load, so a file edited by hand gets it too.
    curriculum = Curriculum.load(args.curriculum)
    rig = RigConfig.open(args.rig)
    print(f"rig: {rig.status()}", file=sys.stderr)
    for warning in (rig.fixation_warning, rig.measured_latency_warning):
        if warning:
            print(f"rig: WARNING: {warning}", file=sys.stderr)
    run_config = RunConfig.from_args(args, curriculum, rig)
    if run_config.skip_load_games:
        print("--skip-load-games: the games are not built before the window; one that cannot "
              "start fails when its block begins", file=sys.stderr)
    problems = run_config.problems(load_games=not run_config.skip_load_games)
    if problems:
        raise ValueError(f"{args.curriculum} on {rig.path}: " + "; ".join(problems))
    run = Run.from_config(run_config)
    previous = signal.signal(signal.SIGINT, quit_like_esc)
    try:
        completed = run.play()
    finally:
        run.close()
        # Last: a terminal's Ctrl+C can come twice (to uv and to us), the second one late.
        signal.signal(signal.SIGINT, previous)
    if not completed:
        # A session script (set -e) must not start the next run after an ESC.
        sys.exit(EXIT_QUIT)


if __name__ == "__main__":
    main()
