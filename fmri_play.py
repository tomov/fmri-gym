"""fmri_play.py -- run any Gymnasium-compatible game as an fMRI task.

One experiment framework across backends: Atari (ALE), stable-retro consoles
(NES/SNES/Genesis/...), and any plain Gymnasium env. The backend is chosen
per game block in the curriculum; the experiment loop is identical for all.

One config file is one run, and this plays it: nothing here loops over runs,
opens an editor or writes a config. A session of several runs is a shell
script with one of these commands per line (README, "Runs and sessions"),
which ``fmri-edit`` writes and any shell plays.

How it is played -- the window, the controller, the audio, where the data
goes -- is the rig's, not the command's: each rig has a file
(:mod:`fmri_gym.rig`), and ``--rig`` names it when the machine has several.

Usage (--ses and --run say which run of which session this is; a session
script passes the same --ses to all of its runs, each with its own --run):
    python fmri_play.py --subject sub-01 --curriculum my.json --ses 1 --run 1
    python fmri_play.py ... --rig scanner3T     # one of this machine's rigs
    python fmri_play.py ... --dummy-trigger     # testing: no experimenter/scanner wait

See configs/demo_mixed.json for a curriculum that mixes all three backends,
and README.md for the config schema.
"""

from __future__ import annotations

import argparse
import signal
import sys

from fmri_gym import Run, checks, rig
from fmri_gym.config import EXIT_QUIT, load_config, validate_config
from fmri_gym.display import quit_like_esc


#: Flags that were fmri-play's before the rig file took them over (:mod:`fmri_gym.rig`).
_MOVED = ("--size", "--fullscreen", "--monitor", "--no-vsync", "--no-pad", "--no-audio",
          "--data-root")


class _Parser(argparse.ArgumentParser):
    """argparse, plus where the numbers it insists on come from."""

    def error(self, message: str) -> None:
        if "--ses" in message or "--run" in message:
            message += ("\n  a run says which run of which session it is: a session script "
                        "passes one --ses to all its runs and gives each its own --run. At the "
                        "desk: `fmri-ses --subject <sub>` prints the next free session, and "
                        "--run 1 is the first run of this task in it")
        moved = [f for f in _MOVED if f in message]
        if moved:
            message += (f"\n  {', '.join(moved)}: how a run is played is the rig's now, in its "
                        "file ~/.config/fmri-gym/rigs/<rig>.json (\"screen\", \"pad\", "
                        "\"audio\", \"data_root\"); --rig picks the rig")
        super().error(message)


def _parser() -> argparse.ArgumentParser:
    """The command line: one run, and the rig that plays it."""
    p = _Parser(description="Run any gym game as an fMRI task.")
    p.add_argument("--subject", default="sub-test", help="BIDS subject: sub-<letters/digits>")
    p.add_argument("--curriculum", required=True, help="config JSON of the run (see README); "
                   "fmri-edit writes one without the JSON")
    p.add_argument("--ses", type=int, required=True,
                   help="BIDS session number, from 1: which scanning session this run belongs "
                   "to. A session script takes it once and passes it to every run (fmri-ses)")
    p.add_argument("--run", type=int, required=True,
                   help="this task's run number in the session, from 1: which run of the "
                   "design this is, so it stays the same however the session went. A run that "
                   "already has data is re-acquired beside it, never overwritten")
    p.add_argument("--rig", help="the rig it is played on, by name: its file "
                   "~/.config/fmri-gym/rigs/<rig>.json says the window, controller, audio and "
                   "data root. Default: this machine's only rig; a rig check makes one")
    p.add_argument("--dummy-trigger", action="store_true")
    return p


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.curriculum)
    problems = validate_config(config)  # the editor's Check, so a file edited by hand gets it too
    if problems:
        raise ValueError(f"{args.curriculum}: " + "; ".join(problems))

    rig_check = None
    # The rig's file first; a rig check makes it when there is none.
    site = checks.open_rig(args.rig, create=checks.has_checks(config))
    vars(args).update(rig.run_args(site))
    if checks.has_checks(config):  # a rig check: a line that opens, first
        rig_check = checks.prepare(config, args.curriculum)
    run = Run.from_config(config, args)
    run.logger.set_extra("rig", site)
    if rig_check is not None:
        run.logger.set_extra("trigger_error", rig_check["trigger_error"])
    previous = signal.signal(signal.SIGINT, quit_like_esc)
    try:
        completed = run.play()
    finally:
        run.close()
        # Last: a terminal's Ctrl+C can come twice (to uv and to us), the second one late.
        signal.signal(signal.SIGINT, previous)
    if rig_check is not None:
        # A failed test is a finding, listed in the report: the session goes on.
        checks.finish(run, args.ses, args.data_root)
    if not completed:
        # A session script (set -e) must not start the next run after an ESC.
        sys.exit(EXIT_QUIT)


if __name__ == "__main__":
    main()
