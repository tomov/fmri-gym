"""fmri_play.py -- run any Gymnasium-compatible game as an fMRI task.

One experiment framework across backends: Atari (ALE), stable-retro consoles
(NES/SNES/Genesis/...), and any plain Gymnasium env. The backend is chosen
per game block in the curriculum; the experiment loop is identical for all.

One config file is one run, and this plays it: nothing here loops over runs,
opens an editor or writes a config. A session of several runs is a shell
script with one of these commands per line (README, "Runs and sessions"),
which ``fmri-edit`` writes and any shell plays.

Usage (--ses and --run say which run of which session this is; a session
script passes the same --ses to all of its runs, each with its own --run):
    python fmri_play.py --subject sub-01 --curriculum my.json --ses 1 --run 1
    python fmri_play.py ... --dummy-trigger    # testing: no experimenter/scanner wait
    python fmri_play.py ... --no-audio         # mute all games

See configs/demo_mixed.json for a curriculum that mixes all three backends,
and README.md for the config schema.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys

from fmri_gym import Run, checks
from fmri_gym.config import EXIT_QUIT, load_config, validate_config
from fmri_gym.display import quit_like_esc


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
    """The command line: one run, and how this machine plays it."""
    p = _Parser(description="Run any gym game as an fMRI task.")
    p.add_argument("--subject", default="sub-test", help="BIDS subject: sub-<letters/digits>")
    p.add_argument("--curriculum", required=True, help="config JSON of the run (see README); "
                   "fmri-edit writes one without the JSON")
    p.add_argument("--data-root", default="data",
                   help="where the BIDS tree goes: <root>/sub-XX/ses-NNN/beh/<run>/")
    p.add_argument("--ses", type=int, required=True,
                   help="BIDS session number, from 1: which scanning session this run belongs "
                   "to. A session script takes it once and passes it to every run (fmri-ses)")
    p.add_argument("--run", type=int, required=True,
                   help="this task's run number in the session, from 1: which run of the "
                   "design this is, so it stays the same however the session went. A run that "
                   "already has data is re-acquired beside it, never overwritten")
    p.add_argument("--size", default="1024x768")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--monitor", type=int, default=0,
                   help="which monitor to open on, by index (0: the first); a wrong one stops "
                   "the run and lists this machine's")
    p.add_argument("--no-vsync", action="store_true",
                   help="do not lock flips to the monitor refresh (default: try to)")
    p.add_argument("--dummy-trigger", action="store_true")
    p.add_argument("--no-audio", action="store_true", help="mute game audio in every block (the curriculum saved "
                   "in the manifest shows \"audio\": false)")
    p.add_argument("--vgdl-repo", default=os.environ.get("VGDL_REPO"),
                   help="path to the language_and_experience checkout (vgdl backend)")
    p.add_argument("--coom-repo", default=os.environ.get("COOM_REPO"),
                   help="path to the TTomilin/COOM checkout (coom backend)")
    return p


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.curriculum)
    problems = validate_config(config)  # the editor's Check, so a file edited by hand gets it too
    if problems:
        raise ValueError(f"{args.curriculum}: " + "; ".join(problems))

    rig_check = None
    if checks.has_checks(config):  # a rig check: its rig file, and a line that opens, first
        rig_check = checks.prepare(config, args.curriculum)
    run = Run.from_config(config, args)
    if rig_check is not None:
        run.logger.set_extra("rig", rig_check["rig"])
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
