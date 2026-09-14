"""fmri_play.py -- run any Gymnasium-compatible game as an fMRI task.

One experiment framework across backends: Atari (ALE), stable-retro consoles
(NES/SNES/Genesis/...), and any plain Gymnasium env. The backend is chosen
per game block in the curriculum; the experiment loop is identical for all.

Usage:
    python fmri_play.py --subject sub-01                        # built-in demo
    python fmri_play.py --subject sub-01 --curriculum my.json
    python fmri_play.py --subject sub-01 --dummy-trigger        # testing
    python fmri_play.py --gui [--curriculum my.json]            # config editor, then run
    python fmri_play.py --curriculum session.json --run 2       # one run of a multi-run config

See configs/demo_mixed.json for a curriculum that mixes all three backends,
and README.md for the curriculum schema.
"""

from __future__ import annotations

import argparse
import os
import sys

from fmri_gym import Display, Session
from fmri_gym.config import (default_outdir, load_config, parse_size, resolve_session, run_dir,
                             runs_of, select_runs)
from fmri_gym.triggers import TriggerError


def build_demo_curriculum() -> list[dict]:
    """Mixed-backend demo: an Atari game, a retro game, and a survival game.

    All three are forgiving, free-roaming games (no instant game-over), so a
    first-time human can actually play them.
    """
    return [
        {"type": "message", "text": "Pong (Atari)", "duration": 2.0},
        {"type": "fixation", "duration": 2.0},
        # Pong's paddle actions are RIGHT=2 / LEFT=3; remap them onto the
        # up/down arrows, which read more naturally for a vertical paddle.
        {"type": "game", "backend": "ale", "game": "ALE/Pong-v5", "mode": "duration",
         "duration": 10.0, "fps": 30, "keys": {"UP": 2, "DOWN": 3}},

        {"type": "message", "text": "Airstriker (Genesis)", "duration": 2.0},
        {"type": "fixation", "duration": 2.0},
        {"type": "game", "backend": "retro", "game": "Airstriker-Genesis-v0", "mode": "duration", "duration": 10.0, "fps": 60},

        # Crafter: an open-world survival game. You wander freely (arrows move,
        # SPACE interacts) with no instant death -- friendlier than CartPole,
        # which topples in ~2 s. Needs `pip install crafter`.
        {"type": "message", "text": "Crafter", "duration": 2.0},
        {"type": "fixation", "duration": 2.0},
        {"type": "game", "backend": "crafter", "game": "crafter", "mode": "episode",
         "n_episodes": 1, "max_duration": 20.0, "fps": 15, "seed": 0},

        {"type": "fixation", "duration": 4.0},
        {"type": "survey", "questions": [
            "I was fully absorbed in the games.",
            "The games were too difficult.",
        ]},
    ]


def _cli_session(args: argparse.Namespace) -> dict:
    """Session settings given on the command line (``None`` = flag absent)."""
    return {"subject": args.subject, "outdir": args.outdir, "size": args.size,
            "fullscreen": args.fullscreen, "dummy_trigger": args.dummy_trigger,
            "vsync": None if args.no_vsync is None else not args.no_vsync}


def _fold_cli_options(curriculum: list[dict], args: argparse.Namespace) -> None:
    """CLI-global backend options fold into the relevant game phases, so each
    per-block EnvAdapter reads everything it needs from its own spec."""
    for phase in curriculum:
        if phase.get("type") != "game":
            continue
        backend = phase.get("backend", "gym")
        if backend == "ale" and args.save_pixels:
            phase.setdefault("save_pixels", True)
        if backend == "vgdl" and args.vgdl_repo:
            phase.setdefault("repo", args.vgdl_repo)


def main() -> None:
    p = argparse.ArgumentParser(description="Run any gym game as an fMRI task.")
    p.add_argument("--subject", help="subject id (default: sub-test)")
    p.add_argument("--curriculum", help="config JSON (a curriculum list, or a dict with sections)")
    p.add_argument("--gui", action="store_true",
                   help="open the config editor first; Run there starts the session")
    p.add_argument("--run", help="multi-run config: play only this run (1-based index or name)")
    p.add_argument("--outdir", help="default: data/<subject>_<timestamp>")
    p.add_argument("--size", help="window size <w>x<h> (default: 1024x768)")
    p.add_argument("--fullscreen", action="store_true", default=None)
    p.add_argument("--no-vsync", action="store_true", default=None,
                   help="do not lock flips to the monitor refresh (default: try to)")
    p.add_argument("--dummy-trigger", action="store_true", default=None)
    p.add_argument("--save-pixels", action="store_true",
                   help="ALE only: also store lossless pixels (large; warns).")
    p.add_argument("--vgdl-repo", default=os.environ.get("VGDL_REPO"),
                   help="path to the language_and_experience checkout (vgdl backend)")
    args = p.parse_args()

    config = (load_config(args.curriculum) if args.curriculum
              else {"curriculum": build_demo_curriculum()})
    # Flags beat the file's "session" section; the editor then shows the result
    # and whatever it hands back is what runs.
    config["session"] = resolve_session(config, _cli_session(args))
    if args.gui:
        from fmri_gym.gui import edit_config
        picked = edit_config(config, args.curriculum)
        if picked is None:
            return
        config, args.run = picked
    try:
        runs = select_runs(runs_of(config), args.run)
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    _play(config, runs, args)


def _play(config: dict, runs: list[tuple[int, dict]], args: argparse.Namespace) -> None:
    """Play ``runs`` in order on one display; ESC in a run ends the session."""
    session = resolve_session(config)
    outdir = session["outdir"] or default_outdir(session["subject"])
    multi = "runs" in config
    display = Display(size=parse_size(session["size"]), fullscreen=session["fullscreen"],
                      vsync=session["vsync"])
    try:
        for index, run in runs:
            _fold_cli_options(run["curriculum"], args)
            target = run_dir(outdir, index, run["name"]) if multi else outdir
            try:
                one = Session(session["subject"], run["curriculum"], display, target,
                              dummy_trigger=session["dummy_trigger"],
                              triggers=config.get("triggers"))
            except (TriggerError, ValueError) as exc:
                # A bad triggers section or an unopenable marker port: stop
                # here, at the desk, with the reason -- not with a participant.
                sys.exit(f"error: {exc}")
            if multi:
                one.logger.set_extra("run", {"index": index, "name": run["name"],
                                             "of": len(runs_of(config))})
            if not one.run():
                break
    finally:
        display.close()


if __name__ == "__main__":
    main()
