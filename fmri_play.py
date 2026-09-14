"""fmri_play.py -- run any Gymnasium-compatible game as an fMRI task.

One experiment framework across backends: Atari (ALE), stable-retro consoles
(NES/SNES/Genesis/...), and any plain Gymnasium env. The backend is chosen
per game block in the curriculum; the experiment loop is identical for all.

Usage:
    python fmri_play.py --subject sub-01 --curriculum my.json
    python fmri_play.py --subject sub-01 --curriculum my.json --dummy-trigger   # testing

See configs/demo_mixed.json for a curriculum that mixes all three backends,
and README.md for the config schema.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from fmri_gym import Audio, Display, Session, Triggers


def load_config(path: str) -> dict:
    """Load a config file: a dict with ``"curriculum"`` and optional sections.

    ``"triggers"`` is the start sync + trigger codes (:mod:`fmri_gym.triggers`);
    ``_``-prefixed keys are notes. One shape only -- a bare list is refused.

    :param path: JSON file path.
    :return: the config dict.
    :raises ValueError: if the file is not a dict with a ``"curriculum"`` list.
    """
    with open(path) as f:
        config = json.load(f)
    if not isinstance(config, dict) or not isinstance(config.get("curriculum"), list):
        raise ValueError(f'{path}: expected a JSON object with a "curriculum" list')
    return config


def main() -> None:
    p = argparse.ArgumentParser(description="Run any gym game as an fMRI task.")
    p.add_argument("--subject", default="sub-test")
    p.add_argument("--curriculum", required=True, help="config JSON (see README)")
    p.add_argument("--outdir")
    p.add_argument("--size", default="1024x768")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--no-vsync", action="store_true",
                   help="do not lock flips to the monitor refresh (default: try to)")
    p.add_argument("--dummy-trigger", action="store_true")
    p.add_argument("--save-pixels", action="store_true",
                   help="ALE only: also store lossless pixels (large; warns).")
    p.add_argument("--vgdl-repo", default=os.environ.get("VGDL_REPO"),
                   help="path to the language_and_experience checkout (vgdl backend)")
    args = p.parse_args()

    config = load_config(args.curriculum)
    curriculum = config["curriculum"]
    w, h = (int(x) for x in args.size.lower().split("x"))
    outdir = args.outdir or os.path.join(
        "data", f"{args.subject}_{time.strftime('%Y%m%d-%H%M%S')}")

    # CLI-global backend options fold into the relevant game phases, so each
    # per-block EnvAdapter reads everything it needs from its own spec.
    for phase in curriculum:
        if phase.get("type") != "game":
            continue
        backend = phase.get("backend", "gym")
        if backend == "ale" and args.save_pixels:
            phase.setdefault("save_pixels", True)
        if backend == "vgdl" and args.vgdl_repo:
            phase.setdefault("repo", args.vgdl_repo)

    # Triggers first: a bad section or an unopenable port stops the run here,
    # at the desk, before any window opens -- not mid-session with a participant.
    triggers = Triggers.from_config(config.get("triggers"))
    print(f"triggers: {triggers.status()}", file=sys.stderr)
    if args.dummy_trigger:
        print("triggers: --dummy-trigger: the experimenter and scanner waits are skipped; "
              "this is a test run, not a session", file=sys.stderr)
    display = Display(size=(w, h), fullscreen=args.fullscreen, vsync=not args.no_vsync)
    audio = Audio()
    session = Session(args.subject, curriculum, display, outdir,
                      audio=audio, triggers=triggers, dummy_trigger=args.dummy_trigger)
    try:
        session.run()
    finally:
        display.close()
        audio.close()
        triggers.close()


if __name__ == "__main__":
    main()
