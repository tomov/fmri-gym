"""fmri_play.py -- run any Gymnasium-compatible game as an fMRI task.

One experiment framework across backends: Atari (ALE), stable-retro consoles
(NES/SNES/Genesis/...), and any plain Gymnasium env. The backend is chosen
per game block in the curriculum; the experiment loop is identical for all.

Usage:
    python fmri_play.py --subject sub-01                        # built-in demo
    python fmri_play.py --subject sub-01 --curriculum my.json
    python fmri_play.py --subject sub-01 --dummy-trigger        # testing

See configs/demo_mixed.json for a curriculum that mixes all three backends,
and README.md for the curriculum schema.
"""

import argparse
import json
import os
import time

from fmri_gym import Display, Session, get_adapter


def build_demo_curriculum():
    """Mixed-backend demo: an Atari game, a retro game, and a plain gym env."""
    return [
        {"type": "message", "text": "Pong (Atari)", "duration": 2.0},
        {"type": "fixation", "duration": 2.0},
        {"type": "game", "backend": "ale", "game": "ALE/Pong-v5",
         "mode": "duration", "duration": 20.0, "fps": 30},

        {"type": "message", "text": "Airstriker (Genesis)", "duration": 2.0},
        {"type": "fixation", "duration": 2.0},
        {"type": "game", "backend": "retro", "game": "Airstriker-Genesis-v0",
         "mode": "duration", "duration": 20.0, "fps": 60},

        {"type": "message", "text": "CartPole", "duration": 2.0},
        {"type": "fixation", "duration": 2.0},
        {"type": "game", "backend": "gym", "game": "CartPole-v1",
         "mode": "episode", "n_episodes": 3, "max_duration": 30.0, "fps": 30,
         "keys": {"LEFT": 0, "RIGHT": 1}},

        {"type": "fixation", "duration": 4.0},
        {"type": "survey", "questions": [
            "I was fully absorbed in the games.",
            "The games were too difficult.",
        ]},
    ]


def load_curriculum(path):
    with open(path) as f:
        data = json.load(f)
    return data["curriculum"] if isinstance(data, dict) else data


def main():
    p = argparse.ArgumentParser(description="Run any gym game as an fMRI task.")
    p.add_argument("--subject", default="sub-test")
    p.add_argument("--curriculum")
    p.add_argument("--outdir")
    p.add_argument("--size", default="1024x768")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--dummy-trigger", action="store_true")
    p.add_argument("--save-pixels", action="store_true",
                   help="ALE only: also store lossless pixels (large; warns).")
    args = p.parse_args()

    curriculum = (load_curriculum(args.curriculum) if args.curriculum
                  else build_demo_curriculum())
    w, h = (int(x) for x in args.size.lower().split("x"))
    outdir = args.outdir or os.path.join(
        "data", f"{args.subject}_{time.strftime('%Y%m%d-%H%M%S')}")

    # Build only the adapters this curriculum actually references.
    backends = {ph.get("backend", "gym") for ph in curriculum
                if ph.get("type") == "game"}
    adapters = {}
    for b in backends:
        kwargs = {"save_pixels": args.save_pixels} if b == "ale" else {}
        adapters[b] = get_adapter(b, **kwargs)

    display = Display(size=(w, h), fullscreen=args.fullscreen)
    session = Session(args.subject, curriculum, adapters, display, outdir,
                      dummy_trigger=args.dummy_trigger)
    try:
        session.run()
    finally:
        display.close()


if __name__ == "__main__":
    main()
