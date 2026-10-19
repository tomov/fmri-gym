"""Play a run through the real loop with nobody at the keyboard and no screen.

``fmri_play.main()`` runs in this process, so the trigger wait, the
instructions screen, the window, the key queue, the pacing and the logger are
the ones a subject gets. Two things stand in for the rig: SDL's dummy video
driver (a window that is never shown; ``--window`` opens a real one) and a key
script registered with :mod:`fmri_gym.keys` the way the gamepad is (:func:`~fmri_gym.keys
.register_event_source` for the presses, :func:`~fmri_gym.keys
.register_held_source` for what counts as held), so every press travels the
route a real key does. The script presses the phase's own keys at random,
after first dismissing the instructions with the key they ask for.

Audio is the one thing a script cannot fake: the run opens the machine's
output unless ``--no-audio`` is passed, which is what a machine without one
(a CI runner) needs. ``--duration`` ends the run on the clock, as the scanner
would, so a five-minute block plays for the seconds a test can afford.

    python -m tests.headless --curriculum configs/curricula/minihack/room5x5.json \\
        --data-root /tmp/headless --duration 25

Run as a module from the repo root: the logger's writer process re-imports
this file, and the ``__main__`` guard is what keeps the child from starting a
second run.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

# Before pygame is imported anywhere: the driver is read when the display opens.
if "--window" not in sys.argv:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402

from fmri_gym import keys  # noqa: E402
from fmri_gym.curriculum import load_curriculum  # noqa: E402

#: Seconds a scripted press is held: long enough for a real-time poll to see it,
#: short enough never to open a hold-to-pause menu.
HOLD_S = 0.08


class KeyScript:
    """Random presses of the phase's keys, the dismiss key first, on a fixed clock.

    :param names: the key names to press at random.
    :param dismiss: the names the instructions take; pressed before the random ones.
    :param rate: presses per second.
    :param seed: the random draw's seed.
    """

    def __init__(self, names: list[str], dismiss: list[str], rate: float, seed: int) -> None:
        self._rng = random.Random(seed)
        self._names = names
        self._queue = list(dismiss) * 2          # instructions, and a message in front of them
        self._period = 1.0 / rate
        self._next = time.perf_counter() + 1.0  # let the opening fixation show first
        self._down: tuple[str, float] | None = None
        self.presses = 0

    def held(self) -> frozenset[str]:
        """The key held right now, for the real-time poll."""
        return frozenset([self._down[0]] if self._down else [])

    def pump(self) -> None:
        """Post the press or release that is due."""
        now = time.perf_counter()
        if self._down and now >= self._down[1]:
            self._post(pygame.KEYUP, self._down[0])
            self._down = None
        if self._down or now < self._next:
            return
        name = self._queue.pop(0) if self._queue else self._rng.choice(self._names)
        self._post(pygame.KEYDOWN, name)
        self._down = (name, now + HOLD_S)
        self._next = now + self._period
        self.presses += 1

    @staticmethod
    def _post(kind: int, name: str) -> None:
        code = keys.KEYCODE_BY_NAME[name]
        unicode = " " if name == "SPACE" else (name.lower() if len(name) == 1 else "")
        pygame.event.post(pygame.event.Event(kind, key=code, mod=0, unicode=unicode, scancode=0))


def script_for(curriculum_path: str, rate: float, seed: int) -> KeyScript:
    """A :class:`KeyScript` over the keys of the config's game phases."""
    phases = [p for p in load_curriculum(curriculum_path)["curriculum"] if p["type"] == "game"]
    names = sorted({k for p in phases for combo in p["keys"] for k in combo.split("+") if k})
    dismiss = sorted({k for p in phases
                      for k in (p.get("instructions") or {}).get("keys_to_dismiss", [])})
    return KeyScript(names, dismiss or ["SPACE"], rate, seed)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--curriculum", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--duration", type=float, default=25.0,
                   help="seconds the run lasts; the phases stop 10 s before")
    p.add_argument("--rate", type=float, default=6.0, help="presses per second")
    p.add_argument("--window", action="store_true",
                   help="open a real window (SDL's default driver) instead of the dummy one")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-audio", action="store_true", help="for a machine without an output")
    args = p.parse_args()

    script = script_for(args.curriculum, args.rate, args.seed)
    keys.register_event_source(script.pump)
    keys.register_held_source(script.held)
    sys.argv = ["fmri-play", "--curriculum", args.curriculum, "--subject", "sub-test",
                "--ses", "1", "--run", "1", "--data-root", args.data_root,
                "--dummy-trigger", "--no-pad", "--no-vsync", "--size", "640x480",
                "--duration", str(args.duration)] + (["--no-audio"] if args.no_audio else [])
    import fmri_play
    try:
        fmri_play.main()
    finally:
        print(f"headless: {script.presses} presses scripted", file=sys.stderr)


if __name__ == "__main__":
    main()
