"""A run builds each of its games once before its window opens, and refuses one that cannot start.

Headless CartPole runs on the ``gym`` backend, which the core CI job plays.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.backends import REPO
from tests.conftest import write_curriculum

pytestmark = pytest.mark.backend("gym")

RUN_S = 12.0
CARTPOLE = {"type": "game", "game": "CartPole-v1", "keys": {"": 0, "LEFT": 0, "RIGHT": 1},
            "fps": 50, "mode": "duration", "duration": 1.0, "show_outcome_duration": 0}


def _run(curriculum: list[dict], tmp_path: Path) -> subprocess.CompletedProcess:
    """Play ``curriculum`` headless; the finished process."""
    config = write_curriculum(curriculum, tmp_path / "loads.json")
    return subprocess.run(
        [sys.executable, "-m", "tests.headless", "--curriculum", str(config),
         "--data-root", str(tmp_path / "data"), "--duration", str(RUN_S), "--no-audio"],
        cwd=REPO, capture_output=True, text=True, timeout=RUN_S + 120,
        env={**os.environ, "SDL_VIDEODRIVER": "dummy"})


def test_each_game_is_loaded_once_and_the_console_says_how_long(tmp_path):
    fixation = {"type": "fixation", "duration": 0.5}
    done = _run([fixation, CARTPOLE, fixation, CARTPOLE], tmp_path)
    assert done.returncode == 0, done.stderr[-3000:]
    loads = [line for line in done.stderr.splitlines() if line.startswith("games: ")]
    assert len(loads) == 1, f"a game played twice, with two seeds, is loaded once: {loads}"
    assert loads[0].startswith("games: phase 1, gym CartPole-v1: built, reset and closed in")


def test_a_game_that_cannot_start_stops_the_run_before_it_plays(tmp_path):
    missing = {**CARTPOLE, "game": "NoSuchGame-v0"}
    done = _run([{"type": "fixation", "duration": 0.5}, CARTPOLE, missing], tmp_path)
    assert done.returncode != 0
    assert "phase 2: gym game 'NoSuchGame-v0' cannot start" in done.stderr
    assert not list((tmp_path / "data").rglob("manifest.json")), "the run played"
