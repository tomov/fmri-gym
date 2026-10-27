"""``--loop``: a curriculum that ends before ``--duration`` is played again.

A 2 s CartPole block (``mode: duration``) in a run several times that long, on
the ``gym`` backend, which the core CI job plays.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fmri_gym.logging import read_events
from tests.backends import REPO
from tests.conftest import write_curriculum

pytestmark = pytest.mark.backend("gym")

RUN_S = 12.0
CURRICULUM = [
    {"type": "fixation", "duration": 0.5, "include_on_resume": True},
    {"type": "game", "game": "CartPole-v1", "keys": {"": 0, "LEFT": 0, "RIGHT": 1}, "fps": 50,
     "mode": "duration", "duration": 2.0, "show_outcome_duration": 0},
    {"type": "fixation", "duration": 0.5},
]


def _run(config: Path, data_root: Path, number: int, *more: str) -> tuple[dict, Path, str]:
    """Play run ``number`` of ``config`` headless; its manifest, its folder and its stderr."""
    done = subprocess.run(
        [sys.executable, "-m", "tests.headless", "--curriculum", str(config),
         "--data-root", str(data_root), "--duration", str(RUN_S), "--no-audio",
         "--run", str(number), *more],
        cwd=REPO, capture_output=True, text=True, timeout=RUN_S + 120,
        env={**os.environ, "SDL_VIDEODRIVER": "dummy"})
    assert done.returncode == 0, f"run {number} exited {done.returncode}:\n{done.stderr[-3000:]}"
    manifest = next(data_root.glob(f"sub-test/ses-001/beh/*_run-{number:03d}/manifest.json"))
    return json.loads(manifest.read_text()), manifest.parent, done.stderr


def test_loop_plays_the_curriculum_again(tmp_path):
    config = write_curriculum(CURRICULUM, tmp_path / "loop.json")
    manifest, run, _ = _run(config, tmp_path / "data", 1, "--loop")
    blocks = [p for p in manifest["phases"] if p["type"] == "game"]
    assert manifest["loop"] is True
    assert len(blocks) >= 3, f"only {len(blocks)} passes in {RUN_S} s"
    names = [b["data_dir"] for b in blocks]
    assert names[:3] == ["block-01_gym_CartPole-v1", "block-01_gym_CartPole-v1_2",
                         "block-01_gym_CartPole-v1_3"]
    seeds = {read_events(str(run / n))[0]["base_seed"] for n in names}
    assert seeds == {manifest["curriculum"][1]["seed"]}, f"a pass changed the seed: {seeds}"
    assert manifest["phases"][-1].get("end_of_run"), "the run did not close on its fixation"


def test_loop_resumes_a_curriculum_played_to_its_end(tmp_path):
    config = write_curriculum(CURRICULUM, tmp_path / "loop.json")
    first, _, _ = _run(config, tmp_path / "data", 1)
    assert [p["index"] for p in first["phases"] if p["type"] == "game"] == [1]
    second, _, stderr = _run(config, tmp_path / "data", 2, "--loop", "--resume")
    assert "--loop starts it again" in stderr
    assert second["resumed_from"] is None
    assert second["phases"][0]["index"] == 0, "the new pass did not start at the beginning"


def test_loop_without_duration_is_refused(tmp_path):
    config = write_curriculum(CURRICULUM, tmp_path / "loop.json")
    done = subprocess.run(
        [sys.executable, "fmri_play.py", "--curriculum", str(config), "--subject", "sub-test",
         "--ses", "1", "--run", "1", "--rig", "configs/examples/rigs/laptop.json", "--loop",
         "--dummy-trigger"],
        cwd=REPO, capture_output=True, text=True, timeout=60,
        env={**os.environ, "SDL_VIDEODRIVER": "dummy"})
    assert done.returncode != 0
    assert "--loop plays the curriculum until --duration" in done.stderr
