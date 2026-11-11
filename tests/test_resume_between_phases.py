"""``--resume`` of a run whose clock stopped it between two game blocks, in a message.

``tests/test_backends.py`` resumes a block the run stopped mid-episode. This
is the other side of :func:`humans.save.bids.last_position`: the run's clock
runs out in a phase that is not a game, after a game block that finished. The
next run must start at that phase, from its beginning, with the curriculum's
``include_on_resume`` opening phases first, and must not play the finished
block again.

The curriculum is built so the clock cannot land anywhere else: a 2 s game
block (``mode: duration``) and then a message far longer than what is left of
the run. The game is CartPole on the ``gym`` backend, which the core CI job plays.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from humans.save.logging import read_events
from tests.backends import REPO
from tests.conftest import write_curriculum

pytestmark = pytest.mark.backend("gym")

#: Seconds each run lasts; its phases stop 1 s before (the closing fixation).
RUN_S = 15.0
#: Seconds the message would last: far past the run's end, wherever the game block ends.
MESSAGE_S = 120.0
CURRICULUM = [
    {"type": "fixation", "duration": 0.5, "include_on_resume": True},
    {"type": "game", "game": "CartPole-v1", "controls": {"": 0, "LEFT": 0, "RIGHT": 1}, "fps": 50,
     "mode": "duration", "duration": 2.0, "show_outcome_duration": 0},
    {"type": "message", "text": "A message the run's clock cuts short.", "duration": MESSAGE_S},
    {"type": "fixation", "duration": 1.0},
]
GAME, MESSAGE = 1, 2


def _run(config: Path, data_root: Path, number: int, *more: str) -> tuple[Path, dict, str]:
    """Play run ``number`` of ``config`` headless; its folder, its manifest and its stderr."""
    done = subprocess.run(
        [sys.executable, "-m", "tests.headless", "--curriculum", str(config),
         "--data-root", str(data_root), "--duration", str(RUN_S), "--no-audio",
         "--run", str(number), *more],
        cwd=REPO, capture_output=True, text=True, timeout=RUN_S + 120,
        env={**os.environ, "SDL_VIDEODRIVER": "dummy"})
    assert done.returncode == 0, f"run {number} exited {done.returncode}:\n{done.stderr[-3000:]}"
    manifest = next(data_root.glob(f"sub-test/ses-001/beh/*_run-{number:03d}/manifest.json"))
    return manifest.parent, json.loads(manifest.read_text()), done.stderr


def test_resume_between_phases(tmp_path):
    config = write_curriculum(CURRICULUM, tmp_path / "between.json")
    run, first, _ = _run(config, tmp_path / "data", 1)
    by_index = {p["index"]: p for p in first["phases"]}
    events = read_events(str(run / by_index[GAME]["data_dir"].split("/")[-1]))
    assert not events[-1]["block_did_not_finish"], "the game block should finish in run 1"
    message = by_index[MESSAGE]
    assert message["offset"] - message["onset"] < MESSAGE_S - 1, (
        "the run's clock should have cut the message short")
    assert set(by_index) == {0, GAME, MESSAGE, None}, f"run 1 played {sorted(by_index, key=str)}"

    _, second, stderr = _run(config, tmp_path / "data", 2, "--resume")
    assert f"at phase {MESSAGE}" in stderr and "mid-block" not in stderr, stderr[-2000:]
    resumed_from = second["resumed_from"]
    assert resumed_from["index"] == MESSAGE, resumed_from
    assert "block_time_elapsed" not in resumed_from, "a finished block has nothing to carry on"
    played = [p["index"] for p in second["phases"]]
    assert played[:2] == [0, MESSAGE], f"run 2 played {played}"
    assert GAME not in played, "run 2 played the finished game block again"
    assert second["phases"][1]["text"] == CURRICULUM[MESSAGE]["text"]
