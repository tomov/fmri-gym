"""Automatic campaign starts use the real loop, existing state logs and replay."""

import json
import os
import subprocess
import sys
from pathlib import Path

from humans.save.logging import read_events
from tests import verify_replay


def test_new_scan_continues_completed_game_with_a_fresh_clock(tmp_path):
    config = tmp_path / "campaign.json"
    phase = {"type": "game", "game": "MountainCar-v0", "fps": 5,
             "controls": {"": 1, "LEFT": 0, "RIGHT": 2},
             "duration": 1.5, "show_outcome_duration": 0, "continue_campaign": True}
    config.write_text(json.dumps({"curriculum": [phase]}))
    data = tmp_path / "data"

    def launch(number, *flags):
        tap = tmp_path / f"tap-{number}.json"
        done = subprocess.run(
            [sys.executable, "-m", "tests.headless", "--curriculum", str(config),
             "--data-root", str(data), "--duration", "3", "--end-fixation", "0.1",
             "--no-audio", "--run", str(number), "--tap", str(tap), *flags],
            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
            timeout=120, env={**os.environ, "SDL_VIDEODRIVER": "dummy"})
        assert done.returncode == 0, done.stderr
        manifest_path = next(data.glob(
            f"sub-test/ses-001/beh/*_run-{number:03d}/manifest.json"))
        manifest = json.loads(manifest_path.read_text())
        game = next(phase for phase in manifest["phases"] if phase["type"] == "game")
        events = read_events(str(manifest_path.parent / game["data_dir"]))
        return manifest_path.parent, manifest, events, json.loads(tap.read_text())

    first_run, first, first_events, first_tap = launch(1, "--new-campaign")
    second_run, second, second_events, second_tap = launch(2)
    assert second["campaign"]["sources"]["0"]["run"] == first["run"]["label"]
    assert first_tap["draws"][-1] == second_tap["draws"][0]
    start = next(event for event in second_events if event["type"] == "episode_start")
    assert start["initial_state"] == first_events[-1]["extra"]["state"]
    assert first_events[-1]["block_did_not_finish"] is False
    assert 1.0 < second_events[-1]["block_time_elapsed"] < 2.5
    replay = verify_replay.check_in_second_process(str(second_run))
    assert replay.returncode == 0, replay.stdout + replay.stderr
    _, third, third_events, _ = launch(3, "--new-campaign")
    assert third["campaign"] == {"new": True, "sources": {}}
    assert next(event for event in third_events if event["type"] == "episode_start")[
        "initial_state"] is None
    assert json.loads((first_run / "manifest.json").read_text()) == first


def test_campaign_policy_validation():
    from fmri_play import _parser
    from humans.config.curriculum import Curriculum
    from humans.config.phases import GamePhase
    from humans.config.rig import RigConfig
    from humans.config.run import RunConfig

    phase = {"type": "game", "game": "MountainCar-v0", "fps": 5,
             "controls": {"": 1}, "continue_campaign": True}
    assert GamePhase.problems({**phase, "continue_campaign": "yes"})
    parser = _parser()
    args = parser.parse_args(["--ses", "1", "--run", "1", "--rig", "unused.json",
                              "--curriculum", "unused.json", "--new-campaign", "--resume"])
    curriculum = Curriculum.from_dict({"curriculum": [phase]}, "campaign.json")
    rig = RigConfig.open("configs/examples/rigs/laptop.json")
    config = RunConfig.from_args(args, curriculum, rig)
    assert any("cannot be combined" in problem for problem in config.problems())
