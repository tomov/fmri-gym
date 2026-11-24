"""Gym/adapter/logger integration, including a real death and a second scan start.

Requires the installed nethack_gym extra and its explicitly built runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pytest

pytestmark = pytest.mark.backend("nethack_gym")


class CampaignRecordingTest(unittest.TestCase):
    def test_recording_rejects_mutable_save_directory_before_launch(self):
        from humans.adapters.nethack_gym import NetHackGymAdapter

        with tempfile.TemporaryDirectory() as temporary:
            player = Path(temporary) / "player"
            spec = {"game": "nethack_gym:NetHackGym-v0",
                    "env_kwargs": {"player_directory": str(player)}}
            with self.assertRaisesRegex(ValueError, "isolated seeded"):
                NetHackGymAdapter(spec)
            self.assertFalse(player.exists())

    def test_death_recording_and_immutable_cross_session_reconstruction(self):
        from humans.adapters.nethack_gym import NetHackGymAdapter
        from humans.save.logging import Logger, read_events, read_frames, read_state
        from humans.save.replay import reconstruct_episode
        from humans.sync.clock import Clock
        from nethack_gym.campaign import encode_state

        spec = {"backend": "nethack_gym", "game": "nethack_gym:NetHackGym-v0",
                "controls": {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3,
                             "Z": 4, "X": 5, "C": 6, "V": 7},
                "env_kwargs": {"game_mode": "modern"}, "is_turn_based": True,
                "frame_stride": 100}
        first = NetHackGymAdapter(spec)
        self.addCleanup(first.close)
        first.reset(1000)
        engine = first.env.unwrapped
        clock = Clock()
        clock.trigger()
        with tempfile.TemporaryDirectory() as temporary:
            logger = Logger(temporary, "sub-check", [spec], clock)
            self.addCleanup(logger.close)
            name = logger.open_block(1, "nethack_gym", spec["game"], spec, 1000)
            logger.log(type="episode_start", episode_id=0, seed=1000, initial_state=None)
            hashes = {}
            step_count = 0

            def press(action):
                nonlocal step_count
                observation, reward, done, truncated, info = first.step(action)
                state = first.capture(observation, info, want_blob=step_count % 100 == 0)
                frame = first.render().copy()
                if step_count % 100 == 0:
                    hashes[step_count] = frame
                logger.log_frame({"episode_id": 0, "ep_frame": step_count,
                                  "action": action, "reward": reward,
                                  "terminated": done, "truncated": truncated,
                                  "variables": state.variables}, frame=frame, state=state.blob)
                step_count += 1
                self.assertFalse(done or truncated)
                return info

            # Character creation is part of the recorded controller stream.
            for code in (ord("s"), ord("a"), ord("m"), 10):
                for _ in range(100):
                    if engine.controller.entries()[engine.controller.cursor].value == code:
                        break
                    press(1)
                else:
                    self.fail("Native keyboard key not reachable")
                press(4)
            for _ in range(6):
                press(4)
            for _ in range(1500):
                info = press(5 if engine.controller.kind == "key" else 4)
                if any(event["event"] == "revival" for event in info["native"]["game_events"]):
                    break
            else:
                self.fail("No real death was exercised")
            expected = first.render().copy()
            variables = info["native"]
            checkpoint = first.block_extra()["state"]
            logger.close_block(extra=first.block_extra())
            logger.close()
            block = Path(temporary) / name
            events = read_events(str(block))
            death = [event for event in events if event["type"] == "frame"][-1]
            self.assertEqual([event["event"] for event in death["variables"]["native"]["game_events"]],
                             ["death", "revival"])
            self.assertEqual(read_state(events[-1]["extra"]["state"]), checkpoint)
            frames, indices = read_frames(str(block))
            for frame, index in zip(frames, indices):
                np.testing.assert_array_equal(frame, hashes[int(index)])

            replay = NetHackGymAdapter(spec)
            self.addCleanup(replay.close)
            reconstruct_episode(str(block), adapter=replay)
            np.testing.assert_array_equal(replay.render(), expected)

            # The second scan contains its starting checkpoint, not a mutable save path.
            continued_spec = json.loads(json.dumps(spec))
            continued_spec["env_kwargs"]["initial_state"] = encode_state(checkpoint)
            second = NetHackGymAdapter(continued_spec)
            self.addCleanup(second.close)
            _, resumed_info = second.reset(9999)
            np.testing.assert_array_equal(second.render(), expected)
            self.assertEqual(resumed_info["native"], variables)
            for action in (4, 5, 6, 2, 5, 5):
                second.step(action)
            end = second.render().copy()
            resumed = NetHackGymAdapter(continued_spec)
            self.addCleanup(resumed.close)
            resumed.reset(1)
            resumed.restore(second.block_extra()["state"])
            np.testing.assert_array_equal(resumed.render(), end)


if __name__ == "__main__":
    unittest.main()
