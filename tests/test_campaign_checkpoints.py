"""Campaign selection uses existing logs, across completed and interrupted runs."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from humans.save.campaign import latest_checkpoint
from humans.save.logging import _json_default


class CampaignCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.phase = {"type": "game", "backend": "gym", "game": "MountainCar-v0"}

    def recording(self, session=1, subject="sub-test", task="task", state=b"first",
                  outcome="playing", closed=True, continuation=None, index=0):
        label = f"{subject}_ses-{session:03d}_task-{task}_run-001"
        folder = self.root / subject / f"ses-{session:03d}" / "beh" / label
        block = folder / f"block-{index:02d}_game"
        block.mkdir(parents=True)
        manifest = {"start_epoch": session, "run": {"label": label}}
        (folder / "manifest.json").write_text(json.dumps(manifest))
        extra = {"state": state}
        if continuation is not None:
            extra["continuation_state"] = continuation
        events = [{"type": "block_start", "block_index": index, "phase": self.phase},
                  {"type": "episode_end", "outcome": outcome}]
        if closed:
            events.append({"type": "block_end", "extra": extra,
                           "block_did_not_finish": False})
        (block / "events.jsonl").write_text(
            "\n".join(json.dumps(event, default=_json_default) for event in events) + "\n")
        return block

    def checkpoint(self, subject="sub-test", task="task", index=0):
        return latest_checkpoint(str(self.root), subject, task + ".json", index)

    def test_completed_scan_continues_latest_game_across_sessions(self):
        self.recording()
        block = self.recording(session=2, state=b"second")
        checkpoint = self.checkpoint()
        self.assertEqual(checkpoint.state, b"second")
        self.assertEqual(checkpoint.block, str(block))
        self.assertEqual(checkpoint.phase, self.phase)
        self.assertEqual(checkpoint.record()["block"], str(block))
        self.assertNotIn("state", checkpoint.record())

    def test_subject_task_and_phase_are_isolated(self):
        self.recording()
        self.recording(session=9, subject="sub-other", state=b"other subject")
        self.recording(session=8, task="different", state=b"other task")
        self.recording(session=7, index=1, state=b"other phase")
        self.assertEqual(self.checkpoint().state, b"first")

    def test_missing_campaign_requires_explicit_creation(self):
        with self.assertRaisesRegex(ValueError, "--new-campaign"):
            self.checkpoint()

    def test_unclosed_latest_block_never_falls_back(self):
        self.recording()
        self.recording(session=2, closed=False)
        with self.assertRaisesRegex(ValueError, "did not close cleanly"):
            self.checkpoint()

    def test_empty_latest_block_never_falls_back(self):
        self.recording()
        block = self.recording(session=2)
        (block / "events.jsonl").write_text("")
        with self.assertRaisesRegex(ValueError, "did not close cleanly"):
            self.checkpoint()

    def test_missing_latest_state_never_falls_back(self):
        self.recording()
        self.recording(session=2, state=None)
        with self.assertRaisesRegex(ValueError, "no saved state"):
            self.checkpoint()

    def test_terminal_episode_requires_new_campaign(self):
        self.recording(outcome="terminated")
        with self.assertRaisesRegex(ValueError, "episode.*ended"):
            self.checkpoint()

    def test_native_save_can_supply_pre_exit_continuation(self):
        self.recording(state=b"finished", outcome="terminated", continuation=b"before save")
        self.assertEqual(self.checkpoint().state, b"before save")

    def test_escape_continues_without_changing_source_recording(self):
        block = self.recording(outcome="quit")
        before = (block / "events.jsonl").read_bytes()
        self.assertEqual(self.checkpoint().state, b"first")
        self.assertEqual((block / "events.jsonl").read_bytes(), before)

    def test_partial_last_line_is_not_a_completed_block(self):
        block = self.recording(closed=False)
        with (block / "events.jsonl").open("a") as output:
            output.write('{"type":"block_end"')
        with self.assertRaisesRegex(ValueError, "did not close cleanly"):
            self.checkpoint()


if __name__ == "__main__":
    unittest.main()
