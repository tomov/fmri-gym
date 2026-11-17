"""Level-entry recovery: native placement, hazards, branches and stable checkpoints."""

from __future__ import annotations

import unittest

import numpy as np

from test_native import build, NativeSession
from test_modern_engine import driver_session, enter_world, acknowledge


class EntryRecoveryTest(unittest.TestCase):
    def driver(self):
        # Share the explicit, isolated driver setup without inheriting its test methods.
        return driver_session(self)

    def test_no_added_altar_or_level_generation_changes(self):
        sessions = []
        try:
            for mode in ("classic", "modern"):
                session = NativeSession(build.default_home() / "runtime", game_mode=mode)
                sessions.append(session)
                enter_world(session)
            np.testing.assert_array_equal(sessions[0].tiles, sessions[1].tiles)
            np.testing.assert_array_equal(sessions[0].glyphs, sessions[1].glyphs)
            self.assertIsNone(sessions[0].checkpoint)
            self.assertEqual(sessions[1].checkpoint["reason"], "start")
        finally:
            for session in sessions:
                session.close()

    def test_level_teleport_stairs_and_retreat_update_checkpoint(self):
        session = self.driver()
        enter_world(session)
        self.assertEqual(session.checkpoint["id"], 1)
        for command, level, reason in ((251, 2, "level teleport"),
                                       (259, 3, "stairs"), (258, 1, "stairs")):
            previous = session.checkpoint["id"]
            session.answer(str(command))
            acknowledge(session)
            self.assertEqual(session.checkpoint["id"], previous + 1)
            self.assertEqual(session.checkpoint["level"], level)
            self.assertEqual(session.checkpoint["reason"], reason)
            checkpoint = session.checkpoint.copy()
            session.answer("250")
            self.assertEqual(session.game_events[-1]["level"], level)
            self.assertEqual(session.checkpoint, checkpoint)
            acknowledge(session)

    def test_same_level_teleport_does_not_update_checkpoint(self):
        session = self.driver()
        enter_world(session)
        checkpoint = session.checkpoint.copy()
        session.answer("260")
        acknowledge(session)
        self.assertEqual(session.checkpoint, checkpoint)
        self.assertFalse(any(event["event"] == "checkpoint" for event in session.game_events))
        session.answer("250")
        self.assertEqual(session.checkpoint, checkpoint)
        revival = session.game_events[-1]
        self.assertEqual((revival["x"], revival["y"]), (checkpoint["x"], checkpoint["y"]))

    def test_blocked_or_hazardous_checkpoint_uses_another_square(self):
        for obstruction in (261, 262, 263):
            with self.subTest(obstruction=obstruction):
                session = self.driver()
                enter_world(session)
                checkpoint = session.checkpoint.copy()
                session.answer(str(obstruction))
                acknowledge(session)
                session.answer("250")
                self.assertEqual(session.checkpoint, checkpoint)
                revival = session.game_events[-1]
                self.assertEqual(revival["event"], "revival")
                self.assertEqual(revival["level"], checkpoint["level"])
                self.assertNotEqual((revival["x"], revival["y"]),
                                    (checkpoint["x"], checkpoint["y"]))
                acknowledge(session)
                self.assertFalse(session.finished)

    def test_portal_entry_is_recorded_but_not_retriggered_on_recovery(self):
        session = self.driver()
        enter_world(session)
        session.answer("264")
        acknowledge(session)
        checkpoint = session.checkpoint.copy()
        self.assertEqual(checkpoint["reason"], "portal")
        self.assertEqual(checkpoint["level"], 2)
        for _ in range(2):
            session.answer("250")
            revival = session.game_events[-1]
            self.assertEqual(revival["level"], 2)
            self.assertNotEqual((revival["x"], revival["y"]),
                                (checkpoint["x"], checkpoint["y"]))
            self.assertEqual(session.checkpoint, checkpoint)
            self.assertFalse(any(event["event"] == "checkpoint" for event in session.game_events))
            acknowledge(session)

    def test_falling_arrival_records_checkpoint(self):
        session = self.driver()
        enter_world(session)
        session.answer("266")
        acknowledge(session)
        checkpoint = session.checkpoint.copy()
        self.assertEqual(checkpoint["reason"], "fall")
        self.assertEqual(checkpoint["level"], 2)
        session.answer("250")
        self.assertEqual(session.game_events[-1]["level"], 2)
        self.assertEqual(session.checkpoint, checkpoint)

    def test_lethal_stair_arrival_recovers_on_new_floor(self):
        session = self.driver()
        enter_world(session)
        session.answer("265")
        events = session.game_events[:]
        self.assertEqual([event["event"] for event in events], ["checkpoint", "death", "revival"])
        self.assertEqual([event["level"] for event in events], [2, 2, 2])
        self.assertIn("stairs", events[1]["cause"])
        self.assertEqual(session.checkpoint["reason"], "stairs")
        self.assertEqual(session.checkpoint["id"], 2)
        checkpoint = session.checkpoint.copy()
        acknowledge(session)
        self.assertEqual(session.checkpoint, checkpoint)

    def test_branch_entry_keeps_branch_identity_after_death(self):
        session = self.driver()
        enter_world(session)
        session.answer("267")
        acknowledge(session)
        checkpoint = session.checkpoint.copy()
        self.assertNotEqual(checkpoint["dungeon"], 0)
        self.assertEqual(checkpoint["reason"], "branch")
        session.answer("250")
        revival = session.game_events[-1]
        self.assertEqual(revival["dungeon"], checkpoint["dungeon"])
        self.assertEqual(revival["level"], checkpoint["level"])
        self.assertEqual(session.checkpoint, checkpoint)


if __name__ == "__main__":
    unittest.main()
