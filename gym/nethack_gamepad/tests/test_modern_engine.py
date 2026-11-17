"""Actual production deaths and isolated driver checks for native revival edge cases."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from test_native import build, NativeSession, Controller, Renderer


def enter_world(session):
    session.start(1000)
    session.answer("sam".encode().hex())
    controller = Controller(session)
    for _ in range(6):
        controller.press(4)
    if controller.kind != "key":
        raise AssertionError("Character creation did not reach the dungeon")
    return controller


def acknowledge(session):
    for _ in range(20):
        if session.finished or session.request["kind"] == "key":
            return
        session.answer("13" if session.request["kind"] == "ack" else "27")
    raise AssertionError("Native game did not return from its message acknowledgement")


def inventory(session):
    session.answer(str(ord("i")))
    controller = Controller(session)
    result = controller.document_lines[:]
    controller.press(5)
    return result


def driver_session(test, mode="modern", player=None):
    directory = Path(os.environ.get("NETHACK_GAMEPAD_TEST_HOME",
                     str(build.default_home()) + "-tests"))
    session = NativeSession(directory / "runtime", player, game_mode=mode,
                            allow_test_driver=True)
    test.addCleanup(session.close)
    test.assertTrue(session.manifest.get("test_driver"),
                    "Build the isolated --test-driver runtime explicitly")
    return session


class ModernEngineTest(unittest.TestCase):
    def test_real_monster_death_in_production_engine(self):
        session = NativeSession(build.default_home() / "runtime", game_mode="modern")
        try:
            controller = enter_world(session)
            for _ in range(1500):
                controller.press(5 if controller.kind == "key" else 4)
                if any(event["event"] == "revival" for event in session.game_events):
                    break
            else:
                self.fail("The production character did not encounter a lethal hazard")
            events = session.game_events[:]
            self.assertEqual([event["event"] for event in events], ["death", "revival"])
            self.assertNotEqual(events[0]["cause"], "desktop test hazard")
            self.assertFalse(session.finished)
            self.assertGreater(events[1]["hp"], 0)
            self.assertEqual(session.revivals, 1)
            self.assertEqual(events[1]["level"], events[0]["level"])
            self.assertEqual(events[1]["checkpoint"], session.checkpoint)
            self.assertEqual(Renderer(session.runtime).draw(session, controller).shape,
                             (720, 1280, 3))
        finally:
            session.close()

    def driver_session(self, mode="modern", player=None):
        return driver_session(self, mode, player)

    def test_same_level_revival_preserves_inventory_and_character(self):
        session = self.driver_session()
        enter_world(session)
        before = inventory(session)
        hpmax = session.request["maxhp"]
        session.answer("250")
        events = session.game_events[:]
        acknowledge(session)
        self.assertEqual(inventory(session), before)
        self.assertEqual(session.request["maxhp"], hpmax)
        self.assertEqual(session.request["hp"], hpmax)
        self.assertEqual(session.score_divisor, 9)
        self.assertEqual(events[1]["sequence"], events[0]["sequence"] + 1)

    def test_cross_level_stoning_and_repeated_fatal_conditions(self):
        session = self.driver_session()
        enter_world(session)
        before = inventory(session)
        for count, hazard in enumerate((252, 253, 254, 255, 256), start=1):
            session.answer("251")  # native level travel to the second floor
            session.answer(str(hazard))
            events = session.game_events[:]
            self.assertEqual(events[0]["level"], 2)
            self.assertEqual(events[-1]["level"], 2)
            self.assertEqual(events[-1]["revivals"], count)
            acknowledge(session)
            self.assertFalse(session.finished)
            self.assertEqual(session.revivals, count)
            self.assertEqual(session.score_divisor, (9, 27, 54, 81, 108)[count - 1])
        self.assertTrue(set(before).issubset(inventory(session)))
        for _ in range(5):
            session.answer(str(ord("s")))
            acknowledge(session)
        self.assertEqual(session.revivals, 5, "Cured sickness/slime must not immediately kill again")
        session.answer("251")
        for _ in range(2000):
            session.answer("115" if session.request["kind"] == "key" else "13")
            if any(event["event"] == "revival" for event in session.game_events):
                break
        else:
            self.fail("No real lethal event occurred on the second level")
        self.assertEqual(session.game_events[0]["level"], 2)
        self.assertEqual(session.game_events[-1]["level"], 2)
        self.assertNotEqual(session.game_events[0]["cause"], "desktop test hazard")

    def test_classic_death_still_ends_the_game(self):
        session = self.driver_session(mode="classic")
        enter_world(session)
        session.answer("250")
        for _ in range(30):
            if session.finished:
                break
            if session.request["kind"] == "question":
                session.answer(str(ord("n")))
            elif session.request["kind"] == "menu":
                session.answer("0")
            else:
                session.answer("13")
        self.assertTrue(session.finished)
        self.assertEqual(session.revivals, 0)

    def test_native_save_restores_mode_and_revival_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            session = self.driver_session(player=Path(temporary))
            enter_world(session)
            session.answer("250")
            acknowledge(session)
            tiles = np.array(session.tiles)
            checkpoint = session.checkpoint.copy()
            session.answer(str(ord("S")))
            session.answer(str(ord("y")))
            acknowledge(session)
            self.assertTrue(session.finished)
            session.start(1000)
            session.answer("sam".encode().hex())
            self.assertEqual(session.mode, "modern")
            self.assertEqual(session.revivals, 1)
            self.assertEqual(session.checkpoint, checkpoint)
            np.testing.assert_array_equal(session.tiles, tiles)

    def test_test_runtime_is_rejected_by_normal_client(self):
        directory = Path(os.environ.get("NETHACK_GAMEPAD_TEST_HOME",
                         str(build.default_home()) + "-tests"))
        with self.assertRaisesRegex(RuntimeError, "participant"):
            NativeSession(directory / "runtime")


if __name__ == "__main__":
    unittest.main()
