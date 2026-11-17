"""Live native-engine checks, also runnable with stdlib unittest without Gym/SDL.

NETHACK_GAMEPAD_HOME must point at an explicitly built runtime. A missing engine
is a failure: these checks must never silently substitute a simulated game.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


PACKAGE = Path(__file__).resolve().parents[1] / "nethack_gamepad_gym"


def module(name):
    specification = importlib.util.spec_from_file_location("native_test_" + name,
                                                         PACKAGE / (name + ".py"))
    result = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = result
    specification.loader.exec_module(result)
    return result


build = module("build")
NativeSession = module("client").NativeSession
Controller = module("controller").Controller
Renderer = module("rendering").Renderer


class NativeTest(unittest.TestCase):
    def setUp(self):
        self.runtime = build.default_home() / "runtime"
        self.sessions = []

    def tearDown(self):
        for session in self.sessions:
            session.close()

    def session(self, persistent=None, seed=1000):
        session = NativeSession(self.runtime, persistent)
        self.sessions.append(session)
        session.start(seed)
        return session

    def start_character(self, session):
        session.answer("sam".encode().hex())
        controller = Controller(session)
        self.assertEqual(controller.kind, "question")
        controller.press(4)  # native default: choose a character manually
        self.assertIn("role", controller.title)
        self.assertTrue(any("Wizard" in entry.label for entry in controller.entries()))
        controller.press(4)
        self.assertIn("race", controller.title)
        controller.press(4)
        self.assertIn("gender", controller.title)
        controller.press(4)
        self.assertIn("alignment", controller.title)
        controller.press(4)
        self.assertEqual(controller.kind, "ack")
        self.assertIn("Quetzalcoatl", " ".join(controller.document_lines))
        controller.press(4)
        self.assertEqual(controller.kind, "key")
        return controller

    def test_native_selection_introduction_and_tiles(self):
        session = self.session()
        controller = self.start_character(session)
        self.assertIn("Archeologist", " ".join(session.messages))
        visible = [tile for row in session.tiles for tile in row if tile >= 0]
        self.assertGreater(len(visible), 20)
        renderer = Renderer(self.runtime)
        for tile in visible:
            self.assertEqual(renderer._tile(tile).size, (16, 16))
        frame = renderer.draw(session, controller)
        self.assertEqual(frame.shape, (720, 1280, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def test_native_map_cursor_and_controller_targeting(self):
        session = self.session()
        controller = self.start_character(session)
        session.answer(str(ord(";")))  # Native quick-look command enters getpos().
        controller = Controller(session)
        self.assertEqual(controller.kind, "position")
        window = next(window for window in session.windows.values() if window["type"] == 3)
        x, y = window["column"], window["row"]
        self.assertEqual((x, y), (session.request["x"], session.request["y"]))
        controller.press(3)
        self.assertEqual(window["column"], max(1, x - 1))
        frame = Renderer(self.runtime).draw(session, controller)
        left, top = 8 + (window["column"] - 1) * 16, 56 + y * 16
        np.testing.assert_array_equal(frame[top, left:left + 16], np.full((16, 3), 255))
        turn = session.request["turn"]
        controller.press(5)
        self.assertEqual(controller.last_reply, "27")
        self.assertEqual(controller.kind, "key")
        self.assertEqual(session.request["turn"], turn)
        session.answer(str(ord(";")))
        controller = Controller(session)
        controller.press(4)
        self.assertEqual(controller.last_reply, str(ord(".")))
        self.assertEqual(controller.kind, "key")
        self.assertEqual(session.request["turn"], turn)

    def test_inventory_and_commands_preserve_map_and_turn(self):
        session = self.session()
        controller = self.start_character(session)
        tiles = np.array(session.tiles)
        turn = session.request["turn"]
        controller.press(7)
        self.assertEqual(controller.kind, "menu")
        self.assertIn("bullwhip", " ".join(controller.document_lines))
        controller.press(2)
        self.assertGreater(controller.scroll, 0)
        controller.press(6)  # must not send a keyboard byte to select_menu
        self.assertIsNone(controller.last_reply)
        controller.press(5)
        self.assertEqual(controller.kind, "key")
        for action in (6, 2, 2, 0, 5):
            controller.press(action)
            self.assertIsNone(controller.last_reply)
        self.assertEqual(session.request["turn"], turn)
        np.testing.assert_array_equal(tiles, session.tiles)
        controller.press(5)  # search is a real turn
        self.assertGreater(session.request["turn"], turn)

    def test_native_object_choice_and_direction_callbacks(self):
        session = self.session()
        controller = self.start_character(session)
        session.answer(str(ord("e")))
        controller = Controller(session)
        self.assertEqual(controller.kind, "menu")
        self.assertIn("food ration", " ".join(entry.label for entry in controller.entries()))
        controller.press(5)
        self.assertEqual(controller.kind, "key")
        session.answer(str(ord("o")))
        controller = Controller(session)
        self.assertEqual(controller.kind, "direction")
        self.assertEqual(controller.entries()[controller.cursor].label, "Here")
        controller.press(5)
        self.assertEqual(controller.kind, "key")

    def test_real_save_restore_and_disconnect_save(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            session = self.session(directory, seed=None)
            self.start_character(session)
            snapshot = session.snapshot()
            session.answer(str(ord("S")))
            self.assertEqual(session.request["kind"], "question")
            session.answer(str(ord("y")))
            self.assertEqual(session.request["kind"], "ack")
            session.answer("13")
            self.assertTrue(session.finished)
            self.assertTrue(list((directory / "save").iterdir()))
            session.start()
            session.answer("sam".encode().hex())
            self.assertEqual(session.request["kind"], "key")
            self.assertEqual(session.request["hp"], snapshot["request"]["hp"])
            np.testing.assert_array_equal(session.tiles, snapshot["tiles"])
            session.close()  # native EOF/hangup path saves and clears locks
            session.start()
            session.answer("sam".encode().hex())
            self.assertEqual(session.request["kind"], "key")
            self.assertIn("welcome back", " ".join(session.messages))

    def test_controller_only_name_and_replay_across_processes(self):
        first = self.session()
        controller = Controller(first)
        actions = []

        def press(action):
            actions.append(action)
            controller.press(action)

        # Navigate the actual DS keyboard with controller presses only.
        for code in (ord("a"), 10):
            for _ in range(100):
                if controller.entries()[controller.cursor].value == code:
                    break
                press(1)
            else:
                self.fail(f"DS keyboard is missing key {code}")
            press(4)
        for _ in range(6):
            press(4)
        self.assertEqual(controller.kind, "key")
        for action in (6, 2, 2, 5, 7, 2, 4, 5, 5, 6, 6, 2):
            press(action)
        second = self.session()
        replay = Controller(second)
        for action in actions:
            replay.press(action)
        self.assertEqual(first.snapshot(), second.snapshot())
        self.assertEqual(controller.state(), replay.state())
        renderer = Renderer(self.runtime)
        np.testing.assert_array_equal(renderer.draw(first, controller),
                                      renderer.draw(second, replay))
        # The log's observation state must survive JSON serialization unchanged.
        self.assertEqual(first.snapshot(), json.loads(json.dumps(first.snapshot())))

    def test_source_reuse_and_matching_command_data(self):
        source = build.source_root()
        commands = (source / "sys/nds/arm9/src/nds_cmd.c").read_text()
        self.assertEqual(build.ds_commands(commands),
                         json.loads((self.runtime / "commands.json").read_text()))
        self.assertEqual(build.ds_keyboard(source),
                         json.loads((self.runtime / "keyboard.json").read_text()))
        selection = (source / "sys/nds/arm9/src/nds_win.c").read_text()
        self.assertEqual(build.function_source(selection, "void nds_player_selection()"),
                         (build.default_home() / "source/src/nds_selection.inc").read_text())

    def test_native_message_history(self):
        session = self.session()
        self.start_character(session)
        session.answer("16")  # native ^P
        controller = Controller(session)
        self.assertEqual(controller.kind, "ack")
        self.assertIn("welcome to NetHack", " ".join(controller.document_lines))
        controller.press(4)
        self.assertEqual(controller.kind, "key")

    def test_name_prompt_ignores_system_username(self):
        for username, login in (("Snastase", "Snastase"), ("other-user", "other-user"),
                                ("", "Snastase"), ("researcher", "different-login")):
            with self.subTest(username=username, login=login):
                with patch.dict(os.environ, {"USER": username, "LOGNAME": login}):
                    session = self.session()
                self.assertEqual(session.request["kind"], "text")
                self.assertEqual(session.request["prompt"], "Who are you?")
                session.answer("Ada".encode().hex())
                self.assertEqual(session.request["kind"], "question")
                self.assertIn("pick a character", session.request["prompt"])

    def test_named_native_save_restores_after_username_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {"USER": "Snastase", "LOGNAME": "Snastase"}):
                session = self.session(Path(temporary))
            session.answer("Ada".encode().hex())
            controller = Controller(session)
            for _ in range(6):
                controller.press(4)
            self.assertIn("Ada", " ".join(session.messages))
            session.answer(str(ord("S")))
            session.answer(str(ord("y")))
            session.answer("13")
            self.assertTrue(session.finished)
            with patch.dict(os.environ, {"USER": "another-account", "LOGNAME": "another-account"}):
                session.start(1000)
            self.assertEqual(session.request["kind"], "text")
            session.answer("Ada".encode().hex())
            self.assertEqual(session.request["kind"], "key")
            self.assertIn("welcome back", " ".join(session.messages))
            self.assertIn("Ada", " ".join(session.messages))

    def test_named_controller_replay_across_username_settings(self):
        outcomes = []
        for username in ("Snastase", "different-account"):
            with patch.dict(os.environ, {"USER": username, "LOGNAME": username}):
                session = self.session()
            controller = Controller(session)
            for code in (ord("a"), ord("d"), ord("a"), 10):
                for _ in range(100):
                    if controller.entries()[controller.cursor].value == code:
                        break
                    controller.press(1)
                else:
                    self.fail("Native keyboard key unreachable")
                controller.press(4)
            for _ in range(6):
                controller.press(4)
            for action in (5, 6, 2, 5, 7):
                controller.press(action)
            outcomes.append((session.snapshot(), controller.state(),
                             Renderer(self.runtime).draw(session, controller)))
        self.assertEqual(outcomes[0][:2], outcomes[1][:2])
        np.testing.assert_array_equal(outcomes[0][2], outcomes[1][2])


if __name__ == "__main__":
    unittest.main()
