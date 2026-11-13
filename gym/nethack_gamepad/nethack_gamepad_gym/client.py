"""Process transport for native NetHack window events; no terminal scraping."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading


class NativeSession:
    """Run the actual game until its next native request for user input."""

    def __init__(self, runtime: Path, player_directory: Path | None = None,
                 game_mode: str = "classic", allow_test_driver: bool = False) -> None:
        self.runtime = Path(runtime).resolve()
        if not (self.runtime / "build.json").is_file():
            raise FileNotFoundError("Native engine is not built; run nethack-gamepad-build")
        self.manifest = json.loads((self.runtime / "build.json").read_text())
        if self.manifest["revision"] != "3.6.0-desktop-4-cursor":
            raise RuntimeError("This backend needs a new runtime: run nethack-gamepad-build")
        if self.manifest.get("test_driver") and not allow_test_driver:
            raise RuntimeError("Test-driver builds cannot be used for participant play")
        if game_mode not in ("classic", "modern"):
            raise ValueError("game_mode must be classic or modern")
        self.game_mode = game_mode
        self.mode, self.revivals = game_mode, 0
        self.score_divisor = 3 if game_mode == "modern" else 1
        self.game_events: list[dict] = []
        self.event_sequence = 0
        self.checkpoint = None
        self.exit_text = ""
        self.player_directory = player_directory
        self.process: subprocess.Popen | None = None
        self._temporary = None
        self._stderr = None
        self.windows: dict[int, dict] = {}
        self.messages: list[str] = []
        self.tiles = [[-1] * 79 for _ in range(21)]
        self.glyphs = [[-1] * 79 for _ in range(21)]
        self.pets = [[False] * 79 for _ in range(21)]
        self.request: dict = {}
        self.finished = False
        self.events = queue.Queue()
        self.commands = json.loads((self.runtime / "commands.json").read_text())

    def start(self, seed: int | None = None) -> dict:
        """Start native character selection or restore, with no predefined role."""
        self.close()
        self.mode, self.revivals = self.game_mode, 0
        self.score_divisor = 3 if self.game_mode == "modern" else 1
        self.game_events, self.event_sequence = [], 0
        self.checkpoint = None
        self.exit_text = ""
        self.windows, self.messages = {}, []
        self.tiles = [[-1] * 79 for _ in range(21)]
        self.glyphs = [[-1] * 79 for _ in range(21)]
        self.pets = [[False] * 79 for _ in range(21)]
        self.finished, self.request, self.events = False, {}, queue.Queue()
        if self.player_directory is None:
            self._temporary = tempfile.TemporaryDirectory(prefix="nethack-gamepad-")
            directory = Path(self._temporary.name)
        else:
            directory = Path(self.player_directory).resolve()
            directory.mkdir(parents=True, exist_ok=True)
        for name in ("nhdat", "symbols", "license"):
            destination = directory / name
            if not destination.exists():
                shutil.copy2(self.runtime / name, destination)
        (directory / "save").mkdir(exist_ok=True)
        for name in ("perm", "record", "logfile", "xlogfile"):
            (directory / name).touch(exist_ok=True)
        # Isolate configuration from personal .nethackrc files; leave game options at defaults.
        configuration = directory / ".desktop.nethackrc"
        configuration.write_text("OPTIONS=windowtype:desktop\n")
        environment = dict(os.environ, NETHACKOPTIONS="@" + str(configuration), TERM="dumb",
                           NETHACK_DESKTOP_MODE=self.game_mode)
        environment.pop("FMRI_NETHACK_SEED", None)
        environment.pop("FMRI_NETHACK_EPOCH", None)
        if seed is not None:
            environment["TZ"] = "UTC"
            environment["FMRI_NETHACK_SEED"] = str(seed)
            environment["FMRI_NETHACK_EPOCH"] = "1609459200"
        self._stderr = (directory / "desktop-stderr.txt").open("w+")
        self.process = subprocess.Popen(
            # Unix whoami() otherwise silently chooses USER/LOGNAME/getlogin().
            # Native -u with an empty name requests NetHack's own askname callback.
            [str(self.runtime / "nethack"), "-d", str(directory), "-u", ""], cwd=directory,
            env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._stderr, text=True, encoding="utf-8", bufsize=1,
        )
        events = self.events
        process = self.process

        def read_events() -> None:
            for line in process.stdout:
                events.put(line)
            events.put(None)

        threading.Thread(target=read_events, daemon=True).start()
        return self._drain()

    def answer(self, response: str) -> dict:
        """Answer the current native input callback and wait for the next callback."""
        if self.finished or self.process is None:
            raise RuntimeError("No active native input request")
        if "\n" in response or "\r" in response:
            raise ValueError("Native replies must be one protocol line")
        self.game_events = []
        self.process.stdin.write(response + "\n")
        self.process.stdin.flush()
        return self._drain()

    def _drain(self) -> dict:
        while True:
            try:
                line = self.events.get(timeout=30)
            except queue.Empty as error:
                raise TimeoutError("NetHack did not reach an input boundary within 30 seconds") from error
            if line is None:
                code = self.process.wait(timeout=2)
                if code:
                    self._stderr.flush()
                    self._stderr.seek(0)
                    diagnostic = self._stderr.read()[-2000:]
                    raise RuntimeError(f"NetHack exited with code {code}: {diagnostic}")
                self.finished = True
                self.request = {"kind": "finished"}
                return self.request
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Unexpected native output: {line[:300]}") from error
            if event["event"] == "input":
                self.request = event
                return copy.deepcopy(event)
            self._event(event)

    def _event(self, event: dict) -> None:
        kind = event["event"]
        window = event.get("window")
        if kind == "mode":
            self.mode, self.revivals = event["mode"], event["revivals"]
            self.score_divisor = event["score_divisor"]
            self.checkpoint = event["checkpoint"]
        elif kind in ("checkpoint", "death", "revival", "recovery_fallback"):
            self.event_sequence += 1
            self.revivals = event["revivals"]
            self.checkpoint = event["checkpoint"]
            self.game_events.append({**event, "sequence": self.event_sequence})
        elif kind == "exit":
            self.exit_text = event["text"]
        elif kind == "create":
            self.windows[window] = {"type": event["type"], "lines": [], "items": [],
                                    "prompt": "", "row": 0, "column": 0}
        elif kind == "destroy":
            self.windows.pop(window, None)
        elif kind == "clear" and window in self.windows:
            self.windows[window]["lines"] = []
            if self.windows[window]["type"] == 3:
                self.tiles = [[-1] * 79 for _ in range(21)]
                self.glyphs = [[-1] * 79 for _ in range(21)]
                self.pets = [[False] * 79 for _ in range(21)]
        elif kind == "cursor" and window in self.windows:
            self.windows[window]["row"] = event["y"]
            self.windows[window]["column"] = event["x"]
        elif kind == "text" and window in self.windows:
            target = self.windows[window]
            if target["type"] == 2:
                row = target["row"]
                while len(target["lines"]) <= row:
                    target["lines"].append("")
                target["lines"][row] = event["text"]
            else:
                target["lines"].append(event["text"])
            if target["type"] == 1:
                self.messages.append(event["text"])
        elif kind == "message":
            self.messages.append(event["text"])
        elif kind == "menu_start":
            self.windows[window]["items"] = []
        elif kind == "menu_item":
            self.windows[window]["items"].append(event)
        elif kind == "menu_end":
            self.windows[window]["prompt"] = event["prompt"]
        elif kind == "glyph":
            x, y = event["x"] - 1, event["y"]
            if 0 <= x < 79 and 0 <= y < 21:
                self.tiles[y][x] = event["tile"]
                self.glyphs[y][x] = event["glyph"]
                self.pets[y][x] = event["pet"]

    def snapshot(self) -> dict:
        """Return independent observable window data for logging and reconstruction."""
        return copy.deepcopy({"tiles": self.tiles, "glyphs": self.glyphs, "pets": self.pets,
                              "mode": self.mode, "revivals": self.revivals,
                              "score_divisor": self.score_divisor, "checkpoint": self.checkpoint,
                              "game_events": self.game_events,
                              "windows": {str(key): value for key, value in self.windows.items()},
                              "messages": self.messages[-100:],
                              "request": self.request, "finished": self.finished})

    def close(self) -> None:
        """Stop the owned process and discard only temporary episode files."""
        if self.process is not None:
            if self.process.poll() is None:
                self.process.stdin.close()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    self.process.wait(timeout=2)
            if not self.process.stdin.closed:
                self.process.stdin.close()
            self.process.stdout.close()
            self.process = None
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
