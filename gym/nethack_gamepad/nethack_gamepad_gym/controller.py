"""Four directions and four buttons over native menus and the DS command table."""

from __future__ import annotations

from dataclasses import dataclass
import json


@dataclass(frozen=True)
class Entry:
    label: str
    value: int | str
    enabled: bool = True
    width: int = 16


class Controller:
    """Translate physical presses; NetHack owns the prompts, menu choices and rules."""

    def __init__(self, session) -> None:
        self.session = session
        self.keyboard = json.loads((session.runtime / "keyboard.json").read_text())
        self.local_mode = ""
        self.cursor = self.scroll = 0
        self.selected: set[int] = set()
        self.typed = ""
        self.shift = self.caps = self.ctrl = self.alt = False
        self.last_reply: str | None = None
        self._sync()

    def _sync(self) -> None:
        self.local_mode = ""
        self.cursor = self.scroll = 0
        self.selected = set()
        self.typed = ""
        self.shift = self.caps = self.ctrl = self.alt = False
        if self.kind == "menu":
            self.selected = {item["index"] for item in self.window["items"]
                             if item["selected"] and item["selectable"]}
        entries = self.entries()
        self.cursor = next((i for i, item in enumerate(entries) if item.enabled), 0)
        if self.kind == "question":
            self.cursor = next((i for i, item in enumerate(entries)
                                if item.value == self.session.request.get("default")), self.cursor)
        elif self.kind == "direction":
            self.cursor = 4

    @property
    def kind(self) -> str:
        return self.local_mode or self.session.request.get("kind", "finished")

    @property
    def window(self) -> dict:
        return self.session.windows.get(self.session.request.get("window"),
                                        {"items": [], "lines": [], "prompt": ""})

    @property
    def title(self) -> str:
        if self.kind == "commands":
            return "Commands"
        if self.kind in ("keyboard", "text"):
            return self.session.request.get("prompt", "") or "Keyboard"
        return self.window["prompt"] or self.session.request.get("prompt", "")

    @property
    def document_lines(self) -> list[str]:
        return self.window["lines"] or [item["text"] for item in self.window["items"]]

    def key_code(self, code: int) -> int:
        # Same modifier order as nds_kbd.c:kbd_mod_code.
        if 97 <= code <= 122 and self.caps != self.shift:
            code -= 32
        elif self.shift:
            shifts = self.keyboard["shifts"]
            position = shifts.find(chr(code))
            if position >= 0 and position % 2 == 0:
                code = ord(shifts[position + 1])
        if self.alt:
            code |= 128
        if self.ctrl:
            code &= 31
        return code

    def keyboard_rows(self) -> list[list[Entry]]:
        labels = {8: "Bksp", 9: "Tab", 10: "Enter", 32: "Space",
                  "K_CAPS": "Caps", "K_SHIFT": "Shift", "K_CTRL": "Ctrl", "K_ALT": "Alt"}
        return [[Entry(labels.get(key["code"], chr(self.key_code(key["code"]))
                                  if isinstance(key["code"], int) else key["code"]),
                       key["code"], width=key["width"]) for key in row]
                for row in self.keyboard["rows"]]

    def entries(self) -> list[Entry]:
        kind = self.kind
        if kind == "commands":
            return [Entry(item["label"], item["key"]) for item in self.session.commands]
        if kind in ("keyboard", "text"):
            return [entry for row in self.keyboard_rows() for entry in row]
        if kind == "menu":
            entries = [Entry(item["text"], item["index"], item["selectable"])
                       for item in self.window["items"]]
            if self.session.request["how"] == 2:
                entries.append(Entry("Finish selection", "finish"))
            return entries
        if kind == "direction":
            return [Entry(label, ord(key)) for label, key in
                    zip(("NW", "N", "NE", "W", "Here", "E", "SW", "S", "SE",
                         "Up", "Down", "Cancel"), "ykuh.lbjn<>\x1b")]
        if kind == "question":
            choices = self.session.request.get("choices", "").split("\x1b")[0]
            labels = {"y": "Yes", "n": "No", "q": "Quit / cancel", "a": "All"}
            return [Entry(labels.get(char, char), ord(char)) for char in dict.fromkeys(choices)]
        return []

    def _send(self, reply: str) -> None:
        self.last_reply = reply
        self.session.answer(reply)
        self._sync()

    def press(self, action: int) -> None:
        """Accept up/right/down/left/Z/X/C/V as actions 0..7, without hidden macros."""
        if not 0 <= int(action) < 8:
            raise ValueError("Expected a controller action from 0 through 7")
        self.last_reply = None
        kind = self.kind
        if kind == "finished":
            return
        if action == 6:
            if self.session.request.get("kind") in ("menu", "ack"):
                return
            self.local_mode = "commands" if kind == "key" else "keyboard"
            self.cursor = 0
            return
        if self.session.request.get("how") == 0 or kind == "ack":
            if action < 4:
                self.scroll = max(0, min(self.scroll + (-1, 10, 1, -10)[action],
                                         max(0, len(self.document_lines) - 1)))
            elif action in (4, 5):
                self._send("0" if kind == "menu" else "13")
            return
        if action == 5:
            if self.local_mode:
                self.local_mode = ""
                self.cursor = 0
            else:
                self._send("115" if kind == "key" else "-1" if kind == "menu"
                           else "1b" if kind == "text" else "27")
            return
        if kind == "position":
            if action < 4:
                self._send(str(ord("kljh"[action])))
            elif action == 4:
                self._send(str(ord(".")))
            return
        if kind == "key":
            if action < 4:
                self._send(str(ord("kljh"[action])))
            elif action in (4, 7):
                self._send(str(ord("," if action == 4 else "i")))
            return
        if action < 4:
            self._move(action)
        elif action == 4:
            self._confirm()

    def _move(self, action: int) -> None:
        entries = self.entries()
        if not entries:
            return
        columns = 3 if self.kind == "direction" else 1
        offset = (-columns, 1, columns, -1)[action]
        if self.kind in ("keyboard", "text") and action in (0, 2):
            rows = self.keyboard_rows()
            starts = [sum(len(row) for row in rows[:index]) for index in range(len(rows))]
            row = max(index for index, start in enumerate(starts) if start <= self.cursor)
            column = self.cursor - starts[row]
            target = (row + (-1 if action == 0 else 1)) % len(rows)
            self.cursor = starts[target] + min(column, len(rows[target]) - 1)
            return
        for _ in entries:
            self.cursor = (self.cursor + offset) % len(entries)
            if entries[self.cursor].enabled:
                return

    def _confirm(self) -> None:
        entries = self.entries()
        if not entries:
            if self.kind == "menu":
                self._send("0")
            return
        entry = entries[self.cursor]
        if not entry.enabled:
            return
        if self.kind in ("keyboard", "text"):
            self._keyboard(entry.value)
        elif self.kind == "menu":
            if self.session.request["how"] == 2:
                if entry.value == "finish":
                    pairs = " ".join(f"{index} -1" for index in sorted(self.selected))
                    self._send(f"{len(self.selected)} {pairs}")
                elif entry.value in self.selected:
                    self.selected.remove(entry.value)
                else:
                    self.selected.add(entry.value)
            else:
                self._send(f"1 {entry.value} -1")
        else:
            self._send(str(entry.value))

    def _keyboard(self, code: int | str) -> None:
        if isinstance(code, str):
            attribute = {"K_SHIFT": "shift", "K_CAPS": "caps",
                         "K_CTRL": "ctrl", "K_ALT": "alt"}[code]
            setattr(self, attribute, not getattr(self, attribute))
            return
        code = self.key_code(code)
        if self.session.request["kind"] == "text":
            if code in (10, 13):
                self._send(self.typed.encode("latin-1").hex())
            elif code == 8:
                self.typed = self.typed[:-1]
            elif len(self.typed) < 255:
                self.typed += chr(code)
        else:
            self._send(str(code))
        self.shift = self.ctrl = self.alt = False

    def state(self) -> dict:
        """Return the visible controller state, including pending text and selections."""
        return {"mode": self.kind, "cursor": self.cursor, "scroll": self.scroll,
                "selected": sorted(self.selected), "typed": self.typed,
                "shift": self.shift, "caps": self.caps, "ctrl": self.ctrl, "alt": self.alt}
