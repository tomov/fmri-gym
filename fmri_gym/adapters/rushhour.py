"""Rush Hour adapter (chrplr/Rush-Hour) -- the DBP "puzzle" pick.

Slide cars out of a 6x6 grid to free the red car. The env (rushhour_gym) wraps a
Go engine binary and renders only ANSI (a letter grid). The action space is
Discrete(32): ``action = slot * 2 + direction`` (0=left/up, 1=right/down).

Humans do not press Discrete indices. Matching the experiment's own
``rushinput.DefaultMap``, we expose a small meta-action keymap:

  * arrows / 3,4  -- select a car (spatial neighbour, or cycle prev/next)
  * 1,2 / , .     -- slide the selected car back (left/up) or forward (right/down)

Select meta-actions update a local highlight and do not call ``env.step``.
Move meta-actions become a Discrete index and are what get logged as
``env_action`` for seed+action replay.

Needs the Go binary ``rushhour-env`` built from the checkout:
    go build -o rushhour-env ./cmd/rushhour-env
Point at it via the phase "binary" field or the RUSHHOUR_ENV_BIN env var; this
adapter also auto-finds the vendored copy under vendor/rush-hour-src/.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .keyspec import SingleKeySpec
from .base import EnvAdapter, FrameState

# Distinct colors for car letters; 'A' (red player car) and exit are special.
_PALETTE: list[tuple[int, int, int]] = [
    (220, 60, 60), (70, 130, 220), (80, 190, 90), (230, 190, 60),
    (170, 90, 200), (230, 140, 60), (90, 200, 200), (230, 120, 170),
    (150, 110, 70), (120, 160, 90), (200, 200, 120), (110, 200, 160),
]
_VENDOR_BIN: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "vendor", "rush-hour-src", "rushhour-env")

# Meta-actions for the human keymap (NOT Discrete env indices). Same vocabulary
# as rushinput.DefaultMap: select a car, then slide it along its axis.
_SELECT_UP, _SELECT_DOWN, _SELECT_LEFT, _SELECT_RIGHT = 0, 1, 2, 3
_SELECT_PREV, _SELECT_NEXT = 4, 5
_MOVE_BACK, _MOVE_FORWARD = 6, 7
_NOOP = -1

# Mirrors https://github.com/chrplr/Rush-Hour/blob/main/internal/rushinput/keymap.go
_DEFAULT_KEYMAP: dict[str, int] = {
    "UP": _SELECT_UP, "DOWN": _SELECT_DOWN,
    "LEFT": _SELECT_LEFT, "RIGHT": _SELECT_RIGHT,
    "3": _SELECT_PREV, "4": _SELECT_NEXT,
    "1": _MOVE_BACK, "2": _MOVE_FORWARD,
    "COMMA": _MOVE_BACK, "PERIOD": _MOVE_FORWARD,
}

# Neighbour scoring constant from rush.Board.Neighbour (select.go).
_SIDEWAYS_PENALTY = 2 * 6


class RushHourAdapter(EnvAdapter):
    name: str = "rushhour"

    def make(self, spec: dict) -> Any:
        import gymnasium as gym
        import rushhour_gym  # noqa: F401  (registers RushHour*-v0)
        binary = spec.get("binary") or os.environ.get("RUSHHOUR_ENV_BIN")
        if not binary and os.path.exists(_VENDOR_BIN):
            binary = _VENDOR_BIN
        if binary:
            os.environ["RUSHHOUR_ENV_BIN"] = binary
        self._last_ansi = ""
        self._labels = ""
        self._cars: list[dict[str, Any]] = []
        self._selected = 0
        self._last_obs: Any = None
        self._last_info: dict = {}
        return gym.make(spec.get("game", "RushHour-Easy-v0"), render_mode="ansi")

    def keymap(self, env: Any) -> SingleKeySpec:
        combos = {frozenset([k]): v for k, v in _DEFAULT_KEYMAP.items()}
        return SingleKeySpec(combos=combos, noop=_NOOP)

    def reset(self, env: Any, seed: int | None, spec: dict) -> tuple[Any, dict]:
        obs, info = env.reset(seed=seed)
        self._last_ansi = env.render() or ""
        self._ingest(info)
        self._selected = 0  # red car; same as the experiment's trial start
        self._last_obs, self._last_info = obs, info
        return obs, info

    def step(self, env: Any, action: Any) -> tuple[Any, float, bool, bool, dict]:
        meta = int(action)
        if meta < 0:
            return self._ui_only()
        if meta <= _SELECT_NEXT:
            self._do_select(meta)
            return self._ui_only()

        dir_bit = 0 if meta == _MOVE_BACK else 1
        discrete = int(self._selected) * 2 + dir_bit
        obs, reward, terminated, truncated, info = env.step(discrete)
        self._last_ansi = env.render() or ""
        self._ingest(info)
        # Keep the highlight on the car that was just acted on when the engine
        # reports a slot (illegal clicks still name a slot).
        if isinstance(info, dict) and "slot" in info:
            try:
                slot = int(info["slot"])
                if 0 <= slot < len(self._cars):
                    self._selected = slot
            except Exception:
                pass
        info = dict(info)
        info["env_action"] = discrete
        self._last_obs, self._last_info = obs, info
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self, env: Any) -> np.ndarray:
        return _board_to_rgb(self._last_ansi, selected=self._selected_label())

    def capture(
        self, env: Any, obs: Any, info: dict, want_blob: bool = True
    ) -> FrameState:
        variables = {}
        if isinstance(info, dict) and "slot" in info:
            try:
                variables["slot"] = int(info["slot"])
            except Exception:
                pass
        variables["selected"] = int(self._selected)
        return FrameState(blob=None, variables=variables)

    # ── Selection / board helpers ─────────────────────────────────────────────

    def _ui_only(self) -> tuple[Any, float, bool, bool, dict]:
        info = dict(self._last_info)
        info["env_action"] = -1
        return self._last_obs, 0.0, False, False, info

    def _ingest(self, info: dict) -> None:
        if not isinstance(info, dict):
            return
        labels = info.get("labels") or ""
        self._labels = str(labels)
        board = info.get("board")
        if board:
            self._cars = _cars_from_board(str(board), self._labels)
        elif self._last_ansi:
            self._cars = _cars_from_board(self._last_ansi, self._labels)

    def _selected_label(self) -> str | None:
        if 0 <= self._selected < len(self._cars):
            return self._cars[self._selected].get("label")
        if self._labels:
            return self._labels[0]
        return "A"

    def _do_select(self, meta: int) -> None:
        if not self._cars:
            return
        cur = self._cars[self._selected] if 0 <= self._selected < len(self._cars) else self._cars[0]
        if meta == _SELECT_PREV:
            nxt = _cycle(self._cars, cur, -1)
        elif meta == _SELECT_NEXT:
            nxt = _cycle(self._cars, cur, 1)
        else:
            d_row, d_col = {
                _SELECT_UP: (-1, 0), _SELECT_DOWN: (1, 0),
                _SELECT_LEFT: (0, -1), _SELECT_RIGHT: (0, 1),
            }[meta]
            nxt = _neighbour(self._cars, cur, d_row, d_col)
        self._selected = int(nxt["slot"])


def _cars_from_board(board: str, labels: str) -> list[dict[str, Any]]:
    """Parse the ANSI/board notation into per-slot car geometries."""
    rows = []
    for line in (board or "").split("\n"):
        if not line:
            continue
        # Drop the " <" exit marker the ansi renderer appends.
        cell = line[:6] if len(line) >= 6 else line.rstrip()
        if cell:
            rows.append(cell)
    positions: dict[str, list[tuple[int, int]]] = {}
    for r, line in enumerate(rows[:6]):
        for c, ch in enumerate(line[:6]):
            if ch.isalpha():
                positions.setdefault(ch.upper(), []).append((r, c))

    cars: list[dict[str, Any]] = []
    # Prefer the engine's slot order; fall back to letters found on the board.
    order = list(labels) if labels else sorted(positions.keys())
    for slot, lab in enumerate(order):
        cells = positions.get(lab, [])
        if not cells:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        horizontal = len(set(rs)) == 1
        cars.append({
            "slot": slot, "label": lab,
            "row": min(rs), "col": min(cs),
            "length": len(cells), "horizontal": horizontal,
            "cells": cells,
        })
    return cars


def _span(car: dict[str, Any]) -> tuple[int, int, int, int]:
    row0, col0 = car["row"], car["col"]
    if car["horizontal"]:
        return row0, row0, col0, col0 + car["length"] - 1
    return row0, row0 + car["length"] - 1, col0, col0


def _center2(car: dict[str, Any]) -> tuple[int, int]:
    r0, r1, c0, c1 = _span(car)
    return r0 + r1, c0 + c1


def _gap(a0: int, a1: int, b0: int, b1: int) -> int:
    d = max(a0, b0) - min(a1, b1)
    return d if d > 0 else 0


def _neighbour(
    cars: list[dict[str, Any]], from_car: dict[str, Any], d_row: int, d_col: int
) -> dict[str, Any]:
    """Port of rush.Board.Neighbour — spatial select with wrap-around."""
    if (d_row != 0) == (d_col != 0):
        return from_car
    from_r2, from_c2 = _center2(from_car)
    f_r0, f_r1, f_c0, f_c1 = _span(from_car)
    best = wrapped = None
    best_score = wrap_score = 0
    for cand in cars:
        if cand is from_car or cand["slot"] == from_car["slot"]:
            continue
        cand_r2, cand_c2 = _center2(cand)
        ahead = (cand_r2 - from_r2) * d_row + (cand_c2 - from_c2) * d_col
        c_r0, c_r1, c_c0, c_c1 = _span(cand)
        sideways = (
            _gap(f_c0, f_c1, c_c0, c_c1) if d_row != 0
            else _gap(f_r0, f_r1, c_r0, c_r1)
        )
        score = ahead + _SIDEWAYS_PENALTY * sideways
        if ahead > 0:
            if best is None or score < best_score:
                best, best_score = cand, score
        else:
            if wrapped is None or score < wrap_score:
                wrapped, wrap_score = cand, score
    return best or wrapped or from_car


def _cycle(
    cars: list[dict[str, Any]], from_car: dict[str, Any], delta: int
) -> dict[str, Any]:
    """Port of rush.Board.Cycle — walk vehicles in board order, wrapping."""
    if not cars:
        return from_car
    idx = next((i for i, c in enumerate(cars) if c["slot"] == from_car["slot"]), 0)
    return cars[(idx + delta) % len(cars)]


def _board_to_rgb(
    ansi: str, cell: int = 64, selected: str | None = None
) -> np.ndarray:
    """Render the ANSI letter grid to a colored pixel board."""
    rows = [r for r in (ansi or "").split("\n") if r != ""]
    if not rows:
        return np.zeros((cell * 6, cell * 6, 3), dtype=np.uint8)
    h = len(rows)
    w = max(len(r) for r in rows)
    img = np.full((h * cell, w * cell, 3), 30, dtype=np.uint8)
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            y0, x0 = r * cell, c * cell
            if ch in (" ", "o"):
                color = (45, 45, 45)          # empty
            elif ch == "<":
                color = (255, 255, 255)       # exit marker
            elif ch == "A":
                color = (230, 40, 40)         # red player car
            elif ch.isalpha():
                color = _PALETTE[(ord(ch.upper()) - ord("A")) % len(_PALETTE)]
            else:
                color = (60, 60, 60)
            # draw a padded block so grid lines show
            img[y0 + 2:y0 + cell - 2, x0 + 2:x0 + cell - 2] = color
            if selected and ch.upper() == selected.upper():
                # Bright border so the selected car is obvious without arrows.
                img[y0:y0 + 3, x0:x0 + cell] = (255, 255, 255)
                img[y0 + cell - 3:y0 + cell, x0:x0 + cell] = (255, 255, 255)
                img[y0:y0 + cell, x0:x0 + 3] = (255, 255, 255)
                img[y0:y0 + cell, x0 + cell - 3:x0 + cell] = (255, 255, 255)
    return img
