"""The controller: a gamepad or button box whose buttons become keypresses.

Some sites' response device is a game controller on an fMRI interface (a
Current Designs 932, say) that reports HID *joystick* buttons rather than
keystrokes. Nothing a phase's ``keys`` names is ever pressed there, so the game
does not move and the run records no responses, with nothing on screen to say
why. This module presses each control as the rig key it is
(:data:`fmri_gym.rig.CONTROLS`: UP DOWN LEFT RIGHT A B X Y LT RT), the names a
curriculum's ``keys`` are written in. :func:`pump` moves the events across; it
registers with :func:`fmri_gym.keys.get_events`, which runs it before every read
of the queue.

The map is fixed on purpose -- the rig keys are this controller's own buttons,
so one curriculum plays at every site. What is *not* fixed is the stick,
which rests off centre, travels further one way than the other, and moves its
centre between sessions; that is measured per rig by ``fmri-pad-calibrate``
(below) into ``~/.config/fmri-gym/pad.json``, and checked in ``fmri-pad-scope``.

It turns itself on when a pad is plugged in, and a rig file's ``"pad": false`` leaves it off.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import pygame

from .keys import is_key, keycode, register_event_source, register_held_source

#: Which rig key each button is. SDL numbers buttons from 0, one less than the
#: HID report does: on the Current Designs pad that is y=0, b=1, x=2, LT=3,
#: a=4, RT=5.
BUTTON_KEYS: dict[int, str] = {0: "Y", 1: "B", 2: "X", 3: "LT", 4: "A", 5: "RT"}
#: Which key a stick axis presses. Axis 0 is horizontal, 1 vertical; SDL reads
#: up and left as negative.
AXIS_KEYS: dict[tuple[int, str], str] = {
    (0, "-"): "LEFT", (0, "+"): "RIGHT", (1, "-"): "UP", (1, "+"): "DOWN",
}
_ARROWS = frozenset(AXIS_KEYS.values())
STICK_AXES = (0, 1)

#: Where each direction sits as an angle of push. A push is classified by which
#: of these it points nearest to, so these four are the whole of what "which
#: direction" means. They are the obvious four: the calibration measures where
#: each cue really pointed (up to 32 degrees off, on the pad this was written
#: for) and reports it, but applies it only under ``--fit-sectors``, since that
#: skew is as much the hand as the stick.
IDEAL_SECTORS: dict[str, float] = {
    "RIGHT": 0.0, "DOWN": math.pi / 2, "LEFT": math.pi, "UP": -math.pi / 2,
}
#: How much nearer another direction must be before the stick changes hands.
_SECTOR_KEEP = math.radians(15)

#: Where this rig's measured stick calibration lives -- deliberately outside the
#: checkout, since it describes the controller on *this* machine, not the task.
CONFIG_ENV = "FMRI_GYM_PAD_CONFIG"

# Defaults for an uncalibrated pad; the calibration overrides every one.
_AXIS_ON, _AXIS_OFF = 0.2, 0.2
#: Whether the stick can press two directions at once, and how far the second
#: must go as a fraction of the first. At 1.0 it cannot, and the push is
#: classified into exactly one direction -- what a game with four discrete
#: actions wants. Below 1.0 the axes are read independently, for one that steers.
_DIAGONAL = 1.0
_DOMINANCE = 1.15           # in that mode, the margin to take the stick over
#: How long a new direction must hold before it is pressed. A thumb leaving rest
#: crosses a neighbouring sector on the way out, for a frame or two; a real push
#: lasted 254 ms at the shortest against transients of 17 ms.
_DWELL = 0.05
#: The stick's centre is taken once, after it has first been used and then let
#: go: cold and warm are not the same place (the gimbal does not return where it
#: started) and the warm one moves between sessions. How far it may be taken
#: from the calibrated centre is what the calibration measured that movement to
#: be, so a stick held slightly off can never be mistaken for centre.
_SETTLE_S, _SETTLE_STILL, _SETTLE_SLACK = 1.0, 0.01, 1.5

_enabled = False
_calibration: dict = {}                 # axis -> {rest, min, max, drift}
_sectors: dict[str, float] = dict(IDEAL_SECTORS)
_held: set[str] = set()                 # the NAMES the pad is holding down
_value: dict[int, float] = {}           # axis -> last reading, calibrated
_raw: dict[int, float] = {}             # axis -> last reading, untouched
_zero: dict[int, float] = {}            # axis -> where it is centred now
_pushed: dict[int, str | None] = {}     # axis -> its side, in diagonal mode
_since: dict[str, float] = {}           # direction -> when it first wanted pressing
_quiet: dict[int, tuple] = {}           # axis -> (since, low, high) while still
_settled: set[int] = set()              # axes whose centre has been taken
_joysticks: list = []
_unmapped: set[int] = set()
_warmed = False
_status = "off (the rig's \"pad\": false)"


def config_path() -> Path:
    """The pad calibration file for this machine."""
    override = os.environ.get(CONFIG_ENV)
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME") or "~/.config"
    return Path(base).expanduser() / "fmri-gym" / "pad.json"


def init(enabled: bool = True) -> str:
    """Open whatever pad is plugged in, unless the rig file said not to.

    :param enabled: ``False`` to leave the pad off for this run.
    :return: a one-line status for the run's stderr banner.
    """
    global _status, _enabled
    if not enabled:
        _status = "off (the rig's \"pad\": false)"
        return _status
    _enabled = True
    _load_calibration()
    register_held_source(held)
    register_event_source(pump)
    if not _open():
        _status = "no controller plugged in; the keyboard answers"
        return _status
    cal = (f"calibrated {config_path()}" if _calibration
           else f"UNCALIBRATED (run fmri-pad-calibrate; writing {config_path()})")
    rule = ("one direction at a time" if _DIAGONAL >= 1.0
            else f"diagonals past {_DIAGONAL:.2f}")
    _status = (f"{', '.join(j.get_name() for j in _joysticks)} -> "
               f"buttons {', '.join(BUTTON_KEYS.values())}, stick UP/DOWN/LEFT/RIGHT "
               f"at {_AXIS_ON:.2f}/{_AXIS_OFF:.2f}, {rule}, {_DWELL * 1000:.0f} ms "
               f"dwell; {cal}")
    return _status


def status() -> str:
    """The line :func:`init` last returned, for the manifest and the banner."""
    return _status


def _load_calibration() -> None:
    """Read this rig's stick measurements, if it has been calibrated."""
    global _warmed
    _calibration.clear()
    # A new calibration means a new centre to find: forget the one taken under
    # the old one rather than carrying it across.
    _zero.clear()
    _settled.clear()
    _quiet.clear()
    _warmed = False
    path = config_path()
    try:
        blob = json.loads(path.read_text())
    except FileNotFoundError:
        return
    except (OSError, ValueError) as exc:
        print(f"pad: ignoring unreadable {path}: {exc}", file=sys.stderr)
        return
    for index, entry in (blob.get("axes") or {}).items():
        try:
            axis = {k: float(entry[k]) for k in ("rest", "min", "max")}
        except (KeyError, TypeError, ValueError):
            print(f"pad: ignoring malformed axis {index} in {path}", file=sys.stderr)
            continue
        axis["drift"] = float(entry.get("drift") or 0.0)
        _calibration[int(index)] = axis
    for key, name in (("axis_on", "_AXIS_ON"), ("axis_off", "_AXIS_OFF"),
                      ("axis_diagonal", "_DIAGONAL"), ("stick_dwell_s", "_DWELL")):
        if isinstance(blob.get(key), (int, float)):
            globals()[name] = float(blob[key])
    _sectors.clear()
    _sectors.update(IDEAL_SECTORS)
    for name, degrees in (blob.get("sectors") or {}).items():
        if name in IDEAL_SECTORS and isinstance(degrees, (int, float)):
            _sectors[name] = math.radians(float(degrees))


def _open() -> bool:
    """Open the pads, or reopen them after something reset the joystick subsystem.

    ``Display`` calls ``pygame.init()`` when the window goes up, which restarts
    every subsystem and quietly closes handles opened before it. Rather than
    depend on the order, they are checked here and taken again when stale.
    """
    if _joysticks and _joysticks[0].get_init():
        return True
    _joysticks.clear()
    if not pygame.joystick.get_init():
        pygame.joystick.init()
    _joysticks.extend(pygame.joystick.Joystick(i)
                      for i in range(pygame.joystick.get_count()))
    return bool(_joysticks)


def held() -> frozenset[str]:
    """The NAMES the pad is holding down this frame."""
    return frozenset(_held)


def centre(index: int) -> float:
    """Where axis ``index`` is currently taken to rest."""
    return _zero.get(index, _calibration.get(index, {}).get("rest", 0.0))


def _normalise(index: int, value: float) -> float:
    """A raw reading as a fraction of that direction's own travel, -1..1.

    Each side is scaled by its own span, so a pad reaching -0.5 one way and +1.0
    the other still reports -1.0 and +1.0 at its stops. The stops do not move,
    so both spans are measured from wherever the stick is currently centred.
    """
    cal = _calibration.get(index)
    if not cal:
        return value
    rest = centre(index)
    span = (cal["max"] - rest) if value >= rest else (rest - cal["min"])
    if span <= 1e-6:
        return 0.0
    return max(-1.0, min(1.0, (value - rest) / span))


def _settle(now: float) -> None:
    """Take the stick's centre, once, after it has been used and let go."""
    if not _warmed or _ARROWS & _held:
        return
    for index, value in _raw.items():
        if index in _settled or index not in _calibration:
            continue
        since, low, high = _quiet.get(index, (now, value, value))
        low, high = min(low, value), max(high, value)
        if high - low > _SETTLE_STILL:
            _quiet[index] = (now, value, value)         # moved: start again
            continue
        _quiet[index] = (since, low, high)
        if now - since < _SETTLE_S:
            continue
        _settled.add(index)
        cal, found = _calibration[index], (low + high) / 2
        allowed = max(cal["drift"] * _SETTLE_SLACK, _SETTLE_STILL)
        if abs(found - cal["rest"]) <= allowed:
            _zero[index] = found
        else:
            print(f"pad: axis {index} settled at {found:+.3f}, {abs(found - cal['rest']):.3f} "
                  f"from the calibrated centre and further than the {cal['drift']:.3f} it was "
                  f"measured to move; keeping the calibrated one. Rerun fmri-pad-calibrate.",
                  file=sys.stderr)


def _press(name: str, down: bool) -> None:
    """Hold or release the rig key ``name``, and post it as a keypress."""
    if down:
        _held.add(name)
    else:
        _held.discard(name)
    # `unicode` is what a "press <char> to go on" screen reads; `pad` marks the
    # event as ours, and `name` is the rig key (LT and RT have no keycode).
    char = name.lower() if len(name) == 1 else ""
    pygame.event.post(pygame.event.Event(
        pygame.KEYDOWN if down else pygame.KEYUP,
        key=keycode(name) if is_key(name) else pygame.K_UNKNOWN, mod=0,
        unicode=char if down else "",
        scancode=0, pad=True, name=name))


def _away(a: float, b: float) -> float:
    """How far apart two angles are, the short way round."""
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def _classified() -> set[str]:
    """The one direction the stick is being pushed, or none.

    How far it is from centre says *whether* a direction is held, and the angle
    says *which*. Classifying the push as a whole is what keeps the horizontal
    axis from answering a push meant for down: that push does move both axes, a
    lot, but it points much closer to down than to left.
    """
    x, y = STICK_AXES
    nx, ny = _value.get(x, 0.0), _value.get(y, 0.0)
    current = _ARROWS & _held
    if math.hypot(nx, ny) < (_AXIS_OFF if current else _AXIS_ON):
        return set()
    theta = math.atan2(ny, nx)
    best = min(_sectors, key=lambda d: _away(theta, _sectors[d]))
    for name in current:                    # the one already pressed keeps it
        if name in _sectors and _away(theta, _sectors[name]) \
                - _away(theta, _sectors[best]) < _SECTOR_KEEP:
            return {name}
    return {best}


def _side(index: int) -> str | None:
    """Which way an axis is pushed, holding whatever it was doing in the gap."""
    value, was = _value.get(index, 0.0), _pushed.get(index)
    if value <= -_AXIS_ON:
        return "-"
    if value >= _AXIS_ON:
        return "+"
    return None if abs(value) < _AXIS_OFF else was


def _independent() -> set[str]:
    """Both axes read separately, for a game that wants diagonals."""
    wanted: dict[int, str] = {}
    for index in list(_value):
        side = _side(index)
        _pushed[index] = side
        if side is not None and (index, side) in AXIS_KEYS:
            wanted[index] = AXIS_KEYS[(index, side)]
    x, y = STICK_AXES
    if x in wanted and y in wanted:
        strong, weak = (x, y) if abs(_value[x]) >= abs(_value[y]) else (y, x)
        if wanted[weak] in _held and wanted[strong] not in _held \
                and abs(_value[weak]) * _DOMINANCE > abs(_value[strong]):
            strong, weak = weak, strong
        lead = abs(_value[strong])
        ratio = abs(_value[weak]) / lead if lead > 1e-6 else 0.0
        if ratio < _DIAGONAL * (0.8 if wanted[weak] in _held else 1.0):
            del wanted[weak]
    return set(wanted.values())


def _dwelt(names: set[str], now: float) -> set[str]:
    """Of the directions asked for, the ones that have asked long enough."""
    for name in [n for n in _since if n not in names]:
        del _since[name]
    for name in names:
        _since.setdefault(name, now)
    return {n for n in names if n in _held or now - _since[n] >= _DWELL}


def _resolve() -> None:
    """Press what the stick's position now means, and release what it does not."""
    global _warmed
    now = time.monotonic()
    _settle(now)
    for index, value in _raw.items():
        _value[index] = _normalise(index, value)
    names = _classified() if _DIAGONAL >= 1.0 else _independent()
    if _DWELL > 0:
        names = _dwelt(names, now)
    for name in sorted(_ARROWS & (_held - names)):
        _press(name, False)
    for name in sorted(names - _held):
        _warmed = True
        _press(name, True)


def pump() -> None:
    """Move the pad's events into the keyboard ones (run by ``keys.get_events``).

    Takes only the joystick events off the queue, so whatever else is waiting
    there is still the caller's to read.
    """
    if not (_enabled and _open()):
        return
    moved = False
    for event in pygame.event.get((pygame.JOYBUTTONDOWN, pygame.JOYBUTTONUP,
                                   pygame.JOYAXISMOTION)):
        if event.type == pygame.JOYAXISMOTION:
            _raw[event.axis] = event.value
            moved = True
        elif event.button in BUTTON_KEYS:
            _press(BUTTON_KEYS[event.button], event.type == pygame.JOYBUTTONDOWN)
        elif event.type == pygame.JOYBUTTONDOWN and event.button not in _unmapped:
            _unmapped.add(event.button)
            print(f"pad: button {event.button} pressed, but no rig key is mapped to it "
                  "(fmri_gym/pad.py: BUTTON_KEYS)", file=sys.stderr)
    if moved:
        _resolve()


# ------------------------------------------------------------ the two tools
# Shared by the calibration task and the scope, which are the same program seen
# before and after: one measures the pad, the other watches what it measured.
SURFACE, PANEL, GRID = (18, 21, 28), (24, 28, 37), (35, 41, 54)
INK, INK2, INK3 = (231, 234, 240), (154, 164, 178), (91, 100, 114)
X_HUE, Y_HUE = (79, 143, 247), (240, 136, 62)
LIVE, WARN = (63, 185, 80), (210, 153, 34)
#: How each control is named and coloured on screen: the face buttons as they
#: are printed on the pad, in the colour they are printed in, so a cue can be
#: matched by colour without reading it.
FACE: dict[int, tuple[str, tuple[int, int, int]]] = {
    0: ("Y", (226, 192, 68)), 1: ("B", (230, 86, 86)), 2: ("X", (79, 143, 247)),
    3: ("LT", INK2), 4: ("A", (63, 185, 80)), 5: ("RT", INK2),
}


def _arrow(surface, direction: str, at: tuple[int, int], size: int, colour) -> None:
    """Draw a direction as the arrow it is, rather than spelling it out."""
    dx, dy = {"LEFT": (-1, 0), "RIGHT": (1, 0), "UP": (0, -1), "DOWN": (0, 1)}[direction]
    px, py = -dy, dx                                    # across the shaft
    cx, cy = at
    tip = (cx + dx * size, cy + dy * size)
    base = (cx - dx * size * 0.4, cy - dy * size * 0.4)
    neck = (cx + dx * size * 0.15, cy + dy * size * 0.15)
    pygame.draw.polygon(surface, colour, [
        tip,
        (neck[0] + px * size * 0.55, neck[1] + py * size * 0.55),
        (neck[0] + px * size * 0.2, neck[1] + py * size * 0.2),
        (base[0] + px * size * 0.2, base[1] + py * size * 0.2),
        (base[0] - px * size * 0.2, base[1] - py * size * 0.2),
        (neck[0] - px * size * 0.2, neck[1] - py * size * 0.2),
        (neck[0] - px * size * 0.55, neck[1] - py * size * 0.55),
    ])


BUTTONS = {i: label for i, (label, _) in FACE.items()}
DIRECTIONS = {"LEFT": (0, -1), "RIGHT": (0, 1), "UP": (1, -1), "DOWN": (1, 1)}
#: The order the task asks for them: round the face the way the buttons sit
#: under a thumb, then round the stick, so the next cue is where the hand
#: already expects it and nobody is surprised into the wrong control.
CUE_BUTTONS = (4, 2, 0, 1, 3, 5)                       # A X Y B LT RT
CUE_DIRECTIONS = ("DOWN", "LEFT", "UP", "RIGHT")

SWEEP_S, REST_S = 14.0, 4.0
CUE_S, ITI_S, FEEDBACK_S = 2.5, 0.7, 0.35
BLOCK_LEAD_S = 3.0          # naming the next control, hands off, before its trials
LOOSE = 0.35                # fraction of measured travel; feedback only, never fitted from
CENTRED = 0.15              # the stick counts as back at rest under this
RECENTRE_S = 2.0            # how long to wait for it before giving up and cueing anyway



# --------------------------------------------------------------------- the fit
def _live_calibration(samples: list[dict], n_axes: int) -> dict:
    """Rest and extremes from what has been recorded so far, for use during the task.

    The cue feedback has to know when the stick has moved, and it cannot ask the
    finished calibration because the task is what produces it. Measuring rest
    from the sweep and rest phases is enough: what it must not do is assume the
    stick rests at 0.0, which on a real pad is off by more than a naive
    threshold's whole margin.
    """
    cal = {}
    for i in range(n_axes):
        col = [s[f"ax{i}"] for s in samples] or [0.0]
        at_rest = [s[f"ax{i}"] for s in samples if s["phase"] == "rest"] or col[:1]
        cal[i] = {"rest": statistics.median(at_rest), "min": min(col), "max": max(col)}
    return cal


def _norm(cal: dict, i: int, v: float) -> float:
    """A reading as a fraction of that direction's travel, -1..1."""
    a = cal.get(i)
    if not a:
        return v
    span = (a["max"] - a["rest"]) if v >= a["rest"] else (a["rest"] - a["min"])
    return 0.0 if span <= 1e-6 else max(-1.0, min(1.0, (v - a["rest"]) / span))


def _still(values: list[float], window: int = 60) -> tuple[float, float]:
    """Where a resting axis sits and how much it dithers, from its quietest second.

    The quietest second rather than the whole phase: "hands off" begins the
    instant the sweep ends, and its first seconds are the stick springing back,
    which averages in as a dither a hundred times the real one.
    """
    if not values:
        return 0.0, 0.0
    window = min(window, len(values))
    first = min(range(len(values) - window + 1),
                key=lambda i: max(values[i:i + window]) - min(values[i:i + window]))
    seg = values[first:first + window]
    return statistics.median(seg), (max(seg) - min(seg)) / 2


def _hands_off(samples: list[dict]) -> list[list[dict]]:
    """The recording's hands-off stretches, one list per block lead-in."""
    blocks, current = [], []
    for s in samples:
        if s["phase"] == "lead":
            current.append(s)
        elif current:
            blocks.append(current)
            current = []
    return blocks + ([current] if current else [])


def fit(samples: list[dict], n_axes: int) -> dict:
    """Turn recorded samples into a calibration: per-axis travel and thresholds.

    :param samples: every recorded frame (raw readings, no threshold applied).
    :param n_axes: how many axes the pad reports.
    :return: the calibration blob, ready to write.
    """
    rest_rows = [s for s in samples if s["phase"] == "rest"]
    cold_rows = [s for s in samples if s["phase"] == "wait"]
    blocks = [b for b in _hands_off(samples) if len(b) >= 30]
    axes, used = {}, {}
    for i in range(n_axes):
        col = [s[f"ax{i}"] for s in samples]
        rest, noise = _still([s[f"ax{i}"] for s in rest_rows] or col[:1])
        # How far the centre moves, which is what a run's one re-centring is
        # later bounded by. Two ways it does: between cold and warm, since the
        # gimbal does not return where it started, and between one block and
        # the next. The hands-off countdown before each block is what makes the
        # second measurable -- still windows spread across the session.
        cold, _ = _still([s[f"ax{i}"] for s in cold_rows]) if cold_rows else (rest, 0.0)
        lo, hi = min(col), max(col)
        # A block whose quietest moment is a long way out is one where the stick
        # was never let go of, whatever the screen asked for; it says nothing
        # about where the centre is, so it is not counted as a sample of it.
        reach = min(hi - rest, rest - lo)
        seen = [_still([s[f"ax{i}"] for s in rows])[0] for rows in blocks]
        settled = [m for m in seen if abs(m - rest) <= 0.25 * reach]
        spread = (max(settled) - min(settled)) if len(settled) >= 3 else 0.0
        used[str(i)] = f"{len(settled)}/{len(seen)}"
        axes[str(i)] = {"rest": round(rest, 5), "min": round(lo, 5), "max": round(hi, 5),
                        "noise": round(noise, 6),
                        "drift": round(max(abs(rest - cold), spread), 5)}

    def norm(i: int, v: float) -> float:
        a = axes[str(i)]
        span = (a["max"] - a["rest"]) if v >= a["rest"] else (a["rest"] - a["min"])
        return 0.0 if span <= 1e-6 else max(-1.0, min(1.0, (v - a["rest"]) / span))

    # How far, in calibrated units, each cued direction was actually pushed, and
    # where it pointed while being pushed there.
    peaks: dict[str, list[float]] = {d: [] for d in DIRECTIONS}
    angles: dict[str, list[float]] = {d: [] for d in DIRECTIONS}
    for name, (axis, sign) in DIRECTIONS.items():
        best: dict[int, tuple[float, float]] = {}
        for s in samples:
            if s["phase"] == "cue" and s["cue"] == name:
                v = norm(axis, s[f"ax{axis}"]) * sign
                if v > best.get(s["trial"], (0.0, 0.0))[0]:
                    best[s["trial"]] = (v, math.atan2(norm(1, s["ax1"]), norm(0, s["ax0"])))
        peaks[name] = sorted(v for v, _ in best.values())
        angles[name] = [a for v, a in best.values() if v >= 0.15]

    # A cued trial with no push is a missed trial, not a weak one: counting it
    # would drag the fit to zero and quietly fall back to a default threshold.
    NO_PUSH = 0.15
    pushed = {d: [v for v in vals if v >= NO_PUSH] for d, vals in peaks.items()}
    skipped = sum(len(vals) - len(pushed[d]) for d, vals in peaks.items())
    reached = [p for vals in pushed.values() for p in vals]
    worst = (min(statistics.median(vals) * 0.8 if len(vals) < 5 else vals[len(vals) // 10]
                 for vals in pushed.values() if vals)
             if all(pushed.values()) else 0.0)
    # Only the axes a direction uses: an undriven one (the 932 reports two
    # pinned at -1.0) has no span to divide by.
    noise_norm = 0.0
    for i in {axis for axis, _ in DIRECTIONS.values()}:
        a = axes.get(str(i))
        if not a:
            continue
        span = min(a["max"] - a["rest"], a["rest"] - a["min"])
        if span > 1e-3:
            noise_norm = max(noise_norm, a["noise"] / span)

    # Halfway to the weakest push made, never within 10x the dither, never past
    # two thirds of travel, which would mean holding the stick against its stop.
    on = min(max(0.5 * worst if worst else 0.5, 10 * noise_norm, 0.08), 0.65)
    off = round(0.6 * on, 3)
    sectors, sector_note = _fit_sectors(angles)
    ideal = {"RIGHT": 0.0, "DOWN": 90.0, "LEFT": 180.0, "UP": -90.0}
    dwell, transients = _dwell(samples, norm, on, ideal)

    return {
        "_comment": "written by fmri-pad-calibrate; rerun it if the controller changes",
        "device": "", "measured": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "axes": axes, "axis_on": round(on, 3), "axis_off": off,
        "axis_diagonal": 1.0, "stick_dwell_s": dwell,
        "_diagnostics": {
            "pushes_recorded": len(reached),
            "trials_with_no_push": skipped,
            "weakest_push_per_direction": {d: round(min(v), 3)
                                           for d, v in pushed.items() if v},
            "max_push_per_direction": {d: round(max(v), 3) for d, v in pushed.items() if v},
            "noise_in_calibrated_units": round(noise_norm, 5),
            "hands_off_blocks_believed": used,
            "crosstalk_median": _crosstalk(samples, axes, norm),
            "sectors_measured": sectors, "sectors_note": sector_note,
            "sector_skew_deg": {d: round(abs((sectors[d] - ideal[d] + 180) % 360 - 180), 1)
                                for d in ideal},
            "wrong_direction_transients_ms": transients,
            "margin_as_shipped_deg": _sector_margin(angles, ideal),
            "margin_if_fitted_deg": _sector_margin(angles, sectors),
        },
    }


def _dwell(samples: list[dict], norm, on: float, ideal: dict[str, float]) -> tuple[float, list[int]]:
    """How long a direction must hold before it is believed, in seconds.

    Half again the longest run of frames where a cued trial was past the
    threshold but pointing somewhere other than its cue -- the stick crossing a
    neighbouring sector on the way out.
    """
    runs: list[float] = []
    for name in DIRECTIONS:
        for trial in {s["trial"] for s in samples
                      if s["phase"] == "cue" and s["cue"] == name}:
            wrong_since = None
            for s in [s for s in samples if s["phase"] == "cue" and s["trial"] == trial]:
                x, y = norm(0, s["ax0"]), norm(1, s["ax1"])
                if math.hypot(x, y) < on:
                    got = None
                else:
                    theta = math.degrees(math.atan2(y, x))
                    got = min(ideal, key=lambda d: abs((theta - ideal[d] + 180) % 360 - 180))
                if got is not None and got != name:
                    wrong_since = s["t"] if wrong_since is None else wrong_since
                elif wrong_since is not None:
                    runs.append(s["t"] - wrong_since)
                    wrong_since = None
    longest = max(runs) if runs else 0.0
    return round(min(max(1.5 * longest, 0.033), 0.15), 3), sorted(round(r * 1000) for r in runs)


def _fit_sectors(angles: dict[str, list[float]]) -> tuple[dict[str, float], str]:
    """Where each direction actually points, in degrees, from the trials.

    Reported always, applied only under ``--fit-sectors`` (see
    :data:`IDEAL_SECTORS`). One more than 45 degrees off is left ideal: past
    that the cue was probably misread, and a blunt sector beats a wrong one.
    """
    ideal = {"RIGHT": 0.0, "DOWN": 90.0, "LEFT": 180.0, "UP": -90.0}
    out, rejected = {}, []
    for name, want in ideal.items():
        got = angles.get(name) or []
        if len(got) < 2:
            out[name], _ = want, rejected.append(f"{name}: not pushed")
            continue
        mean = math.degrees(math.atan2(sum(math.sin(a) for a in got),
                                       sum(math.cos(a) for a in got)))
        if abs((mean - want + 180) % 360 - 180) > 45:
            out[name] = want
            rejected.append(f"{name}: measured {mean:+.0f} deg, too far from {want:+.0f}")
        else:
            out[name] = round(mean, 1)
    return out, "; ".join(rejected) or "all four fitted"


def _sector_margin(angles: dict[str, list[float]], sectors: dict[str, float]) -> float:
    """How much nearer the right direction was than the runner-up, worst trial, in degrees.

    Measured on the trials the sectors came from, so it reads high; the scope
    afterwards is the honest check.
    """
    def away(a: float, b: float) -> float:
        return abs((a - b + 180) % 360 - 180)

    worst = 180.0
    for got in angles.values():
        for a in got:
            ranked = sorted(away(math.degrees(a), v) for v in sectors.values())
            worst = min(worst, ranked[1] - ranked[0])
    return round(worst, 1) if worst < 180 else 0.0


def _crosstalk(samples: list[dict], axes: dict, norm) -> dict:
    """How far the *other* axis moves while one is pushed hard."""
    out = {}
    for name, (axis, sign) in DIRECTIONS.items():
        other = 1 - axis
        if str(other) not in axes:
            continue
        vals = [abs(norm(other, s[f"ax{other}"])) for s in samples
                if s["phase"] == "cue" and s["cue"] == name
                and norm(axis, s[f"ax{axis}"]) * sign > 0.8]
        if vals:
            out[name] = round(statistics.median(vals), 3)
    return out


# ------------------------------------------------------------------- the task
def run(reps: int, shuffle: bool = False,
        screen=None) -> tuple[list[dict], list[dict], str, int, int]:
    """Show the cues and record every frame. Returns (samples, trials, name, axes, buttons).

    :param screen: draw on this surface instead of opening a window, so the rig
        check can run the calibration inside the session it has already opened.
    """
    own = screen is None
    if own:
        pygame.init()
        screen = pygame.display.set_mode((900, 600))
        pygame.display.set_caption("fmri-gym pad calibration")
    w, h = screen.get_size()
    mx, scale = w // 2, min(2.0, max(0.8, h / 600))
    clock = pygame.time.Clock()
    big = pygame.font.SysFont("Menlo,Monaco,monospace", int(62 * scale))
    mid = pygame.font.SysFont("Menlo,Monaco,monospace", int(25 * scale))
    sml = pygame.font.SysFont("Menlo,Monaco,monospace", int(15 * scale))

    pygame.joystick.init()
    sticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
    if not sticks:
        if own:
            pygame.quit()
        raise SystemExit("no controller found: plug it in and run this again")
    j = sticks[0]
    n_ax, n_btn = j.get_numaxes(), j.get_numbuttons()

    samples: list[dict] = []
    trials: list[dict] = []
    t0 = time.perf_counter()
    stop = False

    def draw(lines, colour=INK, sub=None, arrow=None):
        screen.fill(SURFACE)
        if arrow:
            _arrow(screen, arrow, (mx, 235), 74, colour)
        for k, line in enumerate(lines):
            surf = (big if k == 0 else mid).render(line, True, colour if k == 0 else INK2)
            screen.blit(surf, surf.get_rect(center=(mx, (300 if arrow else 235) + k * 66)))
        if sub:
            s = sml.render(sub, True, INK2)
            screen.blit(s, s.get_rect(center=(mx, h - 40)))
        pygame.display.flip()

    def take(phase: str, trial: int, cue: str) -> tuple[list[float], list[int]]:
        pygame.event.pump()
        ax = [j.get_axis(i) for i in range(n_ax)]
        bt = [int(j.get_button(i)) for i in range(n_btn)]
        row = {"t": round(time.perf_counter() - t0, 4), "phase": phase,
               "trial": trial, "cue": cue}
        row.update({f"ax{i}": round(v, 5) for i, v in enumerate(ax)})
        row.update({f"b{i}": v for i, v in enumerate(bt)})
        samples.append(row)
        return ax, bt

    def quit_pressed() -> bool:
        for e in pygame.event.get((pygame.QUIT, pygame.KEYDOWN)):
            if e.type == pygame.QUIT or (not getattr(e, "pad", False)
                                         and e.key == pygame.K_ESCAPE):
                return True
        return False

    def hold(phase: str, seconds: float, lines, sub) -> bool:
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            if quit_pressed():
                return True
            take(phase, -1, "")
            draw(lines + [f"{end - time.perf_counter():.0f}s"], sub=sub)
            clock.tick(60)
        return False

    draw(["ready?"], sub="press any button to start  -  ESC quits and still saves")
    while not stop:
        if quit_pressed():
            stop = True
        elif any(take("wait", -1, "")[1]):
            break
        clock.tick(60)

    if not stop:
        stop = hold("sweep", SWEEP_S, ["roll the stick", "around its edge"],
                    "push it as far as it physically goes, all the way round, a few times")
    if not stop:
        stop = hold("rest", REST_S, ["hands off"], "do not touch the controller")

    live = _live_calibration(samples, n_ax)      # rest + travel measured just now

    # One at a time, in a fixed order: a press aimed at the wrong control
    # measures nothing, and this is measuring the pad, not the person.
    order = ([("button", b) for b in CUE_BUTTONS]
             + [("stick", d) for d in CUE_DIRECTIONS])
    plan = [item for item in order for _ in range(reps)]
    if shuffle:
        random.shuffle(plan)

    previous = None
    for n, (kind, what) in enumerate(plan):
        if stop:
            break
        cue = what if kind == "stick" else BUTTONS[what]
        arrow = cue if kind == "stick" else None
        label = "push" if kind == "stick" else f"press  {cue}"
        ink = INK if kind == "stick" else FACE[what][1]

        if (kind, what) != previous:            # a new control: say so, unhurried
            previous = (kind, what)
            end = time.perf_counter() + BLOCK_LEAD_S
            while time.perf_counter() < end and not stop:
                stop = quit_pressed()
                take("lead", n, cue)
                draw([label, f"{reps} times",
                      f"hands off  {end - time.perf_counter():.0f}"], ink, arrow=arrow,
                     sub=f"{n + 1}-{min(n + reps, len(plan))} of {len(plan)}")
                clock.tick(60)

        # Wait for it to come back, or the trial inherits the last one's
        # deflection and records it as an instant response.
        end = time.perf_counter() + RECENTRE_S
        while time.perf_counter() < end and not stop:
            stop = quit_pressed()
            ax, _ = take("recentre", n, cue)
            if all(abs(_norm(live, i, ax[i])) < CENTRED for i in (0, 1) if i < n_ax):
                break
            draw(["let the stick", "come back"], INK2)
            clock.tick(60)

        end = time.perf_counter() + ITI_S
        while time.perf_counter() < end and not stop:
            stop = quit_pressed()
            take("iti", n, cue)
            draw(["+"], INK2)
            clock.tick(60)

        onset, rt, got = time.perf_counter(), None, ""
        end = onset + CUE_S
        while time.perf_counter() < end and not stop:
            stop = quit_pressed()
            ax, bt = take("cue", n, cue)
            if rt is None:
                if kind == "button" and any(bt):
                    rt, got = time.perf_counter() - onset, BUTTONS.get(bt.index(1), "?")
                elif kind == "stick":
                    best, which = LOOSE, ""
                    for name, (axis, sign) in DIRECTIONS.items():
                        v = _norm(live, axis, ax[axis]) * sign
                        if v >= best:
                            best, which = v, name
                    if which:
                        rt, got = time.perf_counter() - onset, which
            draw([label], LIVE if rt else ink, arrow=arrow,
                 sub=f"trial {n + 1} of {len(plan)}")
            clock.tick(60)
        if stop:
            break

        trials.append({"trial": n, "kind": kind, "cue": cue,
                       "onset_s": round(onset - t0, 4),
                       "rt_s": round(rt, 4) if rt else "",
                       "detected": got, "hit": int(got == cue)})
        end = time.perf_counter() + FEEDBACK_S
        while time.perf_counter() < end:
            take("feedback", n, cue)
            draw([] if got in DIRECTIONS else [got or "-"],
                 LIVE if got == cue else WARN,
                 arrow=got if got in DIRECTIONS else None,
                 sub=f"trial {n + 1} of {len(plan)}")
            clock.tick(60)

    name = j.get_name()
    draw(["measuring..."])
    if own:
        pygame.quit()
    return samples, trials, name, n_ax, n_btn


def _reread(path: Path) -> list[dict]:
    """Load a samples CSV back into the rows :func:`fit` reads."""
    with path.expanduser().open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path}: no samples")
    for r in rows:
        r["trial"] = int(r["trial"])
        for k in r:
            if k == "t" or k.startswith("ax"):
                r[k] = float(r[k])
    return rows


def save(blob: dict, samples: list[dict], trials: list[dict],
         out: Path | None = None) -> Path:
    """Write the calibration and the recording it came from. Returns the samples path.

    The calibration is this site's, not this session's: every run on the rig
    reads it, so it lives beside the other per-machine config rather than in the
    checkout or the session's data. The raw samples go next to it so a later
    question can be answered without asking anyone to push the stick again.
    """
    out = (out or config_path()).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=2) + "\n")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    raw = out.parent / f"pad-samples_{stamp}.csv"
    with raw.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(samples[0]))
        w.writeheader()
        w.writerows(samples)
    if trials:
        with (out.parent / f"pad-trials_{stamp}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(trials[0]))
            w.writeheader()
            w.writerows(trials)
    return raw


def detect() -> str:
    """The name of the controller plugged in, or "" if there is none."""
    if not pygame.joystick.get_init():
        pygame.joystick.init()
    for i in range(pygame.joystick.get_count()):
        return pygame.joystick.Joystick(i).get_name()
    return ""


def calibrate(reps: int, screen=None, extra: dict | None = None) -> dict:
    """Measure the pad and write this site's calibration. Returns the diagnostics.

    For the rig check, which has a window open already and wants the numbers
    back rather than printed. ``extra`` is merged into the stored file, so the
    check can record which site and rig measured it.
    """
    samples, trials, device, n_ax, _ = run(reps, screen=screen)
    if not samples:
        return {"measured": False, "why": "nothing recorded"}
    blob = fit(samples, n_ax)
    blob["device"] = device
    blob.update(extra or {})
    diag = blob.pop("_diagnostics")
    raw = save(blob, samples, trials, None)
    _load_calibration()                      # the rest of the check plays calibrated
    return {"measured": True, "device": device, "config": str(config_path()),
            "samples": str(raw), "trials": len(trials),
            "hits": sum(t["hit"] for t in trials),
            "axis_on": blob["axis_on"], "axis_off": blob["axis_off"],
            "stick_dwell_s": blob["stick_dwell_s"],
            "centre_moves": {i: a["drift"] for i, a in blob["axes"].items()},
            **diag}


def calibrate_main() -> None:
    p = argparse.ArgumentParser(description="Measure this rig's controller.")
    p.add_argument("--reps", type=int, default=6,
                   help="how many times each control is cued (default 6: 60 trials)")
    p.add_argument("--out", type=Path, default=None,
                   help=f"where to write the calibration (default {config_path()})")
    p.add_argument("--dry-run", action="store_true",
                   help="measure and print, but write nothing")
    p.add_argument("--shuffle", action="store_true",
                   help="randomise the trial order instead of one control at a time")
    p.add_argument("--no-scope", action="store_true",
                   help="do not open the scope to check the calibration afterwards")
    p.add_argument("--fit-sectors", action="store_true",
                   help="classify the stick against the directions it actually pointed "
                        "rather than the ideal four; measures the grip as much as the "
                        "pad, so only for a stick that is genuinely skewed")
    p.add_argument("--refit", type=Path, default=None, metavar="SAMPLES.CSV",
                   help="fit a calibration from an earlier recording instead of "
                        "asking anyone to push the stick again")
    args = p.parse_args()

    if args.refit:
        # The recording is the measurement; the fit is an opinion about it. When
        # the opinion is what changed, the five minutes of pushing should not
        # have to happen twice.
        samples, trials = _reread(args.refit), []
        was = (args.out or config_path()).expanduser()
        device = ""                          # keep whose pad this is; the CSV cannot say
        if was.exists():
            try:
                device = json.loads(was.read_text()).get("device", "")
            except ValueError:
                pass
        device = device.split(" (refit of ")[0] or "unknown"
        device = f"{device} (refit of {args.refit.name})"
        n_ax = sum(1 for k in samples[0] if k.startswith("ax"))
    else:
        samples, trials, device, n_ax, _ = run(args.reps, args.shuffle)
    if not samples:
        raise SystemExit("nothing recorded")
    blob = fit(samples, n_ax)
    blob["device"] = device
    if args.fit_sectors:
        blob["sectors"] = blob["_diagnostics"]["sectors_measured"]

    out = (args.out or config_path()).expanduser()
    diag = blob.pop("_diagnostics")
    print(f"\n{device}: {len(samples)} samples, {len(trials)} trials", file=sys.stderr)
    for i, a in blob["axes"].items():
        print(f"  axis {i}: rest {a['rest']:+.3f}  travel {a['min']:+.3f} .. {a['max']:+.3f}"
              f"  (dither {a['noise']:.4f}, centre moves {a['drift']:.3f})",
              file=sys.stderr)
    print(f"  weakest push per direction (calibrated): {diag['weakest_push_per_direction']}",
          file=sys.stderr)
    print(f"  -> axis_on {blob['axis_on']}, axis_off {blob['axis_off']}", file=sys.stderr)
    print(f"  stick pointed (deg): {diag['sectors_measured']}  [{diag['sectors_note']}]",
          file=sys.stderr)
    print(f"  ... off the ideal by: {diag['sector_skew_deg']}", file=sys.stderr)
    print(f"  hands-off blocks the centre was measured over: "
          f"{diag['hands_off_blocks_believed']}", file=sys.stderr)
    print(f"  wrong direction on the way out: {diag['wrong_direction_transients_ms']} ms"
          f"  -> dwell {blob['stick_dwell_s'] * 1000:.0f} ms before a direction is believed",
          file=sys.stderr)
    print(f"  worst margin between two directions: {diag['margin_as_shipped_deg']} deg"
          f" as shipped, {diag['margin_if_fitted_deg']} deg if those angles were fitted"
          f" (--fit-sectors); under ~10 means a push nearly read as its neighbour",
          file=sys.stderr)
    if "sectors" in blob:
        print(f"  --fit-sectors: writing them in, so this rig classifies against "
              f"{blob['sectors']}", file=sys.stderr)
    if trials:
        hits = sum(t["hit"] for t in trials)
        print(f"  cue agreement: {hits}/{len(trials)}", file=sys.stderr)

    if args.dry_run:
        print("\n--dry-run: nothing written\n" + json.dumps(blob, indent=2), file=sys.stderr)
        return

    raw = save(blob, samples, trials, out)
    print(f"\ncalibration -> {out}\nraw samples -> {raw}", file=sys.stderr)

    if not args.no_scope:
        # Look at it before trusting it: a calibration is a claim about the
        # hardware, and the scope is where that claim is checked by pushing the
        # stick and watching the thresholds it just wrote.
        print("opening the scope to check it - ESC closes it", file=sys.stderr)
        _load_calibration()      # pick up what was just written
        scope_main()


W, H = 1020, 680
WINDOW_S, TRAIL_S = 8.0, 0.8

ARROWS = ["UP", "DOWN", "LEFT", "RIGHT"]


def _vy(v: float, cy: int, half: int) -> int:
    """A calibrated value as a y pixel on the scrolling plot."""
    return int(cy - max(-1.0, min(1.0, v)) * half)


def _tx(t: float, now: float, px: int, pw: int) -> int:
    """A timestamp as an x pixel, newest at the right edge."""
    return int(px + pw - (now - t) / WINDOW_S * pw)


def _qpos(vx: float, vz: float, mid: tuple[int, int], q: int) -> tuple[int, int]:
    """A calibrated x/y pair as a point in the square stick plot."""
    return (int(mid[0] + max(-1, min(1, vx)) * q // 2),
            int(mid[1] + max(-1, min(1, vz)) * q // 2))


def _save_thresholds() -> str:
    """Write the live thresholds into the calibration file, keeping the rest of it."""
    path = config_path()
    blob = {}
    if path.exists():
        try:
            blob = json.loads(path.read_text())
        except ValueError:
            pass
    blob["axis_on"], blob["axis_off"] = round(_AXIS_ON, 3), round(_AXIS_OFF, 3)
    blob["axis_diagonal"] = round(_DIAGONAL, 2)
    blob["stick_dwell_s"] = round(_DWELL, 3)
    blob.setdefault("_comment", "thresholds edited in fmri-pad-scope")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob, indent=2) + "\n")
    return (f"saved {blob['axis_on']} / {blob['axis_off']} / "
            f"diag {blob['axis_diagonal']} -> {path}")


def scope_main() -> None:
    global _AXIS_ON, _AXIS_OFF, _DIAGONAL, _DWELL
    argparse.ArgumentParser(
        description="Watch what the controller sends and what fmri-gym makes of it.",
        epilog="UP/DOWN sets axis_on, LEFT/RIGHT sets axis_off, [ and ] set the "
               "dwell a direction must hold for, D switches between "
               "one direction at a time and diagonals, S saves them into the "
               "calibration, R reloads it, ESC quits.").parse_args()
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("fmri-gym pad scope")
    clock = pygame.time.Clock()
    f_big = pygame.font.SysFont("Menlo,Monaco,monospace", 17)
    f = pygame.font.SysFont("Menlo,Monaco,monospace", 13)
    f_sm = pygame.font.SysFont("Menlo,Monaco,monospace", 11)

    status = init()
    watched = {p: (os.path.getmtime(p) if os.path.exists(p) else 0.0)
               for p in (__file__, str(config_path()))}
    flash, flash_until = "", 0.0
    hist: deque = deque()
    log: deque = deque(maxlen=11)
    prev_held: frozenset = frozenset()
    prev_btn: dict[int, bool] = {}
    t0 = time.perf_counter()

    running = True
    while running:
        now = time.perf_counter() - t0
        for e in pygame.event.get((pygame.QUIT, pygame.KEYDOWN)):
            if e.type == pygame.QUIT:
                running = False
                continue
            if getattr(e, "pad", False):
                continue                    # the pad's own presses are data, not UI
            step = 0.05 if e.mod & pygame.KMOD_SHIFT else 0.01
            if e.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            elif e.key == pygame.K_UP:
                _AXIS_ON = round(min(0.99, _AXIS_ON + step), 3)
            elif e.key == pygame.K_DOWN:
                _AXIS_ON = round(max(0.01, _AXIS_ON - step), 3)
            elif e.key == pygame.K_RIGHT:
                _AXIS_OFF = round(min(0.99, _AXIS_OFF + step), 3)
            elif e.key == pygame.K_LEFT:
                _AXIS_OFF = round(max(0.01, _AXIS_OFF - step), 3)
            elif e.key == pygame.K_s:
                flash, flash_until = _save_thresholds(), now + 2.5
                watched[str(config_path())] = os.path.getmtime(config_path())
            elif e.key == pygame.K_r:
                _load_calibration()
                status = init()
                flash, flash_until = "reloaded", now + 1.5
            elif e.key == pygame.K_LEFTBRACKET:
                _DWELL = round(max(0.0, _DWELL - 0.01), 3)
            elif e.key == pygame.K_RIGHTBRACKET:
                _DWELL = round(min(0.3, _DWELL + 0.01), 3)
            elif e.key == pygame.K_d:
                _DIAGONAL = (0.7 if _DIAGONAL >= 1.0
                                 else round(min(1.0, _DIAGONAL + step * 2), 2))
                flash, flash_until = f"axis_diagonal {_DIAGONAL}", now + 1.5
            elif e.key == pygame.K_c:
                hist.clear(); log.clear()

        if now % 1 < 0.02:                  # pick up an edit to either file
            for p, was in list(watched.items()):
                m = os.path.getmtime(p) if os.path.exists(p) else 0.0
                if m != was:
                    watched[p] = m
                    _load_calibration()
                    status = init()
                    flash, flash_until = f"{os.path.basename(p)} changed - reloaded", now + 2.0

        pump()
        pressed = held()
        js = _joysticks[0] if _joysticks else None
        raw = [js.get_axis(i) for i in range(js.get_numaxes())] if js else [0.0, 0.0]
        a0, a1 = _normalise(0, raw[0]), _normalise(1, raw[1])
        btn = {i: bool(js.get_button(i)) for i in range(js.get_numbuttons())} if js else {}

        for name in sorted(pressed - prev_held):
            log.appendleft((now, f"{name:<6} DOWN", LIVE))
        for name in sorted(prev_held - pressed):
            log.appendleft((now, f"{name:<6} up", INK3))
        for i, down in btn.items():
            if down != prev_btn.get(i, False):
                log.appendleft((now, (f"button {BUTTONS.get(i, i):<3} "
                                      f"{'press' if down else 'release':<8}"
                                      f"-> {BUTTON_KEYS.get(i) or 'unmapped'}"),
                                LIVE if down else INK3))
        prev_held, prev_btn = pressed, btn
        hist.append((now, a0, a1, pressed))
        while hist and now - hist[0][0] > WINDOW_S:
            hist.popleft()

        # ------------------------------------------------------------- draw
        screen.fill(SURFACE)
        on, off = _AXIS_ON, _AXIS_OFF
        screen.blit(f_big.render("pad scope", True, INK), (20, 14))
        screen.blit(f.render(status[:74], True, INK2), (140, 18))
        screen.blit(f.render(f"axis_on {on:.2f}   axis_off {off:.2f}", True, INK), (W - 320, 10))
        screen.blit(f_sm.render("UP/DOWN on, LEFT/RIGHT off, S saves to the calibration"
                                if off < on else "OFF >= ON: no hysteresis, expect chatter",
                                True, INK3 if off < on else WARN), (W - 320, 28))
        if flash and now < flash_until:
            screen.blit(f.render(flash, True, LIVE), (20, 40))

        bx, by, bw = 20, 66, 160
        for i in range(6):
            x = bx + i * (bw + 4)
            key, down = BUTTON_KEYS.get(i), btn.get(i, False)
            label, hue = FACE.get(i, (f"b{i}", INK2))
            pygame.draw.rect(screen, hue if down else PANEL, (x, by, bw, 62), border_radius=4)
            pygame.draw.rect(screen, hue if down else GRID, (x, by, bw, 62), 1, border_radius=4)
            ink = SURFACE if down else INK2
            screen.blit(f_big.render(label, True, SURFACE if down else hue), (x + 12, by + 8))
            screen.blit(f_sm.render(f"b{i}", True, ink), (x + 12, by + 40))
            screen.blit(f_big.render(key or "-", True, ink if key else INK3), (x + bw - 46, by + 20))
            if down and key in pressed:
                pygame.draw.circle(screen, SURFACE, (x + bw - 12, by + 12), 4)

        px, py, pw, ph = 20, 176, 630, 232
        cy, half = py + ph // 2, ph // 2
        pygame.draw.rect(screen, PANEL, (px, py, pw, ph))
        for lvl in (-1.0, -0.5, 0.5, 1.0):
            pygame.draw.line(screen, GRID, (px, _vy(lvl, cy, half)), (px + pw, _vy(lvl, cy, half)))
            screen.blit(f_sm.render(f"{lvl:+.1f}", True, INK3), (px + 3, _vy(lvl, cy, half) - 12))
        pygame.draw.line(screen, (60, 68, 86), (px, cy), (px + pw, cy))
        for sgn in (1, -1):
            pygame.draw.line(screen, (120, 130, 150), (px, _vy(sgn * on, cy, half)), (px + pw, _vy(sgn * on, cy, half)))
            for seg in range(px, px + pw, 10):
                pygame.draw.line(screen, (80, 88, 106), (seg, _vy(sgn * off, cy, half)), (seg + 5, _vy(sgn * off, cy, half)))
        if len(hist) > 1:
            for idx, hue in ((1, X_HUE), (2, Y_HUE)):
                pygame.draw.lines(screen, hue, False, [(_tx(h[0], now, px, pw), _vy(h[idx], cy, half)) for h in hist], 2)
            for a, b in zip(hist, list(hist)[1:]):
                if (b[3] - a[3]) & set(ARROWS):
                    pygame.draw.line(screen, LIVE, (_tx(b[0], now, px, pw), py), (_tx(b[0], now, px, pw), py + ph), 1)
        screen.blit(f.render("stick x", True, X_HUE), (px + 8, py + 6))
        screen.blit(f.render("stick y", True, Y_HUE), (px + 78, py + 6))
        screen.blit(f_sm.render("calibrated units", True, INK3), (px + pw - 100, py + 6))

        ry = py + ph + 10
        for r, name in enumerate(ARROWS):
            y = ry + r * 16
            _arrow(screen, name, (px + 8, y + 7), 7,
                   LIVE if name in pressed else INK3)
            pygame.draw.rect(screen, PANEL, (px + 46, y + 2, pw - 46, 10))
            if len(hist) > 1:
                for a, b in zip(hist, list(hist)[1:]):
                    if name in b[3]:
                        x1 = int(px + pw - (now - a[0]) / WINDOW_S * pw)
                        x2 = int(px + pw - (now - b[0]) / WINDOW_S * pw)
                        pygame.draw.rect(screen, LIVE, (max(x1, px + 46), y + 2, max(1, x2 - x1), 10))

        qx, qy, q = 676, 176, 310
        pygame.draw.rect(screen, PANEL, (qx, qy, q, q))
        mid = (qx + q // 2, qy + q // 2)
        pygame.draw.line(screen, GRID, (qx, mid[1]), (qx + q, mid[1]))
        pygame.draw.line(screen, GRID, (mid[0], qy), (mid[0], qy + q))
        # Draw the rule that is actually deciding, so the picture cannot imply a
        # different one: with the classifier that is a distance from rest and a
        # set of fitted directions, not a pair of per-axis boxes.
        classifying = _DIAGONAL >= 1.0
        for frac, col in ((on, (120, 130, 150)), (off, (70, 78, 96))):
            r = int(frac * q // 2)
            if classifying:
                pygame.draw.circle(screen, col, mid, r, 1)
            else:
                pygame.draw.rect(screen, col, (mid[0] - r, mid[1] - r, r * 2, r * 2), 1)
        if classifying:
            ordered = sorted(_sectors.items(), key=lambda kv: kv[1])
            for i, (name, angle) in enumerate(ordered):
                reach = q // 2 - 4
                end = (mid[0] + math.cos(angle) * reach, mid[1] + math.sin(angle) * reach)
                pygame.draw.line(screen, (58, 66, 84), mid, end, 1)
                _arrow(screen, name, (int(end[0]), int(end[1])), 9,
                       LIVE if name in pressed else INK3)
                nxt = ordered[(i + 1) % len(ordered)][1]
                edge = angle + ((nxt - angle) % (2 * math.pi)) / 2   # the boundary between them
                pygame.draw.line(screen, (44, 50, 64), mid,
                                 (mid[0] + math.cos(edge) * reach,
                                  mid[1] + math.sin(edge) * reach), 1)
        for i, h in enumerate([h for h in hist if now - h[0] < TRAIL_S]):
            k = 40 + i // 2
            pygame.draw.circle(screen, (k, k + 6, k + 16), _qpos(h[1], h[2], mid, q), 2)
        cur = _qpos(a0, a1, mid, q)
        pygame.draw.circle(screen, SURFACE, cur, 9)
        pygame.draw.circle(screen, LIVE if pressed & set(ARROWS) else X_HUE, cur, 6)
        screen.blit(f.render("stick position", True, INK2), (qx + 8, qy + 6))
        screen.blit(f_sm.render(f"calibrated {a0:+.3f} {a1:+.3f}", True, INK), (qx + 8, qy + q - 32))
        screen.blit(f_sm.render(f"raw        {raw[0]:+.3f} {raw[1]:+.3f}", True, INK3),
                    (qx + 8, qy + q - 18))
        centre = " ".join(f"{_zero.get(i, _calibration.get(i, {}).get('rest', 0.0)):+.3f}"
                          for i in (0, 1))
        screen.blit(f_sm.render(f"centre     {centre}  (tracked while idle)", True, INK3),
                    (qx + 8, qy + q - 4))
        rule = ("one direction at a time" if _DIAGONAL >= 1.0
                else f"diagonals past {_DIAGONAL:.2f}")
        screen.blit(f_sm.render(f"{rule}   dwell {_DWELL * 1000:.0f} ms"
                                f"   pushed {math.hypot(a0, a1):.2f}", True, INK2),
                    (qx + 8, qy + q - 46))

        ly = ry + 4 * 16 + 14
        screen.blit(f.render("events", True, INK2), (px, ly))
        for i, (t, text, colour) in enumerate(log):
            screen.blit(f_sm.render(f"{t:7.2f}s  {text}", True, colour), (px, ly + 20 + i * 14))
        screen.blit(f_sm.render("UP/DOWN on   LEFT/RIGHT off   [ ] dwell   D diagonals   S save   R reload   "
                                "C clear   ESC quit", True, INK3), (px, H - 22))
        if not js:
            screen.blit(f_big.render("no controller plugged in", True, WARN), (qx + 8, mid[1]))

        pygame.display.flip()
        clock.tick(60)
    pygame.quit()


def main() -> None:
    """``python -m fmri_gym.pad``: name each control and the rig key it presses."""
    pygame.init()
    pygame.display.set_mode((320, 64))
    print(init(), file=sys.stderr)
    if not _joysticks:
        return
    print("Press each control (Ctrl-C to stop).", file=sys.stderr)
    try:
        while True:
            before = set(_held)
            pump()
            for name in sorted(_held - before):
                print(f"  pressed -> {name}")
            for name in sorted(before - _held):
                print(f"  released   {name}")
            pygame.time.wait(5)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
