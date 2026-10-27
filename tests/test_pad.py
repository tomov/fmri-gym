"""The pad's calibration, with no pad: its file, the run's start, the fit and the procedure.

A scripted pad answers the procedure frame by frame, on a fake clock, so a whole
calibration plays in well under a second and writes into a temporary folder.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import types
from datetime import datetime, timezone

import pygame
import pytest

from fmri_gym import pad
from fmri_gym import pad_calibration as cal
from fmri_gym.pad import Stick

MADE = "2026-01-21T14:15:00+00:00"
AXIS = {"rest": 0.0, "low": -1.0, "high": 1.0, "drift": 0.01}
STICK_ENTRY = {"made": MADE, "x": AXIS, "y": AXIS, "on": 0.5, "off": 0.3, "diagonal": 1.0,
               "dwell_s": 0.05}
FILE = {"device": "Scripted", "tuned_from": None,
        "sticks": {"left": STICK_ENTRY, "right": {**STICK_ENTRY, "on": 0.6}}}


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    """Calibrations written to and read from a folder of the test's own."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    return pad.calibrations_dir()


def _write(folder, name: str, blob: dict, mtime: int):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(json.dumps(blob))
    os.utime(path, (mtime, mtime))
    return path


def test_a_calibration_round_trips_through_its_file():
    loaded = pad.Calibration.from_dict(copy.deepcopy(FILE))
    assert loaded.record() == FILE
    assert loaded.sticks[Stick.RIGHT].tuning.on == 0.6
    assert loaded.sticks[Stick.LEFT].made == datetime(2026, 1, 21, 14, 15, tzinfo=timezone.utc)


def _set(blob: dict, path: str, value) -> dict:
    """``blob`` with the field at dotted ``path`` set to ``value`` (or removed, for ...)."""
    out = copy.deepcopy(blob)
    *parents, key = path.split(".")
    d = out
    for p in parents:
        d = d[p]
    if value is ...:
        del d[key]
    else:
        d[key] = value
    return out


@pytest.mark.parametrize("field, value, says", [
    ("sticks.left.on", "0.5", "sticks.left.on: expected a number in (0, 1), got '0.5'"),
    ("sticks.left.off", 0.6, "sticks.left.off: expected a number in (0, on=0.5], got 0.6"),
    ("sticks.left.diagonal", 0, "sticks.left.diagonal: expected a number in (0, 1], got 0"),
    ("sticks.left.dwell_s", 0.5, "sticks.left.dwell_s: expected a number of seconds in [0, 0.3]"),
    ("sticks.left.x.drift", True, "sticks.left.x.drift: expected a number >= 0, got True"),
    ("sticks.left.x.low", 0.5, "sticks.left.x: expected low < rest < high"),
    ("sticks.left.x.noise", 0.001, "sticks.left.x.noise: unknown key"),
    ("sticks.left.made", "2026-10-21T14:15:00", "sticks.left.made: expected a time such as"),
    ("sticks.left.y", ..., "sticks.left.y: missing"),
    ("sticks.middle", STICK_ENTRY, "sticks.middle: unknown stick (expected left, right)"),
    ("device", ..., "device: missing"),
    ("tuned_from", 3, "tuned_from: expected null or a file name, got 3"),
    ("sectors", {}, "sectors: unknown key"),
])
def test_each_wrong_field_is_refused_by_its_file_and_name(tmp_path, field, value, says):
    path = tmp_path / "Scripted_20261021-141500.json"
    path.write_text(json.dumps(_set(FILE, field, value)))
    with pytest.raises(pad.CalibrationError, match=re.escape(f"{path}: {says}")):
        pad.read(path)


def test_a_run_reads_the_newest_calibration_of_the_pad_plugged_in(config_home):
    _write(config_home, "Scripted_a.json", _set(FILE, "sticks.left.on", 0.31), 1000)
    newest = _write(config_home, "Scripted_b.json", _set(FILE, "sticks.left.on", 0.33), 3000)
    _write(config_home, "Scripted_c.json", _set(FILE, "sticks.left.on", 0.32), 2000)
    _write(config_home, "Other_d.json", {**FILE, "device": "Other"}, 4000)   # newer, another pad
    _write(config_home, "Other_e.json", {"device": "Other", "axes": {}}, 5000)  # bad, not ours
    path, found = pad.newest("Scripted")
    assert (path, found.sticks[Stick.LEFT].tuning.on) == (newest, 0.33)
    assert pad.newest("Nobody") is None


def test_a_malformed_newest_calibration_stops_the_run_with_no_fallback(config_home):
    _write(config_home, "Scripted_a.json", FILE, 1000)
    bad = _write(config_home, "Scripted_b.json", _set(FILE, "sticks.left.on", "0.5"), 2000)
    with pytest.raises(pad.CalibrationError, match=re.escape(f"{bad}: sticks.left.on")):
        pad.newest("Scripted")
    bad.write_text("{not json")                    # names no device: it may be this one's
    with pytest.raises(pad.CalibrationError, match="not JSON"):
        pad.newest("Scripted")


class _ScriptedPad:
    """A pad whose readings are a script, one frame each; at its end, ESC."""

    def __init__(self, sticks: tuple[Stick, ...], frames=(), name: str = "Scripted") -> None:
        self.name, self._sticks, self.frames = name, sticks, iter(frames)
        self.layout = types.SimpleNamespace(name="scripted")

    def sticks(self) -> tuple[Stick, ...]:
        return self._sticks

    def controls(self) -> tuple[str, ...]:
        return (*(d for s in self._sticks for d in s.directions), "A", "B")

    def read(self):
        frame = next(self.frames, None)
        if frame is None:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
            return self._axes({}), set()
        return frame

    def _axes(self, moved: dict[int, float]) -> dict[int, float]:
        return {a: moved.get(a, 0.0) for s in self._sticks for a in s.axes}


def _stick_frames(p: _ScriptedPad, stick: Stick, wrong: str | None = None) -> list:
    """Hands off, roll, let go, then each direction twice (``wrong`` pushed once first)."""
    x, y = stick.axes
    still = [(p._axes({x: 0.02}), set())] * 70
    circle = [(p._axes({x: math.cos(2 * math.pi * k / 48), y: math.sin(2 * math.pi * k / 48)}),
               set()) for k in range(48 * 3)]
    rest = [(p._axes({x: -0.05, y: 0.01}), set())] * 70
    out = still + circle + rest
    for d in ([wrong] if wrong else []) + [d for d in cal.CUE_DIRECTIONS for _ in range(2)]:
        axis, sign = cal.DIRECTIONS[dict(zip(pad.STICK, stick.directions))[d]]
        out += [rest[0], (p._axes({x: -0.05, y: 0.01, axis: 0.9 * sign}), set())]
    return out


def _button_frames(p: _ScriptedPad) -> list:
    return [f for b in ("A", "B") for f in [(p._axes({}), set()), (p._axes({}), {b})] * 2]


class _FakeClock:
    """64 frames a second, exactly: a frame's time is never rounded across a window's edge."""

    t = 0.0

    def tick(self) -> float:
        _FakeClock.t += 1 / 64
        return _FakeClock.t


@pytest.fixture
def screen(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setattr(cal, "_FrameClock", _FakeClock)
    pygame.init()
    pygame.event.clear()
    return pygame.Surface((900, 600))


def test_a_calibration_of_both_sticks_measures_each_and_confirms_each_control(config_home,
                                                                              screen):
    p = _ScriptedPad((Stick.LEFT, Stick.RIGHT))
    p.frames = iter(_stick_frames(p, Stick.LEFT, wrong="LEFT") + _stick_frames(p, Stick.RIGHT)
                    + _button_frames(p))
    result = cal.calibrate(p, screen, lambda: None, p.sticks())
    assert (result.confirmed, result.corrections) == (4 + 4 + 2, 1)
    left = result.calibration.sticks[Stick.LEFT]
    assert (left.x.rest, left.x.low, left.x.high, left.x.drift) == (-0.05, -1.0, 1.0, 0.07)
    assert (left.tuning.on, left.tuning.off) == (cal.CAL_ON, cal.CAL_OFF)
    assert set(result.calibration.sticks) == {Stick.LEFT, Stick.RIGHT}
    assert pad.newest("Scripted") == (result.path, result.calibration)
    assert result.path.with_suffix(".csv").exists()
    from fmri_gym.checks import _pad_line
    line = _pad_line({"present": True, **result.record()})
    assert "left stick on past 0.50" in line and "10 controls confirmed, 1 wrong" in line


def test_one_stick_calibrated_alone_keeps_the_other_as_it_was_measured(config_home, screen):
    old = pad.Calibration.from_dict(_set(FILE, "sticks.left.on", 0.4))
    base = pad.write(old)
    p = _ScriptedPad((Stick.LEFT, Stick.RIGHT))
    p.frames = iter(_stick_frames(p, Stick.RIGHT))         # no buttons: not every stick
    result = cal.calibrate(p, screen, lambda: None, (Stick.RIGHT,))
    assert result.path != base and base.read_text() == json.dumps(old.record(), indent=2) + "\n"
    assert result.calibration.sticks[Stick.LEFT] == old.sticks[Stick.LEFT]   # its made too
    assert result.calibration.sticks[Stick.RIGHT].made > old.sticks[Stick.RIGHT].made
    assert result.confirmed == 4


def test_a_calibration_stopped_part_way_writes_nothing(config_home, screen):
    p = _ScriptedPad((Stick.LEFT,))
    p.frames = iter(_stick_frames(p, Stick.LEFT)[:100])
    assert cal.calibrate(p, screen, lambda: None, (Stick.LEFT,)) is None
    assert not config_home.exists()


def test_the_fit_raises_the_threshold_of_a_stick_that_dithers():
    rows = ([{"t": k / 60, "phase": "cold", "cue": "left", "ax0": 0.0, "ax1": 0.0}
             for k in range(60)]
            + [{"t": 1 + k / 60, "phase": "roll", "cue": "left", "ax0": v, "ax1": -v}
               for k, v in enumerate((-0.5, 1.0, 0.2))]
            + [{"t": 2 + k / 60, "phase": "rest", "cue": "left", "ax0": 0.1 + 0.02 * (k % 2),
                "ax1": 0.0} for k in range(60)])
    stick = cal.fit(rows, datetime.now(timezone.utc))[Stick.LEFT]
    assert (stick.x.rest, stick.x.low, stick.x.high, stick.x.drift) == (0.11, -0.5, 1.0, 0.11)
    # dither 0.01 of the 0.61 travel to the low side: ten times it is 0.164, under 0.5
    assert (stick.tuning.on, stick.tuning.off) == (0.5, 0.3)
    shaky = [{**r, "ax0": 0.1 + 0.1 * (k % 2)} if r["phase"] == "rest" else r
             for k, r in enumerate(rows)]
    tuning = cal.fit(shaky, datetime.now(timezone.utc))[Stick.LEFT].tuning
    assert tuning.on > cal.CAL_ON and tuning.off == round(0.6 * tuning.on, 3)


@pytest.fixture
def plugged(monkeypatch):
    """A two-stick pad plugged in, and the pad module on without touching SDL."""
    p = _ScriptedPad((Stick.LEFT, Stick.RIGHT))
    monkeypatch.setattr(pad, "_enabled", True)
    monkeypatch.setattr(pad, "_pads", {1: p})
    monkeypatch.setattr(pad, "find", lambda: p)
    yield p
    pad.use(None)
    pad._held.clear()
    pad._raw.clear()


def test_a_run_refuses_a_bound_stick_with_no_calibration(config_home, plugged):
    with pytest.raises(pad.CalibrationError, match="no calibration of 'Scripted' in .*rig check"):
        pad.init(True, {"UP": "UP", "A": "SPACE"})
    assert "UNCALIBRATED" in pad.init(True, {"A": "SPACE"})            # buttons only: it starts
    status = pad.init(True, {"UP": "UP"}, is_calibration_required=False)  # the rig check
    assert "left stick UNCALIBRATED" in status
    assert pad.tuning(Stick.LEFT) == pad.UNCALIBRATED and not pad.is_calibrated(Stick.LEFT)
    path = pad.write(pad.Calibration.from_dict(_set(FILE, "sticks.right", ...)))
    with pytest.raises(pad.CalibrationError, match="has no right stick, which the rig binds"):
        pad.init(True, {"RS_UP": "W"})
    assert str(path) in pad.init(True, {"UP": "UP"})


def test_each_stick_reads_its_own_tuning(config_home, plugged):
    blob = _set(_set(FILE, "sticks.left.dwell_s", 0), "sticks.right.dwell_s", 0)
    pad.use((config_home / "x.json", pad.Calibration.from_dict(blob)))
    pad._bindings = {}
    pad._raw.update({0: 0.55, 1: 0.0, 2: 0.55, 3: 0.0})   # past the left's 0.5, not the right's
    pad._resolve()
    assert pad.held_controls() == {"RIGHT"}
    pad._raw.update({2: 0.65})
    pad._resolve()
    assert pad.held_controls() == {"RIGHT", "RS_RIGHT"}


def test_a_check_controls_phase_refuses_pad_reps():
    from fmri_gym.config.phases import CheckControls
    assert any("pad_reps" in p for p in CheckControls.problems({"pad_reps": 3}))


@pytest.fixture
def scope(config_home, plugged, screen):
    """The tool's window on the dummy display, with the scripted pad plugged in."""
    from fmri_gym import pad_scope
    plugged.source = lambda control: f"b{control}"
    plugged.read = lambda: (plugged._axes({}), set())
    return pad_scope._Scope({"UP": "UP", "A": "SPACE"}, has_rig=True)


def test_the_keyboard_does_nothing_in_the_tool_but_esc(scope):
    path = pad.write(pad.Calibration.from_dict(copy.deepcopy(FILE)))
    scope.reload("")
    before = (pad.tuning(Stick.LEFT), scope.stick, sorted(pad.calibrations_dir().iterdir()))
    for key in (pygame.K_s, pygame.K_d, pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT,
                pygame.K_RIGHT, pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET, pygame.K_TAB):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0, unicode=""))
    scope.events()
    assert (pad.tuning(Stick.LEFT), scope.stick,
            sorted(pad.calibrations_dir().iterdir())) == before and scope.running
    assert pad.in_use()[0] == path
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0, unicode=""))
    scope.events()
    assert not scope.running


@pytest.mark.parametrize("action", ["on-", "on+", "off-", "off+", "dwell-", "dwell+", "diag"])
def test_each_tuning_button_moves_one_field_within_bounds(action):
    from fmri_gym import pad_scope
    for start in (pad.UNCALIBRATED, pad.StickTuning(0.99, 0.99, 0.7, 0.3),
                  pad.StickTuning(0.5, 0.01, 1.0, 0.0)):
        moved = pad_scope._retuned(start, action)
        assert sum(a != b for a, b in zip(vars(moved).values(), vars(start).values())) <= 1
        pad.StickTuning.from_dict(moved.record(), "")      # what a file allows
    middle = pad.StickTuning(0.5, 0.3, 1.0, 0.05)
    assert pad_scope._retuned(middle, action) != middle


def test_save_writes_a_new_newest_file_and_leaves_its_source(scope):
    from fmri_gym import pad_scope
    source = pad.write(pad.Calibration.from_dict(copy.deepcopy(FILE)))
    os.utime(source, (1000, 1000))
    was = source.read_bytes()
    scope.reload("")
    pad_scope._ACTIONS["on+"](scope)
    scope.click(next(b.rect.center for b in scope.buttons if b.action == "save"))
    assert source.read_bytes() == was
    path, saved = pad.newest("Scripted")
    assert path != source and saved.tuned_from == source.name
    assert saved.sticks[Stick.LEFT].tuning.on == 0.51
    assert saved.sticks[Stick.LEFT].made == pad.read(source).sticks[Stick.LEFT].made
    assert pad.in_use()[0] == path


def test_the_watch_reloads_only_on_its_one_second_tick(scope, monkeypatch):
    reloads = []
    monkeypatch.setattr(scope, "reload", reloads.append)
    start = scope.next_watch
    pad.write(pad.Calibration.from_dict(copy.deepcopy(FILE)))   # a calibration added
    scope.watch(start - 0.5)
    assert reloads == []
    scope.watch(start)
    assert reloads == ["a calibration changed: reloaded"] and scope.next_watch == start + 1.0
    scope.draw()                                                  # and it draws


def test_a_click_is_mapped_back_to_the_picture_at_any_window_size(scope):
    from fmri_gym import pad_scope
    canvas = scope.canvas
    canvas.window.size = (900, 1200)                              # bars above and below
    assert canvas.window.size == (900, 1200)
    s = 900 / pad_scope.W
    top = (1200 - int(pad_scope.H * s)) // 2
    assert canvas.to_canvas((int(100 * s), top + int(40 * s))) == pytest.approx((100, 40), abs=2)
    clear = next(b for b in scope.buttons if b.action == "clear")
    scope.log.append((0.0, "a press", None))
    scope.click(canvas.to_canvas((int(clear.rect.centerx * s), top + int(clear.rect.centery * s))))
    assert not scope.log
