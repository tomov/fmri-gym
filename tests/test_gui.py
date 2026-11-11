"""The rig measurement plan and the windows, built offscreen: a window that does not open is a rig
that cannot be set up without JSON."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from humans.config import curriculum
from humans.config import rig as rig_file
from humans.inputs import ControlNames
from humans.standalone.session_script import read_session, write_session
from humans.sync.triggers import TriggerSettings

RIG = "configs/examples/rigs/laptop.json"
EXAMPLES = rig_file.CONFIGS / "examples" / "rigs"
GEOMETRY = {"width_cm": 53.1, "height_cm": 29.9, "distance_cm": 60.0}


@pytest.fixture
def gui():
    """Skip the test where the gui extra (PySide6) is not installed."""
    pytest.importorskip("PySide6", reason="the gui extra: uv sync --extra gui")


def test_the_measurement_plan_follows_the_rig(gui):
    from humans.standalone.rig_setup.window import measurement_plan

    laptop = measurement_plan(rig_file.RigConfig.open(str(EXAMPLES / "laptop.json")))
    meg = measurement_plan(rig_file.RigConfig.open(str(EXAMPLES / "meg.json")))
    assert "Triggers: not tested" in laptop and "Responses: not tested" in laptop
    assert "serial" in meg and "1000 Hz" in meg and "recording" in meg
    typed = rig_file.RigConfig.from_dict({"screen": {"geometry": GEOMETRY},
                                          "response_keys": {"1": "index"}},
                                         str(EXAMPLES / "typed.json"))
    assert "1 (index)" in measurement_plan(typed)


# -- the windows -------------------------------------------------------------------------


@pytest.fixture
def qt(gui, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", os.environ.get("QT_QPA_PLATFORM", "offscreen"))
    from humans.standalone import widgets
    return widgets.app()


def test_the_session_editor_opens_a_run_and_a_session(qt):
    from humans.standalone.session_editor import _Editor, listed_files

    editor = _Editor(curriculum.new_curriculum(), None)
    assert len(editor.steps) == 1 and editor.session_path is None
    assert "fmri-play" in editor.play_hint.text()
    session = Path(sorted(listed_files("sh"))[0])
    editor.open_session(str(session))
    read = read_session(session.read_text())
    assert editor._session_text() == write_session(read)
    editor.rig_pick.set_path("")
    assert editor._session_text() == write_session(replace(read, rig=None))
    assert "RIG=" in editor.play_hint.text()


def _rig_window(name: str, path: str | None = None, **kwargs):
    from humans.standalone.rig_setup.window import _RigWindow

    source = str(EXAMPLES / f"{name}.json")
    return _RigWindow(rig_file.draft(source), path or source, **kwargs), rig_file.load(source)


@pytest.mark.parametrize("name", ["xbox", "meg"])
def test_the_rig_window_gives_back_the_file(qt, name):
    window, written = _rig_window(name)
    rig = window.rig
    assert {**rig, "triggers": None} == {**written, "triggers": None}
    # the triggers section is written in full: the same settings, every default stated
    assert (TriggerSettings.from_dict(rig["triggers"])
            == TriggerSettings.from_dict(written["triggers"]))


def test_a_rig_without_triggers_is_not_given_any_until_chosen(qt):
    window, written = _rig_window("minimal")
    assert "triggers" not in written and "triggers" not in window.rig
    assert window._triggers_problems()


def test_response_keys_and_the_rig_measurement_are_the_rig_files(qt):
    window, _ = _rig_window("laptop")
    window._add_response_key("1", "index finger")
    window.measurement_form.write("photodiode", "soundcard")
    window.measurement_form.write("has_mic", True)
    rig = window.rig
    assert rig["response_keys"] == {"1": "index finger"}
    assert rig["rig_measurement"] == {"photodiode": "soundcard", "has_mic": True}
    assert "1 (index finger)" in window.plan.text()


def test_a_rig_measurement_waits_for_its_triggers_to_be_chosen(qt):
    window, _ = _rig_window("minimal", need_triggers=True)
    assert window.stack.currentIndex() == 3 and not window.next.isEnabled()
    window.preset.setCurrentIndex(0)
    window._apply_preset()
    assert window.next.isEnabled()
    assert window.rig["triggers"]["sync"]["mode"] == "wait"


def test_only_the_options_that_apply_are_shown(qt):
    window, _ = _rig_window("meg")
    visible = lambda form, key: form.layout().isRowVisible(form.widgets[key])  # noqa: E731
    window.switches["is_fullscreen"].setChecked(True)
    assert not window.size_pick.isEnabled()
    assert not window.code_box.isHidden() and visible(window.trigger_form, "port")
    window.trigger_form.write("backend", "null")
    assert window.code_box.isHidden()
    assert not visible(window.trigger_form, "port") and not visible(window.trigger_form,
                                                                     "on_frame")
    assert not visible(window.measurement_form, "recording_hz")
    window.measurement_form.write("photodiode", "none")
    assert not visible(window.measurement_form, "corner") and not visible(window.measurement_form,
                                                                    "input_device")


def test_a_new_rig_is_filled_step_by_step_then_saved_then_measured(qt, tmp_path):
    path = str(tmp_path / "site" / "rig.json")
    window, _ = _rig_window("minimal", path=path)
    assert not window.next.isEnabled()  # the rig is not described yet
    for key, w in window.fields.items():
        if key == "modality":
            w.setCurrentText("behav")
        elif key == "pi":
            w.setText("MT")
        elif key != "notes":
            w.setText("none")
    for step in range(3):
        assert window.next.isEnabled(), window.problems.text()
        window.next.click()
    assert window.stack.currentIndex() == 3 and not window.next.isEnabled()
    window.preset.setCurrentIndex(window.preset.findText("Behavioural (no scanner)"))
    window._apply_preset()
    window.next.click()
    window.next.click()
    assert window.stack.currentIndex() == 5
    assert window.save.isEnabled() and not window.short_measurement.isEnabled()
    window.save.click()
    assert rig_file.RigConfig.open(path).fields["pi"] == "MT"
    assert window.short_measurement.isEnabled() and not window.save.isEnabled()
    window.switches["has_audio"].setChecked(False)  # a change: save again before a rig measurement
    assert not window.short_measurement.isEnabled() and window.save.isEnabled()


def test_the_screen_step_needs_the_screen_geometry(qt, tmp_path):
    from humans.standalone.rig_setup.window import _RigWindow

    window = _RigWindow({}, str(tmp_path / "site" / "rig.json"))
    window._show_step(1)
    assert not window.next.isEnabled() and "image width" in window.problems.text()
    window.diagonal.setText("81.3")
    window.aspect.setText("16:9")
    window._size_from_diagonal()
    assert window._geometry() == {"width_cm": 70.9, "height_cm": 39.9}
    window.geometry_fields["distance_cm"].setText("120")
    assert window.next.isEnabled(), window.problems.text()
    assert window.rig["screen"]["geometry"] == {"width_cm": 70.9, "height_cm": 39.9,
                                                "distance_cm": 120.0}
    assert "32.9 x 18.9 degrees" in window.geometry_note.text()
    window.geometry_fields["height_cm"].setText("53.2")
    window._diagonal_from_size()
    assert (window.diagonal.text(), window.aspect.text()) == ("88.6", "4:3")


@pytest.mark.parametrize("text, ratio", [("16:9", 16 / 9), ("1.78:1", 1.78), ("1.5", 1.5)])
def test_an_aspect_ratio_is_read_as_typed(gui, text, ratio):
    from humans.standalone.rig_setup.window import aspect_text, parse_aspect

    assert parse_aspect(text) == pytest.approx(ratio)
    assert aspect_text(1920, 1080) == "16:9" and aspect_text(1470, 956) == "1.54:1"


def test_flags_fmri_play_would_refuse_are_problems(gui):
    from humans.standalone.session_editor import _flag_problems

    assert _flag_problems({}) == [] and _flag_problems({"duration": 60.0, "loop": True}) == []
    assert _flag_problems({"duration": 0.0})  # no length at all
    assert _flag_problems({"loop": True})  # looping needs a length


CARTPOLE = {"curriculum": [{"type": "game", "backend": "gym", "game": "CartPole-v1", "fps": 30,
                            "controls": {"": 0, "Q": 1}}]}


def test_check_reports_a_game_key_none_of_the_rigs_inputs_has(qt):
    from humans.standalone.session_editor import _Editor

    a = ControlNames.CONTROLLER_BUTTON_A
    pad = {"curriculum": [{**CARTPOLE["curriculum"][0], "controls": {"": 0, a: 1}}]}
    editor = _Editor(pad, None)
    editor.rig_pick.set_path(str(EXAMPLES / "laptop.json"))
    assert any(a in p and "laptop.json" in p for p in editor._problems())
    editor.rig_pick.set_path(str(EXAMPLES / "xbox.json"))
    assert not any(a in p for p in editor._problems())
    editor.rig_pick.set_path(str(EXAMPLES / "gmpd3.json"))
    assert editor._problems() == []


def test_a_game_phase_shows_and_keeps_only_what_its_mode_uses(qt):
    from humans.standalone.session_editor import _Editor

    editor = _Editor({"curriculum": [{**CARTPOLE["curriculum"][0], "mode": "episode",
                                      "n_episodes": 3}]}, None)
    editor._show_phase(0)
    form = editor.phase_form
    assert form.layout().isRowVisible(form.widgets["n_episodes"])
    form.write("mode", "duration")
    assert not form.layout().isRowVisible(form.widgets["n_episodes"])
    editor._commit_phase()
    assert "n_episodes" not in editor.phases[0]


def test_an_edit_is_known_until_it_is_saved(qt):
    from humans.standalone.session_editor import _Editor

    editor = _Editor(CARTPOLE, None)
    clean = editor._state()
    editor.flags_form.write("duration", "60")
    assert editor._state() != clean


def test_the_controller_line_says_what_to_do(gui):
    from humans.standalone.rig_setup.window import _controller_line

    xbox = {"index": 0, "device_name": "Xbox Wireless Controller", "input": "xbox_controller",
            "calibration": None, "calibrated": None}
    assert "Plug in" in _controller_line({"devices": [], "error": None})
    assert "never been calibrated" in _controller_line({"devices": [xbox], "error": None})
    assert "last calibrated" in _controller_line(
        {"devices": [{**xbox, "calibrated": 1.7e9}], "error": None})
    assert "not a kind of device" in _controller_line(
        {"devices": [{**xbox, "input": None}], "error": None})


def test_the_inputs_ticked_are_the_rigs(qt):
    window, _ = _rig_window("xbox")
    assert window.rig["inputs"] == ["keyboard", "xbox_controller"]
    window.input_boxes["xbox_controller"].setChecked(False)
    assert window.rig["inputs"] == ["keyboard"]
    window.input_boxes["current_designs_gamepad3"].setChecked(True)
    assert window.rig["inputs"] == ["keyboard", "current_designs_gamepad3"]


def test_a_games_keys_are_edited_in_its_phase(qt):
    from humans.standalone.session_editor import _Editor

    editor = _Editor(CARTPOLE, None)
    editor._show_phase(0)
    assert editor.key_table.get() == {"": 0, "Q": 1}
    editor.key_table.add_row("LEFT", "0")
    editor._commit_phase()
    assert editor.phases[0]["controls"] == {"": 0, "Q": 1, "LEFT": 0}
    assert editor.tabs.count() == 2


def test_a_new_rig_has_the_keyboard_alone(qt, tmp_path):
    from humans.standalone.rig_setup.window import _RigWindow

    window = _RigWindow({}, str(tmp_path / "site" / "rig.json"))
    assert window.rig["inputs"] == ["keyboard"]


def test_a_run_opens_on_its_game(qt):
    from humans.standalone.session_editor import _Editor

    editor = _Editor({"curriculum": [{"type": "fixation", "duration": 1.0},
                                     CARTPOLE["curriculum"][0]]}, None)
    assert editor.edit_index == 1 and editor.key_table is not None


def test_a_usb_box_trigger_line_shows_its_box_and_port(qt):
    window, _ = _rig_window("meg")
    visible = lambda key: window.trigger_form.layout().isRowVisible(  # noqa: E731
        window.trigger_form.widgets[key])
    assert not visible("hid_device")
    window.preset.setCurrentIndex(window.preset.findText(
        "MEG (start from the trigger line, USB-1208FS codes over HID)"))
    window._apply_preset()
    assert visible("hid_device") and visible("port") and visible("pulse_ms")
    triggers = window.rig["triggers"]
    assert (triggers["backend"], triggers["hid_device"], triggers["port"]) == ("hid", "mcc-1208fs",
                                                                               "A")
    assert "mcc-1208fs port A" in window.trigger_text.text()


def test_the_rig_window_stays_open_after_a_calibration(qt, monkeypatch):
    from PySide6 import QtCore

    from humans.standalone.rig_setup import window as rig_window

    monkeypatch.setattr(rig_window.subprocess, "run", lambda *_a, **_k: None)
    monkeypatch.setattr(rig_window, "controller_status", lambda: {"devices": [], "error": None})
    window, _ = _rig_window("xbox")
    still_open = []

    def close_if_still_open():
        still_open.append(window.isVisible())
        window.done(7)

    QtCore.QTimer.singleShot(0, window._calibrate)
    QtCore.QTimer.singleShot(50, close_if_still_open)
    assert window.exec() == 7 and still_open == [True]


def test_the_measurements_unticked_are_left_out_and_one_must_stay(qt):
    from PySide6 import QtCore, QtWidgets

    from humans.standalone.rig_setup.measure import MEASUREMENTS
    from humans.standalone.rig_setup.window import pick_measurements

    rig = rig_file.RigConfig.open(str(EXAMPLES / "meg.json"))
    could_start_with_none = []

    def untick_then_start():
        dialog = QtWidgets.QApplication.activeModalWidget()
        boxes = dialog.findChildren(QtWidgets.QCheckBox)
        start, = (b for b in dialog.findChildren(QtWidgets.QPushButton) if b.text() == "Start")
        for box in boxes:
            box.setChecked(False)
        could_start_with_none.append(start.isEnabled())
        for box in boxes[:-1]:
            box.setChecked(True)
        start.click()

    QtCore.QTimer.singleShot(0, untick_then_start)
    assert pick_measurements(rig, is_long=False) == tuple(MEASUREMENTS)[:-1]
    assert could_start_with_none == [False]
