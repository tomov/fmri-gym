"""The rig check plan and the windows, built offscreen: a window that does not open is a rig
that cannot be set up without JSON."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from fmri_gym.config import curriculum
from fmri_gym.config import rig as rig_file
from fmri_gym.session_script import read_session, write_session
from fmri_gym.triggers import TriggerSettings

RIG = "configs/examples/rigs/laptop.json"
EXAMPLES = rig_file.CONFIGS / "examples" / "rigs"


@pytest.fixture
def gui():
    """Skip the test where the gui extra (PySide6) is not installed."""
    pytest.importorskip("PySide6", reason="the gui extra: uv sync --extra gui")


def test_the_check_plan_follows_the_rig(gui):
    from fmri_gym.gui.rig_config import check_plan

    laptop = check_plan(rig_file.RigConfig.open(str(EXAMPLES / "laptop.json")))
    meg = check_plan(rig_file.RigConfig.open(str(EXAMPLES / "meg.json")))
    assert "Triggers: not tested" in laptop and "Responses: not tested" in laptop
    assert "serial" in meg and "1000 Hz" in meg and "recording" in meg
    typed = rig_file.RigConfig.from_dict({"has_pad": False, "response_keys": {"1": "index"}},
                                         str(EXAMPLES / "typed.json"))
    assert "1 (index)" in check_plan(typed)


# -- the windows -------------------------------------------------------------------------


@pytest.fixture
def qt(gui, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", os.environ.get("QT_QPA_PLATFORM", "offscreen"))
    from fmri_gym.gui import widgets
    return widgets.app()


def test_the_session_editor_opens_a_run_and_a_session(qt):
    from fmri_gym.gui.session_editor import _Editor, listed_files

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
    from fmri_gym.gui.rig_config import _RigWindow

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
    window, written = _rig_window("empty")
    assert "triggers" not in written and "triggers" not in window.rig
    assert window._triggers_problems()


def test_a_key_no_game_can_read_is_refused(qt):
    window, _ = _rig_window("xbox")
    window.bindings.cellWidget(0, 1).addItem("NOT_A_KEY")
    window.bindings.cellWidget(0, 1).setCurrentText("NOT_A_KEY")
    assert window._responses_problems()


def test_response_keys_and_the_rig_check_are_the_rig_files(qt):
    window, _ = _rig_window("laptop")
    window._add_response_key("1", "index finger")
    window.check_form.write("photodiode", "soundcard")
    window.check_form.write("has_mic", True)
    rig = window.rig
    assert rig["response_keys"] == {"1": "index finger"}
    assert rig["rig_check"] == {"photodiode": "soundcard", "has_mic": True}
    assert "1 (index finger)" in window.plan.text()


def test_a_rig_check_waits_for_its_triggers_to_be_chosen(qt):
    window, _ = _rig_window("empty", need_triggers=True)
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
    assert not visible(window.check_form, "recording_hz")
    window.check_form.write("photodiode", "none")
    assert not visible(window.check_form, "corner") and not visible(window.check_form,
                                                                    "input_device")


def test_a_new_rig_is_filled_step_by_step_then_saved_then_checked(qt, tmp_path):
    path = str(tmp_path / "site" / "rig.json")
    window, _ = _rig_window("empty", path=path)
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
    assert window.save.isEnabled() and not window.short_check.isEnabled()
    window.save.click()
    assert rig_file.RigConfig.open(path).fields["pi"] == "MT"
    assert window.short_check.isEnabled() and not window.save.isEnabled()
    window.switches["has_audio"].setChecked(False)  # a change: save again before a check
    assert not window.short_check.isEnabled() and window.save.isEnabled()


def test_flags_fmri_play_would_refuse_are_problems(gui):
    from fmri_gym.gui.session_editor import _flag_problems

    assert _flag_problems({}) == [] and _flag_problems({"duration": 60.0, "loop": True}) == []
    assert _flag_problems({"duration": 5.0})  # shorter than its closing fixation
    assert _flag_problems({"loop": True})  # looping needs a length


CARTPOLE = {"curriculum": [{"type": "game", "backend": "gym", "game": "CartPole-v1", "fps": 30,
                            "keys": {"": 0, "Q": 1}}]}


def test_check_reports_a_game_key_the_rigs_controller_cannot_press(qt):
    from fmri_gym.gui.session_editor import _Editor

    editor = _Editor(CARTPOLE, None)
    editor.rig_pick.set_path(str(EXAMPLES / "xbox.json"))
    assert any("Q" in p and "xbox.json" in p for p in editor._problems())
    editor.rig_pick.set_path(RIG)  # a keyboard: every key can be pressed
    assert editor._problems() == []


def test_a_game_phase_shows_and_keeps_only_what_its_mode_uses(qt):
    from fmri_gym.gui.session_editor import _Editor

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
    from fmri_gym.gui.session_editor import _Editor

    editor = _Editor(CARTPOLE, None)
    clean = editor._state()
    editor.flags_form.write("duration", "60")
    assert editor._state() != clean


def test_the_controller_line_says_what_to_do(gui):
    from fmri_gym.gui.rig_config import _controller_line

    assert "Plug in" in _controller_line({"name": None, "calibrated": None, "error": None})
    assert "never been calibrated" in _controller_line(
        {"name": "Xbox Wireless Controller", "calibrated": None, "error": None})
    assert "last calibrated" in _controller_line(
        {"name": "Xbox Wireless Controller", "calibrated": 1.7e9, "error": None})


def test_bindings_are_saved_only_while_the_controller_is_read(qt):
    window, _ = _rig_window("xbox")
    window.switches["has_pad"].setChecked(False)
    assert "bindings" not in window.rig and window.rig["has_pad"] is False
    window.switches["has_pad"].setChecked(True)
    assert window.rig["bindings"]["A"] == "SPACE"


def test_a_games_keys_are_edited_in_its_phase(qt):
    from fmri_gym.gui.session_editor import _Editor

    editor = _Editor(CARTPOLE, None)
    editor._show_phase(0)
    assert editor.key_table.get() == {"": 0, "Q": 1}
    editor.key_table.add_row("LEFT", "0")
    editor._commit_phase()
    assert editor.phases[0]["keys"] == {"": 0, "Q": 1, "LEFT": 0}
    assert editor.tabs.count() == 2


def test_a_new_rig_has_no_controller_until_one_is_said_then_gets_the_standard_map(qt, tmp_path):
    from fmri_gym.gui.rig_config import STANDARD_BINDINGS, _RigWindow

    window = _RigWindow({}, str(tmp_path / "site" / "rig.json"))
    assert window.rig["has_pad"] is False and "bindings" not in window.rig
    window.switches["has_pad"].setChecked(True)
    assert window.rig["bindings"] == STANDARD_BINDINGS


def test_a_run_opens_on_its_game(qt):
    from fmri_gym.gui.session_editor import _Editor

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
