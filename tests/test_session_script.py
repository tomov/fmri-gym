"""Session scripts: what the session editor writes is what it reads back, and what a shell plays."""

from __future__ import annotations

import os
import subprocess

import pytest

from fmri_gym.session_script import Session, play_command, read_session, run_number, write_session

PONG, MARIO = "configs/curricula/pong.json", "configs/curricula/mario.json"
RIG = "configs/examples/rigs/laptop.json"


@pytest.mark.parametrize("steps", [
    [{"config": PONG, "skip": False}],
    [{"config": PONG, "skip": False}, {"config": MARIO, "skip": False},
     {"config": PONG, "skip": False}],
    [{"config": PONG, "skip": True}, {"config": PONG, "skip": False}],
    [{"config": PONG, "skip": False}, {"command": './localizer.sh "$SES"', "skip": False}],
], ids=["one run", "runs", "skipped", "command"])
@pytest.mark.parametrize("head", [("sub-01", RIG, None), ("sub-01", RIG, 3), (None, None, None),
                                  ("sub-01", None, None)],
                         ids=["fixed", "pinned", "given at launch", "rig at launch"])
def test_a_session_reads_back_as_written(steps, head):
    session = Session(steps, *head)
    assert read_session(write_session(session)) == session


def test_a_script_that_spells_out_subject_and_rig_on_each_line_fixes_them():
    line = (f"uv run fmri-play --curriculum {PONG} --subject sub-01 --ses \"$SES\" --run {{}} "
            f"--rig {RIG}")
    text = "SES=${1:-002}\n" + "\n".join(line.format(n) for n in (1, 2)) + "\n"
    assert read_session(text) == Session([{"config": PONG, "skip": False}] * 2, "sub-01", RIG, 2)


def test_a_line_the_editor_cannot_write_is_kept_as_typed():
    session = Session([], "sub-01", RIG, None)
    own_run = (f'uv run fmri-play --curriculum {PONG} --subject "$SUBJECT" --ses "$SES" --run 7 '
               '--rig "$RIG"')
    flags = (f'uv run fmri-play --curriculum {PONG} --subject "$SUBJECT" --ses "$SES" --run 1 '
             '--rig "$RIG" --dummy-trigger')
    text = write_session(session) + own_run + "\n" + flags + "\n"
    assert read_session(text).steps == [{"command": own_run, "skip": False},
                                        {"command": flags, "skip": False}]


def test_a_runs_flags_are_written_and_read_back():
    steps = [{"config": PONG, "skip": False,
              "flags": {"duration": 60.0, "loop": True, "resume": True}},
             {"config": MARIO, "skip": True, "flags": {"duration": 300.0, "end_fixation": 5.0}}]
    text = write_session(Session(steps, "sub-01", RIG, None))
    assert "--duration 60 --loop --resume" in text
    assert read_session(text).steps == steps


def test_a_run_commented_without_a_space_is_a_skipped_run():
    line = (f'#uv run fmri-play --curriculum {PONG} --subject "$SUBJECT" --ses "$SES" --run 1 '
            '--rig "$RIG"')
    text = write_session(Session([], "sub-01", RIG, None)) + line + "\n"
    assert read_session(text).steps == [{"config": PONG, "skip": True}]


def test_a_run_is_numbered_by_its_place_among_its_task():
    steps = [{"config": p, "skip": False} for p in (PONG, MARIO, PONG)]
    assert [run_number(steps, i) for i in range(3)] == [1, 1, 2]


def test_a_session_left_open_stops_until_its_launch_names_the_rig(tmp_path):
    script = tmp_path / "ses.sh"
    text = write_session(Session([{"command": 'echo "$SUBJECT $RIG $SES"', "skip": False}],
                                 "sub-01", None, 4))
    script.write_text(text)
    stopped = subprocess.run(["sh", script], capture_output=True, text=True)
    assert stopped.returncode != 0 and "RIG=" in stopped.stderr
    ran = subprocess.run(["sh", script], capture_output=True, text=True,
                         env={**os.environ, "RIG": RIG})
    assert ran.stdout.strip() == f"sub-01 {RIG} 004"


def test_play_command_names_the_run():
    assert play_command(PONG, "sub-01", "002", 1, RIG) == [
        "uv", "run", "fmri-play", "--curriculum", PONG, "--subject", "sub-01", "--ses", "002",
        "--run", "1", "--rig", RIG]
