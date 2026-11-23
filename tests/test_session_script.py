"""Session scripts: what the session editor writes is what it reads back, and what a shell plays."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time

import pytest

from humans.standalone.session_script import (Session, play_command, read_session, run_number,
                                                write_session)

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
             {"config": MARIO, "skip": True, "flags": {"duration": 300.0, "loop": True}}]
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


def _uv_that_logs(tmp_path) -> dict[str, str]:
    """An environment whose ``uv`` only writes what it is asked into ``uv.log``, and answers
    ``session-logger start`` with a recording's events file."""
    (tmp_path / "bin").mkdir()
    uv = tmp_path / "bin" / "uv"
    uv.write_text(f'#!/bin/sh\necho "$*" >> "{tmp_path / "uv.log"}"\n'
                  'case "$*" in "run session-logger start"*) echo /recordings/1;; esac\n')
    uv.chmod(0o755)
    return {**os.environ, "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}"}


def _asked_of_uv(tmp_path) -> list[str]:
    return (tmp_path / "uv.log").read_text().splitlines() if (tmp_path / "uv.log").exists() \
        else []


def test_a_session_left_open_stops_until_its_launch_names_the_rig(tmp_path):
    script = tmp_path / "ses.sh"
    text = write_session(Session([{"command": 'echo "$SUBJECT $RIG $SES"', "skip": False}],
                                 "sub-01", None, 4))
    script.write_text(text)
    environment = _uv_that_logs(tmp_path)
    stopped = subprocess.run(["sh", script], capture_output=True, text=True, env=environment)
    assert stopped.returncode != 0 and "RIG=" in stopped.stderr
    assert _asked_of_uv(tmp_path) == [], "no recording starts before the head is whole"
    ran = subprocess.run(["sh", script], capture_output=True, text=True,
                         env={**environment, "RIG": RIG})
    assert ran.stdout.strip() == f"sub-01 {RIG} 004"


@pytest.mark.parametrize("last_line, status", [("true", 0), ("false", 1)],
                         ids=["ends", "fails"])
def test_a_session_is_recorded_from_before_its_first_line_until_it_ends(tmp_path, last_line,
                                                                          status):
    script = tmp_path / "ses.sh"
    script.write_text(write_session(Session(
        [{"command": f"echo played >> {tmp_path / 'uv.log'}; {last_line}", "skip": False}],
        "sub-01", RIG, 4)))
    session = subprocess.Popen(["sh", script], env=_uv_that_logs(tmp_path))
    assert session.wait() == status
    assert _asked_of_uv(tmp_path) == [
        f"run session-logger start --subject sub-01 --ses 004 --rig {RIG} "
        f"--session-pid {session.pid}", "played",
        "run session-logger stop /recordings/1"]


@pytest.mark.parametrize("shell", [
    "sh", pytest.param("dash", marks=pytest.mark.skipif(shutil.which("dash") is None,
                                                        reason="no dash here"))])
def test_a_session_interrupted_stops_its_recording(tmp_path, shell):
    script = tmp_path / "ses.sh"
    script.write_text(write_session(Session([{"command": "sleep 30", "skip": False}],
                                            "sub-01", RIG, 4)))
    session = subprocess.Popen([shell, script], env=_uv_that_logs(tmp_path),
                               start_new_session=True)
    deadline = time.monotonic() + 10
    while len(_asked_of_uv(tmp_path)) < 1 and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(0.2)                                    # the sleep has started
    os.killpg(session.pid, signal.SIGINT)              # Ctrl+C, to the terminal's group
    assert session.wait(timeout=10) != 0
    assert _asked_of_uv(tmp_path)[-1] == "run session-logger stop /recordings/1"


def test_the_recording_lines_are_read_as_the_heads():
    text = write_session(Session([{"config": PONG, "skip": False}], "sub-01", RIG, None))
    assert "session-logger start" in text and "session-logger stop" in text
    assert read_session(text).steps == [{"config": PONG, "skip": False}]


def test_play_command_names_the_run():
    assert play_command(PONG, "sub-01", "002", 1, RIG) == [
        "uv", "run", "fmri-play", "--curriculum", PONG, "--subject", "sub-01", "--ses", "002",
        "--run", "1", "--rig", RIG]
