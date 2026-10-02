"""Check that a death a block rolls back is undone in the world and kept in the record.

``"rewind": {...}`` on a game phase (:mod:`fmri_gym.rewind`) is a promise about
something that happens *inside* an episode: a death puts the world back a window
of play and the episode goes on, so the block's clock is still what ends it. Two
halves of that can only be seen from outside the loop -- that the world really
goes back, and that the record still describes the episode afterwards -- so this
checks them the way they are used, through ``fmri_play.py`` in its own process,
with nothing here stubbing the env, the logger or the loop.

    python docs/rewind_check.py

Part 1 is the field: the policy the four shipped crafter configs parse to, and
the ways a bad one is refused, including the two directions of the cross-field
check that keeps a menu entry and a phase field from existing without each
other. Part 2 is the ring on its own, with blobs that are not worlds, because
how far back the head is and when the depth cap rather than the window decides
it are arithmetic and deserve to be read as arithmetic. Part 3 plays a block
that really dies: a level-4 world with nobody pressing anything dies of thirst,
and with ``on_death`` it comes back, so the block logs a rollback every few
frames until its clock stops it. What is checked there is the world (the blob on
each ``rewind`` line is the world the frame it names logged), the record (the
undone frames are still in it, and a death that was undone is a frame flag
rather than an outcome), the screen (the hold really held, with the backend's
own word for a death written over the held frame, which is the one part of this
the subject sees and the one part the record cannot hold: it is measured
through the display's own ``draw_frame``), and the handoff: the
episode's outcome stays ``playing``, so the thread of play survives a death and
the slot is written. Part 4 is the other end of it -- ``max_rewinds`` spent, and
the next death is a death: the episode ends, and the slot goes with it -- and on
the way it plays the world part 3 handed on, which is a world that had been
rolled back. Part 5 reconstructs part 3's episode, which is the half of the
feature an analysis sees: its actions in order are no longer the game it played,
and only the segments cut at its ``rewind`` lines replay it. Part 6 plays the
same phase with a policy instead of a person, where none of this may happen.

Not covered here: the pause menu's ``rewind`` option end to end, which takes a
held key and a two-key choice at a window nobody is sitting at. Its label, its
validation and the branch's effect are checked in parts 1 and 2; what the loop
does with the choice is the one path still read rather than measured.

The curriculum is the real level-4 config with the fields a headless death needs
changed -- real-time pacing instead of turn-based, a smaller frame, a faster
clock, one episode and a wall-clock cap -- plus the ``rewind`` field under test,
which level 4 does not have (level 1 is the level that does). Everything else,
``resume`` included, is whatever ``configs/dbp_games/crafter__crafter_L4.json``
says today.

Last run 2026-10-02 on this branch: 3 blocks, 83 checks, 0 failures.
"""

import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import crafter_gym  # noqa: E402
from fmri_gym import resume, rewind  # noqa: E402
from fmri_gym.adapters import get_adapter  # noqa: E402
from fmri_gym.config import load_config, validate_config  # noqa: E402
from fmri_gym.logging import read_events, read_state  # noqa: E402
from fmri_gym.replay import reconstruct_episode, reconstruction_plan  # noqa: E402

# Unpickling a savestate imports crafter, and the repo's own gym/ directory
# shadows old gym for a process started at the repo root (crafter_gym's own
# import_crafter deals with it). Done here so that reading a block back does not
# depend on a check that builds an env having run first.
crafter_gym.import_crafter()

CONFIGS = os.path.join(ROOT, "configs", "dbp_games")
L4 = os.path.join(CONFIGS, "crafter__crafter_L4.json")
#: Steps a level-4 world takes to die of thirst with nobody pressing anything,
#: at seed 0: measured, and only a guide for the clock below.
TO_DEATH = 209
#: Fast enough that the frames to that death are seconds, slow enough that the
#: loop (160 fps measured at this frame size) paces on the config and not on
#: itself.
FPS = 30
#: Long enough for the death plus a handful of rollbacks after it.
MAX_DURATION = 12.0
#: The window under test, in the subject's own moves, and the pause over the
#: frame that triggered a rollback.
FRAMES_BACK = 5
HOLD = 0.3

failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
    """Record one check's verdict and print it.

    :param name: what was checked.
    :param ok: whether it held.
    :param detail: the measurement behind the verdict.
    """
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def dying_curriculum(path: str, **field) -> str:
    """Write the level-4 game phase as a block that dies headless, and comes back.

    :param path: file to write the config to.
    :param field: what to put in the phase's ``rewind`` field, over the window
        and hold under test.
    :return: ``path``.
    """
    phase = [p for p in load_config(L4)["curriculum"] if p["type"] == "game"][0]
    # Real time wants a name for "no key held", which a turn-based block has no
    # use for: nobody is at the keyboard here, so every frame sends that noop,
    # which is also what makes the world die on its own.
    phase = {**phase, "mode": "episode", "n_episodes": 1, "max_duration": MAX_DURATION,
             "fps": FPS, "turn_based": False, "live_hud": False, "state_stride": 25,
             "keys": {"": 0, **phase["keys"]},
             "env_kwargs": {**phase["env_kwargs"], "size": [128, 128]},
             "rewind": {"frames": FRAMES_BACK, "on_death": True, "hold": HOLD, **field}}
    with open(path, "w") as f:
        json.dump({"triggers": {"sync": {"mode": "none"}, "backend": "null"},
                   "curriculum": [phase]}, f)
    return path


def headless(*args: str, **extra: str) -> subprocess.CompletedProcess:
    """Run one of the repo's own commands with no window, no sound and no scanner.

    :param args: the script and its arguments, after the interpreter.
    :param extra: environment the command needs beyond those.
    :return: the finished process.
    """
    env = {**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
           "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""),
           **extra}
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env,
                          capture_output=True, text=True)


#: ``fmri_play.py``, with every line the display drew over a frame written down
#: on the way out. A notice over a held frame is pixels and nothing else -- no
#: log line says it, and there is no window here to look at -- and
#: :meth:`fmri_gym.display.Display.draw_frame` is the one place a frame reaches
#: the screen, so this is where that half of the feature can be measured. The
#: run itself is untouched: the wrapper records what it was passed and calls the
#: real thing.
DRIVER = '''\
import json
import os
import runpy
import sys

from fmri_gym import display

drawn = []
flip = display.Display.draw_frame


def recording(self, rgb, hud=None, overlay=None):
    if overlay is not None:
        drawn.append(overlay)
    return flip(self, rgb, hud, overlay)


display.Display.draw_frame = recording
sys.argv = [os.environ["FMRI_PLAY"], *sys.argv[1:]]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    with open(os.environ["FMRI_OVERLAY_LOG"], "w") as f:
        json.dump(drawn, f)
'''


def play(curriculum: str, data_root: str, run: int) -> tuple[str, str, list]:
    """Run one ``fmri_play.py`` block in its own process, headless.

    :param curriculum: the config to play, in the folder this writes its own
        driver and overlay log beside.
    :param data_root: the BIDS root the runs share.
    :param run: the run number.
    :return: the block's folder, what the run said on stderr, and every overlay
        the display drew, in order (:data:`DRIVER`).
    :raises RuntimeError: if the process failed or wrote no block.
    """
    work = os.path.dirname(curriculum)
    driver = os.path.join(work, "play_recording.py")
    with open(driver, "w") as f:
        f.write(DRIVER)
    overlays = os.path.join(work, f"overlays-run{run:03d}.json")
    proc = headless(driver, "--curriculum", curriculum,
                    "--subject", "sub-01", "--ses", "1", "--run", str(run),
                    "--data-root", data_root, "--dummy-trigger", "--no-audio",
                    FMRI_PLAY=os.path.join(ROOT, "fmri_play.py"),
                    FMRI_OVERLAY_LOG=overlays)
    if proc.returncode != 0:
        raise RuntimeError(f"run {run} failed:\n{proc.stderr[-2000:]}")
    for line in proc.stderr.splitlines():
        if "resume" in line or "left off" in line or "world is over" in line:
            print(f"    {line.strip()}")
    beh = os.path.join(data_root, "sub-01", "ses-001", "beh")
    folder = [d for d in sorted(os.listdir(beh)) if d.endswith(f"run-{run:03d}")][0]
    run_dir = os.path.join(beh, folder)
    blocks = [d for d in sorted(os.listdir(run_dir)) if d.startswith("block-")]
    if not blocks:
        raise RuntimeError(f"run {run} wrote no block")
    with open(overlays) as f:
        drawn = json.load(f)
    return os.path.join(run_dir, blocks[0]), proc.stderr, drawn


def lines_of(block: str, kind: str) -> list[dict]:
    """Every line of one type in a block's ``events.jsonl``.

    :param block: the block's folder.
    :param kind: the ``type`` field to keep.
    :return: those lines, in the order they were written.
    """
    return [e for e in read_events(block) if e.get("type") == kind]


def world_step(blob: bytes) -> int:
    """How many steps the world in a savestate has taken.

    :param blob: a crafter savestate, as ``restore()`` wants it.
    :return: the engine's own step counter.
    """
    # Through unwrapped, because what the savestate holds is the env the phase
    # built: the menu wrapper, and under it the level's wrapper when the phase
    # names a level (crafter_gym.levels).
    return pickle.loads(blob).unwrapped.game._step


def map_digest(grid: object) -> str:
    """A semantic grid as twelve hex digits, so two can be printed side by side.

    :param grid: the grid, live or as the record's nested lists.
    :return: the first 12 hex digits of its sha256.
    """
    return hashlib.sha256(np.asarray(grid, dtype=np.int64).tobytes()).hexdigest()[:12]


def world_map(adapter: object) -> np.ndarray:
    """What a live world looks like, in the same terms the record logs.

    :param adapter: a crafter adapter mid-episode.
    :return: its semantic grid, as a ``frame`` line's ``variables["semantic"]``.
    """
    return np.asarray(adapter._game._sem_view())


def blob_map(blob: bytes) -> np.ndarray:
    """The same grid, for a world that is still a savestate.

    :param blob: a crafter savestate.
    :return: its semantic grid.
    """
    return np.asarray(pickle.loads(blob).unwrapped.game._sem_view())


def state(ep_frame: int = 0, run_time: float = 0.0, score: float = 0.0) -> rewind.State:
    """One ring entry whose blob is not a world: part 2 is about the arithmetic.

    :param ep_frame: the frame's index in its episode.
    :param run_time: the run clock at its step.
    :param score: the episode's score there.
    :return: the :class:`~fmri_gym.rewind.State`.
    """
    return rewind.State(ep_frame=ep_frame, run_time=run_time, score=score,
                        blob=f"f{ep_frame}".encode())


def part1() -> None:
    """The field: what the shipped configs ask for, and what is refused."""
    print("part 1: the rewind field")
    levels = {n: load_config(os.path.join(CONFIGS, f"crafter__crafter_{n}.json"))
              for n in ("L1", "L2", "L3", "L4")}
    phases = {n: [p for p in c["curriculum"] if p["type"] == "game"][0]
              for n, c in levels.items()}
    check("every shipped crafter level validates",
          all(validate_config(c) == [] for c in levels.values()))
    policy = rewind.policy_of(phases["L1"])
    check("level 1 rolls a death back", policy is not None and policy.on_death)
    # One move, which is the rig's own window read in presses rather than in
    # ticks: --death-restore-s 1.0 at cadence_hz 5.0 is five ticks, but the rig
    # pushes a snapshot on every tick and a tick steps the env whether or not a
    # button is down (core.py:Rig.tick, ButtonMapper.resolve), so the second it
    # goes back holds whichever presses fell inside it and not five of them.
    check("a window of one of the subject's own moves",
          policy.frames == 1 and policy.seconds is None, f"{policy.describe()}")
    check("and the rig's hold over the frame that killed them", policy.hold == 1.2)
    check("so the menu entry it would be offered under says one move",
          policy.label() == "Go back 1 move", policy.label())
    check("nothing caps how many rollbacks the level allows", policy.max_rewinds is None)
    check("the other three levels keep the true death",
          all(rewind.policy_of(phases[n]) is None for n in ("L2", "L3", "L4")))
    check("so does a phase with no field at all", rewind.policy_of({"fps": 30}) is None)

    check("a frames window bounds its own depth",
          rewind.Policy(frames=5).depth == 6 and rewind.Policy(frames=5, stride=2).depth == 4,
          "5 frames: 6 states, every other frame: 4")
    check("a seconds window is bounded by max_states",
          rewind.Policy(seconds=10.0, max_states=12).depth == 12)
    check("the menu says the window in the units it was asked for",
          (rewind.Policy(frames=5).label(), rewind.Policy(frames=1).label(),
           rewind.Policy(seconds=10.0).label())
          == ("Go back 5 moves", "Go back 1 move", "Go back 10 seconds"))
    check("the manifest gets the policy in force",
          rewind.Policy(frames=5, hold=1.2).describe()
          == {"on_death": False, "hold": 1.2, "stride": 1, "depth": 6,
              "max_rewinds": None, "frames": 5})

    ok = {"frames": 5, "on_death": True}
    check("a usable field has nothing to say about it",
          rewind.rewind_problems({"rewind": ok}, []) == [])
    bad = {
        "neither window": {"on_death": True},
        "both windows": {"frames": 5, "seconds": 2.0, "on_death": True},
        "frames 0": {"frames": 0, "on_death": True},
        "frames as a string": {"frames": "5", "on_death": True},
        "seconds 0": {"seconds": 0, "on_death": True},
        "a negative hold": {"frames": 5, "on_death": True, "hold": -1},
        "stride 0": {"frames": 5, "on_death": True, "stride": 0},
        "max_rewinds 0": {"frames": 5, "on_death": True, "max_rewinds": 0},
        "an unknown field": {"frames": 5, "on_death": True, "on_lava": True},
        "a depth the window already fixes": {"frames": 5, "on_death": True, "max_states": 8},
        "nothing to trigger it": {"frames": 5},
    }
    for name, spec in bad.items():
        problems = rewind.rewind_problems({"rewind": spec}, [])
        check(f"refused: {name}", len(problems) == 1,
              problems[0] if problems else "accepted")
    check("refused: a field that is not a field at all",
          len(rewind.rewind_problems({"rewind": 5}, [])) == 1)
    # Both directions of the cross-field check: either half alone is a config
    # that reads as if it had taken and does nothing in the scanner.
    check("refused: a menu option with no field behind it",
          len(rewind.rewind_problems({}, ["rewind", "resume"])) == 1)
    check("a menu option is a trigger, so on_death need not be one",
          rewind.rewind_problems({"rewind": {"frames": 5}}, ["rewind", "resume"]) == [])
    check("a phase with neither is not asked about either",
          rewind.rewind_problems({}, ["reset", "resume"]) == [])
    # And through the config check an experimenter actually runs, which is where
    # the phase index in the message comes from: level 1's own curriculum, so
    # the index is the one a config with a message and a fixation before its
    # game phase really has.
    config = dict(levels["L1"])
    index = [i for i, p in enumerate(config["curriculum"]) if p["type"] == "game"][0]
    config["curriculum"] = [{**p, "rewind": {"seconds": 2.0, "frames": 5, "on_death": True}}
                            if p["type"] == "game" else p for p in config["curriculum"]]
    problems = validate_config(config)
    check("a bad field stops the run, by phase",
          len(problems) == 1 and problems[0].startswith(f"phase {index}: rewind:"),
          problems[0] if problems else "accepted")


def part2() -> None:
    """The ring: where its head is, and what decides it."""
    print("part 2: the ring")
    ring = rewind.Ring(rewind.Policy(frames=FRAMES_BACK))
    check("an episode nobody has played has nowhere to go back to", ring.target() is None)
    ring.push(state(0))
    check("early on, the first frame is as far back as there is", ring.target().ep_frame == 0)
    head = []
    for f in range(1, 20):
        ring.push(state(f))
        head.append(ring.target().ep_frame)
    check("from then on the head is the first frame a whole window back",
          head == [max(0, f - FRAMES_BACK) for f in range(1, 20)],
          f"frame 19 goes back to {head[-1]}")
    check("and never holds more than the window needs", len(ring._q) == 6,
          f"{len(ring._q)} states")
    check("the window did the deciding, not a cap", not ring.capped)

    strided = rewind.Ring(rewind.Policy(frames=FRAMES_BACK, stride=2))
    check("a stride keeps every other frame",
          [strided.due(f) for f in range(4)] == [True, False, True, False])
    for f in range(0, 20, 2):
        strided.push(state(f))
    check("and goes back at least the window, in what it kept",
          18 - strided.target().ep_frame >= FRAMES_BACK,
          f"frame {strided.target().ep_frame}")

    seconds = rewind.Ring(rewind.Policy(seconds=1.0))
    for f in range(20):
        seconds.push(state(f, run_time=f * 0.25))
    check("a seconds window goes back on the run clock",
          19 * 0.25 - seconds.target().run_time == 1.0,
          f"{19 * 0.25 - seconds.target().run_time:g} s back")
    check("which a turn-based block cannot bound in advance", not seconds.capped)

    capped = rewind.Ring(rewind.Policy(seconds=1.0, max_states=4))
    for f in range(20):
        capped.push(state(f, run_time=f * 0.25))
    check("so a cap can be what decides the head, and says so",
          capped.capped and len(capped._q) == 4
          and 19 * 0.25 - capped.target().run_time < 1.0,
          f"{19 * 0.25 - capped.target().run_time:g} s back of the 1.0 asked for")

    # Rig parity (core.py:_apply_restore): the window starts again from the
    # world a rollback restored, or a subject who dies repeatedly in one spot is
    # carried back through the whole block a window at a time.
    again = rewind.Ring(rewind.Policy(frames=FRAMES_BACK))
    for f in range(20):
        again.push(state(f, score=float(f)))
    back = again.target()
    again.reseed(back)
    check("after a rollback the ring starts again from the world it restored",
          again.target() is back and len(again._q) == 1, f"frame {back.ep_frame}")
    for f in range(20, 20 + FRAMES_BACK):
        again.push(state(f, score=float(f)))
    check("so a second death goes back to there and no further",
          again.target() is back, f"frame {again.target().ep_frame}")
    check("and what comes back with the world is that frame's score",
          again.target().score == float(back.ep_frame))


def part3(curriculum: str, data_root: str) -> str:
    """A block that dies, comes back, and is still playing when its clock ends.

    :param curriculum: the dying level-4 config.
    :param data_root: the BIDS root the runs share.
    :return: the block's folder.
    """
    print("part 3: a death rolled back")
    block, said, drawn = play(curriculum, data_root, 1)
    frames = lines_of(block, "frame")
    rewinds = lines_of(block, "rewind")
    check("the block died and came back more than once", len(rewinds) > 1,
          f"{len(rewinds)} rollbacks in {len(frames)} frames")
    check("every one of them was a death", {r["trigger"] for r in rewinds} == {"death"})
    check("and the run had nothing to warn about", "WARNING" not in said)

    # The record keeps what was undone: the frames between the target and the
    # rollback are in the log as played, so the block's frame numbering is still
    # one frame per line, in order.
    check("the undone frames are still in the record",
          [f["ep_frame"] for f in frames] == list(range(len(frames))))
    check("a death that was undone is a frame flag and not an outcome",
          all(frames[r["from_ep_frame"]]["terminated"] for r in rewinds))
    check("and never an env truncation, which is a clock",
          not any(f["truncated"] for f in frames))

    first = rewinds[0]
    check("the first rollback went back the window the config asked for",
          first["frames_back"] == FRAMES_BACK
          and first["to_ep_frame"] == first["from_ep_frame"] - FRAMES_BACK,
          f"frame {first['from_ep_frame']} -> {first['to_ep_frame']}")
    check("at about the time the window is worth at this frame rate",
          abs(first["seconds_back"] - FRAMES_BACK / FPS) < 0.1,
          f"{first['seconds_back']:.3f} s for {FRAMES_BACK} frames at {FPS} fps")
    check("the window and not a cap decided it", not any(r["capped"] for r in rewinds))
    check("frames_back is the distance in the episode's own numbering",
          all(r["frames_back"] == r["from_ep_frame"] - r["to_ep_frame"] for r in rewinds))
    # The second and every later rollback goes back to the world the first one
    # restored, which is the ring starting again from it (rig parity, part 2):
    # so frames_back grows while what each one undid does not, which is what
    # part 5 reads off the segments.
    check("and a later death goes back no further than the first did",
          {r["to_ep_frame"] for r in rewinds} == {first["to_ep_frame"]},
          f"all {len(rewinds)} of them to frame {first['to_ep_frame']}")

    # The blob is the world the frame it names logged: a fresh episode's engine
    # step after frame i is i + 1, so this is the one arithmetic that ties the
    # record's frame numbering to the world it hands back.
    check("each rollback carries the world of the frame it names",
          all(world_step(read_state(r)) == r["to_ep_frame"] + 1 for r in rewinds),
          f"engine step {world_step(read_state(first))} at frame {first['to_ep_frame']}")
    # A frame line logs its own step's reward, so the score a rollback restores
    # is the sum over the frames up to and including the one it went back to.
    check("and the score that frame had",
          all(abs(r["score"] - sum(f["reward"] for f in frames[:r["to_ep_frame"] + 1]))
              < 1e-9 for r in rewinds),
          f"score {first['from_score']:g} at the death, {first['score']:g} back at "
          f"frame {first['to_ep_frame']}")

    # The hold is the one part of this the subject sees: the frame that killed
    # them stays up before the world moves.
    held = [frames[r["from_ep_frame"] + 1]["run_time"] - r["run_time"] for r in rewinds
            if r["from_ep_frame"] + 1 < len(frames)]
    check("the frame that triggered it was held before the world moved",
          len(held) >= len(rewinds) - 1 and all(h >= HOLD for h in held),
          f"{min(held):.2f} s at least, {HOLD} asked for")
    # And what it says while it is up, which no line of the record holds: the
    # backend's own message for this ending, where that backend puts a line on
    # its frame. A rolled-back death never reaches the outcome screen, so
    # without this the subject is shown a world that jumps and told nothing.
    adapter = get_adapter("crafter", read_events(block)[0]["phase"])
    try:
        notice = [[adapter.outcome(True, False)[1]], adapter.overlay_y]
    finally:
        adapter.close()
    check("with the backend's word for the death written over it, once each",
          drawn == [notice] * len(rewinds),
          f"{len(drawn)} lines drawn over {len(rewinds)} rollbacks: "
          f"{notice[0][0]!r} at {notice[1]:.3f} of the frame height")

    end = lines_of(block, "episode_end")[-1]
    check("the episode never ended of the death", end["outcome"] == "playing",
          f"outcome {end['outcome']}, terminated {end['terminated']}")
    check("the record counts the rollbacks", end["n_rewinds"] == len(rewinds)
          and lines_of(block, "block_end")[-1]["n_rewinds"] == len(rewinds))
    manifest = json.load(open(os.path.join(os.path.dirname(block), "manifest.json")))
    entry = [p for p in manifest["phases"] if p["type"] == "game"][0]
    check("and the manifest says what the policy was",
          entry["rewind"] == {"frames": FRAMES_BACK, "on_death": True, "hold": HOLD,
                              "stride": 1, "depth": FRAMES_BACK + 1, "max_rewinds": None},
          f"{entry.get('rewind')}")

    # The handoff: a rolled-back death is not an ending, so the thread of play
    # did not end either and the world carries to the next block.
    resume_dir = os.path.join(data_root, "sub-01", "ses-001", "resume")
    carry = resume.load(resume_dir, "crafter_L4")
    check("so the thread of play survived the death", carry is not None)
    last = rewinds[-1]
    since = len(frames) - 1 - last["from_ep_frame"]
    check("and what it carries is the restored world, played on from there",
          world_step(carry.blob) == world_step(read_state(last)) + since,
          f"engine step {world_step(carry.blob)}: {world_step(read_state(last))} restored "
          f"+ {since} frames since")
    check("which is behind the block's own frame count, by what was undone",
          world_step(carry.blob) < len(frames),
          f"{world_step(carry.blob)} steps of {len(frames)} frames")
    return block


def part4(data_root: str, work: str) -> None:
    """``max_rewinds`` spent: the next death is a death, and takes the world with it.

    :param data_root: the BIDS root the runs share.
    :param work: the folder to write this part's config in.
    """
    print("part 4: a rollback allowance that runs out")
    resume_dir = os.path.join(data_root, "sub-01", "ses-001", "resume")
    was = resume.load(resume_dir, "crafter_L4").blob
    curriculum = dying_curriculum(os.path.join(work, "l4_capped.json"), max_rewinds=1)
    block, said, drawn = play(curriculum, data_root, 2)
    check("the block was handed the world part 3 left, rollbacks and all",
          all(e["resumed"] for e in lines_of(block, "episode_start"))
          and read_state(lines_of(block, "resume")[0]) == was,
          f"engine step {world_step(was)}")
    check("and it had nothing to warn about", "WARNING" not in said)
    rewinds = lines_of(block, "rewind")
    check("the allowance was spent once and not again", len(rewinds) == 1,
          f"{len(rewinds)} rollbacks")
    end = lines_of(block, "episode_end")[-1]
    check("so the next death ended the episode",
          end["outcome"] != "playing" and end["terminated"], f"outcome {end['outcome']}")
    check("the record still counts the one rollback", end["n_rewinds"] == 1)
    # The other side of part 3's notice: a line over the held frame belongs to a
    # death that is about to be taken back, and the one that ends the episode is
    # told on the outcome screen the ending reaches.
    check("only the rollback wrote its line over a frame", len(drawn) == 1,
          f"{len(drawn)} lines drawn over two deaths")
    check("and the thread of play ended with the episode",
          resume.load(resume_dir, "crafter_L4") is None)
    check("the block that ended it is a block whose game ended it",
          "world is over" in said)


def part5(block: str) -> None:
    """The episode an analysis reads back: its actions in order are not the game.

    :param block: part 3's block folder.
    """
    print("part 5: reconstructing an episode that went back")
    plan = reconstruction_plan(block)
    frames = lines_of(block, "frame")
    rewinds = lines_of(block, "rewind")
    check("the plan holds every action the subject sent, undone ones included",
          len(plan["actions"]) == len(frames), f"{len(plan['actions'])} actions")
    check("and cuts the episode into one segment per rollback, plus the first",
          len(plan["segments"]) == len(rewinds) + 1)
    check("the first segment starts from the seed, the rest from a restored world",
          plan["segments"][0]["state"] is None
          and all(s["state"] is not None for s in plan["segments"][1:]))
    check("every action is in exactly one segment, in order",
          [a for s in plan["segments"] for a in s["actions"]] == plan["actions"])
    check("each rollback's record is there without its blob",
          len(plan["rewinds"]) == len(rewinds)
          and not any("state" in r for r in plan["rewinds"]))
    # What the segments are for: the first holds the whole play up to the first
    # death, and each later one the play between two rollbacks, which is what
    # the second of them undid. The ring went back to the same world again
    # (part 3), so frames_back keeps growing by that while the play lost each
    # time does not -- which is why an analysis wants the segment and not the
    # field (:mod:`fmri_gym.rewind`).
    lengths = [len(s["actions"]) for s in plan["segments"]]
    check("a segment is the play between one rollback and the one before it",
          lengths[0] == rewinds[0]["from_ep_frame"] + 1
          and all(lengths[i + 1] == rewinds[i + 1]["from_ep_frame"]
                  - rewinds[i]["from_ep_frame"] for i in range(len(rewinds) - 1)),
          f"segments {lengths}")
    check("so frames_back grows by that, being measured from the same world",
          all(rewinds[i + 1]["frames_back"] - rewinds[i]["frames_back"] == lengths[i + 1]
              for i in range(len(rewinds) - 1)),
          f"frames_back {[r['frames_back'] for r in rewinds]}")
    # And that the segment is what was undone is the engine's own arithmetic:
    # the world the previous rollback restored, played that segment forward,
    # back to the step the frame this line names is at.
    undone = [world_step(read_state(rewinds[i])) + lengths[i + 1]
              - world_step(read_state(rewinds[i + 1])) for i in range(len(rewinds) - 1)]
    check("and the engine's own step counter went back exactly that far",
          undone == lengths[1:len(rewinds)],
          f"{undone} steps undone, against segments {lengths[1:len(rewinds)]}")

    # Where the replay should land: the last frame the block logged, unless its
    # clock ended the block during the last rollback's hold, which leaves no
    # frame after it and the restored world itself as the answer.
    tail = plan["segments"][-1]["actions"]
    ended_at = (frames[-1]["variables"]["semantic"] if tail
                else blob_map(read_state(rewinds[-1])))
    steps = world_step(read_state(rewinds[-1])) + len(tail)
    adapter, _ = reconstruct_episode(block)
    try:
        check("replaying the segments lands on the world the episode ended in",
              map_digest(world_map(adapter)) == map_digest(ended_at)
              and adapter._game._step == steps,
              f"{map_digest(world_map(adapter))} at engine step {adapter._game._step}, "
              f"{len(tail)} frames after the last rollback")
    finally:
        adapter.close()
    # The negative half: the same actions in sequence are a different game from
    # the first rollback on, which is the whole reason the segments exist.
    blind = get_adapter("crafter", read_events(block)[0]["phase"])
    try:
        blind.reset(plan["seed"])
        for action in plan["actions"]:
            blind.step(action)
        check("while stepping them in sequence does not",
              map_digest(world_map(blind)) != map_digest(ended_at),
              f"{map_digest(world_map(blind))} at engine step {blind._game._step}")
    except Exception as exc:
        # Stepping on past where the game ended is not something a run ever
        # does, so an engine that refuses answers the same question.
        check("while stepping them in sequence does not", True,
              f"refused past the death ({type(exc).__name__})")
    finally:
        blind.close()


def part6(data_root: str, work: str) -> None:
    """A model plays the same phase and dies when the game kills it.

    Agent parity is the reason the field is a phase field and not a ``LEVELS``
    rule (:mod:`fmri_gym.rewind`), so it is worth measuring rather than reading:
    the harness a rollout runs in has to see nothing of it.

    :param data_root: the folder to write the rollout under.
    :param work: the folder holding this part's config.
    """
    print("part 6: a rollout of the same phase")
    curriculum = dying_curriculum(os.path.join(work, "l4_agent.json"))
    outdir = os.path.join(data_root, "model-random")
    proc = headless(os.path.join(ROOT, "agents", "agent_play.py"),
                    "--curriculum", curriculum, "--policy", "random",
                    "--n-episodes", "1", "--max-frames", str(TO_DEATH + 60),
                    "--outdir", outdir)
    check("the rollout ran the phase without refusing the field",
          proc.returncode == 0,
          proc.stderr.strip().splitlines()[-1] if proc.returncode else "")
    if proc.returncode != 0:
        return
    block = os.path.join(outdir, sorted(d for d in os.listdir(outdir)
                                        if d.startswith("block-"))[0])
    end = lines_of(block, "block_end")[-1]
    check("and no rollback happened in it",
          not lines_of(block, "rewind") and "n_rewinds" not in end,
          f"outcomes {end['outcomes']} in {end['n_frames']} frames")


def main() -> None:
    """Run all six parts and exit non-zero on any failure."""
    work = tempfile.mkdtemp(prefix="rewind-check-")
    try:
        part1()
        part2()
        data_root = os.path.join(work, "data")
        curriculum = dying_curriculum(os.path.join(work, "l4_dying.json"))
        block = part3(curriculum, data_root)
        part4(data_root, work)
        part5(block)
        part6(data_root, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{len(failures)} failures: {failures}" if failures else "\nall checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
