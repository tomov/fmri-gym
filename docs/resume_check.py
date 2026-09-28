"""Check that a block hands its world to the next block, across two processes.

``"resume": <slot>`` on a game phase is a promise about something that happens
*between* runs: one ``fmri-play`` command leaves the world it was cut off in,
and a later command picks it up. In-process assertions cannot see that promise
break, so this checks it the way it is used -- two subprocesses writing into one
BIDS session -- and nothing here stubs the env, the logger or the loop.

    python docs/resume_check.py

Part 1 is the slot file on its own: a round trip, the four ways a file is
refused, and which backends have the savestate a resume is made of. Part 2 runs
``fmri_play.py`` twice into one session and checks that what run 1 wrote is
byte-for-byte what run 2 was handed, that run 2's own world then moved past it,
and that the two blocks are different games despite the same pinned seed, which
is what proves run 2 did not simply open the seed's world. Part 3 is the one
surprise in the feature: a restored world ignores the block's ``env_kwargs``,
because it comes back exactly as it was pickled, so the run says so out loud
rather than letting an edited config read as if it had taken. Part 4 is the
other half of the rule: a block whose episode the *game* ends loses the slot
with it, and the block after that opens a world of its own, because only an
ending that belongs to the scanner's clock is one the game should not charge
for.

The curriculum is the real level-4 config with three fields changed -- a shorter
duration, a smaller frame, and real-time pacing instead of turn-based, so a
headless run steps without a person pressing buttons. Everything else, the
resume field included, is whatever ``configs/dbp_games/crafter__crafter_L4.json``
says today.

Last run 2026-09-28 on this branch: 5 blocks, 26 checks, 0 failures.
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

from fmri_gym import resume  # noqa: E402
from fmri_gym.adapters import get_adapter  # noqa: E402
from fmri_gym.config import load_config, validate_config  # noqa: E402

L4 = os.path.join(ROOT, "configs", "dbp_games", "crafter__crafter_L4.json")
#: Enough frames to be a world and few enough to be a test.
DURATION = 6.0

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


def short_curriculum(path: str) -> str:
    """Write the level-4 game phase as a headless-playable block.

    :param path: file to write the config to.
    :return: ``path``.
    """
    phase = [p for p in load_config(L4)["curriculum"] if p["type"] == "game"][0]
    # Real time wants a name for "no key held", which a turn-based block has no
    # use for: nobody is at the keyboard here, so every frame sends that noop.
    phase = {**phase, "duration": DURATION, "fps": 8, "turn_based": False,
             "live_hud": False, "state_stride": 5,
             "keys": {"": 0, **phase["keys"]},
             "env_kwargs": {**phase["env_kwargs"], "size": [128, 128]}}
    with open(path, "w") as f:
        json.dump({"triggers": {"sync": {"mode": "none"}, "backend": "null"},
                   "curriculum": [phase]}, f)
    return path


def play(curriculum: str, data_root: str, run: int) -> tuple[str, str]:
    """Run one ``fmri_play.py`` block in its own process, headless.

    :param curriculum: the config to play.
    :param data_root: the BIDS root all the runs share.
    :param run: the run number.
    :return: the block's ``.npz`` path, and what the run said on stderr.
    :raises RuntimeError: if the process failed or wrote no block.
    """
    env = {**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
           "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "fmri_play.py"), "--curriculum", curriculum,
         "--subject", "sub-01", "--ses", "1", "--run", str(run),
         "--data-root", data_root, "--dummy-trigger", "--no-audio"],
        cwd=ROOT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"run {run} failed:\n{proc.stderr[-2000:]}")
    for line in proc.stderr.splitlines():
        if "resume" in line or "left off" in line:
            print(f"    {line.strip()}")
    beh = os.path.join(data_root, "sub-01", "ses-001", "beh")
    folder = [d for d in sorted(os.listdir(beh)) if d.endswith(f"run-{run:03d}")][0]
    blocks = [f for f in os.listdir(os.path.join(beh, folder)) if f.endswith(".npz")]
    if not blocks:
        raise RuntimeError(f"run {run} wrote no block")
    return os.path.join(beh, folder, blocks[0]), proc.stderr


def world_step(blob: bytes) -> int:
    """How many steps the world in a savestate has taken.

    :param blob: a crafter savestate, as ``restore()`` wants it.
    :return: the engine's own step counter.
    """
    return pickle.loads(blob).env.game._step


def world_length(blob: bytes) -> int:
    """The step cap the world in a savestate was built with.

    :param blob: a crafter savestate, as ``restore()`` wants it.
    :return: crafter's own ``length``, 0 for no cap.
    """
    return pickle.loads(blob).env.game._length


def last_anchor(block: np.lib.npyio.NpzFile) -> bytes:
    """The last savestate anchor in a block.

    :param block: a loaded block npz.
    :return: the anchor's blob.
    """
    return [s for s in block["states"] if s is not None][-1]


def part1() -> None:
    """The slot file, the refusals, and which backends can resume."""
    print("part 1: the slot file")
    folder = tempfile.mkdtemp()
    try:
        path = resume.save(folder, "probe", b"0123456789",
                           {"backend": "crafter", "env_kwargs": {"size": [128, 128]}})
        carry = resume.load(folder, "probe")
        check("a slot round trips", carry.blob == b"0123456789")
        check("its header keeps where the world came from",
              carry.header["backend"] == "crafter" and carry.header["slot"] == "probe")
        check("an unwritten slot is not an error", resume.load(folder, "never") is None)
        check("matching env_kwargs are not worth a warning",
              resume.env_drift(carry, {"size": [128, 128]}) == [])
        check("changed ones are", resume.env_drift(carry, {"size": [384, 384]}) == ["size"])
        check("and so is a field the world never had",
              resume.env_drift(carry, {"size": [128, 128], "length": 20}) == ["length"])

        whole = open(path, "rb").read()
        with open(path, "wb") as f:                     # the machine died mid-write
            f.write(whole[:-4])
        check("a half-written file is refused", refuses(folder, "probe", "not finished"))
        with open(path, "wb") as f:
            f.write(b"not json\nxxxx")
        check("a stray .state is refused", refuses(folder, "probe", "header"))
    finally:
        shutil.rmtree(folder, ignore_errors=True)

    phase = [p for p in load_config(L4)["curriculum"] if p["type"] == "game"][0]
    adapter = get_adapter("crafter", phase)
    try:
        check("crafter can resume", resume.supported(adapter))
    finally:
        adapter.close()
    adapter = get_adapter("gym", {"game": "CartPole-v1", "keys": {"": 0, "LEFT": 0},
                                  "fps": 30})
    try:
        check("a backend with no savestate cannot", not resume.supported(adapter))
    finally:
        adapter.close()
    check("the real level-4 config validates", validate_config(load_config(L4)) == [])


def refuses(folder: str, slot: str, wanted: str) -> bool:
    """Whether loading this slot raises, saying ``wanted``.

    :param folder: the slot's folder.
    :param slot: the slot name.
    :param wanted: a phrase the refusal should contain.
    :return: whether it refused for that reason.
    """
    try:
        resume.load(folder, slot)
    except ValueError as exc:
        return wanted in str(exc)
    return False


def part2(curriculum: str, data_root: str) -> str:
    """Two processes, one session: the world run 1 ends in is the one run 2 opens.

    :param curriculum: the short level-4 config.
    :param data_root: the BIDS root the runs share.
    :return: the slot file's path.
    """
    print("part 2: two processes, one session")
    one = np.load(play(curriculum, data_root, 1)[0], allow_pickle=True)
    slot = os.path.join(data_root, "sub-01", "ses-001", "resume", "crafter_L4.state")
    check("run 1 left a world behind", os.path.exists(slot))
    left = resume.load(os.path.dirname(slot), "crafter_L4").blob
    check("run 1 opened no world", not one["episode_resumed"].any()
          and "resume_state" not in one.files)

    block, said = play(curriculum, data_root, 2)
    two = np.load(block, allow_pickle=True)
    check("run 2 is marked resumed", bool(two["episode_resumed"].all()))
    check("and it had nothing to warn run 2 about", "WARNING" not in said)
    check("run 2 was handed exactly what run 1 wrote",
          "resume_state" in two.files and two["resume_state"].tobytes() == left,
          f"{world_step(left)} steps in")
    check("run 2 then played past it",
          world_step(last_anchor(two)) > world_step(left),
          f"{world_step(last_anchor(two))} > {world_step(left)}")
    same_seed = list(one["episode_seeds"]) == list(two["episode_seeds"])
    digest = [hashlib.sha256(np.asarray(b["semantic"][0]).tobytes()).hexdigest()[:12]
              for b in (one, two)]
    check("and it is not the world that seed opens", same_seed and digest[0] != digest[1],
          f"seed {one['episode_seeds'][0]} both runs, first frames {digest[0]} vs {digest[1]}")
    return slot


def part3(curriculum: str, data_root: str, slot: str) -> None:
    """A restored world keeps the ``env_kwargs`` it was built with, and says so.

    ``restore`` replaces the env the block just built, so the block's own
    ``env_kwargs`` reach nothing: a thread of play is fixed at the moment it
    opens. That is defensible and it is also invisible, which is the dangerous
    half, so the check here is that the run warns rather than that it obeys.

    :param curriculum: the short level-4 config, edited to disagree here.
    :param data_root: the BIDS root the runs share.
    :param slot: the slot file part 2 left.
    """
    print("part 3: env_kwargs a restored world will not honour")
    drifted = curriculum.replace(".json", "_drift.json")
    config = json.load(open(curriculum))
    kwargs = {**config["curriculum"][0]["env_kwargs"], "length": 20}
    config["curriculum"][0]["env_kwargs"] = kwargs
    json.dump(config, open(drifted, "w"))
    was = world_length(resume.load(os.path.dirname(slot), "crafter_L4").blob)
    path, said = play(drifted, data_root, 3)
    block = np.load(path, allow_pickle=True)
    check("the run warned that the config would not take",
          "WARNING" in said and "length" in said)
    check("and the world kept the cap it was opened with",
          world_length(last_anchor(block)) == was == 0,
          f"length {kwargs['length']} in the config, {was} in the world")
    # The header follows the world, not the config of whichever block last
    # held it, or the warning would repeat for every block from here on.
    check("the slot still says what the world was opened with",
          (resume.load(os.path.dirname(slot), "crafter_L4").header
           .get("env_kwargs", {}).get("length")) == 0)


def part4(curriculum: str, data_root: str, slot: str) -> None:
    """An ending the game chose ends the thread of play with it.

    The block's clock is the scanner's, and carrying a world across it is what
    keeps the scanner from changing crafter. A death is crafter's own, so it
    has to still cost what crafter charges for it: the slot goes, and the next
    block of that name opens a world of its own rather than putting the subject
    back where the block they died in began.

    :param curriculum: the short level-4 config, played to a real ending here.
    :param data_root: the BIDS root the runs share.
    :param slot: the slot file the runs share.
    """
    print("part 4: an episode the game ended")
    ended = curriculum.replace(".json", "_ended.json")
    config = json.load(open(curriculum))
    # No clock: the block runs until the game ends the episode, which for a
    # player who presses nothing but noop means dying of thirst.
    config["curriculum"][0].update(mode="episode", n_episodes=1)
    json.dump(config, open(ended, "w"))
    path, said = play(ended, data_root, 4)
    block = np.load(path, allow_pickle=True)
    check("a block back on the original config is not warned at", "WARNING" not in said)
    check("the block still resumed", bool(block["episode_resumed"].all()))
    check("the game ended its episode", str(block["episode_outcome"][-1]) != "playing",
          str(block["episode_outcome"][-1]))
    check("and the world went with it", not os.path.exists(slot))

    after, said = play(curriculum, data_root, 5)
    check("so the next block of that name opened its own",
          not np.load(after, allow_pickle=True)["episode_resumed"].any()
          and "no world yet" in said)


def main() -> None:
    """Run all four parts and exit non-zero on any failure."""
    work = tempfile.mkdtemp(prefix="resume-check-")
    try:
        part1()
        curriculum = short_curriculum(os.path.join(work, "l4_short.json"))
        data_root = os.path.join(work, "data")
        slot = part2(curriculum, data_root)
        part3(curriculum, data_root, slot)
        part4(curriculum, data_root, slot)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{len(failures)} failures" if failures else "\nall checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
