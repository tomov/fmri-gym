"""Check that the four crafter levels are the rules they claim to be.

``env_kwargs.level`` is a promise about a game, so every check here measures
the game rather than the code that sets it up: it builds the env a config
builds, reads the engine's own state and the pixels it draws, and compares them
against what ``crafter_gym.levels.LEVELS`` says the level is.

    python docs/levels_check.py
    CRAFTER_RIG=../crafter_rig/core.py python docs/levels_check.py

Part 1 is provenance: the table is a port of the rig's, so with ``CRAFTER_RIG``
pointing at the rig's ``core.py`` it is compared row by row against the
original. Without it that one check says it was skipped, because the rig is not
a dependency of this repo and nothing here can go looking for it. Part 2 is the
rules, one level at a time, each measured where the engine keeps it: hostiles
cleared and kept out through a night, the life stats frozen or not, health
refused or taken, the four icons gone from the item panel and every other icon
still at its own index, the death tint, the sleep button, lava still lethal,
and the achievement count the strip divides by, which is a claim about the
rules: the two defeats are unreachable because the creature is never in the
world, and the other twenty survive a frozen homeostat, shown on the two whose
reward is a stat that is already full.
Each has the engine's own behaviour next to it as a negative control,
because a rule that cannot be shown to change anything is not evidence of
anything. Part 3 is what the levels are for in fMRI: the same seed gives the
same terrain in all four, a level survives the pickle a savestate and a resume
are made of, the rules come back bound to the restored player, a restored env
plays on frame for frame, and the closure style the rig patches on cannot be
pickled at all, which is why these rules are objects. Part 4 runs the real L1
config twice through ``fmri_play.py``, in two processes and one BIDS session, to
show that a levelled world resumes into its own slot, and once more with a
block of another level pointed at that slot, to show it is warned by name.

The in-process envs of parts 2 and 3 are a quarter of crafter's world (see
:data:`AREA`), because worldgen is what a reset costs and there is one env per
level per rule here; part 4's blocks are the configs' own world, at the frame
size ``docs/resume_check.py`` plays at.

Last run 2026-10-02 on this branch: 65 checks, 0 failures (60 without
``CRAFTER_RIG``).
"""

import ast
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
from crafter_gym.levels import LEVELS  # noqa: E402
from fmri_gym import resume  # noqa: E402
from fmri_gym.config import load_config, validate_config  # noqa: E402
from fmri_gym.logging import read_events  # noqa: E402

# Unpickling a savestate imports crafter, and the repo's own gym/ directory
# shadows old gym for a process started at the repo root, which is what
# `crafter_gym.import_crafter` deals with.
crafter = crafter_gym.import_crafter()

CONFIGS = os.path.join(ROOT, "configs", "dbp_games")
#: Small enough to render a few hundred frames, still a whole multiple of the
#: engine's 9x9 view.
SIZE = [128, 128]
#: A quarter of crafter's 64x64 world. Worldgen is the slow part of a reset and
#: these checks want one env per level per rule, so the world is small -- but
#: not so small that no hostile spawns in it, which is the one thing a smaller
#: world would quietly take away (24x24 generates none).
AREA = (32, 32)
#: Deep enough into crafter's night for a zombie to spawn: daylight falls below
#: 0.5 between steps 148 and 272 of each 300-step day.
NIGHT = 200
#: The four move actions of ``crafter.constants.actions``, each with the step it
#: takes, and the index of ``do``.
MOVES = ((1, (-1, 0)), (2, (+1, 0)), (3, (0, -1)), (4, (0, +1)))
DO = 5
#: The two achievements no level without a hostile in it can unlock, in the
#: order ``crafter.constants.achievements`` lists them.
HOSTILE_ACHIEVEMENTS = ["defeat_skeleton", "defeat_zombie"]

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


def build(level: str | None, seed: int = 1, menu: bool = False) -> tuple:
    """One env in one level, reset and ready to play.

    :param level: a key of :data:`LEVELS`, or ``None`` for stock crafter.
    :param seed: the episode seed.
    :param menu: build it behind the eight-button menu, as the configs do.
    :return: ``(env, game, player)``.
    """
    make = crafter_gym.make_menu if menu else crafter_gym.make_plain
    env = make(area=AREA, size=SIZE, length=0, level=level)
    env.reset(seed=seed)
    game = env.unwrapped.game
    return env, game, game._player


def level_wrapper(env: object) -> object:
    """The level wrapper in this chain, wherever it sits in it.

    Found rather than taken off the outside, because a level that names tasks
    has the task chain above it and Gymnasium 1.3 forwards no attribute through
    a wrapper. What a block reads is the level in the chain, which is this
    wrapper's own ``level`` (``crafter_gym.levels.level_of``).

    :param env: any env, wrapped or not.
    :return: the :class:`~crafter_gym.levels.LevelWrapper`, or ``None`` if the
        chain has none.
    """
    while hasattr(env, "env"):
        if isinstance(env, crafter_gym.LevelWrapper):
            return env
        env = env.env
    return None


def unit_of(game: object) -> np.ndarray:
    """The pixel size of one tile, by the engine's own arithmetic.

    :param game: the ``crafter.Env``.
    :return: crafter's ``unit``, as ``render`` computes it.
    """
    return np.array(game._size) // np.array(game._view)


def hostiles(game: object) -> list:
    """Every zombie, skeleton and arrow in a world.

    :param game: the ``crafter.Env``.
    :return: those objects.
    """
    kinds = (crafter.objects.Zombie, crafter.objects.Skeleton, crafter.objects.Arrow)
    return [o for o in game._world.objects if isinstance(o, kinds)]


def free_neighbour(game: object, player: object) -> tuple:
    """A cell beside the player with nothing standing in it.

    Every check that puts something within reach needs one: ``world.add``
    refuses to share a cell, and so does the ``is_free`` of a move.

    :param game: the ``crafter.Env``.
    :param player: its ``Player``.
    :return: ``(move action, facing, cell)`` for that neighbour.
    """
    pos = np.array(player.pos)
    return next((a, d, tuple(pos + d)) for a, d in MOVES
                if game._world[tuple(pos + d)][1] is None)


def part1() -> None:
    """The table is the rig's, row for row."""
    print("part 1: the table is a port")
    path = os.environ.get("CRAFTER_RIG")
    if not path:
        print("  [skip] set CRAFTER_RIG=<rig>/core.py to compare against the "
              "original; the rig is not a dependency of this repo")
        return
    # Parsed rather than imported: importing the rig pulls in pygame and asks
    # for a display, and the one thing being read here is a literal.
    source = open(path).read()
    assign = next(n for n in ast.parse(source).body
                  if isinstance(n, ast.Assign)
                  and getattr(n.targets[0], "id", None) == "LEVELS")
    namespace: dict = {"dict": dict}
    exec(ast.get_source_segment(source, assign), namespace)  # noqa: S102
    theirs = namespace["LEVELS"]
    check("the same levels, in the same order", list(LEVELS) == list(theirs),
          " ".join(LEVELS))
    for name in theirs:
        check(f"{name} is the rig's own row", LEVELS.get(name) == theirs[name],
              json.dumps(LEVELS.get(name), default=str, sort_keys=True))


def part2() -> None:
    """Each level's rules, measured on the engine's own state and pixels."""
    print("part 2: the rules")
    for level in LEVELS:
        env, game, _ = build(level)
        at_reset = len(hostiles(game))
        for _ in range(NIGHT):
            env.step(0)
        after = len(hostiles(game))
        want = LEVELS[level]["hostiles"]
        check(f"{level}: {'some' if want else 'none'} at worldgen and through a night",
              (at_reset > 0) == want and (after > 0) == want,
              f"{at_reset} at reset, {after} at step {NIGHT}")
        env.close()
    _, game, _ = build(None)
    check("stock crafter: the same world has hostiles", len(hostiles(game)) > 0,
          f"{len(hostiles(game))} at reset")

    for level in ("L1_affordance", "L2_homeostasis"):
        env, _, player = build(level)
        for _ in range(30):
            env.step(0)
        stats = {k: player.inventory[k] for k in ("food", "drink", "energy")}
        frozen = bool(LEVELS[level].get("frozen_stats"))
        # Stock ticks each counter once a step and spends a point past its own
        # threshold, so 30 noops cost food and drink but not yet energy.
        check(f"{level}: the life stats are {'frozen at 9' if frozen else 'ticking'}"
              " after 30 steps",
              (stats == {"food": 9, "drink": 9, "energy": 9}) == frozen,
              json.dumps(stats))
        env.close()

    for level in LEVELS:
        _, _, player = build(level)
        # The engine spends a point of health when _recover falls past -15 with
        # a necessity missing (objects.py:160), which is the only way the
        # homeostat can kill.
        player.inventory["food"] = 0
        player._recover = -15
        player._degen_or_regen_health()
        held = player.health == 9
        check(f"{level}: starvation {'cannot' if held else 'can'} cost health",
              held == (not LEVELS[level]["homeostatic_death"]),
              f"health {player.health}")
    for level in LEVELS:
        _, _, player = build(level)
        # And it hands one back when _recover passes 25, which it can do on the
        # very tick a death set health to 0.
        player.health = 0
        player._recover = 25
        player._degen_or_regen_health()
        check(f"{level}: dead stays dead", player.health == 0, f"health {player.health}")
    _, _, player = build(None)
    player.health = 0
    player._recover = 25
    player._degen_or_regen_health()
    check("stock crafter: the same state revives", player.health == 1,
          f"health {player.health}")

    env, game, player = build("L1_affordance")
    hidden = LEVELS["L1_affordance"]["hidden_items"]
    unit, panel = unit_of(game), game._item_view
    stock = panel.view
    shown, blanked = stock(player.inventory, unit), panel(player.inventory, unit)
    # A cell's position is the item's index in the whole inventory dict, and
    # the engine draws an item and its count inside that one cell, so hiding by
    # zeroing a copy has to leave every pixel outside those four cells alone.
    cells = np.zeros(shown.shape[:2], bool)
    for name in hidden:
        index = list(player.inventory).index(name)
        x, y = index % stock._grid[0], index // stock._grid[0]
        cells[x * unit[0]:(x + 1) * unit[0], y * unit[1]:(y + 1) * unit[1]] = True
    check("L1: the four stat cells of the item panel are empty",
          blanked[cells].max() == 0 and shown[cells].max() > 0, ", ".join(hidden))
    check("L1: and every other icon is where the stock game draws it",
          np.array_equal(blanked[~cells], shown[~cells]),
          f"{len(player.inventory) - len(hidden)} other slots")
    env.close()
    for level in ("L2_homeostasis", "L3_predation", "L4_survival"):
        _, game, _ = build(level)
        check(f"{level}: the item panel is the engine's own",
              type(game._item_view) is crafter.engine.ItemView,
              type(game._item_view).__name__)

    for level in LEVELS:
        _, game, player = build(level)
        view, unit = game._local_view, unit_of(game)
        alive = np.array_equal(view(player, unit), view.view(player, unit))
        player.health = 0
        plain = np.asarray(view.view(player, unit), np.float64)
        want = 0.4 * plain + 0.6 * np.array((128.0, 0.0, 0.0))
        check(f"{level}: the view is tinted dead and untouched alive",
              alive and np.array_equal(view(player, unit), want),
              "0.4 canvas + 0.6 (128, 0, 0)")

    for level in ("L1_affordance", "L2_homeostasis"):
        env, _, player = build(level)
        env.step(6)
        slept = bool(player.sleeping)
        env.step(0)
        frozen = bool(LEVELS[level].get("frozen_stats"))
        check(f"{level}: sleep at full energy {'naps' if slept else 'is refused'}",
              slept == frozen, f"wake_up {player.achievements['wake_up']}")
        if frozen:
            check("L1: and the nap wakes on the next step, unlocking wake_up",
                  player.achievements["wake_up"] == 1 and not player.sleeping)
        env.close()

    env, game, player = build("L1_affordance")
    action, _, target = free_neighbour(game, player)
    game._world[target] = "lava"
    _, _, terminated, truncated, info = env.step(action)
    check("L1: lava still ends the episode",
          terminated and not truncated and info["inventory"]["health"] == 0,
          f"health {info['inventory']['health']}, terminated {terminated}")
    env.close()

    # What the rules cost the scoreboard. The strip divides by the reachable
    # count, so the count is a claim about the rules and is measured as one: the
    # two defeats go because the creature is never in the world (the hostile
    # checks above), and the rest stay because an unlock hangs off the action
    # and not off the stat it feeds, which the two live checks below show at the
    # only stat value where the difference is visible.
    names = list(crafter.constants.achievements)
    check("crafter still ships 22 achievements", len(names) == 22, str(len(names)))
    check("stock crafter: every one of them is reachable",
          list(crafter_gym.reachable_achievements(None)) == names)
    for level in LEVELS:
        reach = crafter_gym.reachable_achievements(level)
        hostile = LEVELS[level]["hostiles"]
        gone = sorted(set(names) - set(reach))
        check(f"{level}: {len(names) if hostile else len(names) - 2} of the "
              f"{len(names)} are reachable, in crafter's order",
              gone == ([] if hostile else HOSTILE_ACHIEVEMENTS)
              and list(reach) == [n for n in names if n not in gone],
              f"{len(reach)} reachable, missing {', '.join(gone) or 'none'}")
        env, _, _ = build(level)
        check(f"{level}: the wrapper says the same of the env it wraps",
              level_wrapper(env).achievements == reach)
        env.close()

    env, game, player = build("L1_affordance")
    _, facing, target = free_neighbour(game, player)
    game._world[target] = "water"
    # Water is not walkable, so a move would only turn the player; face it and
    # drink in one step. `collect.water` requires nothing and leaves water.
    player.facing = facing
    env.step(DO)
    check("L1: collect_drink unlocks on drinking with the stat already full",
          player.achievements["collect_drink"] == 1
          and player.inventory["drink"] == 9,
          f"drink {player.inventory['drink']}")
    env.close()

    env, game, player = build("L1_affordance")
    _, facing, target = free_neighbour(game, player)
    cow = crafter.objects.Cow(game._world, np.array(target))
    # A kill in one hit, which the player lands before the cow's own update can
    # walk it out of the faced cell (`env.step` updates the player first).
    cow.health = 1
    game._world.add(cow)
    player.facing = facing
    env.step(DO)
    check("L1: eat_cow unlocks on the kill with the stat already full",
          player.achievements["eat_cow"] == 1 and player.inventory["food"] == 9,
          f"food {player.inventory['food']}")
    env.close()


def part3() -> None:
    """What a level has to survive: a seed, a pickle, and a restore."""
    print("part 3: a level across a savestate")
    maps = {}
    for level in LEVELS:
        _, game, _ = build(level, seed=7)
        maps[level] = game._world._mat_map.copy()
    first = list(LEVELS)[0]
    check("the same seed is the same terrain in all four levels",
          all(np.array_equal(maps[first], m) for m in maps.values()),
          f"{maps[first].shape[0]}x{maps[first].shape[1]} tiles, "
          f"{len(np.unique(maps[first]))} materials")

    env, game, player = build("L1_affordance", seed=3, menu=True)
    for action in (3, 3, 2, 5, 0, 1):
        env.step(action)
    blob = pickle.dumps(env, protocol=5)
    back = pickle.loads(blob)
    bg = back.unwrapped.game
    rule = bg._player._degen_or_regen_health
    check("the rules come back with the world",
          type(rule).__name__ == "_NoHomeostaticDeath"
          and type(bg._item_view).__name__ == "_HiddenItems"
          and type(bg._local_view).__name__ == "_DeathTint"
          and type(bg._balance_object).__name__ == "_NoHostiles",
          f"{len(blob)} bytes")
    check("bound to the restored player and not to the original",
          rule.player is bg._player and rule.inner.player is bg._player
          and rule.player is not player)
    check("and the wrapper still knows which level it is",
          level_wrapper(back).level == "L1_affordance"
          and level_wrapper(back).rules == LEVELS["L1_affordance"])
    # The point of a savestate: the restored world is where play continues, so
    # it has to produce the frames the original would have.
    here = [env.step(a) for a in (4, 4, 5, 17, 18, 0)]
    there = [back.step(a) for a in (4, 4, 5, 17, 18, 0)]
    check("a restored world plays on frame for frame",
          all(np.array_equal(a[0], b[0]) for a, b in zip(here, there))
          and [a[1:4] for a in here] == [b[1:4] for b in there],
          f"{len(there)} frames")
    check("and keeps the hostiles out on the far side",
          not hostiles(bg) and not hostiles(game))
    env.close()
    back.close()

    _, _, fresh = build("L4_survival")
    check("a player with these rules on it pickles",
          bool(pickle.dumps(fresh, protocol=5)))
    original = crafter.objects.Player._degen_or_regen_health

    def patched() -> None:          # the rig's style: a closure over this player
        original(fresh)

    fresh._degen_or_regen_health = patched
    try:
        pickle.dumps(fresh, protocol=5)
        raised = ""
    except Exception as exc:
        raised = f"{type(exc).__name__}: {exc}"
    check("while a closure patched onto one does not", bool(raised), raised[:110])


def short_config(level: int, work: str, slot: str | None = None) -> str:
    """One level's real config, as a block a headless process can play.

    :param level: 1 to 4.
    :param work: folder to write the config into.
    :param slot: override the phase's resume slot, to point this block at
        another level's world.
    :return: the path written.
    """
    src = os.path.join(CONFIGS, f"crafter__crafter_L{level}.json")
    phase = [p for p in load_config(src)["curriculum"] if p["type"] == "game"][0]
    # Real time wants a name for "no key held", which a turn-based block has no
    # use for: nobody is at the keyboard here, so every frame sends that noop.
    phase = {**phase, "duration": 6.0, "fps": 8, "turn_based": False,
             "live_hud": False, "state_stride": 5,
             "keys": {"": 0, **phase["keys"]},
             "env_kwargs": {**phase["env_kwargs"], "size": SIZE}}
    if slot:
        phase["resume"] = slot
    name = f"l{level}_short{'_into_' + slot if slot else ''}.json"
    path = os.path.join(work, name)
    with open(path, "w") as f:
        json.dump({"triggers": {"sync": {"mode": "none"}, "backend": "null"},
                   "curriculum": [phase]}, f)
    return path


def play(curriculum: str, data_root: str, run: int) -> tuple[str, str]:
    """Run one ``fmri_play.py`` block in its own process, headless.

    :param curriculum: the config to play.
    :param data_root: the BIDS root the runs share.
    :param run: the run number.
    :return: the block's folder, and what the run said on stderr.
    :raises RuntimeError: if the process failed or wrote no block.
    """
    env = {**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
           "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")}
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, "fmri_play.py"),
         "--curriculum", curriculum, "--subject", "sub-01", "--ses", "1",
         "--run", str(run), "--data-root", data_root,
         "--dummy-trigger", "--no-audio"],
        cwd=ROOT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"run {run} failed:\n{proc.stderr[-2000:]}")
    beh = os.path.join(data_root, "sub-01", "ses-001", "beh")
    folder = [d for d in sorted(os.listdir(beh)) if d.endswith(f"run-{run:03d}")][0]
    run_dir = os.path.join(beh, folder)
    blocks = [d for d in sorted(os.listdir(run_dir)) if d.startswith("block-")]
    if not blocks:
        raise RuntimeError(f"run {run} wrote no block")
    return os.path.join(run_dir, blocks[0]), proc.stderr


def lines_of(block: str, kind: str) -> list[dict]:
    """Every line of one type in a block's ``events.jsonl``.

    :param block: the block's folder.
    :param kind: the ``type`` field to keep.
    :return: those lines, in the order they were written.
    """
    return [e for e in read_events(block) if e.get("type") == kind]


def part4(work: str) -> None:
    """The four configs, and one level twice through the run loop."""
    print("part 4: a level in a run")
    slots = []
    for level in range(1, 5):
        config = load_config(os.path.join(CONFIGS, f"crafter__crafter_L{level}.json"))
        phase = [p for p in config["curriculum"] if p["type"] == "game"][0]
        slots.append(phase["resume"])
        check(f"the real level-{level} config validates and names its own level",
              validate_config(config) == []
              and phase["env_kwargs"]["level"] in LEVELS
              and phase["resume"] == f"crafter_L{level}",
              f'{phase["env_kwargs"]["level"]} into {phase["resume"]!r}')
    check("and no two of them share a slot", len(set(slots)) == 4, " ".join(slots))

    data_root = os.path.join(work, "data")
    first, _ = play(short_config(1, work), data_root, 1)
    check("the block records the level it played",
          read_events(first)[0]["phase"]["env_kwargs"]["level"] == "L1_affordance")
    folder = os.path.join(data_root, "sub-01", "ses-001", "resume")
    slot = os.path.join(folder, "crafter_L1.state")
    check("and left its world in this level's own slot", os.path.exists(slot),
          os.path.relpath(slot, data_root))
    check("whose header names the level too",
          resume.load(folder, "crafter_L1").header["env_kwargs"].get("level")
          == "L1_affordance")

    second, said = play(short_config(1, work), data_root, 2)
    check("the next block of that level continues the world",
          all(e["resumed"] for e in lines_of(second, "episode_start"))
          and len(lines_of(second, "resume")) == 1,
          f"{len(lines_of(second, 'episode_start'))} episodes")
    check("with nothing to warn it about", "WARNING" not in said)

    # Pointing an L2 block at L1's slot is exactly what one shared slot name
    # would have done by itself, on every block, silently.
    _, said = play(short_config(2, work, slot="crafter_L1"), data_root, 3)
    warned = [ln for ln in said.splitlines() if "WARNING" in ln and "level" in ln]
    check("a block of another level is warned, and both are named",
          bool(warned) and "L1_affordance" in warned[0]
          and "L2_homeostasis" in warned[0],
          warned[0].split("WARNING ")[-1][:120] if warned else "no warning")


def main() -> None:
    """Run all four parts and exit non-zero on any failure."""
    work = tempfile.mkdtemp(prefix="levels-check-")
    try:
        part1()
        part2()
        part3()
        part4(work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{len(failures)} failures" if failures else "\nall checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
