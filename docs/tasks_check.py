"""Check that the task chain asks for one thing at a time, and says so on screen.

A task level promises the subject a sequence: one achievement is named above the
frame, the frame is held when it is reached to say so, and the next one is named.
That promise is made in four places at once -- the chain in the env
(:mod:`crafter_gym.tasks`), the strip and the held messages in the crafter
adapter, the hold itself in the run loop, and the fields a block logs -- so every
check here measures the thing promised rather than the code that arranges it: the
engine's own state, the pixels the display packed, the lines it drew, and the
block's own record.

    python docs/tasks_check.py
    CRAFTER_RIG=../crafter_rig/core.py python docs/tasks_check.py

Part 1 is provenance: the chain, its wording and the two hold lengths are a port
of the rig's, so with ``CRAFTER_RIG`` pointing at the rig's ``core.py`` they are
compared against the original, including both cue functions over every entry.
Without it that part says it was skipped, because the rig is not a dependency of
this repo. What it checks either way is the one thing the port cannot inherit:
which entries crafter scores by itself, and which one the chain therefore has to
score, at both ends of the fraction. Part 2 is the chain as state, measured
through ``info``: a frame that changes nothing says nothing, an unlock earned
before the chain asks for it moves no pointer and is stepped over when the
pointer arrives, a composite entry waits for both halves, a skip puts the task at
the back of this episode's own order, a death outranks all of it, and the whole
wrapper leaves the world it wraps untouched -- same map, same inventory, same
achievements, same random state. Part 3 is the one entry crafter has no
achievement for, against nine built worlds: what seals a room and what only looks
like it does, that the test is read-only, and that being sealed before the chain
asks does not pre-complete it. Part 4 is the adapter: the field that sets the two
holds and every way a bad one is refused, the strip's three fields with the task
between them and the count the shelter widens, every label packed into one row at
the shipped window size, the two messages a completion produces, and the fields a
frame and a block record. Part 5 plays a real block through ``fmri_play.py`` with
a policy in place of the subject, where the half of this nothing else can see is
measured: the messages reached the screen in order, each hold really held, and the
block's clock paid for them -- the same play, ending earlier, rather than a stall.

Part 5 measures wall-clock on the machine it runs on, and its two blocks are
played one after the other, so it wants that machine otherwise idle: a block
that is competing for a core plays slower, and what a hold cost is read from
the difference between the two. The rest of the file is state and pixels and
does not care.

Not covered here: the homeostat's own task interrupts, which are not ported
(:mod:`crafter_gym.tasks`, "What is not here"), and the shipped 300-second
turn-based block, which has a person at the keyboard. Part 5 plays the same phase
in real time at a smaller frame and a faster clock.

Last run 2026-10-02 on this branch: 2 blocks, 79 checks, 0 failures (73 without
``CRAFTER_RIG``).
"""

import ast
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import numpy as np

# Before fmri_gym.display is imported: part 4 opens a real window to measure what
# the strip packs into, and there is no screen here.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import crafter_gym  # noqa: E402
from crafter_gym import tasks  # noqa: E402
from fmri_gym import display, resume  # noqa: E402
from fmri_gym.adapters import get_adapter  # noqa: E402
from fmri_gym.adapters.crafter import _TASK_HOLDS, _task_holds  # noqa: E402
from fmri_gym.config import load_config, validate_config  # noqa: E402
from fmri_gym.logging import read_events  # noqa: E402

# Unpickling a savestate imports crafter, and the repo's own gym/ directory
# shadows old gym for a process started at the repo root, which is what
# `crafter_gym.import_crafter` deals with.
crafter = crafter_gym.import_crafter()

CONFIGS = os.path.join(ROOT, "configs", "dbp_games")
#: The level the shipped task configs are read from, and the one whose chain is
#: the whole of what it asks for: no hostiles and a frozen homeostat, so nothing
#: but the player moves the chain.
LEVEL = "L1_affordance"
#: Small enough to reset a few dozen worlds, still a whole multiple of the
#: engine's 9x9 view.
SIZE = [128, 128]
#: A quarter of crafter's world, for the reason ``docs/levels_check.py`` uses it.
AREA = (32, 32)
#: How far around the player part 3 flattens the ground before building: wider
#: than the widest wall it builds, so what is being measured is the wall.
CLEARING = 14
#: The shipped window (``fmri_play.py --size``). The HUD font is a fixed 24 px
#: whatever the window is, so this is the tightest the strip is ever packed, and
#: the frame it is packed against is the shipped ``env_kwargs.size``.
WINDOW = (1024, 768)
FRAME = 384
#: Part 5's block: a real-time clock fast enough that a few tasks are reached in
#: twenty seconds, and holds short enough that three of them fit in it.
FPS = 8
DURATION = 20.0
DONE_S = 0.4
NEXT_S = 0.6

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


# --- the env ---------------------------------------------------------------

def build(level: str | None = LEVEL, seed: int = 1, menu: bool = True,
          chain: tuple[str, ...] | None = None) -> tuple:
    """One env in one level, reset and ready to play.

    :param level: a key of ``crafter_gym.LEVELS``, or ``None`` for stock crafter.
    :param seed: the episode seed.
    :param menu: build it behind the eight-button menu, as the configs do.
    :param chain: ask for these entries instead of the whole chain, which is how
        a short episode reaches the end of one.
    :return: ``(env, game, tracker, info)`` for the episode it opened, the tracker
        ``None`` on a level that names no tasks. The game is handed back because
        a reset builds a new one, so a handle taken across a reset is stale and
        granting an achievement on it reaches a player nobody is playing.
    """
    make = crafter_gym.make_menu if menu else crafter_gym.make_plain
    if chain is None:
        env = make(area=AREA, size=SIZE, length=0, level=level)
    else:
        env = tasks.TaskWrapper(make(area=AREA, size=SIZE, length=0), chain=chain)
    _, info = env.reset(seed=seed)
    return env, env.unwrapped.game, crafter_gym.tracker_of(env), info


def player_cell(game: object) -> tuple[int, int]:
    """The player's cell as a pair of ints, which is how the world is indexed.

    :param game: the ``crafter.Env``.
    :return: ``(x, y)``.
    """
    return tuple(int(p) for p in game._player.pos)


def grant(game: object, *names: str) -> None:
    """Unlock achievements behind the chain's back, as play would have.

    The chain reads crafter's own counters (``TaskWrapper._unlocked_in``), so
    setting one is the whole of what an unlock is as far as it can tell, and it
    costs none of the hours of play the later entries would.

    :param game: the ``crafter.Env``.
    :param names: achievement names.
    """
    for name in names:
        game._player.achievements[name] += 1


def close_through(stepper: object, game: object, tracker: object, entry: str) -> dict:
    """Unlock everything the chain asks for before ``entry``, and step once.

    The step is what makes the chain notice: unlocks reach it through ``info``,
    so a grant is not seen until the next frame.

    :param stepper: the env or adapter to step.
    :param game: the ``crafter.Env``.
    :param tracker: the :class:`~crafter_gym.tasks.TaskWrapper`.
    :param entry: the entry to leave as the pointer.
    :return: that step's ``info``.
    """
    for name in tracker.chain[:tracker.chain.index(entry)]:
        for goal in tasks.TASK_GOALS.get(name, (name,)):
            if goal in crafter.constants.achievements:
                grant(game, goal)
    return stepper.step(0)[4]


def skip(env: object, tracker: object) -> dict:
    """Put the current task off from the menu, the way a subject does.

    :param env: the env, behind a menu.
    :param tracker: the :class:`~crafter_gym.tasks.TaskWrapper`.
    :return: the confirming step's ``info``.
    """
    menu = crafter_gym.menu_of(env)
    while menu.selected != tasks.SKIP_ENTRY:
        env.step(menu.cycle)
    return env.step(menu.confirm)[4]


def rng_digest(game: object) -> str:
    """The world's random state as twelve hex digits.

    What a predicate must not move: every later draw the engine makes comes out
    of this, so a read-only test leaves it exactly here.

    :param game: the ``crafter.Env``.
    :return: the first 12 hex digits of its sha256.
    """
    return hashlib.sha256(game._world.random.get_state()[1].tobytes()).hexdigest()[:12]


# --- worlds to build in ----------------------------------------------------

def clear(game: object, radius: int = CLEARING, floor: str = "grass") -> tuple[int, int]:
    """Flatten the ground around the player and take everything off it.

    :param game: the ``crafter.Env``.
    :param radius: half-width of the square to flatten.
    :param floor: the material to lay down.
    :return: the player's cell.
    """
    world, start = game._world, player_cell(game)
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            cell = (start[0] + dx, start[1] + dy)
            material, obj = world[cell]
            if material is None:
                continue
            if obj is not None and obj is not game._player:
                world.remove(obj)
            world[cell] = floor
    return start


def ring(game: object, start: tuple[int, int], radius: int, material: str,
         interior: str | None = None, door: tuple[int, int] | None = None) -> None:
    """Build a square wall around a cell.

    :param game: the ``crafter.Env``.
    :param start: the cell to build around.
    :param radius: half-width of the wall.
    :param material: what the wall is made of.
    :param interior: a material to lay inside it, if not the floor already there.
    :param door: a direction to leave one cell of the wall open in.
    """
    world = game._world
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            cell = (start[0] + dx, start[1] + dy)
            if max(abs(dx), abs(dy)) == radius:
                world[cell] = material
            elif interior and cell != start:
                world[cell] = interior
    if door:
        world[(start[0] + door[0] * radius, start[1] + door[1] * radius)] = "grass"


def fenced(game: object, start: tuple[int, int], radius: int = 1) -> None:
    """Build the same wall out of fences, which are objects and not materials.

    :param game: the ``crafter.Env``.
    :param start: the cell to build around.
    :param radius: half-width of the wall.
    """
    world = game._world
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if max(abs(dx), abs(dy)) == radius:
                world.add(crafter.objects.Fence(world, (start[0] + dx, start[1] + dy)))


def lake(game: object, start: tuple[int, int], radius: int) -> None:
    """Flood everything around the player, leaving them their own cell.

    :param game: the ``crafter.Env``.
    :param start: the cell to leave dry.
    :param radius: half-width of the water.
    """
    world = game._world
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if (dx, dy) != (0, 0):
                world[(start[0] + dx, start[1] + dy)] = "water"


# --- the rig ---------------------------------------------------------------

def rig_objects(path: str, names: tuple[str, ...]) -> dict:
    """The rig's own values for these top-level names.

    Parsed and exec'd statement by statement rather than imported: importing the
    rig pulls in pygame and asks for a display, and what is wanted here is a
    handful of literals and two functions that read one of them.

    :param path: the rig module to read.
    :param names: the assignments and functions to take.
    :return: a namespace holding them.
    """
    source = open(path).read()
    namespace: dict = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            wanted = getattr(node.targets[0], "id", None) in names
        elif isinstance(node, ast.FunctionDef):
            wanted = node.name in names
        else:
            wanted = False
        if wanted:
            exec(ast.get_source_segment(source, node), namespace)  # noqa: S102
    return namespace


def rig_default(path: str, flag: str) -> float:
    """The default of one of the rig's command-line flags.

    :param path: the rig module holding its argument parser.
    :param flag: the flag, as it is written on the command line.
    :return: its default.
    :raises LookupError: if the parser does not define that flag.
    """
    for node in ast.walk(ast.parse(open(path).read())):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"
                and node.args and getattr(node.args[0], "value", None) == flag):
            return next(k.value.value for k in node.keywords if k.arg == "default")
    raise LookupError(f"the rig's parser has no {flag}")


def part1() -> None:
    """The chain is the rig's, and what crafter does not pay for is counted."""
    print("part 1: the chain is a port")
    path = os.environ.get("CRAFTER_RIG")
    if not path:
        print("  [skip] set CRAFTER_RIG=<rig>/core.py to compare against the "
              "original; the rig is not a dependency of this repo")
    else:
        names = ("TASK_CHAIN", "TASK_GOALS", "TASK_LABELS", "SKIP_ENTRY",
                 "TASK_DONE_TEXT", "task_text", "next_task_text")
        theirs = rig_objects(path, names)
        check("the same entries, in the same order",
              tasks.TASK_CHAIN == theirs["TASK_CHAIN"], " ".join(tasks.TASK_CHAIN))
        check("the same composite entries",
              tasks.TASK_GOALS == theirs["TASK_GOALS"], json.dumps(tasks.TASK_GOALS))
        # As a mapping: the rig lists the shelter's label last, this module keeps
        # it next to the entry it belongs to.
        check("the same wording, label for label",
              tasks.TASK_LABELS == theirs["TASK_LABELS"],
              f"{len(tasks.TASK_LABELS)} labels")
        check("the same name for the entry that puts a task off, and the same "
              "line for a completion",
              (tasks.SKIP_ENTRY, tasks.TASK_DONE_TEXT)
              == (theirs["SKIP_ENTRY"], theirs["TASK_DONE_TEXT"]),
              f"{tasks.SKIP_ENTRY!r}, {tasks.TASK_DONE_TEXT!r}")
        asked = [*dict.fromkeys((*tasks.TASK_CHAIN, *tasks.TASK_LABELS)), None]
        check("and both cues word every entry, and a finished chain, as the rig does",
              all(tasks.task_text(t) == theirs["task_text"](t)
                  and tasks.next_task_text(t) == theirs["next_task_text"](t)
                  for t in asked),
              f"{len(asked)} entries, {tasks.task_text(None)!r} at the end")
        frontend = os.path.join(os.path.dirname(path), "frontend_pygame.py")
        rig_holds = {"done": rig_default(frontend, "--task-done-s"),
                     "next": rig_default(frontend, "--task-hold-s")}
        check("the two messages are held for the rig's own defaults",
              _TASK_HOLDS == rig_holds,
              ", ".join(f"{k} {v} s" for k, v in _TASK_HOLDS.items()))

    check("the chain asks for each entry once",
          len(set(tasks.TASK_CHAIN)) == len(tasks.TASK_CHAIN),
          f"{len(tasks.TASK_CHAIN)} entries")
    # The port's own question, and the reason the fraction above the frame is not
    # just a count of achievements: an entry crafter scores needs nothing added at
    # either end of it, and an entry crafter has never heard of needs both.
    unpaid = [e for e in tasks.TASK_CHAIN
              if any(g not in crafter.constants.achievements
                     for g in tasks.TASK_GOALS.get(e, (e,)))]
    check("every entry but one is an achievement crafter scores itself",
          unpaid == ["build_stone_shelter"],
          f"{len(tasks.TASK_CHAIN) - len(unpaid)} of {len(tasks.TASK_CHAIN)}, "
          f"not {', '.join(unpaid)}")
    check("so that one is the chain's own to score",
          list(tasks.TASK_PREDICATES) == unpaid)
    _, _, tracker, _ = build()
    check("and it is what the chain tells a scoreboard to widen by",
          tracker.scorable == tuple(unpaid), f"{tracker.scorable}")
    levels = [n for n in crafter_gym.LEVELS if tasks.has_tasks(n)]
    check("the levels that name tasks are the ones the table says",
          levels == [n for n, r in crafter_gym.LEVELS.items() if r.get("tasks")]
          and not tasks.has_tasks(None), " ".join(levels))
    _, _, none, _ = build(level="L4_survival")
    check("and a level that names none is played without a chain at all", none is None)


def part2() -> None:
    """The chain as state: what moves the pointer, and what the wrapper costs."""
    print("part 2: the pointer")
    env, game, tracker, info = build()
    check("an episode opens on the chain's first entry",
          (tracker.task, tracker.passed, tracker.chain)
          == (tasks.TASK_CHAIN[0], frozenset(), list(tasks.TASK_CHAIN)),
          f"{tracker.task!r}")
    check("and says so in the reset info",
          (info["task"], info["task_done"], info["task_moved"], info["task_skip"],
           info["task_passed"]) == (tasks.TASK_CHAIN[0], False, False, False, ""))
    info = env.step(0)[4]
    check("a frame that changes nothing says nothing",
          (info["task"], info["task_moved"], info["task_done"]) == ("collect_wood", False, False))

    grant(game, "collect_diamond")
    info = env.step(0)[4]
    check("an unlock the chain has not asked for yet moves no pointer",
          (info["task"], info["task_moved"], info["task_done"])
          == ("collect_wood", False, False),
          "collect_diamond is the chain's eighth entry")
    grant(game, "collect_wood")
    info = env.step(0)[4]
    check("the entry in hand completes, and the frame that did it knows both",
          (info["task"], info["task_moved"], info["task_done"], tracker.task)
          == ("collect_wood", True, True, "place_table"),
          f'{info["task"]!r} -> {tracker.task!r}')
    env.close()

    env, game, tracker, _ = build()
    grant(game, "collect_wood", "place_table")
    info = env.step(0)[4]
    check("an entry already closed when the pointer reaches it is stepped over",
          (info["task"], info["task_done"], tracker.task)
          == ("collect_wood", True, "collect_place_plant"),
          "place_table was unlocked on the same frame as collect_wood")
    grant(game, "collect_sapling")
    info = env.step(0)[4]
    check("a composite entry waits for both of its halves",
          (tracker.task, info["task_moved"]) == ("collect_place_plant", False),
          "collect_sapling without place_plant")
    grant(game, "place_plant")
    info = env.step(0)[4]
    check("and completes on the second", (info["task"], info["task_done"], tracker.task)
          == ("collect_place_plant", True, "make_wood_pickaxe"))
    env.close()

    # A chain of its own, short enough that an episode reaches the end of it: the
    # same wrapper, asked for less (`TaskWrapper(chain=...)`).
    env, game, tracker, _ = build(level=None, chain=("collect_wood", "place_table"))
    menu = crafter_gym.menu_of(env)
    check("a task level puts the entry that puts a task off at the end of the menu",
          menu.menu_names[-1] == tasks.SKIP_ENTRY and len(menu.menu_names) == 11,
          f"{len(menu.menu_names)} entries, last {menu.menu_names[-1]!r}")
    was = game._step
    info = skip(env, tracker)
    check("putting one off moves it to the back of this episode's own order",
          (info["task"], info["task_skip"], info["task_moved"], info["task_done"],
           tracker.chain, tracker.task)
          == ("collect_wood", True, True, False, ["place_table", "collect_wood"],
              "place_table"),
          f"{tracker.chain}")
    check("the gesture costs the engine a turn per press and gives it no action",
          game._step - was == len(menu.menu_names) and info["env_action"] == 0,
          f"{game._step - was} steps, {info['menu_sel']!r} confirmed as noop")
    grant(game, "place_table")
    env.step(0)
    info = skip(env, tracker)
    check("and putting off the last task left does nothing at all",
          (info["task"], info["task_skip"], info["task_moved"], tracker.chain)
          == ("collect_wood", False, False, ["place_table", "collect_wood"]),
          "the chain cannot ask for anything else")
    env.close()

    env, game, tracker, _ = build()
    grant(game, "collect_wood")
    game._player.health = 0
    info = env.step(0)[4]
    check("a death outranks the chain: the task is recorded, the move is not",
          (info["task"], info["task_moved"], info["task_done"], tracker.task)
          == ("collect_wood", False, False, "place_table"),
          "the episode has an ending of its own to show")
    env.close()

    # What the chain costs the world, which is the agent-parity claim of the
    # module: the same presses have to leave the same game.
    actions = (3, 3, 2, 5, 0, 1, 5, 4, 5, 0)
    worlds = []
    for tasked in (True, False):
        env = crafter_gym.with_level(
            crafter_gym.CrafterEnv(area=AREA, size=SIZE, length=0), LEVEL)
        if tasked:
            env = tasks.TaskWrapper(env)
        env.reset(seed=5)
        game = env.unwrapped.game
        worlds.append({"frames": [env.step(a)[0] for a in actions],
                       "map": game._world._mat_map.copy(),
                       "inventory": dict(game._player.inventory),
                       "achievements": dict(game._player.achievements),
                       "random": rng_digest(game)})
        env.close()
    chained, plain = worlds
    check("the chain leaves the world it wraps untouched",
          all(np.array_equal(a, b)
              for a, b in zip(chained["frames"], plain["frames"]))
          and np.array_equal(chained["map"], plain["map"])
          and all(chained[k] == plain[k]
                  for k in ("inventory", "achievements", "random")),
          f"{len(actions)} presses, random state {chained['random']} either way")

    env, game, tracker, _ = build(seed=3)
    grant(game, "collect_wood")
    env.step(0)
    skip(env, tracker)
    blob = pickle.dumps(env, protocol=5)
    back = pickle.loads(blob)
    restored = crafter_gym.tracker_of(back)
    check("a chain mid-episode survives the pickle a savestate is made of",
          restored is not None and restored.chain == tracker.chain
          and restored.task == tracker.task and restored.passed == tracker.passed,
          f"{len(blob)} bytes, pointer {restored.task!r}")
    here = [env.step(a) for a in (4, 4, 5, 0)]
    there = [back.step(a) for a in (4, 4, 5, 0)]
    check("and the restored world plays on frame for frame, chain included",
          all(np.array_equal(a[0], b[0]) for a, b in zip(here, there))
          and [a[4]["task"] for a in here] == [b[4]["task"] for b in there])
    env.close()
    back.close()


def part3() -> None:
    """The one entry the chain scores itself: nine worlds, and a read-only test."""
    print("part 3: what counts as a shelter")
    predicate = tasks.TASK_PREDICATES["build_stone_shelter"]
    cases = []
    env, game, _, _ = build(menu=False)
    clear(game)
    cases.append(("an open field is not a shelter", game, False))
    for name, want, make in (
        ("a wall of stone one step out is", True,
         lambda g, s: ring(g, s, 1, "stone")),
        ("the same wall with one cell open is not", False,
         lambda g, s: ring(g, s, 1, "stone", door=(1, 0))),
        ("a moat one cell wide is not, because an arrow flies over it", False,
         lambda g, s: ring(g, s, 1, "water")),
        ("a lake five cells wide is, because nothing can stand in it", True,
         lambda g, s: lake(g, s, 5)),
        ("a fence is, though the ground under it is still grass", True,
         lambda g, s: fenced(g, s)),
        ("a room three steps across is", True,
         lambda g, s: ring(g, s, 3, "stone")),
        ("a walled meadow is not, because it breeds its own zombie", False,
         lambda g, s: ring(g, s, 6, "stone")),
        ("and a sealed region too large to be a room is not", False,
         lambda g, s: ring(g, s, 11, "stone", interior="sand")),
    ):
        env, game, _, _ = build(menu=False)
        start = clear(game)
        make(game, start)
        cases.append((name, game, want))
    moved = []
    for name, game, want in cases:
        before = rng_digest(game)
        got = predicate(game)
        if rng_digest(game) != before:
            moved.append(name)
        check(name, got == want, f"{got}")
    check("and asking the question never moves the world's random state",
          not moved, f"{len(cases)} worlds, digest unchanged in each")

    env, game, tracker, _ = build()
    start = clear(game)
    ring(game, start, 1, "stone")
    info = env.step(0)[4]
    check("sealing yourself in before the chain asks does not pre-complete it",
          (info["task_passed"], tracker.passed) == ("", frozenset()),
          f'the pointer is {info["task"]!r}')
    close_through(env, game, tracker, "build_stone_shelter")
    before = dict(game._player.achievements)
    info = env.step(0)[4]
    check("and being sealed in while it is asked completes it",
          (info["task"], info["task_passed"], info["task_done"], tracker.passed)
          == ("build_stone_shelter", "build_stone_shelter", True,
              frozenset({"build_stone_shelter"})),
          f"the pointer moved on to {tracker.task!r}")
    check("without crafter having scored anything, which is why it is counted twice",
          dict(info["achievements"]) == before
          and "build_stone_shelter" not in crafter.constants.achievements,
          "no achievement of that name exists")
    env.close()


# --- the adapter -----------------------------------------------------------

def spec(level: int = 1, **over: object) -> dict:
    """One level's real game phase, as a block an in-process adapter can build.

    :param level: 1 to 4.
    :param over: phase fields to replace. A ``None`` value drops the field,
        which no shipped config has one of.
    :return: the phase dict.
    """
    src = os.path.join(CONFIGS, f"crafter__crafter_L{level}.json")
    phase = [p for p in load_config(src)["curriculum"] if p["type"] == "game"][0]
    phase = {**phase, "env_kwargs": {**phase["env_kwargs"], "size": SIZE, "area": AREA},
             **over}
    return {k: v for k, v in phase.items() if v is not None}


def adapter_on(level: int = 1, seed: int = 1, **over: object) -> tuple:
    """One adapter on one level's phase, reset and stepped once.

    The step is what gives the strip something to say: an adapter's own fields
    are set by ``step``, so a block's first frame is the earliest any of this can
    be read.

    :param level: 1 to 4.
    :param seed: the episode seed.
    :param over: phase fields to replace.
    :return: ``(adapter, game, tracker, info)``.
    """
    adapter = get_adapter("crafter", spec(level, **over))
    adapter.reset(seed)
    info = adapter.step(0)[4]
    return adapter, adapter._game, adapter._tracker, info


def refusal(call: object) -> str:
    """The message a bad field was refused with, or ``""`` if it was accepted.

    :param call: a thunk that builds whatever is under test.
    :return: the ``ValueError``'s message.
    """
    try:
        call()
    except ValueError as exc:
        return str(exc)
    return ""


def strip_rows(screen: object, strip: list[str]) -> tuple[int, int]:
    """How many rows the display packs a strip into, and how wide its longest line is.

    Measured through a real ``draw_frame`` rather than by adding up widths: the
    width the lines are packed to is the frame's, after the strip itself has
    shrunk a height-limited frame, and that loop is the display's own.

    :param screen: the :class:`~fmri_gym.display.Display` to pack on.
    :param strip: the adapter's HUD lines.
    :return: ``(rows, pixels)``.
    """
    seen = []
    real = display.Display._hud_rows

    def recording(self: object, lines: list[str], width: int) -> list:
        rows = real(self, lines, width)
        seen.append(rows)
        return rows

    display.Display._hud_rows = recording
    try:
        screen.draw_frame(np.zeros((FRAME, FRAME, 3), np.uint8), list(strip))
    finally:
        display.Display._hud_rows = real
    return len(seen[-1]), max(screen.hud_font.size(line)[0] for line in strip)


def part4() -> None:
    """The adapter: the field, the strip, the two messages, and the record."""
    print("part 4: what the subject is shown")
    phase = spec()
    check("the shipped level-1 config validates and asks for the two holds",
          validate_config(load_config(os.path.join(CONFIGS, "crafter__crafter_L1.json")))
          == [] and phase["tasks"] == _TASK_HOLDS, json.dumps(phase["tasks"]))
    check("a phase that asks for nothing gets the rig's own timings",
          _task_holds({}, True) == _TASK_HOLDS)
    check("and one that asks for half of it keeps the other half",
          _task_holds({"tasks": {"done": 0.4}}, True) == {"done": 0.4, "next": 2.0})
    said = refusal(lambda: _task_holds({"tasks": {"done": 1.0}}, False))
    check("a hold on a level that names no tasks is refused, with the table to read",
          "names none" in said and "LEVELS" in said, said[:90])
    check("so is a field that is not the two of them",
          all(refusal(lambda asked=asked: _task_holds({"tasks": asked}, True))
              for asked in (1.0, [1.0], {"hold": 1.0}, {"done": 1.0, "hold": 1.0})),
          "a number, a list, an unknown key, a good key beside a bad one")
    check("and a hold that is not a length of time",
          all(refusal(lambda v=v: _task_holds({"tasks": {"next": v}}, True))
              for v in (True, "1.0", -0.5, None)),
          "a flag, a string, a negative number, nothing")
    said = refusal(lambda: get_adapter("crafter", spec(tasks={"done": -1})))
    check("refused where a block would meet it, before a subject is in the bore",
          "seconds to hold" in said, said[:90])

    adapter, game, tracker, info = adapter_on()
    check("an adapter on a task level finds the chain in the env it built",
          tracker is not None and adapter._holds == _TASK_HOLDS)
    total = len(adapter._achievements) + len(tracker.scorable)
    check("the strip is the clock, the task, then the count",
          adapter.hud(0.0, 12.4) == ["12 s", "TASK: collect wood", f"0 / {total}"],
          " | ".join(adapter.hud(0.0, 12.4)))
    check("whose denominator is what the level pays for, the chain's own entry included",
          total == len(adapter._achievements) + 1 and total == 21,
          f"{len(adapter._achievements)} reachable achievements + "
          f"{len(tracker.scorable)} chain entry")
    start = clear(game)
    ring(game, start, 1, "stone")
    close_through(adapter, game, tracker, "build_stone_shelter")
    before = adapter.hud(0.0, 1.0)
    info = adapter.step(0)[4]
    check("and the entry crafter does not pay for moves it like any other",
          (before[-1], adapter.hud(0.0, 1.0)[-1]) == (f"6 / {total}", f"7 / {total}"),
          f"{before[-1]} -> {adapter.hud(0.0, 1.0)[-1]}")
    adapter.close()

    adapter, _, _, _ = adapter_on(show_score=False)
    check("a block that shows the score instead of the count still names the task",
          adapter.hud(2.0, 12.4) == ["12 s", "TASK: collect wood", "Score: 2"],
          " | ".join(adapter.hud(2.0, 12.4)))
    adapter.close()

    # The one thing a label is measured against: the strip holds the clock and
    # the count too, and a label that wraps pushes the frame down mid-block.
    screen = display.Display(size=WINDOW, vsync=False)
    try:
        widest, rows = (0, ""), 0
        for entry in (*tasks.TASK_CHAIN, None):
            strip = ["300 s", tasks.task_text(entry), f"20 / {total}"]
            packed, pixels = strip_rows(screen, strip)
            rows = max(rows, packed)
            widest = max(widest, (pixels, tasks.task_text(entry)))
        check("every label the chain can show fits one row of the shipped window",
              rows == 1, f"{widest[1]!r} is {widest[0]} px of "
                         f"{FRAME} at {WINDOW[0]}x{WINDOW[1]}")
    finally:
        screen.close()

    adapter, _, tracker, _ = adapter_on()
    check("a frame that moved no pointer is held up for nothing",
          adapter._task_notices({"task_moved": False, "task_done": True}) == [])
    check("a completion is announced, and then what comes next",
          adapter._task_notices({"task_moved": True, "task_done": True})
          == [([tasks.TASK_DONE_TEXT], 1.5), (["Next task: collect wood"], 2.0)],
          f"{tasks.TASK_DONE_TEXT!r} for 1.5 s, then the next task for 2.0 s")
    check("a pointer that moved without scoring says only what comes next",
          adapter._task_notices({"task_moved": True, "task_done": False})
          == [(["Next task: collect wood"], 2.0)],
          "an entry closed by an unlock the chain had not yet asked for")
    adapter.close()
    quiet, game, tracker, _ = adapter_on(tasks={"done": 0})
    check("and a block that asks for no announcement is not given one",
          quiet._task_notices({"task_moved": True, "task_done": True})
          == [(["Next task: collect wood"], 2.0)], "tasks.done = 0")
    quiet.close()

    adapter, game, tracker, _ = adapter_on()
    grant(game, "collect_wood")
    obs, _, _, _, info = adapter.step(0)
    check("a real completion leaves the two messages for the loop to show",
          [lines for lines, _ in adapter.notices()]
          == [[tasks.TASK_DONE_TEXT], ["Next task: place table"]],
          f'{len(adapter.notices())} messages at task {info["task"]!r}')
    blob = adapter.capture(obs, info, want_blob=True).blob
    variables = adapter.capture(obs, info, want_blob=False).variables
    check("the frame records the task it was played under, and what it did to the chain",
          (variables["task"], variables["task_done"], variables["task_passed"],
           variables["task_skip"]) == ("collect_wood", True, "", False),
          ", ".join(k for k in variables if k.startswith("task")))
    adapter.restore(blob)
    check("a restored frame is held up for nothing, since nobody pressed anything "
          "to reach it", adapter.notices() == [])
    adapter.reset(2)
    check("and neither is the first frame of a new episode", adapter.notices() == [])
    extra = adapter.block_extra()
    chain = crafter_gym.tracker_of(adapter.env).base_chain
    check("the block records the chain it asked for, in the words it used",
          list(extra["task_chain"]) == list(chain)
          and list(extra["task_labels"]) == [tasks.task_text(t) for t in chain],
          f"{len(chain)} entries")
    check("and which of them it scores itself",
          [c for c, s in zip(extra["task_chain"], extra["task_scorable"]) if s]
          == list(tracker.scorable)
          and list(extra["menu_names"])[-1] == tasks.SKIP_ENTRY,
          f"{', '.join(tracker.scorable)}, offered as {tasks.SKIP_ENTRY!r}")
    adapter.close()

    adapter, _, tracker, info = adapter_on(level=4)
    check("a level with no chain records none of this",
          tracker is None and not any(k.startswith("task") for k in info)
          and not any(k.startswith("task") for k in adapter.block_extra()),
          "level 4 keeps its unlocks anonymous")
    check("and its strip is the clock and the count alone",
          len(adapter.hud(0.0, 12.4)) == 2, " | ".join(adapter.hud(0.0, 12.4)))
    adapter.close()


# --- a real block ----------------------------------------------------------

#: ``fmri_play.py`` with a player and an eye. The player is a policy in place of
#: the subject's hands (``fmri_gym.keys.register_held_source``), stateless by
#: construction: it reads the live game every frame, so it presses what a
#: cooperative subject would and cannot drift out of step with a block whose
#: frame count it does not control. The eye writes down every strip and every
#: line drawn over a frame, because a held message is pixels and nothing else --
#: :meth:`fmri_gym.display.Display.draw_frame` is the one place a frame reaches
#: the screen, and there is no window here to look at. The run itself is
#: untouched: both hooks read, and call the real thing.
DRIVER = '''\
import json
import os
import runpy
import sys

from fmri_gym import display, keys, run

drawn = []
flip = display.Display.draw_frame


def recording(self, rgb, hud=None, overlay=None):
    drawn.append([hud, overlay])
    return flip(self, rgb, hud, overlay)


display.Display.draw_frame = recording

live = {}
build = run.get_adapter


def keeping(backend, phase):
    adapter = build(backend, phase)
    live["adapter"] = adapter
    return adapter


run.get_adapter = keeping

WALK = {(-1, 0): "LEFT", (1, 0): "RIGHT", (0, -1): "UP", (0, 1): "DOWN"}
GROUND = ("grass", "sand", "path")


def toward(game, start, material):
    """A first step along the shortest walk to a cell of `material`."""
    seen = {start}
    queue = [(start, None)]
    while queue:
        cell, first = queue.pop(0)
        for delta in WALK:
            side = (cell[0] + delta[0], cell[1] + delta[1])
            if game._world[side][0] == material:
                return (first or delta), cell == start
            if side in seen:
                continue
            ground, obj = game._world[side]
            if ground not in GROUND or obj is not None:
                continue
            seen.add(side)
            queue.append((side, first or delta))
    return None, False


def policy():
    """Chop the nearest tree, then put the next task off."""
    adapter = live.get("adapter")
    if adapter is None:
        return frozenset()
    game, tracker = adapter._game, adapter._tracker
    player = game._player
    if not player.achievements["collect_wood"]:
        delta, adjacent = toward(game, tuple(int(p) for p in player.pos), "tree")
        if delta is None:
            return frozenset()
        if adjacent and tuple(player.facing) == delta:
            return frozenset({"D"})
        return frozenset({WALK[delta]})
    if tracker.task == "place_table":
        return frozenset({"A" if adapter._menu.selected == "skip_task" else "W"})
    return frozenset()


keys.register_held_source(policy)
sys.argv = [os.environ["FMRI_PLAY"], *sys.argv[1:]]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
finally:
    with open(os.environ["FMRI_DRAWN_LOG"], "w") as f:
        json.dump(drawn, f)
'''


def block_config(path: str, **over: object) -> str:
    """Write the level-1 game phase as a block a policy plays headless.

    :param path: file to write the config to.
    :param over: phase fields to replace, over the real-time ones.
    :return: ``path``.
    """
    phase = [p for p in load_config(os.path.join(CONFIGS, "crafter__crafter_L1.json"))
             ["curriculum"] if p["type"] == "game"][0]
    # Real time wants a name for "no key held", which a turn-based block has no
    # use for, and no window to redraw between presses: the strip is drawn on
    # every frame of a real-time block already.
    phase = {**phase, "duration": DURATION, "fps": FPS, "turn_based": False,
             "live_hud": False, "state_stride": 5, "keys": {"": 0, **phase["keys"]},
             "env_kwargs": {**phase["env_kwargs"], "size": SIZE},
             "tasks": {"done": DONE_S, "next": NEXT_S}, **over}
    with open(path, "w") as f:
        json.dump({"triggers": {"sync": {"mode": "none"}, "backend": "null"},
                   "curriculum": [phase]}, f)
    return path


def play(curriculum: str, data_root: str) -> tuple[str, list]:
    """Run one block of it in its own process, headless, with the driver in front.

    :param curriculum: the config to play, in the folder this writes its own
        driver and screen log beside.
    :param data_root: the BIDS root to write the block under. One per block: two
        blocks of the same level in one session share a resume slot, and the
        second would open the first one's world.
    :return: the block's folder, and everything the display drew (:data:`DRIVER`).
    :raises RuntimeError: if the process failed or wrote no block.
    """
    work = os.path.dirname(curriculum)
    name = os.path.splitext(os.path.basename(curriculum))[0]
    driver = os.path.join(work, "play_recording.py")
    with open(driver, "w") as f:
        f.write(DRIVER)
    log = os.path.join(work, f"drawn-{name}.json")
    env = {**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy",
           "PYTHONPATH": ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""),
           "FMRI_PLAY": os.path.join(ROOT, "fmri_play.py"), "FMRI_DRAWN_LOG": log}
    proc = subprocess.run(
        [sys.executable, driver, "--curriculum", curriculum, "--subject", "sub-01",
         "--ses", "1", "--run", "1", "--data-root", data_root,
         "--dummy-trigger", "--no-audio"],
        cwd=ROOT, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed:\n{proc.stderr[-2000:]}")
    run_dir = os.path.join(data_root, "sub-01", "ses-001", "beh")
    run_dir = os.path.join(run_dir, sorted(os.listdir(run_dir))[0])
    blocks = [d for d in sorted(os.listdir(run_dir)) if d.startswith("block-")]
    if not blocks:
        raise RuntimeError(f"{name} wrote no block")
    with open(log) as f:
        drawn = json.load(f)
    return os.path.join(run_dir, blocks[0]), drawn


def lines_of(block: str, kind: str) -> list[dict]:
    """Every line of one type in a block's ``events.jsonl``.

    :param block: the block's folder.
    :param kind: the ``type`` field to keep.
    :return: those lines, in the order they were written.
    """
    return [e for e in read_events(block) if e.get("type") == kind]


def phase_entry(block: str) -> dict:
    """The manifest's entry for the one game phase of a block's run.

    :param block: the block's folder.
    :return: the phase entry, which is where its onset and offset are.
    """
    manifest = json.load(open(os.path.join(os.path.dirname(block), "manifest.json")))
    return [p for p in manifest["phases"] if p["type"] == "game"][0]


def part5(work: str) -> None:
    """A block that really holds: the screen, the record, and what it costs.

    :param work: the folder to write the configs, drivers and data under.
    """
    print("part 5: a block with a player in it")
    block, drawn = play(block_config(os.path.join(work, "l1_held.json")),
                        os.path.join(work, "data_held"))
    frames = lines_of(block, "frame")
    notices = lines_of(block, "notice")
    wanted = [[tasks.TASK_DONE_TEXT], ["Next task: place table"],
              ["Next task: collect and place plant"]]
    check("the player finished a task and put the next one off",
          [n["lines"] for n in notices] == wanted,
          f"{len(notices)} messages in {len(frames)} frames")
    if len(notices) != len(wanted):
        return
    overlay_y = 3.5 / 9.0
    shown = [o for _, o in drawn if o is not None and o[0] in wanted]
    check("each reached the screen once, in that order, over the frame that earned it",
          shown == [[lines, overlay_y] for lines in wanted],
          f"{len(shown)} drawn at {overlay_y:.3f} of the frame height")
    strips = [hud for hud, o in drawn if o is not None and o[0] in wanted]
    check("and the strip beside them already named the task they announce",
          [s[1] for s in strips]
          == ["TASK: place table", "TASK: place table",
              "TASK: collect and place plant"]
          and strips[0][2].startswith("1 /"),
          f"{strips[0][1]!r} at {strips[0][2]!r}")

    # Each hold is the gap between the message going up and the next thing
    # happening, which is the next message or the next frame: both are logged, so
    # the timeline the subject saw is the log's own order.
    timeline = [(e["flip_time"] if e["type"] == "notice" else e["run_time"], e)
                for e in read_events(block) if e.get("type") in ("frame", "notice")]
    held = []
    for index, (at, line) in enumerate(timeline):
        if line["type"] == "notice" and index + 1 < len(timeline):
            held.append((timeline[index + 1][0] - at, line["duration"]))
    check("every message was held for at least as long as it asked for",
          len(held) == len(notices) and all(was >= asked for was, asked in held),
          ", ".join(f"{was:.2f} s for {asked} s" for was, asked in held))
    check("and the holds are the two lengths the config asked for",
          [n["duration"] for n in notices] == [DONE_S, NEXT_S, NEXT_S])

    done, skipped = notices[0]["ep_frame"], notices[2]["ep_frame"]
    check("the frame that completed the task says so in the record",
          (frames[done]["variables"]["task"], frames[done]["variables"]["task_done"])
          == ("collect_wood", True), f"frame {done}")
    row = frames[skipped]
    check("and the frame the task was put off on says that",
          (row["variables"]["task"], row["variables"]["task_skip"],
           row["variables"]["menu_sel"], row["env_action"])
          == ("place_table", True, tasks.SKIP_ENTRY, 0),
          f"frame {skipped}, button {row['action']}")
    carry = resume.load(os.path.join(work, "data_held", "sub-01", "ses-001", "resume"),
                        "crafter_L1")
    tracker = crafter_gym.tracker_of(pickle.loads(carry.blob))
    check("the world the next block opens has the chain the subject left it with",
          tracker.chain[-1] == "place_table" and tracker.task == "collect_place_plant",
          f"{tracker.task!r} next, {tracker.chain[-1]!r} at the back")

    # What a hold costs, which is the question a block's design asks: the same
    # play in the same wall-clock block, reaching fewer frames.
    silent, _ = play(block_config(os.path.join(work, "l1_silent.json"),
                                  tasks={"done": 0, "next": 0}),
                     os.path.join(work, "data_silent"))
    quiet = lines_of(silent, "frame")
    check("a block that holds for nothing logs no message at all",
          not lines_of(silent, "notice"), f"{len(quiet)} frames")
    check("and plays exactly what the held block played, as far as it got",
          [f["action"] for f in quiet][:len(frames)] == [f["action"] for f in frames],
          f"{len(frames)} presses either way")
    durations = [phase_entry(b)["offset"] - phase_entry(b)["onset"]
                 for b in (block, silent)]
    # Counted in the time one press got rather than in frames, because `fps` is
    # not the rate either block reached: a real-time block gets whatever the
    # engine and the display leave it, which on this machine is short of 8. So
    # the holds come out of the held block's own clock exactly when the two
    # blocks spent the same time per press, the holding taken out of the first.
    # Both have to have run at the same rate for that to mean anything, so part
    # 5 wants a machine that is otherwise idle (see this module's note).
    presses = [(durations[0] - (DONE_S + 2 * NEXT_S)) / len(frames),
               durations[1] / len(quiet)]
    check("so the holds are paid out of the block's own clock, not added to it",
          abs(presses[0] - presses[1]) < 0.05 * presses[1],
          f"{len(quiet) - len(frames)} frames fewer, and {presses[0] * 1000:.0f} ms "
          f"a press against {presses[1] * 1000:.0f} ms")
    check("the block itself lasted the same either way",
          abs(durations[0] - durations[1]) < 0.5,
          f"{durations[0]:.2f} s held, {durations[1]:.2f} s silent")
    check("and neither one ever fell behind its own clock",
          all(lines_of(b, "block_end")[-1]["n_pacing_resets"] == 0
              for b in (block, silent)))


def main() -> None:
    """Run all five parts and exit non-zero on any failure."""
    work = tempfile.mkdtemp(prefix="tasks-check-")
    try:
        part1()
        part2()
        part3()
        part4()
        part5(work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{len(failures)} failures: {failures}" if failures else "\nall checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
