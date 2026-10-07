"""Crafter's task chain: the one thing the player is asked to do next.

Crafter's 22 achievements are a tech tree, and a player dropped into it with no
guidance spends their first hour rediscovering that stone wants a wood pickaxe
and that iron wants coal and a furnace. :data:`TASK_CHAIN` is the paradigm's
answer (crafter-for-brain-scan v0.33, ``core.py``, ported entry for entry): one
achievement at a time is named on screen, in an order that is a feasible path
through the tree, and the next one is named as soon as the current one is
reached. The levels that name tasks say so in their row of
:data:`~crafter_gym.levels.LEVELS` (``tasks``); stock crafter's level keeps its
unlocks anonymous, so there the whole tree is the task.

The chain adds nothing to the world. Every entry but one is an achievement
crafter already scores, naming it changes no rule, and the two things the
player can do about it cost a turn of the engine and nothing else: asking for
the current task to be put off steps crafter with ``noop``
(:data:`SKIP_ENTRY`), and the one entry crafter has no achievement for is a
read-only test on the map (:data:`TASK_PREDICATES`). So a block's actions
replay into the same world whether or not tasks were on, and a policy trained
against this wrapper meets the same game.

Why it is an env wrapper
------------------------

Because the pointer is state, and state the game is played through. Which task
is named decides what a cooperative player does next, the skip reorders the
rest of the episode, and both have to come back with a savestate: the crafter
adapter's savestate is a pickle of the env chain, so a wrapper's fields are in
the blob and an adapter's fields are not, and a block that resumes a world
would otherwise resume it with the chain back at ``collect wood`` (the same
argument :mod:`crafter_gym.levels` makes for the rules and
:mod:`crafter_gym.menu` for the cursor). It also keeps the goal on the
environment side of the comparison this repo exists for: a goal-conditioned
agent reads :data:`TASK_CHAIN` and each step's ``info["task"]``, which is the
text the subject was shown.

The pointer is derived
----------------------

:attr:`TaskWrapper.task` is computed on every read as the first entry of this
episode's chain that is still open, never stored and advanced. That is what
makes out-of-order play behave: an achievement unlocked long before the chain
asks for it leaves the pointer where it is and scores the moment it unlocks,
and when the pointer arrives at that entry it is already closed, so it is
stepped over without any bookkeeping having had to notice. An entry that is a
composite (:data:`TASK_GOALS`) is open until all of its achievements are
unlocked, so one sapling of a sapling-and-plant pair does not close it.

What the chain scores
---------------------

The chain is the whole of the score: :attr:`TaskWrapper.completed` against
:attr:`TaskWrapper.base_chain` is the fraction above the frame, and a ``+1``
sounds when and only when that numerator moves. Crafter's own 22 achievement
counters are untouched and still logged every frame, but the ones the chain
never names stop being a point the subject is paid: ten of the 22 on a level
that asks for :data:`TASK_CHAIN` alone, eleven on L1, and seven on a level
whose interrupts name three more (below).

That is a rule about which achievements mean anything on the level being
played (Fan, 2026-10-06). Crafter awards an unlock for the act and never for its
effect: ``collect_drink`` is paid for drinking water with the stat already at
9, because ``_do_material`` checks a collect's ``require`` and never its
``receive`` (``crafter/objects.py:219``), and the point then lands on an
inventory cell that ``update`` clamps straight back to full
(``objects.py:127``). On L1, where :class:`~crafter_gym.levels._FrozenLifeStats`
pins all four stats and :class:`~crafter_gym.levels._HiddenItems` hides them,
that is a point for an act with no consequence the subject can even see. Four
achievements hang on a live homeostat this way: ``collect_drink``, ``eat_cow``,
``wake_up`` and ``eat_plant``. None of the four is in the chain of a level that
freezes the stats, so none of them is paid there, and all four are in the chain
of a level whose stats run, the first three as the interrupts below. Deciding
that per level is :func:`chain_for`'s whole job.

``build_stone_shelter`` is the chain's one entry crafter has no achievement
for, so nothing in a block's log accounts for it on its own:
:attr:`TaskWrapper.passed` is where it is closed and
:attr:`TaskWrapper.scorable` marks it in the log for an analysis that joins
the chain to the achievement columns. It needs no correction to the
denominator any more, because the denominator is the chain and the entry is in
it.

The low-stat interrupts
-----------------------

The base chain is playable thirsty, so on its own it lets a subject ignore the
homeostat that L2 exists to ask about. A level that says so in its row
(``stat_tasks``) therefore carries three entries more, one per stat
(:data:`STAT_TASKS`), and each becomes askable the first time that stat is seen
at or below :data:`STAT_TASK_LOW` in an episode. Until then the pointer steps
over it, and on the step it arms it is moved up the episode's own order to just
ahead of the chain the subject was working through, which is what makes it an
interrupt rather than a thirteenth task. Closing it hands the pointer straight
back to the entry it cut in front of.

Fan settled three things about them (2026-10-07), and the derived pointer gives
all three out of the order :attr:`TaskWrapper.chain` already keeps for skips,
plus one per-episode set of the entries that have armed:

- **Eight of nine, not the rig's six.** Awake, a stat loses a point every 21
  (drink), 26 (food) or 31 (energy) ticks (``objects.py:_update_life_stats``),
  so from a full start a threshold of 8 puts the three crossings at ticks 21,
  26 and 31 against 63, 78 and 93 for the rig's 6. In a turn-based block a tick
  is a press, so this is the difference between an errand that arrives while
  the subject is still on the early chain and one that arrives a third of the
  way through the block, and at 8 the stat is one point down rather than
  two ticks from the crisis the rig's comment was worried about.
- **Drink, then eat, then sleep, one at a time.** An interrupt is asked for
  behind one already being asked for, never over the top of it, so the next
  appears when the one before it is finished or put off; and the ones waiting
  are in that fixed order whatever order their stats ran low in
  (:data:`_STAT_ORDER`), which can differ from it in a world a block resumed
  with more than one stat already down. Both are where the entry is inserted
  (:meth:`TaskWrapper._arm`), where the rig needed a queue walk for the first of
  them.
- **One reminder per stat per episode.** Arming is a one-way latch, so a stat
  that recovers and runs low a second time asks nothing and one hovering at the
  threshold cannot ask twice, and an entry closed before its stat ever ran low
  is simply never asked for: a subject who drank in the first twenty presses is
  paid for it like any other entry of the chain and then left alone. Putting one
  off is the subject's own doing and is a skip like any other, back once the
  rest of the chain is done.

Nothing new is logged for them. The crossing is in the inventory columns every
frame already carries, the entries are in the chain the block writes, and the
score is the chain's as it is for everything else: what changes on a
``stat_tasks`` level is that the denominator is 15 rather than 12.

What is not here
----------------

Nothing of the rig's task system now, but two things about it are deliberately
not the rig's: :data:`STAT_TASK_LOW` is 8 and not 6, and an interrupt is a
chain entry from the first frame rather than a row spliced into the chain when
it fires, so the fraction above the frame does not grow a denominator
mid-block.
"""

from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np

from .env import import_crafter
from .levels import require_level
from .menu import menu_of

__all__ = ["SKIP_ENTRY", "STAT_TASKS", "STAT_TASK_LOW", "TASK_CHAIN", "TASK_DONE_TEXT",
           "TASK_GOALS", "TASK_LABELS", "TASK_PREDICATES", "TaskWrapper", "chain_for",
           "has_tasks", "next_task_text", "task_text", "tracker_of", "with_tasks"]

#: The order the tasks are named in, and the whole of what a task level asks
#: for before :func:`chain_for` drops what the level's own rules make empty. It
#: is a feasible path through crafter's tech tree rather than the tree itself:
#: each entry is reachable from the ones before it with what they leave in the
#: inventory, and the ingredients in between (coal for the furnace, stone for
#: the table) are left for the player to discover, unnamed and unpaid. The
#: sapling goes in the ground five tasks before it is eaten because a plant
#: needs 300 in-range steps to ripen (``objects.py``: ``grown > 300``, counted
#: only within twice the view), and the shelter follows the stone pickaxe,
#: where the material is already in hand.
#:
#: The three swords are one task per material, so each earns its own hold, and
#: they are asked for on every task level including L1 even though nothing
#: there can be hurt with them: the damage ladder is 1 / 2 / 3 / 5 and a cow's
#: health is 3 (``objects.py:268``), so on a peaceful level a sword buys at
#: most a shorter cow kill and on L1, with food frozen, not even that. They
#: stay because L1 asks what the world affords, and a recipe is an affordance
#: whether or not its product has a use (Fan, 2026-10-06). ``eat_plant`` has no
#: such defence and is dropped there; see :func:`chain_for`.
TASK_CHAIN = (
    "collect_wood", "place_table",
    "collect_place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe", "build_stone_shelter", "make_iron_pickaxe",
    "collect_diamond",
    "make_wood_sword", "make_stone_sword", "make_iron_sword",
    "eat_plant",
)

#: Entries of :data:`TASK_CHAIN` whose completion is only worth a point while
#: the homeostat runs: each is scored for putting food, drink or energy back,
#: and a level that freezes the four life stats pays it for nothing. Only
#: ``eat_plant`` is one. The other three achievements of that kind are the
#: interrupts (:data:`STAT_TASKS`), which such a level does not carry at all.
#: Keyed off the level's own ``frozen_stats`` rule rather than off its name, so
#: a later level that freezes the stats inherits this for free.
_FROZEN_STAT_ENTRIES = ("eat_plant",)

#: The low-stat interrupts a ``stat_tasks`` level adds to its chain: the stat
#: to watch, and the entry to ask for when it falls. In the order they are
#: asked for, which is this order and not the order the stats happen to cross
#: (Fan, 2026-10-07): drink, then eat, then sleep. Each entry is one of
#: crafter's own achievements, so it closes and scores the way the rest of the
#: chain does, and it is labelled by what to do rather than by the achievement
#: that completes it (:data:`TASK_LABELS`).
STAT_TASKS: tuple[tuple[str, str], ...] = (
    ("drink", "collect_drink"),
    ("food", "eat_cow"),
    ("energy", "wake_up"),
)

#: The stat each interrupt entry watches, which is also the test for whether an
#: entry is an interrupt at all: the pointer reaches these only once they are
#: armed, where it walks the base entries in chain order.
_STAT_OF: dict[str, str] = {entry: stat for stat, entry in STAT_TASKS}

#: Where an interrupt goes in the queue of interrupts waiting to be asked for,
#: which is :data:`STAT_TASKS`' own order and not the order the stats ran low in
#: (Fan, 2026-10-07). It matters only in a world that was resumed with more than
#: one stat already down: from a full start the stats cross in this order by
#: themselves, 21 ticks for drink against 26 and 31.
_STAT_ORDER: dict[str, int] = {entry: i for i, (_, entry) in enumerate(STAT_TASKS)}

#: A stat at or below this, of a full 9, arms its interrupt. Eight and not the
#: rig's six (Fan, 2026-10-07): awake, a stat loses a point every 21 (drink),
#: 26 (food) or 31 (energy) ticks (``objects.py:_update_life_stats``), so from
#: a full start 8 puts the three crossings 21, 26 and 31 presses into a
#: turn-based block against 63, 78 and 93 at 6. A subject meets the homeostat
#: while they are still on the early chain, and meets it one point down rather
#: than near the crisis the rig's threshold was chosen to keep clear of.
STAT_TASK_LOW = 8

#: Entries that need more than one achievement: id -> the achievements that
#: must ALL unlock to close it. An id that is not in here is itself the single
#: achievement it asks for. The ids are what a log and a replay carry, so a
#: composite costs nothing downstream.
TASK_GOALS: dict[str, tuple[str, ...]] = {
    "collect_place_plant": ("collect_sapling", "place_plant"),
}

#: Wording for the entries where ``id.replace("_", " ")`` reads badly. Short on
#: purpose: the strip above the frame holds the clock and the achievement count
#: as well, and a long label is what overlaps them on a scanner display
#: (measured in ``docs/tasks_check.py``). The three consume tasks are labelled
#: by what to do rather than by the achievement that completes them.
TASK_LABELS: dict[str, str] = {
    "collect_place_plant": "collect and place plant",
    "eat_plant": "eat a ripe plant",
    "build_stone_shelter": "build a stone shelter",
    "collect_drink": "drink water",
    "eat_cow": "eat a cow",
    "wake_up": "sleep",
}

#: The pseudo-entry a task level adds to the menu
#: (:meth:`~crafter_gym.menu.MenuWrapper.set_extra`): a task the player cannot
#: finish right now is put off from there, goes to the back of this episode's
#: chain and comes around again once the rest are done. It is the escape hatch
#: the chain needs because some entries can stall on something the world did --
#: a cow walking next to a sapling destroys it (``Plant.health`` is 1), and
#: ``eat_plant`` then waits out another 300 steps of ripening.
SKIP_ENTRY = "skip_task"


# --- build_stone_shelter --------------------------------------------------
#
# What a shelter is for: a hostile cannot hurt you. That is a connectivity
# question about the room the player is standing in, not a test on the four
# cells they happen to touch, since walls one step further out enclose just as
# well -- hence a flood fill, and one condition per hostile.

#: What zombies and skeletons walk on. Neither class overrides
#: ``Object.walkable``, so it is plain ``constants.walkable``. It is NOT the
#: player's own set: ``Player.walkable`` adds lava, so a player can step into
#: lava, and die, where a zombie cannot follow.
_HOSTILE_WALKABLE = ("grass", "path", "sand")

#: A zombie attacks at Manhattan distance <= 1 (``Zombie.update``:
#: ``dist <= 1``, and ``Object.distance`` is the L1 norm) and cannot stand on
#: the player's own cell, since ``is_free`` wants ``obj is None``. So "a zombie
#: can hurt you" is exactly "a zombie can walk to one of your four
#: neighbours", and diagonals are safe.
_SHELTER_DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0))

#: Where a sealed room stops being safe. ``env.py:_balance_chunk`` spawns
#: hostiles inside the room as readily as outside it: zombies on a ``grass``
#: cell at Manhattan distance >= 6 from the player, skeletons on a ``path``
#: cell at >= 7 (its ``span_dist``, gated on ``player.distance(pos) >=
#: span_dist``). Wall off a big enough meadow and it breeds its own zombie.
#: Nothing spawns on sand. This is also what bounds the fill.
_SPAWN_RANGE = {"grass": 6, "path": 7}

#: Cost guard: the fill runs every step while this entry is the pointer, and a
#: sealed region this large is terrain rather than a shelter. Only reachable
#: through sand, which :data:`_SPAWN_RANGE` does not bound.
_SHELTER_MAX_CELLS = 200

#: What an arrow crosses. ``Arrow.walkable`` is ``constants.walkable +
#: ['water', 'lava']``, so an arrow flies over the moat that stops every
#: walker: water and lava seal a room against zombies and not against archers.
#: Rock, ore, tree, table and furnace stop one.
_ARROW_WALKABLE = ("grass", "path", "sand", "water", "lava")

#: A skeleton fires from Manhattan distance <= 5 (``Skeleton.update``) along
#: ONE axis: ``toward(player)`` with ``long_axis=True`` returns an
#: axis-aligned unit step, so an archer outside the player's row and column
#: shoots past them. From distance 1 it cannot fire at all, because ``_shoot``
#: needs the cell between the two to be ``is_free`` and that cell holds the
#: player. So the whole threat is the four rays, cells 2 through 5.
_ARROW_RANGE = 5
_ARROW_MIN_DIST = 2


def _arrow_safe(world: Any, start: tuple[int, int], room: set, static: tuple[type, ...]) -> bool:
    """Whether no archer has a clear shot down any of the player's four rays.

    A ray is safe once something stops the arrow: the map edge, a material
    outside :data:`_ARROW_WALKABLE`, or a fence. Water and lava do not stop it,
    they only deny the archer a place to stand, so a five-wide lake is cover
    while a one-cell moat is not. Cells of the sealed room itself are skipped
    as firing positions: nothing can walk in (the caller's fill) and nothing
    can spawn in (:data:`_SPAWN_RANGE`), so only ground outside the room can
    hold an archer.

    A table and a furnace stop an arrow by being destroyed (``Arrow.update``
    rewrites them to ``path``), so a wall made of them opens a hole after the
    first hit. It does stop the arrow that hits it, which is what this asks.

    :param world: the crafter ``World``, indexed by cell.
    :param start: the player's cell.
    :param room: the cells of the sealed room, as the fill found them.
    :param static: object classes that stop an arrow where they stand.
    :return: whether every ray is stopped before an archer could stand in it.
    """
    for dx, dy in _SHELTER_DIRS:
        for k in range(1, _ARROW_RANGE + 1):
            cell = (start[0] + dx * k, start[1] + dy * k)
            material, obj = world[cell]
            if material is None or material not in _ARROW_WALKABLE:
                break
            if isinstance(obj, static):
                break
            if k >= _ARROW_MIN_DIST and cell not in room and material in _HOSTILE_WALKABLE:
                return False
    return True


def _shelter_around(game: Any) -> bool:
    """Whether the player is standing somewhere no hostile could reach them.

    Melee: flood-fill from the player across the cells a zombie could cross
    and require the fill to stay trapped. A cell stops it when its material is
    not :data:`_HOSTILE_WALKABLE` (rock, ore, table, furnace, tree, water,
    lava) or when a fence stands on it -- a fence is an object rather than a
    material, so the ground under it still reads ``grass`` while ``is_free``
    refuses the step. Placed or natural makes no difference, so walling up
    against a cliff face counts. Two things that look like walls are not:
    creatures move, so a room held shut by a cow in the doorway is not a room,
    and a sapling has ``health`` 1 and is removed by ``Plant.update`` as soon
    as anything stands beside it. Arrows: :func:`_arrow_safe`.

    :param game: the ``crafter.Env`` whose world and player to read.
    :return: whether the player is enclosed, against both hostiles. ``False``
        if the fill escapes off the map (an edge is not a wall), reaches a cell
        a hostile could spawn on inside with them, or runs past
        :data:`_SHELTER_MAX_CELLS`.
    """
    static = (import_crafter().objects.Fence,)
    world = game._world
    start = tuple(int(p) for p in game._player.pos)
    seen = {start}
    stack = [start]
    while stack:
        x, y = stack.pop()
        for dx, dy in _SHELTER_DIRS:
            cell = (x + dx, y + dy)
            if cell in seen:
                continue
            material, obj = world[cell]
            if material is None:
                return False
            if material not in _HOSTILE_WALKABLE or isinstance(obj, static):
                continue
            reach = _SPAWN_RANGE.get(material)
            if reach is not None and abs(cell[0] - start[0]) + abs(cell[1] - start[1]) >= reach:
                return False
            seen.add(cell)
            if len(seen) > _SHELTER_MAX_CELLS:
                return False
            stack.append(cell)
    return _arrow_safe(world, start, seen, static)


#: Entries closed by a test on the live world instead of by an achievement
#: unlock. Tested only while the entry is the pointer, so sealing yourself in
#: before being asked does not pre-complete the task: the test is positional
#: and instantaneous, and what it is about is being enclosed while asked.
#: Each counts toward the player's score like an unlock (see "What the chain
#: scores"), and each must be read-only -- a predicate that touched
#: ``world.random`` would move every later draw and break the replay.
TASK_PREDICATES: dict[str, Callable[[Any], bool]] = {
    "build_stone_shelter": _shelter_around,
}


def task_text(task: str | None) -> str:
    """The cue naming the task in effect, as the player is shown it.

    Here rather than in whatever draws it, because what a subject was told is
    data: two displays of the same block must not word it differently.

    :param task: the task in effect, or ``None`` for a finished chain.
    :return: one line.
    """
    if task is None:
        return "ALL TASKS DONE! Free play."
    return "TASK: " + TASK_LABELS.get(task, task.replace("_", " "))


#: Shown alone over the frozen frame an entry was completed on, before
#: :func:`next_task_text` names what comes next. It promises a point, so it is
#: only honest on a frame that scored one: an entry can also close because the
#: player unlocked its achievement long before the chain asked for it, and
#: crafter pays for a first unlock and not for the pointer moving.
TASK_DONE_TEXT = "Task completed! Score +1"


def next_task_text(task: str | None) -> str:
    """The line naming what the pointer has just moved to.

    :param task: the task now in effect, or ``None`` for a finished chain.
    :return: one line.
    """
    if task is None:
        return "ALL TASKS DONE! Free play."
    return "Next task: " + TASK_LABELS.get(task, task.replace("_", " "))


class TaskWrapper(gym.Wrapper):
    """Crafter with one task named at a time, out of :data:`TASK_CHAIN`.

    Outermost of the crafter wrappers, over the menu if there is one: the
    rules are the game, the buttons are how it is played, and this is what the
    player is playing it for. It has to be outermost because the press that
    asks for a task to be put off arrives here (:data:`SKIP_ENTRY`, which it
    installs in the menu below it when there is one; with no menu there is
    nowhere to put the entry, and a task level played on the seventeen-key env
    is the cue and the score without the escape hatch).

    The action space, the frame and crafter's own ``info`` are untouched.
    ``info`` gains five fields:

    - ``task`` -- the task in effect DURING this step, or ``None`` with the
      chain finished. The record of what the player was asked for while they
      pressed; :attr:`task` is the live pointer, which a completion has
      already advanced.
    - ``task_passed`` -- the predicate entry whose world test passed on this
      step, else ``""``. The one completion crafter does not pay for, so
      whatever pays a ``+1`` has to read it.
    - ``task_done`` -- the task named during this step is now closed, so a
      completion can be announced as one and a ``+1`` paid for it.
    - ``task_moved`` -- the pointer moved, for whatever reason.
    - ``task_skip`` -- the pointer moved because the player asked for the task
      to be put off.

    :param env: a crafter env, menu or not.
    :param chain: the entries to ask for, in order.
    """

    def __init__(self, env: gym.Env, chain: tuple[str, ...] = TASK_CHAIN) -> None:
        super().__init__(env)
        self.base_chain = tuple(chain)
        #: This episode's own order: a skip moves an entry to the back of it, a
        #: stat going low moves its interrupt up to the front of what is left.
        self.chain: list[str] = list(self.base_chain)
        self._unlocked: set[str] = set()
        self._passed: set[str] = set()
        self._armed: set[str] = set()
        menu = menu_of(env)
        if menu is not None:
            menu.set_extra((SKIP_ENTRY,))

    @property
    def action_names(self) -> list[str]:
        """The names of the action ids, as the env below reports them.

        Re-exposed because Gymnasium 1.3 forwards no attribute through a
        wrapper, and a config's ``keys`` are read against these.
        """
        return list(self.env.action_names)

    @property
    def task(self) -> str | None:
        """The first entry of this episode's order that is open and askable.

        Still derived and never stored, and still one walk of :attr:`chain`,
        which is where the order lives: a skip moved an entry to the back of it
        and a crossing moved an interrupt to the front of what was left
        (:meth:`_arm`). The one thing an entry can be besides open or closed is
        unreachable: an interrupt whose stat has not gone low this episode is
        not askable, so the pointer steps over it and a chain of nothing but
        those reads as finished.

        ``None`` once every askable entry is closed, which is the chain finished
        and the rest of the episode free play.
        """
        for entry in self.chain:
            if (entry not in _STAT_OF or entry in self._armed) and self._entry_open(entry):
                return entry
        return None

    @property
    def armed(self) -> frozenset[str]:
        """The interrupts whose stat has gone low this episode, so are askable.

        Membership only: which of them is asked for first is :attr:`chain`'s to
        say, the way it is for every other entry.
        """
        return frozenset(self._armed)

    @property
    def passed(self) -> frozenset[str]:
        """The predicate entries whose test has passed this episode."""
        return frozenset(self._passed)

    @property
    def completed(self) -> tuple[str, ...]:
        """The closed entries of :attr:`base_chain`, in the order it asks them.

        The numerator above the frame, and the only thing a ``+1`` sounds for
        (see "What the chain scores"). Of :attr:`base_chain` and not of
        :attr:`chain`, so a skip reorders what is asked and never what is
        scored.
        """
        return tuple(e for e in self.base_chain if not self._entry_open(e))

    @property
    def scorable(self) -> tuple[str, ...]:
        """The entries crafter has no achievement for, in chain order.

        The log's marker for the entries whose completion is in no achievement
        column (``task_scorable``), so an analysis joining the chain to those
        columns knows which ones it has to read :attr:`passed` for instead. Not
        a correction to the denominator any more: the denominator is the chain.
        """
        return tuple(e for e in self.base_chain if e in TASK_PREDICATES)

    def _entry_open(self, entry: str) -> bool:
        """Whether this entry is still unfinished this episode.

        :param entry: a chain entry.
        :return: ``True`` while it is open. A predicate entry closes when its
            world test has passed; a composite one (:data:`TASK_GOALS`) only
            when all of its achievements have unlocked.
        """
        if entry in TASK_PREDICATES:
            return entry not in self._passed
        return any(g not in self._unlocked for g in TASK_GOALS.get(entry, (entry,)))

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict]:
        """Start an episode with the chain back at its first entry.

        The chain is per episode in both of its parts: a new world has unlocked
        nothing, and the order a skip left behind belongs to the episode that
        was played in it.

        :param kwargs: passed through (``seed``, ``options``).
        :return: ``(obs, info)`` with the task fields added.
        """
        obs, info = self.env.reset(**kwargs)
        self.chain = list(self.base_chain)
        self._unlocked = set()
        self._passed = set()
        # A new world starts the homeostat full (``data.yaml``: food, drink and
        # energy all `initial: 9`), so nothing is armed yet and the first
        # crossing is the first thing `_arm_stats` can see.
        self._armed = set()
        return obs, self._with_tasks(info, self.task)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Apply the press, then see what it did to the chain.

        In cause order: the task in effect can complete, by an unlock or by its
        world test; a stat that has just run low can cut an interrupt in ahead
        of it; and only then can a skip requeue it, and only if neither of those
        moved the pointer. So a press that completes the task, or is answered
        with an errand, is never also read as a request to put the task off:
        it is off the screen either way, and an errand hands it straight back
        to be put off on a later press if the subject still wants to. A death
        outranks all of it: the episode is over, and what the player needs told
        is that rather than which task they were on.

        :param action: an index into the action space, unchanged by this wrapper.
        :return: ``(obs, reward, terminated, truncated, info)``; ``info``
            carries the five task fields.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Read the pointer before the unlocks it is derived from: this is the
        # task the press was made under.
        task = self.task
        self._unlocked = self._unlocked_in(info)
        if terminated:
            return obs, reward, terminated, truncated, self._with_tasks(info, task)
        # After the unlocks and before the pointer is read again: a stat that
        # just went low can take the pointer on this very step, and what the
        # press was made under is `task`, read above.
        self._arm_stats(info)
        passed = self._pass_predicate(task)
        # A completion is this entry closing, which is the one thing the banner
        # claims and the one thing that moves the count. An unlock the chain
        # never named moves neither, which is what keeps drinking water on a
        # level with no thirst from being paid for (Fan, 2026-10-06).
        done = task is not None and not self._entry_open(task)
        moved_to, skipped = self.task, False
        if moved_to == task and info.get("menu_extra") == SKIP_ENTRY and task is not None:
            moved_to = self._requeue(task)
            skipped = moved_to != task
        return obs, reward, terminated, truncated, self._with_tasks(
            info, task, passed=passed, done=done, moved=moved_to != task,
            skipped=skipped)

    def _arm_stats(self, info: dict) -> None:
        """Arm the interrupt of every stat seen at or below :data:`STAT_TASK_LOW`.

        A one-way latch, which is the whole of "one reminder per stat per
        episode": arming happens on the step an interrupt goes from unarmed to
        armed and never again, so a stat that recovers and dips a second time
        asks nothing, and one hovering at the threshold cannot ask twice.
        Reading the stat as it stands rather than watching for the step it
        crossed on is the same thing for a stat that only decays, and it is the
        right thing for a world restored mid-episode, where the crossing
        happened in a block that is already over.

        Only the entries this level carries can arm, so a level with no
        interrupts in its chain never grows one, and a stat crafter does not
        report is read as full rather than as low. Which of several goes first
        is :meth:`_arm`'s to say and not this walk's.

        :param info: an ``info`` dict from the env below.
        """
        inventory = info.get("inventory")
        if inventory is None:
            return
        for stat, entry in STAT_TASKS:
            if (entry in self.chain and entry not in self._armed
                    and inventory.get(stat, 9) <= STAT_TASK_LOW):
                self._arm(entry)

    def _arm(self, entry: str) -> None:
        """Make one interrupt askable, and move it up to where it interrupts.

        Both halves of Fan's second rule (2026-10-07) are this one move. The
        entry being asked for right now is read first and left exactly where it
        is, so nothing is ever replaced on screen before it is finished or put
        off. Everything else waiting goes just ahead of the first base entry
        still open, in :data:`_STAT_ORDER`, so the queue is drink then food then
        sleep however the stats themselves ran low. With the base chain finished
        there is nothing to interrupt and the queue goes on the end.

        An interrupt the subject put off is not in that queue: a skip left it
        behind the open chain, which is where "back once the rest is done" lives,
        and only a position before the first open base entry counts as waiting.
        One whose entry closed before its stat ever ran low moves nothing at all,
        since there is nothing left to ask for.

        :param entry: an entry of :data:`STAT_TASKS`, not yet armed.
        """
        held = self.task
        self._armed.add(entry)
        if not self._entry_open(entry):
            return
        waiting = [e for e in self.chain[:self._first_base_open()]
                   if e in self._armed and e != held and self._entry_open(e)]
        if entry not in waiting:
            waiting.append(entry)
        waiting.sort(key=_STAT_ORDER.__getitem__)
        for queued in waiting:
            self.chain.remove(queued)
        at = self._first_base_open()
        self.chain[at:at] = waiting

    def _first_base_open(self) -> int:
        """Where the chain the subject was working through starts.

        :return: the index in :attr:`chain` of the first entry that is open and
            is not an interrupt, or the length of the chain if there is none.
        """
        return next((i for i, e in enumerate(self.chain)
                     if e not in _STAT_OF and self._entry_open(e)), len(self.chain))

    def _pass_predicate(self, task: str | None) -> str:
        """Test a predicate entry that is the pointer, and close it if it passes.

        :param task: the task in effect.
        :return: ``task`` if its world test has just passed, else ``""``.
        """
        if task not in TASK_PREDICATES or task in self._passed:
            return ""
        if not TASK_PREDICATES[task](self.env.unwrapped.game):
            return ""
        self._passed.add(task)
        return task

    def _requeue(self, task: str) -> str | None:
        """Put a task off: move it to the back of this episode's chain.

        The entries it moves past are the ones the pointer had already stepped
        over, closed or not yet armed, so the chain's order is the only thing
        this changes; whether it changes the pointer is the caller's question,
        and it does not when the task put off is the last one askable.

        :param task: the task in effect.
        :return: the pointer after the move, which is ``task`` itself if
            nothing else is left to do.
        """
        self.chain.remove(task)
        self.chain.append(task)
        return self.task

    @staticmethod
    def _unlocked_in(info: dict) -> set[str]:
        """The achievements unlocked so far, out of crafter's own counts.

        :param info: an ``info`` dict from the env below.
        :return: the names whose count is above zero.
        """
        return {name for name, count in info["achievements"].items() if count > 0}

    def _with_tasks(self, info: dict, task: str | None, passed: str = "", done: bool = False,
                    moved: bool = False, skipped: bool = False) -> dict:
        """Add the task fields to an ``info`` dict.

        :param info: the wrapped env's info.
        :param task: the task in effect during the step.
        :param passed: the predicate entry that passed, or ``""``.
        :param done: the task in effect closed, which implies ``moved``.
        :param moved: the pointer moved.
        :param skipped: the pointer moved because of a skip.
        :return: the same dict, with the five fields set.
        """
        info = dict(info)
        info["task"] = task
        info["task_passed"] = passed
        info["task_done"] = done
        info["task_moved"] = moved
        info["task_skip"] = skipped
        return info


def chain_for(level: str | None) -> tuple[str, ...]:
    """The chain this level asks for: :data:`TASK_CHAIN` less what it cannot pay.

    The level decides, not the session, for the reason
    :mod:`crafter_gym.levels` gives for the rules themselves: a model compared
    against a subject has to be asked for the same things, and the chain
    pickles into the savestate with the rest of the env.

    Both of the rules it reads are about the homeostat, from the two ends of
    it. A level that freezes the stats drops the entries that only mean
    something while they run (:data:`_FROZEN_STAT_ENTRIES`), and a level that
    says ``stat_tasks`` adds the three it can ask for when they run low
    (:data:`STAT_TASKS`), which is why the two branches are exclusive in
    practice as well as in code. The interrupts go on the end, where they are
    out of the order the base chain is walked in and in the order they are
    asked for.

    :param level: a key of :data:`~crafter_gym.levels.LEVELS`, or ``None``.
    :return: the entries to ask for, in :data:`TASK_CHAIN`'s order, the
        interrupts after it. Empty on a level that names no tasks, which is what
        :func:`with_tasks` reads as "do not wrap".
    :raises ValueError: if ``level`` is not one of the levels.
    """
    if not has_tasks(level):
        return ()
    config = require_level(level)
    if config.get("frozen_stats"):
        return tuple(e for e in TASK_CHAIN if e not in _FROZEN_STAT_ENTRIES)
    if config.get("stat_tasks"):
        return TASK_CHAIN + tuple(entry for _, entry in STAT_TASKS)
    return TASK_CHAIN


def has_tasks(level: str | None) -> bool:
    """Whether this level names a task at a time.

    :param level: a key of :data:`~crafter_gym.levels.LEVELS`, or ``None`` for
        stock crafter.
    :return: the level's ``tasks`` flag; ``False`` with no level at all.
    :raises ValueError: if ``level`` is not one of the levels.
    """
    return level is not None and bool(require_level(level).get("tasks"))


def tracker_of(env: gym.Env) -> TaskWrapper | None:
    """The task chain in this env chain, or ``None`` if it has none.

    Reads the chain rather than the level that was asked for, for the reason
    :func:`~crafter_gym.levels.level_of` does: a block can open a world that
    was recorded on another level, and the env is the authority on what is
    being played.

    :param env: any env, wrapped or not.
    :return: the :class:`TaskWrapper`, or ``None``.
    """
    while isinstance(env, gym.Wrapper):
        if isinstance(env, TaskWrapper):
            return env
        env = env.env
    return None


def with_tasks(env: gym.Env, level: str | None) -> gym.Env:
    """Wrap an env in its level's task chain, or hand it back unchanged.

    :param env: a crafter env, menu or not.
    :param level: a key of :data:`~crafter_gym.levels.LEVELS`, or ``None``.
    :return: the env, wrapped in :func:`chain_for` if this level names tasks.
    :raises ValueError: if ``level`` is not one of the levels.
    """
    chain = chain_for(level)
    return TaskWrapper(env, chain) if chain else env
