"""Crafter's four levels: the rule variants the scanner paradigm plays.

The paradigm does not play one game, it plays four, and they differ by rules
rather than by worlds: L1 takes the homeostat away and hides it, L2 gives it
back, L3 adds the things that fight back, L4 is stock crafter. The definition
lives in the rig that ran them first (crafter-for-brain-scan v0.33,
``core.py:LEVELS``), and :data:`LEVELS` here is that table, field for field.

Why the rules are an env wrapper
--------------------------------

A level is part of the game, so it has to be part of the environment: a level
imposed by an adapter, or by a frontend, would be a game only humans play, and
the models this paradigm compares against subjects would be answering a
different question. That is the same reason the eight-button interface is a
wrapper (:mod:`crafter_gym.menu`) rather than a key map in the run loop.

It also has to survive a savestate. The crafter adapter's ``capture`` pickles
the whole env chain and ``restore`` unpickles it, which is how a block resumes
a world the previous block left off in (see "Resuming a world" in the repo
README), and how a model is rolled out from a subject's own frame. A wrapper
pickles with the env and comes back with its rules intact. Patched-on methods
only come back if they are picklable, which is what the rule objects below are
for: the rig installs closures over the live player, and ``pickle`` cannot
serialise a local function at all, so it has to strip every patch before its
own deep copy and re-impose them afterwards. These rules are instances of
module-level classes instead, so a pickle round-trip and a ``deepcopy`` both
reproduce the chain -- and, because each rule reaches its predecessor through
``type(obj)`` and never through the patched instance's own attribute, the cycle
between a player and the rules installed on it unpickles without resolving
back through the dict being restored.

What a level can score
----------------------

Rules decide what there is to get. Two of crafter's 22 achievements are a
creature L1 and L2 keep out of the world, so 20 is the ceiling on those two
levels: :func:`reachable_achievements` is that set, and the count the subject
is shown while playing is divided by it rather than by 22. Nothing else is lost
to the rest of the table, for the reasons recorded with
:data:`_HOSTILE_ACHIEVEMENTS`. Which level an env is actually in is a question
for the chain and not for the config that built it, since a block can open
another level's world: :func:`level_of` answers it.

What is not here
----------------

``tasks`` and ``stat_tasks`` are in the table because they are part of the
table, but nothing in fmri-gym reads them yet: the task chain is cue text, a
HUD row and a menu entry to skip a task, so it spans the adapter and the menu's
action space rather than the game's rules. A block run from the L1-L3 configs
is the level's game without its task cues.
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar

import gymnasium as gym
import numpy as np

from .env import import_crafter

__all__ = ["LEVELS", "LevelWrapper", "level_of", "reachable_achievements",
           "with_level"]

# The rig's own table (core.py:LEVELS), which the agent-side training env is
# built from as well. In every level an episode ends on death only (the configs
# pass `length: 0`), health cannot regenerate from 0, and the death frame is
# tinted red.
#   L1_affordance  - no hostiles, no homeostatic death, the four life stats
#                    frozen at 9 and hidden from the HUD: what is left is what
#                    the world affords, acted on without a clock running out.
#   L2_homeostasis - the homeostat back on and visible, still nothing hostile.
#   L3_predation   - zombies and skeletons too.
#   L4_survival    - stock crafter.
LEVELS: dict[str, dict[str, Any]] = {
    "L1_affordance": dict(hostiles=False, homeostatic_death=False,
                          hidden_items=("health", "food", "drink", "energy"),
                          tasks=True, frozen_stats=True),
    "L2_homeostasis": dict(hostiles=False, homeostatic_death=True,
                           hidden_items=(), tasks=True, stat_tasks=True),
    "L3_predation": dict(hostiles=True, homeostatic_death=True,
                         hidden_items=(), tasks=True, stat_tasks=True),
    "L4_survival": dict(hostiles=True, homeostatic_death=True,
                        hidden_items=()),
}

#: The two of crafter's 22 achievements a level with nothing hostile in it can
#: never unlock: both are a creature :class:`_NoHostiles` keeps out of the
#: world. The other twenty survive the rest of the table, a frozen homeostat
#: included, because an unlock hangs off the action and not off the stat it
#: feeds: ``eat_cow`` is awarded on the kill however full the player is,
#: ``collect_drink`` on drinking water with the stat already at 9 (the engine
#: checks a collect's ``require``, never its ``receive``), and
#: :class:`_InstantSleep` is in the table precisely so ``wake_up`` stays
#: reachable with energy frozen. Measured in ``docs/levels_check.py``.
_HOSTILE_ACHIEVEMENTS = ("defeat_skeleton", "defeat_zombie")


def _require_level(level: str) -> dict[str, Any]:
    """This level's row of :data:`LEVELS`.

    :param level: a key of :data:`LEVELS`.
    :return: the row itself, not a copy.
    :raises ValueError: if ``level`` is not one of them.
    """
    try:
        return LEVELS[level]
    except KeyError:
        raise ValueError(f"unknown crafter level {level!r}; the paradigm's "
                         f"levels are {', '.join(LEVELS)}") from None


def reachable_achievements(level: str | None) -> tuple[str, ...]:
    """The achievements this level can unlock, in crafter's own id order.

    Crafter scores a run by how many of its achievements were unlocked, and
    the count is only a score against the number there are to get. On a level
    with no hostiles that number is not 22 (see :data:`_HOSTILE_ACHIEVEMENTS`),
    and a subject shown a denominator they cannot reach is being told the run
    failed at something it never contained.

    :param level: a key of :data:`LEVELS`, or ``None`` for stock crafter.
    :return: the reachable names, a subsequence of
        ``crafter.constants.achievements``.
    :raises ValueError: if ``level`` is not one of :data:`LEVELS`.
    """
    names = tuple(import_crafter().constants.achievements)
    if level is None or _require_level(level)["hostiles"]:
        return names
    return tuple(n for n in names if n not in _HOSTILE_ACHIEVEMENTS)


class _PlayerRule:
    """One rule installed over a player method, as a picklable callable.

    ``slot`` names the method on ``crafter.objects.Player`` this rule replaces
    on one player instance. ``inner`` is the rule underneath it, or ``None``
    for the engine's own method, which is reached through the class and not
    through the instance: the instance attribute is this object, so an instance
    lookup would be a loop -- and during unpickling, with the player's dict
    already restored, a bound method pickled by name would resolve to exactly
    that loop.

    :param player: the player this rule is installed on.
    :param inner: the rule it wraps, or ``None`` for the engine's method.
    """

    slot: ClassVar[str] = ""

    def __init__(self, player: Any, inner: Callable[[], None] | None = None) -> None:
        self.player = player
        self.inner = inner

    def _call_inner(self) -> None:
        """Run whatever this rule wraps."""
        if self.inner is None:
            getattr(type(self.player), self.slot)(self.player)
        else:
            self.inner()


class _NoRevive(_PlayerRule):
    """Health must not regenerate from 0, in any level.

    Stock ``_degen_or_regen_health`` can add +1 on the very tick lava or a
    killing blow set health to 0, and the env then counts the player as alive
    again: dead has to stay dead, or a death is not a reliable episode end and
    the adapter's ``terminated`` is not either.
    """

    slot = "_degen_or_regen_health"

    def __call__(self) -> None:
        dead = self.player.health <= 0
        self._call_inner()
        if dead:
            self.player.health = 0


class _NoHomeostaticDeath(_PlayerRule):
    """L1: starvation, dehydration and exhaustion cannot cost health.

    Only the health decrement is blocked. Hunger, thirst and fatigue still
    tick (where :class:`_FrozenLifeStats` has not stopped them), sleep still
    restores energy, regeneration still works, and lava stays lethal because it
    zeroes health in the action phase, before this runs.
    """

    slot = "_degen_or_regen_health"

    def __call__(self) -> None:
        before = self.player.health
        self._call_inner()
        if self.player.health < before:
            self.player.health = before


class _FrozenLifeStats(_PlayerRule):
    """L1: the four life stats are constants at 9.

    ``_update_life_stats`` is the only place hunger, thirst and fatigue move
    food, drink and energy (``crafter/objects.py:133``), so a no-op there pins
    all three at their spawn value, and health -- already shielded by
    :class:`_NoHomeostaticDeath` -- has nothing left to move it but lava. The
    stats are hidden in L1 anyway; freezing them makes the hidden layer inert
    instead of merely invisible.
    """

    slot = "_update_life_stats"

    def __call__(self) -> None:
        pass


class _InstantSleep(_PlayerRule):
    """L1: sleep must still do something with energy frozen at full.

    Stock ``update`` refuses to fall asleep at maximum energy (the sleep branch
    of ``crafter/objects.py:99``), which would leave the sleep button a silent
    no-op in the one level whose energy never drops. This honours the press
    after the stock update ran, so the player sleeps this tick and the stock
    wake branch -- also energy-at-max -- wakes them on the next one: a one-tick
    nap that unlocks ``wake_up``.
    """

    slot = "update"

    def __call__(self) -> None:
        player = self.player
        want = not player.sleeping and player.action == "sleep"
        self._call_inner()
        if want and not player.sleeping:
            player.sleeping = True


class _HiddenItems:
    """Blank the named cells of the item panel without moving the others.

    crafter's ``ItemView`` skips any item whose amount is below 1, and a cell's
    position is the item's index in the full inventory dict, so zeroing a copy
    hides those icons while every other icon stays where the stock game draws
    it.

    :param view: the engine's own ``ItemView``.
    :param hidden: inventory names to blank.
    """

    def __init__(self, view: Any, hidden: tuple[str, ...]) -> None:
        self.view = view
        self.hidden = tuple(hidden)

    def __call__(self, inventory: dict, unit: Any) -> np.ndarray:
        inventory = dict(inventory)
        for name in self.hidden:
            inventory[name] = 0
        return self.view(inventory, unit)


class _DeathTint:
    """Red-tint the world view while the player is dead, in every level.

    This is the engine's own commented-out tint, in the lines directly before
    ``LocalView.__call__`` returns (``crafter/engine.py:165``), so wrapping the
    return is the identical arithmetic: the uint8 truncation happens where it
    would have, on assignment into the frame. It is the minimal death signal L1
    needs, since with the health icon hidden a lava death is otherwise
    invisible, and it is applied in every level so that the signal always means
    the same thing.

    :param view: the engine's own ``LocalView``.
    """

    def __init__(self, view: Any) -> None:
        self.view = view

    def __call__(self, player: Any, unit: Any) -> np.ndarray:
        canvas = self.view(player, unit)
        if player.health < 1:
            canvas = (0.4 * np.asarray(canvas, np.float64)
                      + 0.6 * np.array((128.0, 0.0, 0.0)))
        return canvas


class _NoHostiles:
    """Stop the periodic respawn from placing anything hostile.

    The engine rebalances creatures per chunk every tenth step, so clearing a
    world once is not enough for the length of an episode.

    :param game: the ``crafter.Env`` whose ``_balance_object`` this replaces.
    """

    def __init__(self, game: Any) -> None:
        self.game = game

    def __call__(self, chunk: Any, objs: Any, cls: type, *args: Any, **kwargs: Any) -> None:
        if issubclass(cls, _hostile_classes()):
            return None
        return type(self.game)._balance_object(self.game, chunk, objs, cls,
                                               *args, **kwargs)


def _hostile_classes() -> tuple[type, ...]:
    """The object classes a peaceful level has none of.

    Arrows are in the list although only a skeleton makes one: the engine's
    rebalance is called for one class at a time, and a level that is checked
    against the whole set cannot be undone by a new caller.

    :return: ``(Zombie, Skeleton, Arrow)``.
    """
    objects = import_crafter().objects
    return (objects.Zombie, objects.Skeleton, objects.Arrow)


def _clear_hostiles(game: Any) -> None:
    """Remove what worldgen just placed.

    Worldgen runs before any rule can, and it draws its creatures from the
    world's RNG either way, so a peaceful level is a world generated like every
    other one with the hostiles deleted. That is deliberate as well as
    unavoidable: the same seed then gives the same terrain in all four levels,
    and a world is comparable across them.

    :param game: the ``crafter.Env``.
    """
    hostile = _hostile_classes()
    world = game._world
    for obj in world.objects:
        if isinstance(obj, hostile):
            world.remove(obj)


def _unwrap(current: Any, rule: type) -> Any:
    """The engine's own callable, past a rule of this kind if one is installed.

    Lets :meth:`LevelWrapper._impose` run on an env it has already run on --
    ``reset()`` with no seed reuses the game, and so does a restored env -- and
    leave exactly the level's own rules behind rather than a second layer.

    :param current: what is installed now.
    :param rule: the rule class to look past.
    :return: the callable underneath, or ``current`` if it is not that rule.
    """
    return current.view if isinstance(current, rule) else current


class LevelWrapper(gym.Wrapper):
    """One of :data:`LEVELS`, imposed on a :class:`~crafter_gym.env.CrafterEnv`.

    Sits innermost, under the menu if there is one: the rules are the game, the
    buttons are how it is played. The wrapper changes no space and no API --
    same 17 actions, same frame, same ``info`` -- so everything that can play
    stock crafter can play all four levels.

    :param env: the env to impose the level on, a ``CrafterEnv`` or something
        wrapping one.
    :param level: a key of :data:`LEVELS`.
    :raises ValueError: if ``level`` is not one of them.
    """

    def __init__(self, env: gym.Env, level: str) -> None:
        _require_level(level)
        super().__init__(env)
        self.level = level

    @property
    def rules(self) -> dict[str, Any]:
        """This level's row of :data:`LEVELS`, copied."""
        return dict(LEVELS[self.level])

    @property
    def achievements(self) -> tuple[str, ...]:
        """What this level can unlock (:func:`reachable_achievements`)."""
        return reachable_achievements(self.level)

    @property
    def action_names(self) -> list[str]:
        """crafter's own names for the action ids, in id order.

        Re-exposed because Gymnasium 1.3 forwards no attribute through a
        wrapper, and the menu above this one reads it.
        """
        return list(self.env.action_names)

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict]:
        """Generate a new world and impose the level on it.

        The rules have to go on again after every reset: crafter builds a new
        player per episode, and a new world comes with fresh hostiles. The
        frame is then drawn again, because the one the reset returned was drawn
        before any of this (``crafter/env.py:70`` generates the world and
        renders in the same call), and it would show the hostiles that have
        just been removed and the stat icons L1 hides.

        :param kwargs: ``seed`` and ``options``, as Gymnasium's.
        :return: ``(frame, info)``.
        """
        _, info = self.env.reset(**kwargs)
        self._impose()
        return self.env.unwrapped.redraw(), info

    def _impose(self) -> None:
        """Install exactly this level's rules, on an env in any prior state."""
        cfg = LEVELS[self.level]
        game = self.env.unwrapped.game
        player = game._player

        # Observation side: on the game, which outlives an episode.
        item_view = _unwrap(game._item_view, _HiddenItems)
        game._item_view = (_HiddenItems(item_view, cfg["hidden_items"])
                           if cfg["hidden_items"] else item_view)
        game._local_view = _DeathTint(_unwrap(game._local_view, _DeathTint))

        # Hostiles.
        if cfg["hostiles"]:
            game.__dict__.pop("_balance_object", None)
        else:
            _clear_hostiles(game)
            game._balance_object = _NoHostiles(game)

        # Player side: on the player, which does not.
        health: _PlayerRule = _NoRevive(player)
        if not cfg["homeostatic_death"]:
            health = _NoHomeostaticDeath(player, health)
        player._degen_or_regen_health = health
        if cfg.get("frozen_stats"):
            player._update_life_stats = _FrozenLifeStats(player)
            player.update = _InstantSleep(player)
        else:
            player.__dict__.pop("_update_life_stats", None)
            player.__dict__.pop("update", None)


def level_of(env: gym.Env) -> str | None:
    """The level imposed on this env chain, or ``None`` for stock crafter.

    Reads the chain rather than the ``level`` that was asked for, because the
    two part ways: a block that opens another level's ``resume`` slot plays the
    world in the slot, rules and all, and says so (``fmri_gym.resume``). The
    env is the authority on which game is being played.

    :param env: any env, wrapped or not.
    :return: the :class:`LevelWrapper`'s level, or ``None`` if there is none.
    """
    while isinstance(env, gym.Wrapper):
        if isinstance(env, LevelWrapper):
            return env.level
        env = env.env
    return None


def with_level(env: gym.Env, level: str | None) -> gym.Env:
    """Wrap an env in a level, or hand it back unchanged.

    :param env: a ``CrafterEnv`` or something wrapping one.
    :param level: a key of :data:`LEVELS`, or ``None`` for stock crafter with
        no level rules at all (which is not the same as ``L4_survival``: that
        level still forbids reviving at 0 health and still tints a death).
    :return: the env, wrapped if there is a level to wrap it in.
    """
    return env if level is None else LevelWrapper(env, level)
