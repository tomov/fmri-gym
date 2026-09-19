"""Check CrafterAdapter._predict against the engine's own behaviour.

`_predict` decides which cue a frame earns by reimplementing the branches of
crafter's ``Player.update``. That is the one place in the adapter that can
drift away from the engine silently: a crafter release that changed a tool
requirement would leave a subject hearing "no effect" for a press that worked.
Run this from the repo root after any crafter version bump.

    python docs/crafter_cue_check.py

Ground truth without reimplementing anything a second time: deep-copy the env
twice, set the player's action to the pressed one on one copy and to `noop` on
the other, run ONLY `Player.update()` on each (which is exactly what Env.step
does first, before anything else has moved), and diff the two resulting states.

- refused  <=>  the two states are identical, i.e. pressing the button was
  indistinguishable from pressing nothing.
- hit      <=>  the health of whatever stood on the faced tile is lower in the
  pressed copy than in the noop copy.

Homeostasis (hunger/thirst/fatigue/health) runs in both copies, so it cancels.
`noop` itself is excluded: it is the absence of a press, not a refused one, and
against this baseline it is trivially "refused".

Part 1 is a random sweep over several seeds; part 2 drives the branches random
play barely reaches (creatures, ripe plants, arrows, tool requirements, crafting
utilities, sleeping) by building the situation and asking the engine anyway.
Part 2 is not optional. Over 12 x 400 random presses the sweep faced a creature
and pressed `do` eleven times, so the hit branch is almost all part 2's, and an
earlier version of this file skipped those cases without saying so.

Last run 2026-09-19 against chengfanbrain/crafter@deterministic: 4545 presses
checked, 0 mismatches.
"""

import pickle
import sys
import types
from collections import Counter

import numpy as np

# Headless box: no PortAudio. Nothing under test plays a sound, but importing
# the package pulls in fmri_gym.audio, which imports sounddevice at module
# scope. Stub it so the import resolves.
_sd = types.ModuleType("sounddevice")
_sd.default = types.SimpleNamespace(dtype=["float32", "float32"])
_sd.OutputStream = object
sys.modules.setdefault("sounddevice", _sd)

import crafter
from crafter import objects as co

from fmri_gym.adapters.crafter import CrafterAdapter

SIZE = {"env_kwargs": {"size": (64, 64), "length": 0}}


def signature(env):
    p, w = env._player, env._world
    return (
        tuple(p.pos), tuple(p.facing), p.sleeping,
        tuple(sorted(p.inventory.items())),
        tuple(sorted(p.achievements.items())),
        p._hunger, p._thirst, p._fatigue, p._recover,
        w._mat_map.tobytes(), w._obj_map.tobytes(),
        tuple(sorted((type(o).__name__, tuple(o.pos), o.inventory.get("health"))
                     for o in w.objects)),
    )


def apply_only_player(env, action_name):
    copy = pickle.loads(pickle.dumps(env, protocol=5))
    copy._player.action = action_name
    copy._player.update()
    return copy


def health_at(env, pos):
    _, obj = env._world[pos]
    return None if obj is None else obj.health


def truth(env, action):
    """Engine ground truth for ``action``, plus the tag naming its branch."""
    player = env._player
    faced = (player.pos[0] + player.facing[0], player.pos[1] + player.facing[1])
    material, obj = env._world[faced]
    name = env.action_names[action]
    pressed, idle = apply_only_player(env, name), apply_only_player(env, "noop")
    refused = signature(pressed) == signature(idle)
    hp_p, hp_i = health_at(pressed, faced), health_at(idle, faced)
    hit = hp_p is not None and hp_i is not None and hp_p < hp_i
    what = type(obj).__name__.lower() if obj else material
    return hit, refused, f"{name.split('_')[0]}/{what}"


def check(env, action, cov, bad, where):
    """Compare prediction to truth for one press; record coverage."""
    name = env.action_names[action]
    if name == "noop":
        return
    player = env._player
    faced = (player.pos[0] + player.facing[0], player.pos[1] + player.facing[1])
    material, obj = env._world[faced]
    # `do` on bare grass is a 0.1-probability draw with leaves: grass, so a
    # failed roll leaves state identical to noop. Deliberately silent, and so
    # not comparable against a state-diff ground truth. Only BARE grass: an
    # object standing on it is handled by `_do_object`, which runs first and is
    # entirely deterministic, so those presses must still be checked.
    if name == "do" and material == "grass" and obj is None:
        cov["do/grass (skipped: probabilistic)"] += 1
        return
    adapter = CrafterAdapter.__new__(CrafterAdapter)
    adapter.env = env
    pred = adapter._predict(action)
    hit, refused, tag = truth(env, action)
    cov[f"{tag} -> {'hit' if hit else 'refused' if refused else 'effect'}"] += 1
    if (pred["hit"], pred["refused"]) != (hit, refused):
        bad.append((where, name, material, pred, {"hit": hit, "refused": refused}))


def random_sweep(seeds, steps, cov, bad):
    for seed in seeds:
        ad = CrafterAdapter({"seed": seed, **SIZE})
        ad.reset(seed)
        rng = np.random.RandomState(seed)
        for t in range(steps):
            action = int(rng.randint(0, 17))
            check(ad.env, action, cov, bad, f"seed{seed}/step{t}")
            _, _, term, trunc, _ = ad.step(action)
            if term or trunc:
                ad.reset(seed * 1000 + t)


def scene(material="grass", obj=None, inventory=None, facing=(0, 1),
          sleeping=False, nearby=()):
    """Build an env with a known tile, object and inventory in front."""
    env = crafter.Env(seed=7, **SIZE["env_kwargs"])
    env.reset()
    p, w = env._player, env._world
    p.facing = facing
    faced = (p.pos[0] + facing[0], p.pos[1] + facing[1])
    # Clear the player's neighbourhood so only what a scenario asks for is
    # within reach of `make`'s 3x3 read.
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            pos = (p.pos[0] + dx, p.pos[1] + dy)
            if w[pos][1] is not None and w[pos][1] is not p:
                w.remove(w[pos][1])
            w[pos] = "grass"
    w[faced] = material
    for i, util in enumerate(nearby):
        w[(p.pos[0] - 1, p.pos[1] - 1 + i)] = util
    if obj is not None:
        pos = np.array(faced)
        if obj in (co.Zombie, co.Skeleton):
            w.add(obj(w, pos, p))
        elif obj is co.Arrow:
            w.add(obj(w, pos, (0, 1)))
        else:
            w.add(obj(w, pos))
    for k, v in (inventory or {}).items():
        p.inventory[k] = v
    p.sleeping = sleeping
    return env


def directed(cov, bad):
    A = {n: i for i, n in enumerate(crafter.constants.actions)}
    cases = [
        ("cow bare-handed", scene("grass", co.Cow), "do"),
        ("cow, iron sword (kills)", scene("grass", co.Cow, {"iron_sword": 1}), "do"),
        ("zombie", scene("grass", co.Zombie), "do"),
        ("skeleton", scene("grass", co.Skeleton), "do"),
        # No `co.Fence` case: crafter never constructs one (no place_fence
        # action, no `fence` inventory slot), and `_do_object`'s fence branch
        # raises KeyError if you build one by hand. Unreachable in a real game.
        ("arrow in flight", scene("grass", co.Arrow), "do"),
        ("unripe plant", scene("grass", co.Plant), "do"),
        ("stone, no pickaxe", scene("stone"), "do"),
        ("stone, wood pickaxe", scene("stone", None, {"wood_pickaxe": 1}), "do"),
        ("coal, no pickaxe", scene("coal"), "do"),
        ("iron, wood pickaxe", scene("iron", None, {"wood_pickaxe": 1}), "do"),
        ("iron, stone pickaxe", scene("iron", None, {"stone_pickaxe": 1}), "do"),
        ("diamond, stone pickaxe", scene("diamond", None, {"stone_pickaxe": 1}), "do"),
        ("diamond, iron pickaxe", scene("diamond", None, {"iron_pickaxe": 1}), "do"),
        ("tree", scene("tree"), "do"),
        ("water", scene("water"), "do"),
        ("sand", scene("sand"), "do"),
        ("path", scene("path"), "do"),
        ("lava", scene("lava"), "do"),
        ("table", scene("table"), "do"),
        ("sleep, energy full", scene(), "sleep"),
        ("sleep, energy low", scene(inventory={"energy": 4}), "sleep"),
        ("move while asleep", scene(inventory={"energy": 4}, sleeping=True), "move_left"),
        ("do while asleep", scene("tree", inventory={"energy": 4}, sleeping=True), "do"),
        ("walk into tree, facing it", scene("tree"), "move_down"),
        ("walk into tree, turning", scene("tree"), "move_left"),
        ("walk into water, facing it", scene("water"), "move_down"),
        ("walk into lava, facing it", scene("lava"), "move_down"),
        ("walk onto grass", scene("grass"), "move_down"),
        ("walk into a cow", scene("grass", co.Cow), "move_down"),
        ("place stone, none held", scene("grass"), "place_stone"),
        ("place stone on grass", scene("grass", None, {"stone": 1}), "place_stone"),
        ("place stone on water", scene("water", None, {"stone": 1}), "place_stone"),
        ("place stone on tree", scene("tree", None, {"stone": 1}), "place_stone"),
        ("place stone on a cow", scene("grass", co.Cow, {"stone": 1}), "place_stone"),
        ("place table, 1 wood", scene("grass", None, {"wood": 1}), "place_table"),
        ("place table, 2 wood", scene("grass", None, {"wood": 2}), "place_table"),
        ("place furnace, 4 stone", scene("grass", None, {"stone": 4}), "place_furnace"),
        ("place plant, no sapling", scene("grass"), "place_plant"),
        ("place plant, sapling", scene("grass", None, {"sapling": 1}), "place_plant"),
        ("place plant on sand", scene("sand", None, {"sapling": 1}), "place_plant"),
        ("make wood pickaxe, no table", scene(inventory={"wood": 1}), "make_wood_pickaxe"),
        ("make wood pickaxe, table", scene(inventory={"wood": 1}, nearby=["table"]),
         "make_wood_pickaxe"),
        ("make wood pickaxe, table but no wood", scene(nearby=["table"]),
         "make_wood_pickaxe"),
        ("make stone sword, table, no stone",
         scene(inventory={"wood": 1}, nearby=["table"]), "make_stone_sword"),
        ("make stone sword, table, stocked",
         scene(inventory={"wood": 1, "stone": 1}, nearby=["table"]), "make_stone_sword"),
        ("make iron pickaxe, table only",
         scene(inventory={"wood": 1, "coal": 1, "iron": 1}, nearby=["table"]),
         "make_iron_pickaxe"),
        ("make iron pickaxe, table + furnace",
         scene(inventory={"wood": 1, "coal": 1, "iron": 1},
               nearby=["table", "furnace"]), "make_iron_pickaxe"),
    ]
    for label, env, action_name in cases:
        check(env, A[action_name], cov, bad, label)
    # Ripe plant needs one extra nudge: `ripe` is grown > 300.
    env = scene("grass", co.Plant)
    env._world[(env._player.pos[0], env._player.pos[1] + 1)][1].grown = 400
    check(env, A["do"], cov, bad, "ripe plant")


if __name__ == "__main__":
    cov, bad = Counter(), []
    random_sweep(seeds=range(1, 13), steps=400, cov=cov, bad=bad)
    n_random = sum(cov.values())
    print(f"random sweep done: {n_random} presses, {len(bad)} mismatches",
          flush=True)
    directed(cov, bad)
    print(f"random sweep: {n_random} presses over 12 seeds x 400 steps")
    print(f"total checked: {sum(cov.values())}   mismatches: {len(bad)}")
    print("\nbranch coverage:")
    for tag, n in sorted(cov.items()):
        print(f"  {n:6d}  {tag}")
    for row in bad[:40]:
        print("MISMATCH", row)
