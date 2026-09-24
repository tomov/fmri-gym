"""Crafter adapter (danijar/crafter).

Crafter uses the OLD gym API shape -- reset() returns obs only, step() returns a
4-tuple (obs, reward, done, info) with no `truncated`, and it doesn't register
cleanly under gymnasium. We wrap crafter.Env directly and normalize it to the
gymnasium contract the session loop expects.

The observation IS the RGB frame (default 64x64x3; bump via env_kwargs.size), so
render() just returns obs. Choose a `size` that is an integer fraction of the
display height (384 -> x2 in a 1024x768 window): display.draw_frame scales
unfiltered, so a fractional factor smears crafter's pixel art, and the HUD --
four status icons plus 16 inventory counters, drawn by the env itself into the
bottom two of the nine tile rows -- is the part that suffers first.

Two traps, both about reproducibility:

- `crafter.Env.reset()` takes no seed. The world seed is
  hash((env seed, episode counter)) and the counter increments on every reset,
  so a second reset() on one object silently builds a DIFFERENT world and the
  session's `episode_seeds` would be a fiction. We rebuild crafter.Env(seed=...)
  per episode instead, which costs 91 ms on top of a ~2 s worldgen reset
  (measured at size 384) and makes the seed alone determine the episode.
- `length` defaults to a 10000-step cap. A scanner block is bounded by its own
  duration, so configs pass `length: 0` (no cap) and every `done` is a real
  death. If a config does set a cap, the step-limit case is reported as
  `truncated` and only death as `terminated`.

Seed + actions replay crafter exactly, but only on a fork, because the bug this
config works around is an engine one rather than a rig one. Every tenth step
`Env.step` rebalances creatures per chunk, and `World.chunks` holds each chunk's
objects in a Python *set*; set order follows object id(), so the despawn pick
`creatures[random.randint(...)]` lands on a different animal between two runs of
the same seed and actions. Terrain is bit-identical either way, since worldgen is
seeded properly; creatures, and with them the rendered frame, are not. Sorting
that list by position is the whole fix, and this config expects it:

    pip install git+https://github.com/chengfanbrain/crafter.git@deterministic

Measured 2026-09-15. Stock 1.8.3 leaves its own trajectory at step 69 of a
300-random-action replay; on the fork, every one of the 6 episodes in a real
300 s block replayed from `episode_seeds` + `actions` into both the logged
symbolic state and the logged pixels, bit for bit. `log_frames` stays on anyway:
the stored frames are the record that does not depend on whoever opens the block
later having the right build installed. Cost at 2.5 fps (756 frames, size 384):
7.0 KB for a median daylit frame, but crafter mixes per-pixel noise into the view
at night, so that block's 34 night frames ran to 189 KB and the frames totalled
11.2 MB inside a 10.2 MB npz.

Rendering is not side-effect free at night, which is a second trap. That noise
is drawn from `world.random`, the same stream the creatures use, so any render
outside the step loop shifts every later draw. `render` therefore hands back the
frame `step` already produced rather than calling the engine again, and
`restore` puts the RNG back after the one render it cannot avoid. Measured:
restoring a night anchor and stepping to the end of the episode reproduces the
log exactly, and does not if one extra render is inserted first.

Crafter has no savestate API, but the whole Env pickles to ~195 KB in ~1.2 ms
(and loads back in ~7.7 ms), so `capture` returns that blob and `restore` loads
it. Storing one per frame would cost ~147 MB a block, hence `state_stride` in
the config (25 = one anchor every 10 s at 2.5 fps, counted per episode). On the
fork an anchor is a real branch point: all 33 in the measured block restored and
then played their episode out with every frame and every semantic grid identical
to the seed replay, which is what a model rollout from a subject's own state
needs. On stock crafter most of them do not, because unpickling gives the objects
new id()s and the next rebalance picks a different animal. A restored copy also
re-renders the same frame bit-for-bit by day; at night only the state is
preserved, since the noise is a fresh draw (4 of those 33 anchors).

`info` carries the whole symbolic state -- 16 inventory counters (health, food,
drink, energy, then materials and tools), 22 achievement counters, the player's
world position, and a 64x64 grid of material/object ids -- so `capture` logs all
four and `block_extra` ships the tables that decode them.

The env's HUD covers inventory and the four status bars but not the achievement
count, so `show_score` turns on an `overlay` that puts it in the letterbox bar
beside the frame. It reports only what `info` already carries, which is what
keeps humans and models on the same game: `agent_play.py` hands a policy the
same lines as text.

`menu` swaps the 16-key map for the eight buttons a scanner button box has, the
scheme crafter-for-brain-scan (v0.33, `core.py:ButtonMapper`) plays in: six
buttons drive move/do/sleep directly, and the other ten actions sit in a list
that `cycle` advances and `confirm` fires. That is a control scheme rather than
a game change -- the space is still Discrete(17) and every action stays
reachable -- so it belongs here and not in the env. The two meta-buttons take
ids 17 and 18, just past the space, and never reach the engine: `cycle` steps it
with `noop`, since moving the cursor still costs the subject a turn, and
`confirm` steps it with whatever the cursor is on. So `actions` in the npz
records what was pressed, and `capture` logs `env_action`, `menu_idx` and
`menu_sel` beside it, which is what a replay needs to rebuild both the world and
the screen.

The selection is drawn over the player rather than in the letterbox bar, because
that is where the eyes already are between presses, and only while the last
press was a `cycle` or a `confirm`. The rig instead fades it out on a 3 s wall
clock (`frontend_pygame.py --menu-s`), which a turn-based block cannot copy: it
repaints only when a key is pressed, so nothing would clear the line until the
next press anyway. "While you are using it" is the closest equivalent, and it
has the better property of being reconstructible from the logged actions alone.

Crafter ships no sound at all (56 assets, every one a PNG), which leaves a
subject in the bore unable to tell three situations apart, all of which look
like a frame where nothing much moved: the press landed on a creature but did
not kill it (a zombie takes five bare-handed blows), the press was refused by
the engine (no pickaxe for that rock, no table in reach), and the press was
dropped by the rig. `cues` fills that in, with the three waveforms in
`fmri_gym.cues` and a rule per frame: score > hit > blocked, at most one,
because audio output queues rather than mixes.

Which cue to play is decided BEFORE the step, by reading the state the engine is
about to act on, not by diffing state afterwards. Afterwards is not enough: a
non-lethal blow changes nothing that reaches `info`, and a refusal is
indistinguishable from a refusal-plus-a-zombie-walking-past. The prediction is
exact rather than heuristic because the player is `world.objects[0]` -- added in
`reset` before worldgen -- and `Env.step` updates objects in that order, so
nothing can move between the read and `Player.update`. It reimplements the
branches of `Player.update` over `constants.collect / place / make`, reading
crafter's own tables, and touches neither the world nor `world.random`.

Reimplementing engine branches is the part of this file that can go stale
without anyone noticing, so it is checked rather than asserted:
`docs/crafter_cue_check.py` runs each press twice against a deep copy of the
live env -- once as pressed, once as `noop` -- and calls the prediction wrong if
the two resulting states disagree with it. 4545 presses, 0 mismatches on
2026-09-19. Re-run it after a crafter version bump.

Collecting grass is deliberately silent: `data.yaml` gives it `probability: 0.1`
and `leaves: grass`, so a press that yields nothing is the roll failing rather
than the engine refusing, and the honest cue for luck is no cue.

The four resulting columns (`hit`, `no_effect`, `cue`, `target`) are logged
whether or not `cues` is on, since they describe the frame rather than the
feedback. What is gated is delivery, and each player gets one channel: `cues`
plays the sound, and `cue_overlay` writes the same thing as a line in the
letterbox bar. Scanner configs set the first and not the second, because a
subject who has already heard the cue would only be reading a repeat of it, and
every glance at the margin is a glance away from the frame. `agent_play.py`
forces the second on, since a policy cannot hear one: the information a model
reads is still exactly the information the subject got.
"""

from __future__ import annotations

import pickle
import zlib
from typing import Any

import numpy as np

from .. import cues as cue_bank
from .base import EnvAdapter, FrameState, Sound
from .keyspec import SingleKeySpec

# Crafter's Discrete(17): 0=noop, 1-4 move left/right/up/down, 5=do, 6=sleep,
# 7-10 place stone/table/furnace/plant, 11-13 make wood/stone/iron pickaxe,
# 14-16 make wood/stone/iron sword. Arrows + SPACE + S are the six keys the
# scanner paradigm already teaches; R/T/F/P and 1-6 are the letters crafter's
# own run_gui.py uses for the other ten, so every action is reachable (noop =
# no key held). Without them place and make are dead keys and the tech tree
# above "collect wood" cannot be played at all.
# Source: https://github.com/danijar/crafter/blob/master/crafter/data.yaml
_DEFAULT_KEYMAP = {
    "LEFT": 1, "RIGHT": 2, "UP": 3, "DOWN": 4, "SPACE": 5, "S": 6,
    "R": 7, "T": 8, "F": 9, "P": 10,
    "1": 11, "2": 12, "3": 13, "4": 14, "5": 15, "6": 16,
}

# The six actions `menu` mode leaves on their own button, and the keys the rig's
# own frontend puts them on (`frontend_pygame.py:KEYMAP` -- D and S beside the
# arrows, W and A for cycle and confirm), so a subject who has practised in
# crafter-for-brain-scan keeps their fingers. The remaining ten are not listed
# anywhere: they are whatever is left of the action space, so a crafter that
# gained an action would put it in the menu rather than drop it.
_MENU_KEYMAP = {"LEFT": 1, "RIGHT": 2, "UP": 3, "DOWN": 4, "D": 5, "S": 6}

#: What ``move_<name>`` displaces the player by, copied from ``Player._move``.
_DIRECTIONS = {"left": (-1, 0), "right": (+1, 0), "up": (0, -1), "down": (0, +1)}

#: Overlay text per cue -- the model's copy of what the subject just heard.
_CUE_LINES = {"score": ["+1"], "hit": ["HIT"], "blocked": ["NO EFFECT"]}

#: The outcome of a frame that has not happened yet (reset, or no env).
_NO_OUTCOME = {"hit": False, "refused": False, "target": ""}


def _semantic_names(env: Any) -> list[str] | None:
    """Names for the ids in crafter's ``info["semantic"]`` grid, in id order.

    Read off the env's own two tables rather than hardcoded: crafter appends a
    fresh id for any material it is asked to place that was not in its initial
    list, so the mapping is only fully known once the block has been played.

    :param env: the crafter env that produced the semantic grids.
    :return: names indexed by id, or ``None`` if this crafter has no semantic
        view (then the grid is logged without a legend).
    """
    view = getattr(env, "_sem_view", None)
    if view is None:
        return None
    names = {i: mat or "void" for mat, i in view._mat_ids.items()}
    names.update({i: cls.__name__.lower() for cls, i in view._obj_ids.items()})
    return [names.get(i, "?") for i in range(max(names) + 1)]


class CrafterAdapter(EnvAdapter):
    name: str = "crafter"

    def _make(self, spec: dict) -> Any:
        import crafter
        # Built here so `env` is valid before the first reset; reset() then
        # rebuilds it per episode from that episode's seed.
        self._kwargs = dict(spec.get("env_kwargs", {}))
        self._log_frames = bool(spec.get("log_frames", False))
        self._show_score = bool(spec.get("show_score", False))
        self._cues = bool(spec.get("cues", False))
        self._cue_overlay = bool(spec.get("cue_overlay", False))
        # Synthesized once per block rather than per frame: each is a few tens
        # of thousands of samples, and the loop wants them at 2.5 Hz.
        self._cue_sounds = {
            "score": cue_bank.score_cue(),
            "hit": cue_bank.hit_cue(),
            "blocked": cue_bank.blocked_cue(),
        } if self._cues else {}
        self._n_achievements = len(crafter.constants.achievements)
        self._unlocked: set[str] = set()
        self._last_unlock: str | None = None
        self._last_obs = None
        self._cue = ""
        self._outcome: dict = _NO_OUTCOME
        env = self._build(spec.get("seed"))
        self._menu_mode = bool(spec.get("menu", False))
        self._setup_menu(env)
        return env

    def _build(self, seed: int | None) -> Any:
        """Construct a crafter env whose first episode is fixed by ``seed``."""
        import crafter
        return crafter.Env(seed=seed, **self._kwargs)

    def _setup_menu(self, env: Any) -> None:
        """Work out the menu list and the two meta-action ids.

        The menu is the complement of the directly-mapped six within the env's
        own ``action_names``, so it is read off the space rather than written
        down twice. Set up even when ``menu`` is off: the ids are then unused,
        and one branch fewer beats a half-built adapter.

        :param env: the env being built, not yet stored as ``self.env``.
        """
        n = len(env.action_names)
        direct = set(_MENU_KEYMAP.values()) | {0}
        self._menu = [i for i in range(n) if i not in direct]
        self._cycle, self._confirm = n, n + 1
        self._menu_idx = 0
        self._env_action = 0
        self._show_menu = False

    def _keyspec(self) -> SingleKeySpec:
        keymap = _DEFAULT_KEYMAP
        if self._menu_mode:
            keymap = {**_MENU_KEYMAP, "W": self._cycle, "A": self._confirm}
        combos = {frozenset([k]): v for k, v in keymap.items()}
        return SingleKeySpec(combos=combos, noop=0)

    def reset(self, seed: int | None) -> tuple[Any, dict]:
        """Start an episode on a world determined by ``seed`` alone.

        crafter.Env has no ``reset(seed=...)``: its RNG is fixed at
        construction and never changes after, so re-seeding an episode means
        rebuilding the env, and closing the one it replaces.

        :param seed: episode seed, from the run's fold of the design.
        :return: ``(obs, info)``; crafter's reset returns no info, so ``{}``.
        """
        self.env.close()
        self.env = self._build(seed)
        self._unlocked = set()
        self._last_unlock = None
        self._cue = ""
        self._outcome = _NO_OUTCOME
        # `_menu_idx` deliberately survives: the rig's mapper keeps the cursor
        # across episodes within a run, and an adapter lives exactly one block.
        self._show_menu = False
        self._env_action = 0
        self._last_obs = np.asarray(self.env.reset())
        return self._last_obs, {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        # In menu mode the press and the action part ways here; everything
        # downstream -- cue prediction, engine, log -- uses what the env gets.
        self._env_action = self._menu_action(int(action))
        # Read what this press is about to meet, before the engine acts on it.
        self._outcome = self._predict(self._env_action)
        obs, reward, done, info = self.env.step(self._env_action)
        self._last_obs = np.asarray(obs)
        scored = self._note_unlocks(info["achievements"])
        # One cue at a time, highest first: the killing blow that unlocks
        # defeat_zombie is a "+1", not a thud, and a refusal never outranks
        # something that actually happened.
        self._cue = ("score" if scored else
                     "hit" if self._outcome["hit"] else
                     "blocked" if self._outcome["refused"] else "")
        # crafter's `done` is death OR its own step cap; only death is terminal,
        # and health rides along in `info`, so the two are separable here.
        alive = info["inventory"]["health"] > 0
        return self._last_obs, reward, done and not alive, done and alive, info

    def _menu_action(self, action: int) -> int:
        """Translate a button press into the action the engine will receive.

        Only the two meta-buttons are translated; the six direct ones are
        already engine actions and pass through untouched, as does everything
        when ``menu`` is off.

        :param action: the id the keyspec resolved the press to.
        :return: a real crafter action. ``cycle`` returns ``noop``, because the
            turn has to advance either way: a free look at the menu would let a
            subject reach any of the ten for the price of one, and the rig does
            not give them that.
        """
        if not self._menu_mode:
            return action
        self._show_menu = action in (self._cycle, self._confirm)
        if action == self._cycle:
            self._menu_idx = (self._menu_idx + 1) % len(self._menu)
            return 0
        if action == self._confirm:
            return self._menu[self._menu_idx]
        return action

    def _menu_name(self) -> str:
        """The action name the menu cursor is currently on."""
        return self.env.action_names[self._menu[self._menu_idx]]

    def _predict(self, action: int) -> dict:
        """What the pending action will do, read off the pre-step state.

        A reimplementation of the branches of ``Player.update``, against
        crafter's own ``constants.collect / place / make`` tables rather than a
        copy of their contents. Read-only: it indexes the world and the
        inventory, and never touches ``world.random``, so inserting it into the
        loop cannot perturb a replay.

        "Refused" means the engine ran the action and returned having changed
        nothing -- a rock without the pickaxe for it, a craft with no table in
        reach, a walk into a wall the player already faces. Turning to face a
        new direction counts as an effect even when the step itself is blocked,
        because ``Player._move`` sets ``facing`` before it tests the tile.

        :param action: index into ``env.action_names``.
        :return: ``{"hit": bool, "refused": bool, "target": str}``, where
            ``target`` names whatever the player is facing (object class if one
            is there, else the material, else ``""`` off-map).
        """
        import crafter
        from crafter import objects as crafter_objects

        player = getattr(self.env, "_player", None)
        world = getattr(self.env, "_world", None)
        if player is None or world is None:
            return _NO_OUTCOME
        name = self.env.action_names[action]
        facing = tuple(player.facing)
        material, obj = world[(player.pos[0] + facing[0],
                               player.pos[1] + facing[1])]
        out = dict(_NO_OUTCOME)
        out["target"] = type(obj).__name__.lower() if obj else (material or "")
        items = crafter.constants.items
        if name == "noop":
            return out
        if player.sleeping and player.inventory["energy"] < items["energy"]["max"]:
            # Asleep, `Player.update` overwrites the action with `sleep`, so
            # every press in this state is swallowed whole.
            return {**out, "refused": True}

        if name.startswith("move_"):
            direction = _DIRECTIONS[name[len("move_"):]]
            blocked = not player.is_free(player.pos + np.array(direction))
            return {**out, "refused": blocked and direction == facing}

        if name == "do":
            if obj is not None:
                if isinstance(obj, crafter_objects.Plant):
                    return {**out, "refused": not obj.ripe}
                if isinstance(obj, (crafter_objects.Zombie,
                                    crafter_objects.Skeleton,
                                    crafter_objects.Cow)):
                    return {**out, "hit": True}
                # Everything else on a tile is an arrow in flight, and
                # `_do_object` has no branch for one. Crafter's `Fence` would
                # be the other case, but no code path constructs one: there is
                # no `place_fence` action, no `fence` inventory slot and no
                # `collect_fence` achievement, so its pickup branch raises
                # KeyError. Treated as the arrow it must be, not special-cased.
                return {**out, "refused": True}
            info = crafter.constants.collect.get(material)
            if not info:
                return {**out, "refused": True}
            short = any(player.inventory[k] < v
                        for k, v in info["require"].items())
            return {**out, "refused": short}

        if name == "sleep":
            asleep = player.inventory["energy"] >= items["energy"]["max"]
            return {**out, "refused": asleep}

        if name.startswith("place_"):
            info = crafter.constants.place[name[len("place_"):]]
            refused = (obj is not None
                       or material not in info["where"]
                       or any(player.inventory[k] < v
                              for k, v in info["uses"].items()))
            return {**out, "refused": refused}

        if name.startswith("make_"):
            info = crafter.constants.make[name[len("make_"):]]
            # Same 3x3 read `_make` does; slicing the maps, no RNG.
            nearby, _ = world.nearby(player.pos, 1)
            refused = (not all(util in nearby for util in info["nearby"])
                       or any(player.inventory[k] < v
                              for k, v in info["uses"].items()))
            return {**out, "refused": refused}
        return out

    def sound(self) -> Sound | None:
        """The cue for the frame just stepped, or ``None``.

        :return: one :class:`~fmri_gym.adapters.base.Sound`, or ``None`` when
            ``cues`` is off or the frame earned no cue.
        """
        return self._cue_sounds.get(self._cue)

    def render(self) -> np.ndarray:
        # obs IS the RGB frame, and re-rendering it would consume the world RNG
        # after dark (see the module docstring), so hand back what step made.
        return self._last_obs

    def _note_unlocks(self, achievements: dict) -> bool:
        """Track which achievements are unlocked, and the newest one.

        :param achievements: crafter's per-achievement counts for this frame.
        :return: whether this frame unlocked at least one new achievement.
        """
        unlocked = {name for name, n in achievements.items() if n > 0}
        new = unlocked - self._unlocked
        if new:
            # Two can land on one step (eat_cow completes collect_drink's
            # sibling, say); taking the min just makes the pick reproducible.
            self._last_unlock = min(new)
        self._unlocked = unlocked
        return bool(new)

    def overlay(self) -> list[str] | None:
        """Score and cue lines for the display margin, when asked for.

        Crafter draws its own HUD -- four status bars and the inventory -- but
        never the achievement count, which is the score its paper reports and
        the only feedback that the tech tree moved. A subject who cannot see it
        is guessing. The numbers come straight out of the ``info`` dict the env
        already returns every step, so the model harness reads the identical
        values and rule 1 holds: this shows env state, it does not add any.

        The cue line is the same bit of information the subject just heard,
        written down, so a policy reading these lines as text is told what a
        human in the bore is told and no more. It has its own flag because the
        two players receive it differently: a subject hears it, and a line in
        the margin only repeats that while pulling the eyes off the frame, so
        scanner configs leave `cue_overlay` off. A policy has no ears, so
        `agent_play.py` turns it on and reads as text what the subject heard.

        :return: the lines to draw, or ``None`` when there are none.
        """
        lines: list[str] = []
        if self._show_score:
            lines += ["SCORE", f"{len(self._unlocked)} / {self._n_achievements}"]
            if self._last_unlock:
                # Underscores would not wrap inside the 128 px bar; spaces do.
                lines += ["", "LAST", self._last_unlock.replace("_", " ")]
        if self._cue_overlay and self._cue:
            lines += ([""] if lines else []) + _CUE_LINES[self._cue]
            if self._cue == "hit":
                lines.append(self._outcome["target"])
        return lines or None

    def on_frame_overlay(self) -> tuple[list[str], float] | None:
        """The menu cursor, drawn over the player, while it is being used.

        The one thing on screen the subject is aiming with rather than reading,
        so it goes where they are already looking instead of into the margin.
        0.5 would be the middle of the frame; crafter draws its own HUD into the
        bottom two of the nine tile rows, so the played part is the top seven
        and the player stands at row 3.5 of 9.

        :return: ``(lines, y_frac)`` for :meth:`fmri_gym.display.Display
            .draw_frame`, or ``None`` when there is nothing to show.
        """
        if not (self._menu_mode and self._show_menu):
            return None
        # Underscores are the logged id; the screen gets the readable form.
        return (["> " + self._menu_name().replace("_", " ")], 3.5 / 9.0)

    def capture(
        self, obs: Any, info: dict, want_blob: bool = True
    ) -> FrameState:
        """Log this frame's symbolic state, and its pixels if asked to.

        :param obs: the frame crafter returned from ``step``.
        :param info: crafter's info dict for the same step.
        :param want_blob: whether this frame is a savestate anchor.
        :return: the frame's :class:`FrameState`.
        """
        variables = {
            "inventory": list(info["inventory"].values()),
            "achievements": list(info["achievements"].values()),
            "player_pos": np.asarray(info["player_pos"]),
            # SemanticView hands out a fresh array today; copy anyway, because a
            # view onto the live map would be overwritten in place next step.
            "semantic": np.asarray(info["semantic"]).copy(),
            # Outcome of the press that produced this frame (see _predict).
            # Logged whether or not `cues` is on: `cue` is the label of what
            # happened, and only its playback is optional.
            "hit": self._outcome["hit"],
            "no_effect": self._outcome["refused"],
            "cue": self._cue,
            "target": self._outcome["target"],
        }
        if self._menu_mode:
            # What the engine got, next to what was pressed: `actions` in the
            # npz holds the button, and for cycle/confirm the two differ. Menu
            # state is read after the press, so a cycle row names where the
            # cursor landed, which is also what the frame shows.
            variables["env_action"] = self._env_action
            variables["menu_idx"] = self._menu_idx
            variables["menu_sel"] = self._menu_name()
        if self._log_frames:
            # Kept as uint8 rather than bytes: a list of bytes becomes a numpy
            # "S" array, which strips the trailing NULs a zlib stream can end
            # with. Decode with zlib.decompress -> frombuffer -> frame_shape.
            packed = zlib.compress(self._last_obs.tobytes(), 6)
            variables["frame_zlib"] = np.frombuffer(packed, np.uint8)
        blob = pickle.dumps(self.env, protocol=5) if want_blob else None
        return FrameState(blob=blob, variables=variables)

    def restore(self, blob: bytes) -> None:
        """Replace the env with the pickled one in ``blob``.

        :param blob: bytes previously returned as :attr:`FrameState.blob`.
        """
        self.env = pickle.loads(blob)
        # The restored frame is one nobody pressed a button to reach.
        self._cue = ""
        self._outcome = _NO_OUTCOME
        self._show_menu = False
        world = getattr(self.env, "_world", None)
        if world is None:
            self._last_obs = np.asarray(self.env.render())
            return
        # Rewind the stream this render just drew night noise from, so the
        # continuation sees the draws the recorded run saw.
        state = world.random.get_state()
        self._last_obs = np.asarray(self.env.render())
        world.random.set_state(state)

    def block_extra(self) -> dict:
        """Block-level legends for the per-frame variables.

        :return: id-indexed name arrays for the action space, the inventory
            slots, the achievements and (when available) the semantic grid.
        """
        import crafter
        # constants.items / .achievements are what the player's dicts are built
        # from, so their order is the order `capture` logs the values in
        # (checked against a live env, 2026-09-15).
        extra = {
            "action_names": np.array(self.env.action_names),
            "inventory_names": np.array(list(crafter.constants.items)),
            "achievement_names": np.array(list(crafter.constants.achievements)),
        }
        if self._menu_mode:
            # `actions` now holds button ids, so the legend has to cover the two
            # that are not engine actions; `env_action` still indexes the first
            # 17 of it. menu_names decodes menu_idx, and is the order the
            # subject cycles through.
            extra["action_names"] = np.array(
                list(self.env.action_names) + ["cycle", "confirm"]
            )
            extra["menu_names"] = np.array(
                [self.env.action_names[i] for i in self._menu]
            )
        names = _semantic_names(self.env)
        if names:
            extra["semantic_names"] = np.array(names)
        if self._log_frames:
            extra["frame_shape"] = np.array(self._last_obs.shape)
        return extra
