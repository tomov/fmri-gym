"""Crafter adapter (danijar/crafter), via ``crafter-gym``.

``crafter_gym`` (``gym/crafter/``) is the Gymnasium env: ``crafter.Env`` speaks
the old ``gym`` API and is seeded at construction only, and that package puts
the Gymnasium contract in front of it, so ``reset(seed=)`` rebuilds the world
and a seed alone fixes an episode. A phase's ``game`` picks between the two
envs it registers: ``Crafter-v0``, and ``CrafterMenu-v0``, the same game behind
the eight buttons of a scanner button box. Both are the environment's business
rather than this file's: an adapter that owned the interface would stop a model
from meeting the game a subject met.

So are the paradigm's four levels, for the same reason, and for one more: a
savestate is a pickle of the env chain, so rules that are a wrapper's
parameters come back with a resumed world while rules an adapter imposed on a
live player would not. ``env_kwargs.level`` names one of
``crafter_gym.levels.LEVELS`` and nothing in this file reads it -- it goes
through to the env with the rest of ``env_kwargs``, which also means the
``resume`` slot header carries it and a block that opens an earlier level's
world is told so.

The observation IS the RGB frame (64x64 by default; ``env_kwargs.size``), so
``render`` returns it. Choose a ``size`` that is an integer fraction of the
display height (384 -> x2 in a 1024x768 window): display.draw_frame scales
unfiltered, so a fractional factor smears crafter's pixel art, and the HUD --
four status icons plus 16 inventory counters, drawn by the env itself into the
bottom two of the nine tile rows -- is the part that suffers first.

``length`` defaults to a 10000-step cap. A scanner block is bounded by its own
duration, so configs pass ``length: 0`` (no cap) and every ending is a real
death.

What this file owns is the fMRI side of crafter: the state a block logs, the
savestates it branches from, the score the env never draws, and the sounds a
game of PNGs cannot make.

Replay needs a fork
-------------------

Seed + actions replay crafter exactly, but only on a fork, because the bug is
an engine one rather than a rig one. Every tenth step ``Env.step`` rebalances
creatures per chunk, and ``World.chunks`` holds each chunk's objects in a
Python *set*; set order follows object id(), so the despawn pick
``creatures[random.randint(...)]`` lands on a different animal between two runs
of the same seed and actions. Terrain is bit-identical either way, since
worldgen is seeded properly; creatures, and with them the rendered frame, are
not. Sorting that list by position is the whole fix, and this config expects
it::

    pip install git+https://github.com/chengfanbrain/crafter.git@deterministic

Measured 2026-09-28, recording and replaying in two processes that differ only
in ``PYTHONHASHSEED``. Stock 1.8.3 leaves its own trajectory at step 30 of a
225-action episode and ends it a step early; on the fork, all 5 episodes of a
750-frame block replayed from ``episode_seeds`` + ``actions`` into both the
logged symbolic state and the logged pixels, bit for bit. The block's
``frames.h5`` keeps the displayed frames anyway: they are the record that does
not depend on whoever opens the block later having the right build installed.
Cost, measured with per-frame zlib at 2.5 fps and size 384: 7.4 KB for a median
daylit frame, but crafter mixes per-pixel noise into the view at night, so its
107 night frames ran to a median 185 KB and 217 KB at the worst, 17.4 MB between
them; the frames totalled 22.5 MB.

That night noise is drawn from ``world.random``, the same stream the creatures
use, so a render outside the step loop would shift every later draw. Nothing
here renders: ``CrafterEnv`` hands back the frame ``step`` already produced,
which is also what ``restore`` shows for a restored anchor. The one render out
of band is a level's, on the first frame of an episode, where the noise is not
drawn at all (``CrafterEnv.redraw``).

Savestates
----------

Crafter has no savestate API, but the whole env pickles, so ``capture`` returns
that blob and ``restore`` loads it. At size 384 a state is 2.3 MB and ~3 ms
either way, of which the game is 0.16 MB: the rest is the observation space's
constant bounds and the frame ``render`` hands back, both of which zlib takes
down to ~57 KB an anchor once compressed. Storing one per frame would still be
1.7 GB raw a block, hence ``state_stride`` in the config (25 = one anchor every
10 s at 2.5 fps, counted per episode). On the fork an anchor is a real branch
point: all 32 in the measured block restored and then played their episode out
with every frame and every semantic grid identical to the seed replay, which is
what a model rollout from a subject's own state needs. On stock crafter most of
them do not, because unpickling gives the objects new id()s and the next
rebalance picks a different animal.

``info`` carries the whole symbolic state -- 16 inventory counters (health,
food, drink, energy, then materials and tools), 22 achievement counters, the
player's world position, and a 64x64 grid of material/object ids -- so
``capture`` logs all four and ``block_extra`` ships the tables that decode
them. In menu mode ``info`` also carries what the engine was given
(``env_action``) and where the cursor stands, which ``capture`` adds to the
same row, beside the button ``actions`` holds.

The score, and the cursor
-------------------------

The env's HUD covers inventory and the four status bars but not the achievement
count, so ``show_score`` puts it in the strip above the frame. It reports only
what ``info`` already carries, which is what keeps humans and models on the same
game: a policy is handed the same lines as text. The count is shown rather than
the block's cumulative reward, which is the same quantity written less legibly:
crafter pays +1 per first unlock and (health - last_health)/10 per step, and
the health terms telescope to at most -0.9 over an episode, so a return of 2.3
means three achievements.

``menu`` mode draws the cursor over the player rather than in the strip,
because that is where the eyes already are between presses, and only while the
last press was a ``cycle`` or a ``confirm``. The rig instead fades it out on a
3 s wall clock (``frontend_pygame.py --menu-s``), which a turn-based block
cannot copy: it repaints only when a key is pressed, so nothing would clear the
line until the next press anyway. "While you are using it" is the closest
equivalent, and it has the better property of being reconstructible from the
logged actions alone.

Cues
----

Crafter ships no sound at all (56 assets, every one a PNG), which leaves a
subject in the bore unable to tell three situations apart, all of which look
like a frame where nothing much moved: the press landed on a creature but did
not kill it (a zombie takes five bare-handed blows), the press was refused by
the engine (no pickaxe for that rock, no table in reach), and the press was
dropped by the rig. The three waveforms below fill that in, one per frame at
most, score > hit > blocked, because audio output queues rather than mixes.

Which cue to play is decided BEFORE the step, by reading the state the engine
is about to act on, not by diffing state afterwards. Afterwards is not enough:
a non-lethal blow changes nothing that reaches ``info``, and a refusal is
indistinguishable from a refusal-plus-a-zombie-walking-past. The prediction is
exact rather than heuristic because the player is ``world.objects[0]`` -- added
in ``reset`` before worldgen -- and ``Env.step`` updates objects in that order,
so nothing can move between the read and ``Player.update``. It reimplements the
branches of ``Player.update`` over ``constants.collect / place / make``,
reading crafter's own tables, and touches neither the world nor
``world.random``.

Reimplementing engine branches is the part of this file that can go stale
without anyone noticing, so it is checked rather than asserted:
``docs/crafter_cue_check.py`` runs each press twice against a deep copy of the
live env -- once as pressed, once as ``noop`` -- and calls the prediction wrong
if the two resulting states disagree with it. 4545 presses, 0 mismatches on
2026-09-28. Re-run it after a crafter version bump.

Collecting grass is deliberately silent: ``data.yaml`` gives it
``probability: 0.1`` and ``leaves: grass``, so a press that yields nothing is
the roll failing rather than the engine refusing, and the honest cue for luck
is no cue.

The four resulting columns (``hit``, ``no_effect``, ``cue``, ``target``) are
logged whether or not ``cues`` is on, since they describe the frame rather than
the feedback. What is gated is delivery, and each player gets one channel:
``cues`` plays the sound, and ``cue_overlay`` writes the same thing as a field
in the HUD strip. Scanner configs set the first and not the second, because a
subject who has already heard the cue would only be reading a repeat of it, and
every glance at the strip is a glance away from the frame. A policy has no
ears, so the agent harness turns the second on: the information a model reads
is still exactly the information the subject got.
"""

from __future__ import annotations

import pickle
from typing import Any

import gymnasium as gym
import numpy as np

from .base import EnvAdapter, FrameState, Sound

#: The two envs ``crafter_gym`` registers, and what a phase's ``game`` picks
#: between: the sixteen-key game, and the same game behind eight buttons.
_PLAIN, _MENU = "Crafter-v0", "CrafterMenu-v0"

#: What ``move_<name>`` displaces the player by, copied from ``Player._move``.
_DIRECTIONS = {"left": (-1, 0), "right": (+1, 0), "up": (0, -1), "down": (0, +1)}

#: HUD text per cue -- the model's copy of what the subject just heard.
_CUE_LINES = {"score": "+1", "hit": "HIT", "blocked": "NO EFFECT"}

#: The outcome of a frame nobody pressed a button for: a reset, or a restore.
_NO_OUTCOME = {"hit": False, "refused": False, "target": ""}

#: The crafter internals this adapter reads past the gym API: the pre-step
#: state `_predict` needs, and the semantic legend's tables.
_INTERNALS = ("_player", "_world", "_sem_view")


def _check_internals(game: Any) -> None:
    """Fail at start-up if this crafter is not the one the adapter reads.

    A crafter that renamed one of these would not announce itself. It would log
    every frame as "nothing happened", or write a semantic grid with no legend,
    and the block would be wrong rather than missing -- so the check is here,
    before a subject is in the bore, rather than at the read.

    :param game: the ``crafter.Env`` inside the env just built.
    :raises RuntimeError: naming what is missing and the build to install.
    """
    missing = [name for name in _INTERNALS if not hasattr(game, name)]
    if missing:
        raise RuntimeError(
            f"this crafter's Env has no {', '.join(missing)}, which the adapter "
            "reads for cue prediction and the semantic legend. Install the "
            "build the config expects: pip install "
            "git+https://github.com/chengfanbrain/crafter.git@deterministic")


def _semantic_names(game: Any) -> list[str]:
    """Names for the ids in crafter's ``info["semantic"]`` grid, in id order.

    Read off the env's own two tables rather than hardcoded: crafter appends a
    fresh id for any material it is asked to place that was not in its initial
    list, so the mapping is only fully known once the block has been played.

    :param game: the crafter env that produced the semantic grids.
    :return: names indexed by id.
    """
    view = game._sem_view
    names = {i: mat or "void" for mat, i in view._mat_ids.items()}
    names.update({i: cls.__name__.lower() for cls, i in view._obj_ids.items()})
    return [names.get(i, "?") for i in range(max(names) + 1)]


def _reachable(env: Any) -> tuple[str, ...]:
    """The achievements this env's level can unlock, in crafter's id order.

    Asked of the env rather than of the phase's ``env_kwargs``, so that it
    follows a restored world into the level that world was recorded on, which
    is the level actually being played (``crafter_gym.levels.level_of``).

    :param env: the env chain, as ``_make`` built it or ``restore`` unpickled it.
    :return: the reachable names, a subsequence of
        ``crafter.constants.achievements``.
    """
    import crafter_gym

    return crafter_gym.reachable_achievements(crafter_gym.level_of(env))


# --- the three cues -------------------------------------------------------
#
# Separated by TIMBRE, not pitch: `hit` is a noise burst, `score` a harmonic
# chime, `blocked` a low sine, so the one that can immediately precede another
# is never mistaken for it. Queueing makes length a timing constraint too: a
# cue longer than a frame delays the next one by the difference, and all three
# fit inside the scanner block's 400 ms turn (hit 60 ms, blocked 100, score
# exactly 400). `score` is the one that had to be cut to get there: it is a
# port of the rig's `celebrate_wav`, whose third note rings for 0.45 s (670 ms
# in all), and at that length an unlock followed immediately by another cue
# delayed it ~270 ms, with three unlocks on consecutive frames drifting
# ~800 ms. Shortened to 0.18 s on 2026-09-20; the attack that identifies it is
# untouched. Everything is synthesized from these numbers rather than shipped
# as a wav, so the exact stimulus a session presented is recoverable from the
# commit hash, and `hit` draws its noise from a fixed seed for the same reason.

#: Cue playback rate. 44.1 kHz because that is what the rig's own celebration
#: jingle was authored at and what any scanner-side audio chain expects.
_SAMPLE_RATE = 44100

#: Peak amplitude as a fraction of full scale. The jingle was raised to this
#: after a pilot found the earlier cue inaudible over scanner-adjacent
#: playback; the other two are matched to it so relative loudness is a property
#: of the waveform, not of three different normalisations.
_PEAK = 0.92


def _pcm(wave: np.ndarray, peak: float = _PEAK) -> np.ndarray:
    """Normalise a float waveform to int16 mono PCM shaped ``(n, 1)``.

    :param wave: float samples, any scale.
    :param peak: target peak as a fraction of full scale.
    :return: ``(n_samples, 1)`` int16, the shape :class:`~fmri_gym.audio.Audio`
        opens the output stream from.
    """
    wave = np.asarray(wave, np.float64)
    wave = wave / max(np.abs(wave).max(), 1e-12) * peak
    return (wave * 32767).astype("<i2")[:, None]


def _score_cue(rate: int = _SAMPLE_RATE) -> Sound:
    """The "+1" chime: a three-note ascending arpeggio, exactly 0.40 s.

    B5, E6, B6, each a sine plus a quieter octave harmonic under a 5 ms attack
    and an exponential decay. The two short notes that make it recognisable are
    the rig's own.

    :param rate: sample rate in Hz.
    :return: the cue as a :class:`~fmri_gym.adapters.base.Sound`.
    """
    parts = []
    for freq, dur in ((987.77, 0.11), (1318.51, 0.11), (1975.53, 0.18)):
        t = np.arange(int(rate * dur)) / rate
        tone = np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(4 * np.pi * freq * t)
        envelope = np.minimum(t / 0.005, 1.0) * np.exp(-t / (0.7 * dur))
        parts.append(tone * envelope)
    return Sound(_pcm(np.concatenate(parts)), rate)


def _hit_cue(rate: int = _SAMPLE_RATE) -> Sound:
    """The "that landed" thud: a 60 ms noise burst with a dropping body.

    Deliberately not a note. It fires up to five times on one zombie and would
    have to be agreed with the ``score`` chime it can immediately precede, so
    it is separated by texture: a smoothed noise burst (the impact) over a
    220 Hz sine sliding down an octave (the body), under a 1 ms attack and a
    fast decay. Short enough that five in a row at 2.5 Hz stay distinct.

    :param rate: sample rate in Hz.
    :return: the cue as a :class:`~fmri_gym.adapters.base.Sound`.
    """
    dur = 0.06
    t = np.arange(int(rate * dur)) / rate
    noise = np.random.default_rng(20260916).standard_normal(t.size)
    # Boxcar-smooth the noise: drops the hiss that makes a raw burst read as
    # static rather than as an impact.
    noise = np.convolve(noise, np.ones(8) / 8, mode="same")
    body = np.sin(2 * np.pi * 220 * t * np.exp(-t / dur))
    envelope = np.minimum(t / 0.001, 1.0) * np.exp(-t / (0.25 * dur))
    return Sound(_pcm((0.7 * noise + 0.6 * body) * envelope), rate)


def _blocked_cue(rate: int = _SAMPLE_RATE) -> Sound:
    """The "nothing happened" blip: 0.1 s of low sine, quieter than the rest.

    Low and dull on purpose. It reports a non-event, it can repeat while a
    subject probes the tech tree, and it must never be mistaken for the reward
    chime an octave and a half above it.

    :param rate: sample rate in Hz.
    :return: the cue as a :class:`~fmri_gym.adapters.base.Sound`.
    """
    dur = 0.1
    t = np.arange(int(rate * dur)) / rate
    tone = np.sin(2 * np.pi * 160 * t) + 0.25 * np.sin(2 * np.pi * 80 * t)
    envelope = np.minimum(t / 0.004, 1.0) * np.minimum(1.0, (dur - t) / 0.02)
    return Sound(_pcm(tone * envelope, peak=0.55), rate)


class CrafterAdapter(EnvAdapter):
    name: str = "crafter"

    #: 0.5 would be the middle of the frame; crafter draws its own HUD into the
    #: bottom two of the nine tile rows, so the played part is the top seven and
    #: the player stands at row 3.5 of 9. Anything written over the frame goes
    #: there, which is where the rig puts its own lines too.
    overlay_y: float = 3.5 / 9.0

    def _make(self, spec: dict) -> gym.Env:
        import crafter_gym

        # Through the env package rather than `import crafter`: the repo's own
        # gym/ directory shadows old gym for a process started at the repo
        # root, and `crafter_gym.import_crafter` is where that is dealt with.
        # Called for that, not for the module: once it has run, the plain
        # `import crafter` of `block_extra` works too.
        crafter_gym.import_crafter()

        self._show_score = bool(spec.get("show_score", False))
        self._cues = bool(spec.get("cues", False))
        self._cue_overlay = bool(spec.get("cue_overlay", False))
        # Synthesized once per block rather than per frame: each is a few tens
        # of thousands of samples, and the loop wants them at 2.5 Hz.
        self._cue_sounds = {
            "score": _score_cue(), "hit": _hit_cue(), "blocked": _blocked_cue(),
        } if self._cues else {}
        self._unlocked: set[str] = set()
        self._cue = ""
        self._outcome: dict = _NO_OUTCOME
        game = spec.get("game", _PLAIN)
        if game not in (_PLAIN, _MENU):
            raise ValueError(f"crafter backend: game must be {_PLAIN!r} or "
                             f"{_MENU!r} (the menu is the same game behind a "
                             f"button box), not {game!r}")
        self._menu_mode = game == _MENU
        kwargs = dict(spec.get("env_kwargs", {}), seed=spec.get("seed"))
        # Built directly rather than through gym.make: the menu's cursor is on
        # the wrapper, and gymnasium's OrderEnforcing would hide it.
        env = (crafter_gym.make_menu(**kwargs) if self._menu_mode
               else crafter_gym.make_plain(**kwargs))
        _check_internals(env.unwrapped.game)
        self._achievements = _reachable(env)
        return env

    @property
    def _game(self) -> Any:
        """The ``crafter.Env`` itself, past the Gymnasium env and any wrapper."""
        return self.env.unwrapped.game

    def reset(self, seed: int | None) -> tuple[Any, dict]:
        """Start an episode on a world determined by ``seed`` alone.

        :param seed: episode seed, from the run's fold of the design.
        :return: ``(obs, info)``.
        """
        self._unlocked = set()
        self._cue = ""
        self._outcome = _NO_OUTCOME
        return super().reset(seed)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        # In menu mode the press and the action part ways; the cue is about
        # what the engine is given, so ask the wrapper before it moves.
        env_action = (self.env.env_action(action) if self._menu_mode
                      else int(action))
        # Read what this press is about to meet, before the engine acts on it.
        self._outcome = self._predict(env_action)
        obs, reward, terminated, truncated, info = super().step(action)
        scored = self._note_unlocks(info["achievements"])
        # One cue at a time, highest first: the killing blow that unlocks
        # defeat_zombie is a "+1", not a thud, and a refusal never outranks
        # something that actually happened.
        self._cue = ("score" if scored else
                     "hit" if self._outcome["hit"] else
                     "blocked" if self._outcome["refused"] else "")
        return obs, reward, terminated, truncated, info

    def _predict(self, action: int) -> dict:
        """What the pending action will do, read off the pre-step state.

        A reimplementation of the branches of ``Player.update``, against
        crafter's own ``constants.collect / place / make`` tables rather than a
        copy of their contents. Read-only: it indexes the world and the
        inventory, and never touches ``world.random``, so inserting it into the
        loop cannot perturb a replay. Called from ``step`` only, so the episode
        has been reset and the player exists.

        "Refused" means the engine ran the action and returned having changed
        nothing -- a rock without the pickaxe for it, a craft with no table in
        reach, a walk into a wall the player already faces. Turning to face a
        new direction counts as an effect even when the step itself is blocked,
        because ``Player._move`` sets ``facing`` before it tests the tile.

        :param action: index into crafter's own ``action_names``.
        :return: ``{"hit": bool, "refused": bool, "target": str}``, where
            ``target`` names whatever the player is facing (object class if one
            is there, else the material, else ``""`` off-map).
        """
        import crafter
        from crafter import objects as crafter_objects

        game = self._game
        player, world = game._player, game._world
        name = game.action_names[action]
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

    def _note_unlocks(self, achievements: dict) -> bool:
        """Track which achievements are unlocked, and the newest one.

        :param achievements: crafter's per-achievement counts for this frame.
        :return: whether this frame unlocked at least one new achievement.
        """
        unlocked = {name for name, n in achievements.items() if n > 0}
        new = unlocked - self._unlocked
        self._unlocked = unlocked
        return bool(new)

    def sound(self) -> Sound | None:
        """The cue for the frame just stepped, or ``None``.

        :return: one :class:`~fmri_gym.adapters.base.Sound`, or ``None`` when
            ``cues`` is off or the frame earned no cue.
        """
        return self._cue_sounds.get(self._cue)

    def autoplay(self, info: dict) -> int | None:
        """``noop`` while the player is asleep, otherwise ``None``.

        Crafter takes no action from a sleeping player: ``Player.update``
        overwrites it with ``sleep`` until energy is full, and the step on
        which it fills is the one that wakes them and runs their action
        normally (``crafter/objects.py``). So under ``turn_based`` the presses
        a sleep costs buy nothing; they only turn the world's crank, and stock
        crafter charges eleven steps per point of energy, which is sixty-six
        presses to go from three to nine. Stepping those frames here leaves
        every rule alone -- the night still passes at its own rate, hunger,
        thirst and health still move at their sleeping rates, a zombie still
        closes in and still deals 7 to a sleeper -- and only stops asking the
        subject to supply the crank.

        The wake is what the subject needs to see, so it is drawn like any
        other frame and the loop stops on it: waking from a blow clears
        ``sleeping`` (``_wake_up_when_hurt``), so the last autoplayed frame is
        the one with the zombie next to them and the health bar already down.
        A frame is logged for every one of these steps, with ``noop`` as its
        action, so the log and a replay see exactly what happened; the ``noop``
        also keeps a press made mid-sleep from moving the menu cursor, which
        is wrapper state the engine would not have discarded.

        :param info: crafter's info for the step just taken.
        :return: 0 (``noop``) while asleep, else ``None``.
        """
        return 0 if info.get("sleeping") else None

    def hud(self, score: float, time_remaining: float) -> list[str]:
        """Time left, then the achievement count.

        Crafter draws its own HUD -- four status bars and the inventory -- but
        never the achievement count, which is the score its paper reports and
        the only feedback that the tech tree moved. A subject who cannot see it
        is guessing. The numbers come straight out of the ``info`` dict the env
        already returns every step, so the model harness reads the identical
        values: this shows env state, it does not add any.

        The denominator is what the level being played can unlock, which on the
        two levels with nothing hostile in them is 20 rather than crafter's 22
        (``crafter_gym.levels.reachable_achievements``). A count against 22
        there would ask the subject for two achievements the world does not
        contain, and would score the block against them afterwards.

        The cue field is the same bit of information the subject just heard,
        written down, so a policy reading these lines as text is told what a
        human in the bore is told and no more. It has its own flag because the
        two players receive it differently (see ``cue_overlay`` above).

        :param score: the episode's cumulative reward, shown only when
            ``show_score`` is off and the count is therefore not.
        :param time_remaining: seconds until the block ends.
        :return: the fields, laid out left to right above the frame.
        """
        lines = [f"{max(0, int(time_remaining))} s"]
        if self._cue_overlay and self._cue:
            cue = _CUE_LINES[self._cue]
            if self._cue == "hit" and self._outcome["target"]:
                cue = f"{cue} {self._outcome['target']}"
            lines.append(cue)
        if not self._show_score:
            return [*lines, f"Score: {score:g}"]
        return [*lines, f"{len(self._unlocked)} / {len(self._achievements)}"]

    def overlay(self) -> tuple[list[str], float] | None:
        """The menu cursor, drawn over the player, while it is being used.

        The one thing on screen the subject is aiming with rather than reading,
        so it goes where they are already looking (:attr:`overlay_y`) instead of
        into the strip.

        :return: ``(lines, y_frac)``, or ``None`` when the menu is off or idle.
        """
        if not (self._menu_mode and self.env.showing):
            return None
        # Underscores are the logged id; the screen gets the readable form.
        return (["> " + self.env.selected.replace("_", " ")], self.overlay_y)

    def outcome(self, terminated: bool, truncated: bool) -> tuple[str, str]:
        # The env already separates the two endings crafter reports as one
        # flag, so terminated is death and truncated is its step cap.
        if terminated:
            return "lost", "You died"
        return super().outcome(terminated, truncated)

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
            # True on every frame the player spent asleep, which under
            # turn_based are the frames the loop stepped itself (see
            # `autoplay`): without it a run of noops is indistinguishable from
            # a subject who pressed nothing.
            "sleeping": bool(info.get("sleeping", False)),
        }
        if self._menu_mode:
            # What the engine was given is `info["env_action"]`, which the run
            # logs beside the button `action` holds. Menu state is read after
            # the press, so a cycle row names where the cursor landed, which is
            # what the frame shows.
            variables["menu_idx"] = info["menu_idx"]
            variables["menu_sel"] = info["menu_sel"]
        blob = pickle.dumps(self.env, protocol=5) if want_blob else None
        return FrameState(blob=blob, variables=variables)

    def restore(self, blob: bytes) -> None:
        """Replace the env with the pickled one in ``blob``.

        The picture comes back with it: ``CrafterEnv`` keeps the frame its last
        step produced, so a restored env shows that frame without rendering,
        and the night-noise draws the continuation makes are the ones the
        recorded run made.

        :param blob: bytes previously returned as :attr:`FrameState.blob`.
        """
        self.env = pickle.loads(blob)
        # The level came back with the world, and it is the level now being
        # played even if the phase asked for another one (`fmri_gym.resume`
        # warns about that before the block opens), so the strip's denominator
        # follows the world rather than the config.
        self._achievements = _reachable(self.env)

        self._unlocked = set()
        for name, count in self._game._player.achievements.items():
            if count > 0:
                self._unlocked.add(name)
        # The restored frame is one nobody pressed a button to reach.
        self._cue = ""
        self._outcome = _NO_OUTCOME

    def block_extra(self) -> dict:
        """Block-level legends for the per-frame variables.

        :return: id-indexed name arrays for the action space, the inventory
            slots, the achievements and the semantic grid, plus which of the
            achievements the level played could unlock at all.
        """
        import crafter
        # constants.items / .achievements are what the player's dicts are built
        # from, so their order is the order `capture` logs the values in
        # (checked against a live env, 2026-09-15).
        names = list(crafter.constants.achievements)
        extra = {
            # In menu mode this covers the two buttons that are not crafter
            # actions, since `actions` holds buttons; `env_action` indexes the
            # first 17 of the same list either way.
            "action_names": np.array(self.env.action_names),
            "inventory_names": np.array(list(crafter.constants.items)),
            "achievement_names": np.array(names),
            # Aligned with the names above and with every frame's achievement
            # counts: False marks one the level kept out of the world, which is
            # the denominator the subject was scored against (see `hud`) and
            # the column an analysis has to leave out of a per-level total.
            "achievements_reachable": np.array([n in self._achievements
                                                for n in names]),
            "semantic_names": np.array(_semantic_names(self._game)),
        }
        if self._menu_mode:
            # Decodes menu_idx, and is the order the subject cycles through.
            extra["menu_names"] = np.array(self.env.menu_names)
        return extra
