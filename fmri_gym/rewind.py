"""Going back a few seconds, instead of back to the start.

A block has two kinds of ending and :mod:`fmri_gym.resume` turns on the
difference: the scanner's clock cutting an episode off, which the game should
not charge for, and the game ending it, which it should. A death is the second
kind, so the world ends with it and the next episode opens the seed's. That is
right for a level, and for the affordance level of the crafter paradigm it
throws away the thing being measured: nothing there is hostile and the life
stats are frozen, so the only death left is a mistake about what the world
affords, and paying for it with the minute of scanner time it takes to walk back
out to where they were buys no data. The rig plays it by rolling the world back
instead (``crafter_rig/core.py``: ``DEATH_RESTORE_LEVELS``,
``--death-restore-s``, default ``1.0``).

This is that, and the subject's own version of it: a game phase may say

    "rewind": {"frames": 5, "on_death": true, "hold": 1.0}
    "rewind": {"seconds": 10.0, "stride": 2, "max_states": 32}

and the block keeps the last few frames of play as savestates, so a death
(``on_death``) or the pause menu's ``rewind`` option puts the world back a
window and the episode goes on from there. The episode does not end, so the
block's clock is still what ends it, and a slot the block carries
(:mod:`fmri_gym.resume`) is still handed on: the ending was undone, so the
thread of play never ended.

**Beside ``resume``, and not a rule of the level.** The rig is explicit that
its own rollback is not a ``LEVELS`` field (``core.py``, above
``DEATH_RESTORE_LEVELS``), and the reason is agent parity: a model compared
against a subject plays the same rules, and it must not be handed a second
chance the scanner invented for a person who cannot be given back their
minute. So this is session policy, which in a config means a phase field, and
a rollout of the same level sees nothing of it.

**How far back is two different questions.** In a real-time block the frame
rate is the config's, so ``seconds`` says what it means. A turn-based block
steps on presses and waits as long as the subject thinks, so a second of its
run clock is worth one press or none, and the window that matters there is
``frames``: five frames is five of the subject's own moves, whatever they spent
deciding. Exactly one of the two is required, and a phase that gives the one
that does not suit its pacing gets a window that drifts with how fast the
subject plays. The rig's default is ``1.0`` s at ``cadence_hz=5.0``, which is
five of its ticks, and the turn-based form of that is ``"frames": 5``.

**Memory, and what the ring holds.** A savestate is a blob the backend hands
over (``capture(..., want_blob=True)``), and crafter's is a pickle of the env
chain: 2.3 MB at ``size`` 384. The ring holds them raw, because zlib on
megabytes is tens of milliseconds and this runs inside the frame loop, so the
depth is what bounds the cost. The ``frames`` form bounds its own
(:attr:`Policy.depth` states, ``frames // stride`` rounded up plus one); the
``seconds`` form cannot, since presses have no rate, so it is capped by
``max_states`` and a rollback that the cap, rather than the window, decided the
far end of says so on its own record (``capped``). ``stride`` trades
granularity for both.

**Reusing the savestates the log already takes is not enough.** The anchors a
block writes come every ``state_stride`` frames, 25 in the crafter configs, so
in a turn-based block the newest anchor is around 25 presses old and anything
nearer is not kept at all. The ring therefore captures on its own stride, and
a frame that is due for both pays for one capture (:mod:`fmri_gym.run`).

**What the record says.** A rollback writes one ``rewind`` line: what triggered
it, the frame it happened on, the frame it went back to, how far back that was
in frames and in seconds, the score before and after, and the world it restored
to, as the ``resume`` line carries one. That blob is what makes the episode
replayable, because actions alone no longer describe it: the frames between the
target and the rollback were played and then undone, so an analysis that steps
them in sequence plays a different game from here on. ``reconstruction_plan``
cuts the episode into segments at each ``rewind`` line for exactly that reason,
and a block with no rollbacks has one segment and reads as it always did.

**A second death cannot walk further back than the first.** After a rollback
the ring starts again from the world it restored, which is the rig's own
behaviour (``core.py:_apply_restore``) and keeps a subject who dies repeatedly
in one spot from being carried back through the whole block a window at a time.
``max_rewinds`` is the other end of it: once an episode has had that many, a
death is a death and the episode ends the way it would have.

**Which makes ``frames_back`` a distance rather than an amount of play.** It is
measured in the episode's own frame numbering, from the frame the rollback
happened on to the frame the restored world is, and that is what the subject
lost on an episode's first rollback only. From the second on it keeps growing
while what was lost is the play since the rollback before it: a block that died
five times in one spot logs ``frames_back`` 5, 10, 15, 20, 25 for five rollbacks
that each cost five frames. The number an analysis usually wants is therefore
the length of the segment a rollback ended, which is the frame lines since the
previous ``rewind`` line (:func:`fmri_gym.replay.reconstruction_plan`), and not
the field on the line.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

#: The phase field, and the pause-menu option that also triggers one.
FIELD = "rewind"
#: How far back, one of which a policy must name; see the module docstring.
WINDOWS = ("frames", "seconds")
_DEFAULTS: dict = {"on_death": False, "hold": 0.0, "stride": 1, "max_states": 32,
                   "max_rewinds": None}


@dataclass(frozen=True)
class Policy:
    """A phase's ``"rewind"`` field, parsed.

    :ivar frames: window as the subject's own moves; ``None`` if given in
        seconds.
    :ivar seconds: window on the run clock; ``None`` if given in frames.
    :ivar on_death: whether a death rolls the world back instead of ending the
        episode. The pause menu's ``rewind`` option is the other trigger and
        needs no field here: it is in the menu's own ``options``.
    :ivar hold: seconds the frame that triggered the rollback is left on screen
        first, so the subject sees what happened before the world moves
        (the rig's ``--death-hold-s``).
    :ivar stride: capture a state for the ring every this many frames.
    :ivar max_states: the ``seconds`` form's cap on ring depth, which the
        ``frames`` form computes instead.
    :ivar max_rewinds: at most this many rollbacks in one episode; ``None`` =
        as many as the block has time for.
    """

    frames: int | None = None
    seconds: float | None = None
    on_death: bool = False
    hold: float = 0.0
    stride: int = 1
    max_states: int = 32
    max_rewinds: int | None = None

    @property
    def depth(self) -> int:
        """How many states the ring may hold at once.

        The ``frames`` form knows: the oldest entry it needs is the first one
        at least a window back, which is ``frames / stride`` rounded up, plus
        itself. The ``seconds`` form cannot know, because a turn-based block's
        frames have no rate, so it is :attr:`max_states`.
        """
        if self.frames is not None:
            return -(-self.frames // self.stride) + 1
        return self.max_states

    def label(self) -> str:
        """What the pause menu calls this, in the units it was asked for."""
        if self.frames is not None:
            return f"Go back {self.frames} move{'s' if self.frames != 1 else ''}"
        return f"Go back {self.seconds:g} seconds"

    def describe(self) -> dict:
        """The policy in force, for the manifest."""
        return {"on_death": self.on_death, "hold": self.hold, "stride": self.stride,
                "depth": self.depth, "max_rewinds": self.max_rewinds,
                **({"frames": self.frames} if self.frames is not None
                   else {"seconds": self.seconds})}


def policy_of(phase: dict) -> Policy | None:
    """This game phase's rollback policy, or ``None`` if it has no rewind.

    :param phase: a game-phase config, already past :func:`rewind_problems`.
    :return: the :class:`Policy`, or ``None``.
    """
    spec = phase.get(FIELD)
    if not spec:
        return None
    return Policy(**{**_DEFAULTS, **{k: v for k, v in spec.items()}})


def rewind_problems(phase: dict, menu_options: list[str]) -> list[str]:
    """What a bad ``"rewind"`` field looks like, for ``validate_config``.

    The menu's own ``rewind`` option is checked here too, in both directions.
    An option with no policy behind it and a policy nothing can trigger are
    both a field that reads as if it had taken and does nothing in the
    scanner, which is the one failure a config check exists to prevent.

    :param phase: a game-phase config.
    :param menu_options: the options the phase's pause menu offers
        (:func:`fmri_gym.menu.options_of`), empty when it has no menu.
    :return: human-readable problems, empty when the field is absent or usable.
    """
    by_menu = FIELD in menu_options
    if FIELD not in phase:
        return [f'menu.options: "{FIELD}" is offered, but the phase has no "{FIELD}" field '
                f"saying how far back to go, so the option would pause the game and do "
                f'nothing. Add e.g. "{FIELD}": {{"frames": 5}}'] if by_menu else []
    spec = phase[FIELD]
    if not isinstance(spec, dict):
        return [f'{FIELD}: expected a dict naming how far back to go, e.g. '
                f'{{"frames": 5, "on_death": true}} or {{"seconds": 10.0}}, got {spec!r}']
    out = []
    unknown = sorted(set(spec) - set(WINDOWS) - set(_DEFAULTS))
    if unknown:
        out.append(f"{FIELD}: unknown fields {unknown}; the fields are "
                   f"{sorted((*WINDOWS, *_DEFAULTS))}")
    given = [w for w in WINDOWS if w in spec]
    if len(given) != 1:
        out.append(f"{FIELD}: name exactly one of {list(WINDOWS)} (got {given or 'neither'}): "
                   '"frames" is the subject\'s own moves, which is the window a turn-based '
                   'block has, and "seconds" is the run clock, which is the one a real-time '
                   "block has")
    if "frames" in spec and (isinstance(spec["frames"], bool)
                             or not isinstance(spec["frames"], int) or spec["frames"] < 1):
        out.append(f"{FIELD}.frames: expected a whole number of frames >= 1, "
                   f"got {spec['frames']!r}")
    if "seconds" in spec and (isinstance(spec["seconds"], bool)
                              or not isinstance(spec["seconds"], (int, float))
                              or spec["seconds"] <= 0):
        out.append(f"{FIELD}.seconds: expected seconds > 0, got {spec['seconds']!r}")
    if not isinstance(spec.get("on_death", False), bool):
        out.append(f"{FIELD}.on_death: expected true or false, got {spec['on_death']!r}")
    hold = spec.get("hold", _DEFAULTS["hold"])
    if isinstance(hold, bool) or not isinstance(hold, (int, float)) or hold < 0:
        out.append(f"{FIELD}.hold: expected seconds (>= 0) to leave the frame that triggered "
                   f"the rollback on screen, got {hold!r}")
    for field in ("stride", "max_states"):
        value = spec.get(field, _DEFAULTS[field])
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            out.append(f"{FIELD}.{field}: expected a whole number >= 1, got {value!r}")
    if spec.get("max_rewinds") is not None and (
            isinstance(spec["max_rewinds"], bool) or not isinstance(spec["max_rewinds"], int)
            or spec["max_rewinds"] < 1):
        out.append(f"{FIELD}.max_rewinds: expected a whole number >= 1, or null for as many "
                   f"as the block has time for, got {spec['max_rewinds']!r}")
    if "max_states" in spec and "frames" in spec:
        out.append(f"{FIELD}.max_states: only for the \"seconds\" form, whose depth cannot be "
                   f"known in advance. A \"frames\" window bounds its own: "
                   f"frames / stride rounded up, plus one")
    if not spec.get("on_death", False) and not by_menu:
        out.append(f'{FIELD}: nothing can trigger it -- "on_death" is false and the phase\'s '
                   f'menu does not offer "{FIELD}" (menu.options). Set "on_death": true, or '
                   f'add "{FIELD}" to the menu\'s options, or drop the field')
    return out


@dataclass(frozen=True)
class State:
    """One frame of play, kept in case the block goes back to it.

    :ivar ep_frame: the frame's index within its episode.
    :ivar run_time: the run clock at the step that produced it.
    :ivar score: the episode's cumulative reward up to and including it, which
        a rollback puts back with the world.
    :ivar blob: the savestate, as ``restore`` wants it.
    """

    ep_frame: int
    run_time: float
    score: float
    blob: bytes


class Ring:
    """The last window of one episode's play, as savestates, oldest first.

    One per episode. It holds the oldest frame that is still a whole window
    back and everything the stride kept after it, so its head is where a
    rollback goes.

    :param policy: the phase's :class:`Policy`.
    """

    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self._q: collections.deque[State] = collections.deque()
        #: Set when the depth cap, rather than the window, decided the head.
        self.capped = False

    def due(self, ep_frame: int) -> bool:
        """Whether this frame is one the ring keeps.

        :param ep_frame: the frame's index within the episode.
        """
        return ep_frame % self.policy.stride == 0

    def push(self, state: State) -> None:
        """Keep this frame, and drop what the window and the depth no longer need.

        :param state: the frame just played.
        """
        self._q.append(state)
        while len(self._q) > 1 and self._covers(self._q[1], state):
            self._q.popleft()
        while len(self._q) > self.policy.depth:
            # The window asked for more than the cap allows, so the rollback
            # will be shorter than the config reads. Said out loud on the
            # rewind line rather than quietly shortening the window.
            self.capped = True
            self._q.popleft()

    def target(self) -> State | None:
        """The world a rollback would restore, or ``None`` if there is none yet.

        ``None`` is an episode nobody has played a frame of, and early in an
        episode the head is simply the first frame, which is as far back as
        there is to go. The rig's empty deque falls back to a fresh reset the
        same way (``core.py``), which here is leaving the ending alone.
        """
        return self._q[0] if self._q else None

    def reseed(self, state: State) -> None:
        """Start the window again from the world a rollback just restored.

        Rig parity (``core.py:_apply_restore``): without this, a subject who
        dies again where they died the first time is carried back another
        window from there, and a bad spot walks them to the start of the block.

        :param state: the restored world.
        """
        self._q.clear()
        self._q.append(state)

    def _covers(self, state: State, now: State) -> bool:
        """Whether ``state`` is already a whole window behind ``now``."""
        if self.policy.frames is not None:
            return now.ep_frame - state.ep_frame >= self.policy.frames
        return now.run_time - state.run_time >= (self.policy.seconds or 0.0)
