"""The phases of a curriculum, typed: one frozen dataclass per phase type.

A curriculum's JSON is read once, at load (:class:`~fmri_gym.config.curriculum.Curriculum`),
and each phase becomes the object of its ``type``. Each class states its fields'
defaults in one place, its ``from_dict``, and what a phase of its type must
look like in its ``problems``, which the editor's Check and the load share, so
a wrong value is refused before the window opens and never read as a default.

Every phase keeps the JSON it was read from as ``spec``: the manifest records
it, a resumed run compares it, and a game phase's adapter reads its backend's
own keys from it (``env_kwargs``, ``level``, ...), which are the adapter's
business, not the loop's. Nothing here reads a rig.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from ..display import image_line_problems
from ..inputs import ESCAPE, is_control_of_any_input_device
from ..menu import menu_problems

#: An episode's possible outcomes: EnvAdapter.outcome's names, and the subject's own endings.
OUTCOMES = frozenset({"won", "lost", "terminated", "truncated", "playing", "forfeit", "reset"})
#: Outcomes a block may rewind on: the game's own endings. Not `truncated`,
#: the env's step cap, which a rewound episode would hit again for ever.
REWINDABLE = frozenset({"won", "lost", "terminated"})
#: What a frames check plays its rates under: nothing else, or every core kept busy.
LOADS = ("none", "cpu")


class BlockMode(str, Enum):
    """What ends a game block: its clock, or its count of episodes."""

    #: the block lasts ``duration`` seconds
    DURATION = "duration"
    #: the block lasts ``n_episodes`` episodes, or ``duration`` seconds at most
    EPISODE = "episode"


def join_text(text: str | list | tuple) -> str:
    """A message's ``text``: a string, or a list of lines joined by newlines, so a
    curriculum can keep long instructions readable without ``\\n`` escapes."""
    if isinstance(text, (list, tuple)):
        return "\n".join("" if line is None else str(line) for line in text)
    return str(text)


@dataclass(frozen=True, slots=True)
class Phase(ABC):
    """One phase of a curriculum, read off its JSON once: what every phase has.

    A subclass is one phase ``type``: it names it, states its own fields'
    defaults in :meth:`from_dict` and the shape its JSON must have in
    :meth:`problems`.
    """

    #: the phase's ``type`` in the JSON
    type: ClassVar[str]
    #: the JSON it was read from: the manifest records it, a resumed run compares
    #: it, and a game phase's adapter reads its backend's own keys from it
    spec: dict
    #: a resumed run plays it before carrying on (only the non-game phases a
    #: curriculum opens with may say so)
    include_on_resume: bool

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict, index: int) -> Phase:
        """Read the phase off its JSON (which :meth:`problems` passed), defaults filled in.

        :param index: its place in the curriculum.
        """

    @staticmethod
    @abstractmethod
    def problems(d: dict) -> list[str]:
        """What keeps ``d`` from being a phase of this type, as text; empty when it is one."""

    @staticmethod
    def _shared(d: dict) -> dict[str, Any]:
        """The fields every phase reads the same way, for :meth:`from_dict`."""
        return {"spec": d, "include_on_resume": d.get("include_on_resume", False)}


@dataclass(frozen=True, slots=True)
class CheckPhase(Phase, ABC):
    """A rig check's phase (:mod:`fmri_gym.checks`): it measures the rig, plays no game."""


@dataclass(frozen=True, slots=True)
class Fixation(Phase):
    """A fixation cross for ``duration`` seconds."""

    type: ClassVar[str] = "fixation"
    duration: float

    @classmethod
    def from_dict(cls, d: dict, index: int) -> Fixation:
        return cls(**cls._shared(d), duration=d.get("duration", 2.0))

    @staticmethod
    def problems(d: dict) -> list[str]:
        return _positive_problems(d, ("duration",))


@dataclass(frozen=True, slots=True)
class Message(Phase):
    """Text on screen until its ``duration``, a press, or whichever comes first.

    ``controls_to_dismiss`` ends it at a press of one of them, and with a
    ``duration`` at whichever comes first; without them an untimed message
    waits for the control ``control`` (``"any"``: any control).
    """

    type: ClassVar[str] = "message"
    text: str
    duration: float | None
    #: the control that dismisses an untimed message without ``controls_to_dismiss``
    control: str
    controls_to_dismiss: tuple[str, ...] | None
    align: str

    @classmethod
    def from_dict(cls, d: dict, index: int) -> Message:
        dismiss = d.get("controls_to_dismiss")
        return cls(**cls._shared(d), text=join_text(d.get("text", "")),
                   duration=d.get("duration"), control=d.get("control", "SPACE"),
                   controls_to_dismiss=None if dismiss is None else tuple(dismiss),
                   align=d.get("align", "center"))

    @staticmethod
    def problems(d: dict) -> list[str]:
        out = _dismiss_problems("controls_to_dismiss", d.get("controls_to_dismiss"))
        control = d.get("control", "SPACE")
        if control != "any" and not _is_possible_on_any_input_device(control):
            out.append(f'control: expected a control name ({_NAMES_ARE}) or "any", '
                       f"got {control!r}")
        if d.get("align", "center") not in ("center", "left"):
            out.append(f"align: expected 'center' or 'left', got {d['align']!r}")
        out += [f"text: {p}" for p in image_line_problems(join_text(d.get("text", "")))]
        return out + _positive_problems(d, ("duration",))


@dataclass(frozen=True, slots=True)
class Instructions:
    """The screen before a game block: a heading with its title over its text."""

    title: str
    text: str
    align: str
    duration: float | None
    controls_to_dismiss: tuple[str, ...] | None

    @classmethod
    def from_dict(cls, d: dict) -> Instructions:
        dismiss = d.get("controls_to_dismiss")
        return cls(title=d["title"], text=join_text(d.get("text", "")),
                   align=d.get("align", "center"), duration=d.get("duration"),
                   controls_to_dismiss=None if dismiss is None else tuple(dismiss))

    def message(self, resumed: bool) -> Message:
        """As the message shown, its heading marking a block a run carries on."""
        heading = f"{'Resuming:' if resumed else 'Starting:'} {self.title}"
        return Message(spec={}, include_on_resume=False, text=f"{heading}\n\n{self.text}",
                       duration=self.duration, control="SPACE",
                       controls_to_dismiss=self.controls_to_dismiss, align=self.align)

    @staticmethod
    def problems(spec: Any) -> list[str]:
        if not isinstance(spec, dict):
            return [f"instructions: expected an object, got {spec!r}"]
        out = []
        unknown = set(spec) - {"title", "text", "align", "duration", "controls_to_dismiss"}
        if unknown:
            out.append(f"instructions: unknown field(s) {sorted(unknown)}; the fields are title, "
                       "text, align, duration and controls_to_dismiss")
        if not isinstance(spec.get("title"), str) or not spec["title"]:
            out.append("instructions: title: missing (the screen's heading names the game)")
        if spec.get("align", "center") not in ("center", "left"):
            out.append(f"instructions: align: expected 'center' or 'left', got {spec['align']!r}")
        out += [f"instructions: {p}" for p in _positive_problems(spec, ("duration",))]
        out += [f"instructions: text: {p}"
                for p in image_line_problems(join_text(spec.get("text", "")))]
        dismiss = spec.get("controls_to_dismiss")
        out.extend(_dismiss_problems("instructions: controls_to_dismiss", dismiss))
        if spec.get("duration") is None and dismiss is None:
            out.append("instructions: give a duration, controls_to_dismiss, or both")
        return out


@dataclass(frozen=True, slots=True)
class Survey(Phase):
    """Likert questions, each answered with one press of the control of a point."""

    type: ClassVar[str] = "survey"
    questions: tuple[str, ...]
    #: one control per point of the scale, from its lowest
    controls: tuple[str, ...]

    @classmethod
    def from_dict(cls, d: dict, index: int) -> Survey:
        return cls(**cls._shared(d), questions=tuple(d.get("questions", ())),
                   controls=tuple(d.get("controls", [str(i) for i in range(1, 8)])))

    @staticmethod
    def problems(d: dict) -> list[str]:
        questions = d.get("questions", [])
        out = [] if isinstance(questions, list) and all(isinstance(q, str) for q in questions) \
            else [f"questions: expected a list of questions, got {questions!r}"]
        if "n_points" in d:
            out.append('"n_points" is gone: list one control per point under "controls"')
        controls = d.get("controls", ["1", "2"])
        if not (isinstance(controls, list) and len(controls) >= 2
                and len(set(map(str, controls))) == len(controls)
                and all(_is_possible_on_any_input_device(c) for c in controls)):
            out.append(f"controls: expected a list of two or more distinct control names, one "
                       f"per point from the lowest (names {_NAMES_ARE}), got {controls!r}")
        return out


@dataclass(frozen=True, slots=True)
class GamePhase(Phase):
    """A game block: the env it plays, how long, at what rate, and what ends it.

    The controls a block maps (``controls``) and its menu are its spec's, read
    by the adapter's keymap and :class:`~fmri_gym.menu.Menu`; the backend's own
    keys stay in ``spec`` for the adapter.
    """

    type: ClassVar[str] = "game"
    backend: str
    #: the env id, as the backend reads it
    game: str
    mode: BlockMode
    #: seconds the block lasts (``mode`` ``duration``, default 30) or lasts
    #: at most (``mode`` ``episode``, default 300)
    duration: float
    #: episodes an ``episode`` block lasts
    n_episodes: int
    #: A blocking block: only an episode whose outcome is listed (usually
    #: ``["won"]``) counts toward ``n_episodes``; any other replays the same
    #: instance (the seed follows the count), so the subject stays on a level
    #: until they clear it, ``duration`` runs out, or they forfeit from the
    #: menu. ``None``: every episode counts, whatever its outcome.
    advance_on: tuple[str, ...] | None
    #: An episode ending in one of these outcomes sends the world back
    #: ``rewind_frames`` frames and play goes on, as a new episode from that
    #: state: session policy, not a rule of the game, so a model played on
    #: the same env ends as the game says. Empty: no ending is rewound.
    rewind_on: tuple[str, ...]
    #: how far back a rewound episode goes; ``None`` when nothing is rewound
    rewind_frames: int | None
    #: seconds the final score and outcome stay on screen after an episode
    show_outcome_duration: float
    #: the block's base seed: its episode ``e`` plays ``seed + e``; ``None`` until
    #: the run derives one (:meth:`~fmri_gym.config.curriculum.Curriculum.with_seeds`)
    seed: int | None
    #: a savestate every this many frames: 1 keeps one per frame, larger
    #: trades savestate density for disk (retro states are ~1 MB/frame)
    state_stride: int
    #: Turn-based games (grid worlds: FrozenLake, CliffWalking, Taxi, ...)
    #: advance ONE step per deliberate key PRESS, not once per frame. In a
    #: real-time loop they would auto-step every frame with the noop action
    #: (which for e.g. FrozenLake is action 0 = LEFT), so the agent "moves on
    #: its own" and a single held key fires many times. The standing frame is
    #: redrawn a few times a second while the block waits for a press, so the
    #: HUD's clock keeps time; that redraw is pixels only (no step, no log row,
    #: no trigger, no sound).
    is_turn_based: bool
    #: the game has sound to play, and the block plays it (with the rig's own
    #: ``has_audio``); muting never changes what is logged
    has_audio: bool
    #: the block's steps per second. Every game phase states its own: what
    #: the block plays at is the curriculum's business, not a default that
    #: changes with the engine underneath it.
    fps: float
    #: the screen before the block, if any
    instructions: Instructions | None
    #: the loading screen's label instead of the env id, if any
    text: str | None

    @property
    def frame_period(self) -> float:
        """Seconds per frame."""
        return 1.0 / self.fps

    @classmethod
    def from_dict(cls, d: dict, index: int) -> GamePhase:
        """Read a game phase's dict, filling in the defaults (:meth:`problems` checked it)."""
        mode = BlockMode(d.get("mode", "duration"))
        advance_on = d.get("advance_on")
        instructions = d.get("instructions")
        return cls(
            spec=d, include_on_resume=False,     # a game phase is never replayed on resume
            backend=d.get("backend", "gym"), game=d["game"], mode=mode,
            duration=d.get("duration", 30.0 if mode is BlockMode.DURATION else 300.0),
            n_episodes=d.get("n_episodes", 1),
            advance_on=None if advance_on is None else tuple(advance_on),
            rewind_on=tuple(d.get("rewind_on", ())),
            rewind_frames=d.get("rewind_frames"),
            show_outcome_duration=float(d.get("show_outcome_duration", 2.0)),
            seed=d.get("seed"), state_stride=max(1, int(d.get("state_stride", 1))),
            is_turn_based=d.get("is_turn_based", False),
            has_audio=d.get("has_audio", True),
            fps=d["fps"],
            instructions=None if instructions is None else Instructions.from_dict(instructions),
            text=d.get("text"))

    @staticmethod
    def problems(d: dict) -> list[str]:
        out = []
        if not d.get("game"):
            out.append("game: missing env id")
        if d.get("mode", "duration") not in [mode.value for mode in BlockMode]:
            out.append(f"mode: expected 'duration' or 'episode', got {d.get('mode')!r}")
        out += (_fps_problems(d) + _controls_problems(d) + _advance_problems(d)
                + _rewind_problems(d))
        out += _bool_problems(d, ("is_turn_based", "has_audio"))
        seconds = d.get("show_outcome_duration", 0)
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or seconds < 0:
            out.append(f"show_outcome_duration: expected seconds >= 0, got {seconds!r}")
        if "menu" in d:
            out.extend(menu_problems(d["menu"]))
        if "instructions" in d:
            out.extend(Instructions.problems(d["instructions"]))
        return out


@dataclass(frozen=True, slots=True)
class CheckDisplay(CheckPhase):
    """A rig check: are flips locked to the refresh; missed refreshes."""

    type: ClassVar[str] = "check_display"
    #: flip intervals to measure
    n: int
    #: flip for this long instead of ``n`` flips, if given
    seconds: float | None

    @classmethod
    def from_dict(cls, d: dict, index: int) -> CheckDisplay:
        return cls(**cls._shared(d), n=d.get("n", 60), seconds=d.get("seconds"))

    @staticmethod
    def problems(d: dict) -> list[str]:
        return _positive_problems(d, ("n", "seconds"))


@dataclass(frozen=True, slots=True)
class CheckFrames(CheckPhase):
    """A rig check: the test pattern played through the run's own game loop, per rate and load."""

    type: ClassVar[str] = "check_frames"
    rates: tuple[float, ...]
    loads: tuple[str, ...]
    #: seconds per rate and load
    seconds: float

    @classmethod
    def from_dict(cls, d: dict, index: int) -> CheckFrames:
        return cls(**cls._shared(d), rates=tuple(d.get("rates", (60,))),
                   loads=tuple(d.get("loads", ("none",))), seconds=d.get("seconds", 2.0))

    @staticmethod
    def problems(d: dict) -> list[str]:
        rates, loads = d.get("rates", [60]), d.get("loads", ["none"])
        out = _positive_problems(d, ("seconds",)) + _rig_owned(d, ("recording_hz",))
        if not (isinstance(rates, list) and rates and all(_positive(r) for r in rates)):
            out.append(f"rates: expected frames per second, e.g. [60, 30], got {rates!r}")
        if not (isinstance(loads, list) and loads and all(x in LOADS for x in loads)):
            out.append(f"loads: expected a list of {LOADS}, got {loads!r}")
        return out


@dataclass(frozen=True, slots=True)
class CheckTriggers(CheckPhase):
    """A rig check: every line and code sent on the rig's trigger line; scanner pulses counted."""

    type: ClassVar[str] = "check_triggers"
    #: passes of every line and code
    repeat: int
    hold_ms: float
    gap_ms: float
    #: scanner pulses to count, when the run waits for the scanner
    pulses: int
    #: count pulses for this long instead, if given
    seconds: float | None
    #: seconds allowed without a pulse
    timeout_s: float

    @classmethod
    def from_dict(cls, d: dict, index: int) -> CheckTriggers:
        return cls(**cls._shared(d), repeat=d.get("repeat", 1), hold_ms=d.get("hold_ms", 40),
                   gap_ms=d.get("gap_ms", 40), pulses=d.get("pulses", 3),
                   seconds=d.get("seconds"), timeout_s=d.get("timeout_s", 60.0))

    @staticmethod
    def problems(d: dict) -> list[str]:
        return _positive_problems(d, ("repeat", "hold_ms", "gap_ms", "pulses", "seconds",
                                      "timeout_s"))


@dataclass(frozen=True, slots=True)
class CheckControls(CheckPhase):
    """A rig check: each control of the rig's ``"inputs"`` and each of its
    ``"response_keys"`` (:mod:`fmri_gym.config.rig`), asked for on screen and
    pressed on the device."""

    type: ClassVar[str] = "check_controls"
    #: seconds to press each key
    timeout_s: float
    #: calibrate the stick axes of the rig's input devices first
    calibrates: bool

    @classmethod
    def from_dict(cls, d: dict, index: int) -> CheckControls:
        return cls(**cls._shared(d), timeout_s=d.get("timeout_s", 10.0),
                   calibrates=d.get("calibrates", True))

    @staticmethod
    def problems(d: dict) -> list[str]:
        out = _positive_problems(d, ("timeout_s",)) + _bool_problems(d, ("calibrates",))
        if "pad_reps" in d:
            out.append("pad_reps: the calibration asks for each control until it comes twice "
                       "in a row, with no count to set; remove it")
        return out + _rig_owned(d, ("controls", "keys"))


@dataclass(frozen=True, slots=True)
class CheckPhotodiode(CheckPhase):
    """A rig check: the flip-to-photon offset, and on the same flashes the audio test.

    Which of the two it measures, and with what, is the rig's ``"rig_check"``
    (:class:`~fmri_gym.config.rig.RigCheckConfig`): this is only how many flashes and
    how they are timed."""

    type: ClassVar[str] = "check_photodiode"
    #: the sound-card input's rate
    samplerate: int
    #: flashes
    n: int
    on_ms: float
    gap_min_ms: float
    gap_max_ms: float
    settle_ms: float
    seed: int

    @classmethod
    def from_dict(cls, d: dict, index: int) -> CheckPhotodiode:
        return cls(**cls._shared(d), samplerate=d.get("samplerate", 48000), n=d.get("n", 10),
                   on_ms=d.get("on_ms", 50.0), gap_min_ms=d.get("gap_min_ms", 150.0),
                   gap_max_ms=d.get("gap_max_ms", 250.0), settle_ms=d.get("settle_ms", 300.0),
                   seed=d.get("seed", 0))

    @staticmethod
    def problems(d: dict) -> list[str]:
        out = _positive_problems(d, ("n", "on_ms", "gap_min_ms", "gap_max_ms", "settle_ms",
                                     "samplerate"))
        out += _rig_owned(d, ("readout", "input_device", "has_mic", "has_audio", "corner",
                              "patch_px"))
        low, high = d.get("gap_min_ms", 150), d.get("gap_max_ms", 250)
        if _positive(low) and _positive(high) and low > high:
            out.append(f"gap_min_ms ({low}) is above gap_max_ms ({high})")
        return out


#: Each phase ``type`` and the class it is read as.
PHASE_CLASSES: dict[str, type[Phase]] = {cls.type: cls for cls in (
    Fixation, Message, Survey, GamePhase, CheckDisplay, CheckFrames, CheckTriggers,
    CheckControls, CheckPhotodiode)}
#: The rig check's phase classes (:mod:`fmri_gym.checks`).
CHECK_PHASES: tuple[type[CheckPhase], ...] = tuple(
    cls for cls in PHASE_CLASSES.values() if issubclass(cls, CheckPhase))


def phase_from_dict(d: dict, index: int) -> Phase:
    """The phase of ``d["type"]``, read off ``d`` (which :func:`phase_problems` passed)."""
    return PHASE_CLASSES[d["type"]].from_dict(d, index)


def phase_problems(d: dict) -> list[str]:
    """What keeps ``d`` from being a phase of its ``type``, as text."""
    return PHASE_CLASSES[d["type"]].problems(d)


# -- the checks the phases share ------------------------------------------------------


def _positive(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0


def _rig_owned(d: dict, keys: tuple[str, ...]) -> list[str]:
    """Keys a check phase may not set: they describe the rig, so its rig file says them."""
    return [f"{k}: this is the rig's, not the curriculum's: set it in the rig file "
            "(rig-config: the Rig check equipment step, or Response devices for response keys)"
            for k in keys if k in d]


def _positive_problems(d: dict, keys: tuple[str, ...]) -> list[str]:
    """The ``keys`` ``d`` sets that are not a positive number (``None``: not set)."""
    return [f"{k}: expected a positive number, got {d[k]!r}"
            for k in keys if d.get(k) is not None and not _positive(d[k])]


def _bool_problems(d: dict, keys: tuple[str, ...]) -> list[str]:
    return [f'"{k}" must be true or false, got {d[k]!r}'
            for k in keys if k in d and not isinstance(d[k], bool)]


#: Where a curriculum's control names are listed, for its messages.
_NAMES_ARE = ("as the keyboard's in fmri_gym/inputs/keyboard.py and each device's in "
              "fmri_gym/inputs/<device>.py; ESCAPE quits the run")


def _is_possible_on_any_input_device(name: Any) -> bool:
    """Whether a curriculum may map ``name``: a control of some device, not the run's quit."""
    return isinstance(name, str) and is_control_of_any_input_device(name) and name != ESCAPE


def _dismiss_problems(where: str, controls: Any) -> list[str]:
    """``controls_to_dismiss``, if given: a non-empty list of control names."""
    if controls is None or (isinstance(controls, list) and controls
                            and all(map(_is_possible_on_any_input_device, controls))):
        return []
    return [f"{where}: expected a list of control names ({_NAMES_ARE}), got {controls!r}"]


def _controls_problems(d: dict) -> list[str]:
    """``controls`` is required, and each name in it must be a control of some device.

    There is no default map, so the file states it in full. A name that is no
    control of any device can never be pressed, and the binding would be dead
    without a word; ``""`` is the no-control action. Whether the rig has the
    device is the run's check (:mod:`fmri_gym.config.run`); whether the values
    fit the env's action space is checked when the env is built
    (:mod:`fmri_gym.adapters.keymap`).
    """
    controls = d.get("controls")
    if not isinstance(controls, dict) or not controls:
        return ['controls: missing; map each control to the env action it sends, e.g. {"": 0, '
                '"LEFT": 3, "RIGHT": 2, "UP+SPACE": 5} ("" is nothing held; names '
                f"{_NAMES_ARE}; there is no default map)"]
    unknown = sorted({c for combo in controls if combo for c in combo.split("+")
                      if not _is_possible_on_any_input_device(c)})
    if unknown:
        return [f"controls: {unknown} are not control names; use those {_NAMES_ARE}"]
    return []


def _fps_problems(d: dict) -> list[str]:
    """``fps`` is required: a block's rate is the curriculum's, not the engine's."""
    fps = d.get("fps")
    if fps is None:
        return ["fps: missing; state the block's steps per second (the engine's own rate is "
                "not assumed: 60 for console cores and Atari, 35 / frame_skip for Doom)"]
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or fps <= 0:
        return [f"fps: expected a positive number of steps per second, got {fps!r}"]
    return []


def _advance_problems(d: dict) -> list[str]:
    """``advance_on`` gates ``n_episodes``, so it means nothing to a block that ends on the clock."""
    if "advance_on" not in d:
        return []
    names = d["advance_on"]
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        return [f"advance_on: expected a list of outcome names, got {names!r}"]
    if set(names) - OUTCOMES:
        return [f"advance_on: unknown {sorted(set(names) - OUTCOMES)}; the names are "
                f"{sorted(OUTCOMES)}"]
    if d.get("mode", "duration") != "episode":
        return ["advance_on: only in mode 'episode' (a duration block ends on the clock)"]
    return []


def _rewind_problems(d: dict) -> list[str]:
    """``rewind_on`` (outcomes) and ``rewind_frames`` (how far back) come as a pair."""
    if "rewind_on" not in d and "rewind_frames" not in d:
        return []
    names, frames = d.get("rewind_on"), d.get("rewind_frames")
    out = []
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        out.append(f"rewind_on: expected a list of outcome names, got {names!r}")
    elif set(names) - REWINDABLE:
        out.append(f"rewind_on: {sorted(set(names) - REWINDABLE)} cannot be rewound; the "
                   f"game's own endings are {sorted(REWINDABLE)}")
    if isinstance(frames, bool) or not isinstance(frames, int) or frames < 1:
        out.append(f"rewind_frames: expected the frames to go back, >= 1, got {frames!r}")
    return out
