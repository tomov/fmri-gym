"""A subject's block, replayed into the turns of a conversation with a language model.

A block's log holds each episode's seed and every action (``events.jsonl``),
which rebuilds the game frame for frame (:mod:`fmri_gym.replay`). This replays
it once more, through the block's own adapter, and at every ``stride``-th frame
asks the backend's :class:`~agents.lrm.symbolic.SymbolicReader` what the player
was looking at just before the logged action. That moment and that action are
one :class:`Turn`; a block's turns, with a :class:`TranscriptHeader`, are its
transcript.

The transcript is a file (``transcript.jsonl``: the header, then one turn per
line) because the two halves of the pipeline need different machines: the
replay needs the game installed, the model needs a GPU. Nothing downstream
opens the block again.

The replay is held to the log: every step's reward and ``terminated`` must be
the logged one, or it stops, because the turns after a divergence would
describe a game the subject never played.

Usage::

    python -m agents.lrm.transcript --block data/<run>/block-01_vizdoom_VizdoomDefendCenter-v1
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .symbolic import Observation, SymbolicReader, get_reader

if TYPE_CHECKING:
    from fmri_gym.adapters.base import EnvAdapter

#: Per-game rules for the system prompt: ``<backend>/<game>.txt``.
RULES_DIR = Path(__file__).parent / "prompts" / "game_rules"


@dataclass(frozen=True)
class TranscriptHeader:
    """What a transcript was made from, and what every turn of it shares.

    :param block: the block's folder, as given.
    :param backend: the block's backend.
    :param game: the block's game.
    :param subject: who played the block.
    :param reader: the :class:`SymbolicReader` that wrote the observations.
    :param stride: one turn every this many frames.
    :param actions: every action name a turn can carry.
    :param rules: the game's rules, for the system prompt.
    :param rules_source: where ``rules`` came from: a rules file, or the
        instructions the subject read.
    """

    block: str
    backend: str
    game: str
    subject: str
    reader: str
    stride: int
    actions: tuple[str, ...]
    rules: str
    rules_source: str

    def record(self) -> dict:
        return {"type": "header", **{k: getattr(self, k) for k in self.__dataclass_fields__},
                "actions": list(self.actions)}

    @classmethod
    def from_dict(cls, d: dict) -> TranscriptHeader:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k != "actions"},
                   actions=tuple(d["actions"]))


@dataclass(frozen=True)
class Turn:
    """One sampled frame: what the player saw, and the key they pressed on it.

    :param episode_id: the episode, as logged.
    :param ep_frame: the frame within the episode.
    :param frame: the frame within the block (the log's ``frame``), which
        lines the turn up with the frames, and the scanner, of the block.
    :param run_time: the frame's logged ``run_time``.
    :param observation: the state before the action.
    :param action: the action's name (``"TURN_LEFT + ATTACK"``).
    :param reward: what that action earned.
    """

    episode_id: int
    ep_frame: int
    frame: int
    run_time: float
    observation: Observation
    action: str
    reward: float

    def record(self) -> dict:
        return {"type": "turn", "episode_id": self.episode_id, "ep_frame": self.ep_frame,
                "frame": self.frame, "run_time": self.run_time,
                "observation": self.observation.record(), "action": self.action,
                "reward": self.reward}

    @classmethod
    def from_dict(cls, d: dict) -> Turn:
        return cls(d["episode_id"], d["ep_frame"], d["frame"], d["run_time"],
                   Observation.from_dict(d["observation"]), d["action"], d["reward"])


def transcribe(block: str, stride: int = 1) -> tuple[TranscriptHeader, list[Turn]]:
    """Replay every episode of a block into turns.

    :param block: the block's folder.
    :param stride: one turn every this many frames of an episode (its
        ``ep_frame`` 0, ``stride``, ...); the actions in between are played
        but not shown to the model.
    :return: the header and the turns, in log order.
    :raises RuntimeError: if the replay leaves the log.
    """
    # fmri_gym is imported here and not at the top: the package brings the run
    # loop and its audio stack, which the replay needs and a reader of a
    # transcript (the GPU side) has no use for.
    from fmri_gym import get_adapter
    from fmri_gym.logging import read_events
    if stride < 1:
        raise ValueError(f"stride must be at least 1, not {stride}")
    events = read_events(block)
    start = events[0]
    reader = get_reader(start["backend"])
    adapter = get_adapter(start["backend"], reader.replay_spec(start["phase"]))
    try:
        rules, rules_source = _rules(start)
        header = TranscriptHeader(block, start["backend"], start["game"], start["subject"],
                                  reader.name, stride, reader.action_names(adapter),
                                  rules, rules_source)
        episodes = sorted({e["episode_id"] for e in events if e["type"] == "episode_start"})
        block_end = _block_end(start, events)
        turns = [t for ep in episodes
                 for t in _episode(block, ep, events, adapter, reader, stride, block_end)]
    finally:
        adapter.close()
    return header, turns


def _episode(block: str, episode_id: int, events: list[dict], adapter: EnvAdapter,
             reader: SymbolicReader, stride: int, block_end: float) -> list[Turn]:
    """One episode's turns, replayed from its seed (and starting savestate)."""
    from fmri_gym.replay import reconstruction_plan
    plan = reconstruction_plan(block, episode_id)
    observation, info = adapter.reset(plan.seed)
    if plan.initial_state is not None:
        adapter.restore(plan.initial_state)
    reader.reset(adapter)
    variables = adapter.capture(observation, info, want_blob=False).variables
    frames = [e for e in events if e["type"] == "frame" and e["episode_id"] == episode_id]
    turns, score = [], 0.0
    for f in frames:
        if f["ep_frame"] % stride == 0:
            screen = _screen_text(adapter, score, block_end - f["run_time"])
            turns.append(Turn(episode_id, f["ep_frame"], f["frame"], f["run_time"],
                              reader.observe(adapter, variables, score, screen),
                              reader.action_name(adapter, f["action"]), float(f["reward"])))
        observation, reward, terminated, _, info = adapter.step(f["action"])
        _check(f, float(reward), bool(terminated))
        reader.after_step(adapter)
        variables = adapter.capture(observation, info, want_blob=False).variables
        score += float(reward)
    return turns


def _block_end(start: dict, events: list[dict]) -> float:
    """The ``run_time`` the block's clock ran out at, for the time left the subject saw.

    A block lasts its phase's ``duration`` from its start, which the log does
    not stamp; its first frame follows it by the first reset, a fraction of a
    second, which the whole seconds shown absorb. A block resumed mid-way, or
    cut short by the run's own end, ran on a shorter clock than this.
    """
    from fmri_gym.config.phases import GamePhase
    first = next(e for e in events if e["type"] == "frame")
    return first["run_time"] + GamePhase.from_dict(start["phase"], start["block_index"]).duration


def _screen_text(adapter: EnvAdapter, score: float, time_left: float) -> tuple[str, ...]:
    """The lines the subject read beside and over the frame, from the calls the run makes."""
    overlay = adapter.overlay()
    return tuple(adapter.hud(score, time_left) or ()) + tuple(overlay[0] if overlay else ())


def _check(logged: dict, reward: float, terminated: bool) -> None:
    """Raise if a replayed step is not the logged one."""
    if reward == float(logged["reward"]) and terminated == logged["terminated"]:
        return
    raise RuntimeError(
        f"the replay left the log at episode {logged['episode_id']} frame "
        f"{logged['ep_frame']}: reward {reward} / terminated {terminated}, logged "
        f"{logged['reward']} / {logged['terminated']}. Check the backend replays "
        "(python -m tests.verify_replay) and that the reader's replay_spec changes "
        "nothing the game plays by.")


def _rules(start: dict) -> tuple[str, str]:
    """The game's rules text: its rules file, or else the subject's instructions.

    :param start: the block's ``block_start`` record.
    :return: ``(rules, source)``, the source a path or ``"instructions"``.
    :raises ValueError: if the game has neither.
    """
    path = RULES_DIR / start["backend"] / f"{start['game']}.txt"
    if path.exists():
        return path.read_text().strip(), str(path.relative_to(RULES_DIR))
    text = start["phase"].get("instructions", {}).get("text")
    if not text:
        raise ValueError(f"no rules for {start['backend']}/{start['game']}: write "
                         f"{path} (the block's phase has no instructions to fall back on)")
    return "\n".join(text).strip(), "instructions"


def write_transcript(path: str, header: TranscriptHeader, turns: list[Turn]) -> None:
    """Write a transcript: the header line, then one line per turn."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for record in [header.record()] + [t.record() for t in turns]:
            f.write(json.dumps(record) + "\n")


def read_transcript(path: str) -> tuple[TranscriptHeader, list[Turn]]:
    """Inverse of :func:`write_transcript`."""
    with open(path) as f:
        records = [json.loads(line) for line in f]
    if not records or records[0].get("type") != "header":
        raise ValueError(f"{path}: not a transcript (its first line is not a header)")
    return TranscriptHeader.from_dict(records[0]), [Turn.from_dict(r) for r in records[1:]]


def main() -> None:
    p = argparse.ArgumentParser(description="Replay a logged block into an LRM transcript.")
    p.add_argument("--block", required=True, help="the block's folder")
    p.add_argument("--stride", type=int, default=1, help="one turn every this many frames")
    p.add_argument("--out", help="default: <block>/lrm/transcript_stride<N>.jsonl")
    p.add_argument("--show", type=int, default=2,
                   help="print this many turns as the model reads them")
    args = p.parse_args()

    from .prompting import show
    header, turns = transcribe(args.block, args.stride)
    out = args.out or os.path.join(args.block, "lrm", f"transcript_stride{args.stride}.jsonl")
    write_transcript(out, header, turns)
    print(f"{len(turns)} turns ({header.reader} reader, rules from {header.rules_source}) -> {out}")
    print(show(turns[:args.show]))


if __name__ == "__main__":
    main()
