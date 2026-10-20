"""A transcript's turns as chat messages: the system prompt, then observation / action pairs.

The format is the one a model would play in: a ``user`` message per turn with
the observation, an ``assistant`` message with the action the subject took::

    [USER t=596]
    Observation:
    HUD: 287 s, Score: 1
    Inventory: [Ammo: 21, Health: 100%, Arms: 2, Armor: 0%, Weapon: Pistol, ...]
    Objects:
    - Demon: 12 deg left, 410 away
    Events: Your shots hit 1 time(s); You killed 1
    [ASSISTANT t=596]
    Action: TURN_LEFT + ATTACK

so the model reads the subject's game as if it had played it. One episode is
one conversation; an episode longer than the model's context is read in
overlapping :class:`Window` s (:func:`windows`).
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from .transcript import TranscriptHeader, Turn

SYSTEM_TEMPLATE = Path(__file__).parent / "prompts" / "system.txt"


class Window(NamedTuple):
    """A run of turns read in one pass, and which of them it reports.

    ``targets`` is the back of the window, so every reported turn has at least
    ``len(window) - len(targets)`` turns of history before it (all of them in
    the episode's first window, which has no earlier history to give).
    """

    #: first turn in the window (an index into the episode's turns)
    start: int
    #: one past the last
    end: int
    #: first turn whose state this window reports
    first_target: int


def system_prompt(header: TranscriptHeader) -> str:
    """The system message: the template with the game's rules and action names."""
    return SYSTEM_TEMPLATE.read_text().format(rules=header.rules,
                                              actions=", ".join(header.actions)).strip()


def observation_text(turn: Turn) -> str:
    """A turn's user message."""
    return "Observation:\n" + turn.observation.text()


def action_text(turn: Turn) -> str:
    """A turn's assistant message."""
    return f"Action: {turn.action}"


def messages(system: str, turns: list[Turn]) -> list[dict]:
    """The chat for ``turns``: the system message, then a user and an assistant message each.

    :param system: the system prompt.
    :param turns: consecutive turns of one episode.
    :return: ``[{"role", "content"}, ...]``; turn ``i``'s action is message ``2 + 2 * i``.
    """
    chat = [{"role": "system", "content": system}]
    for turn in turns:
        chat.append({"role": "user", "content": observation_text(turn)})
        chat.append({"role": "assistant", "content": action_text(turn)})
    return chat


def windows(n_turns: int, size: int) -> list[Window]:
    """Cover ``n_turns`` turns with windows of ``size``, each reporting its back half.

    Windows step by half their size, so every turn is reported exactly once
    and sees between ``size // 2`` and ``size - 1`` turns before it (fewer only
    at the start of the episode).

    :param n_turns: turns in the episode.
    :param size: turns per window (at least 2).
    :return: the windows, in order.
    """
    if size < 2:
        raise ValueError(f"a window holds at least 2 turns, not {size}")
    if n_turns <= size:
        return [Window(0, n_turns, 0)] if n_turns else []
    hop = size // 2
    out = [Window(0, size, 0)]
    while out[-1].end < n_turns:
        start = out[-1].start + hop
        out.append(Window(start, min(start + size, n_turns), out[-1].end))
    return out


def show(turns: list[Turn]) -> str:
    """Turns as a human reads a transcript: ``[USER t=..]`` / ``[ASSISTANT t=..]`` blocks."""
    return "\n".join(f"[USER  t={t.frame}]\n{observation_text(t)}\n"
                     f"[ASSISTANT t={t.frame}]\n{action_text(t)}" for t in turns)
