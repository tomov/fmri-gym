"""A growing chat for one block: the model's own past turns stay in view as text.
"""

from __future__ import annotations

#: rough text-size proxy for the truncation budget: no tokenizer call, so this can run
#: on every frame without the cost of one.
CHARS_PER_TOKEN = 3.0


class Conversation:
    """One block's turns, oldest first, ahead of a system prompt that never changes.

    :param system: the system prompt, fixed for the whole conversation.
    :param token_budget: once the chat's estimated size passes this many tokens, the
        oldest turns are dropped; ``None`` or ``0`` never drops.
    :param min_turns: turns kept regardless of ``token_budget``.
    """

    def __init__(self, system: str, token_budget: int | None = None,
                min_turns: int = 10) -> None:
        self.system = system
        self.token_budget = token_budget or None
        self.min_turns = min_turns
        self._turns: list[tuple[list[dict], str]] = []
        self._chars = 0  # running total over _turns, kept in step by add()/_truncate()

    @property
    def has_turns(self) -> bool:
        return bool(self._turns)

    def chat(self, content: list[dict]) -> list[dict]:
        """The full chat to send for a new turn with ``content``, past turns first.

        :param content: this turn's content parts (images included); not yet recorded.
        :return: ``[system, *(user, assistant) pairs so far, user(content)]``.
        """
        messages = [{"role": "system", "content": self.system}]
        for user_content, reply in self._turns:
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": content})
        return messages

    def add(self, content: list[dict], reply: str) -> None:
        """Record a turn: its text (any image parts dropped) and the model's reply.

        :param content: the user content that was sent this turn.
        :param reply: the model's reply text, ``"(no reply)"`` for a dropped call.
        """
        stored = [part for part in content if part.get("type") != "image"]
        reply = reply or "(no reply)"
        self._turns.append((stored, reply))
        self._chars += _turn_chars(stored, reply)
        self._truncate()

    def _truncate(self) -> None:
        if self.token_budget is None:
            return
        while len(self._turns) > self.min_turns and self._size() > self.token_budget:
            stored, reply = self._turns.pop(0)
            self._chars -= _turn_chars(stored, reply)

    def _size(self) -> int:
        return int((len(self.system) + self._chars) / CHARS_PER_TOKEN)


def _turn_chars(stored: list[dict], reply: str) -> int:
    """A stored turn's text size, in characters: its text parts plus the reply."""
    return sum(len(part.get("text", "")) for part in stored) + len(reply)
