"""A growing chat for one episode: the model's own past turns stay in view.

"""

from __future__ import annotations


class Conversation:
    """One episode's turns, oldest first, ahead of a system prompt that never changes."""

    def __init__(self, system: str, max_turns: int) -> None:
        self.system = system
        self.max_turns = max_turns
        self._turns: list[tuple[list[dict], str]] = []

    def chat(self, content: list[dict]) -> list[dict]:
        """The full chat to send for a new turn with ``content``, past turns first.

        :param content: this turn's content parts (not yet recorded; see :meth:`add`).
        :return: ``[system, *(user, assistant) pairs so far, user(content)]``.
        """
        messages = [{"role": "system", "content": self.system}]
        for user_content, reply in self._turns:
            messages.append({"role": "user", "content": user_content})
            messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": content})
        return messages

    def add(self, content: list[dict], reply: str) -> None:
        """Record a turn's content and the model's reply, oldest dropped past ``max_turns``.

        :param content: the user content that was sent.
        :param reply: the model's reply text, ``""`` for a dropped or empty call.
        """
        self._turns.append((content, reply or "(no reply)"))
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns:]
