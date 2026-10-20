"""What the harness needs from a language model: its internal state over a chat, or its reply.

Two things a model can do here, and not every backend can do both. ``encode``
reads a chat in one forward pass and returns hidden states, which needs the
weights in this process (``transformers``, the DeepSeek runtimes). ``generate``
answers a chat with text, which any model can (``vllm``, ``openrouter`` for the
models only an API serves). A backend says which it does with
:attr:`LRMBackend.can_encode` / :attr:`LRMBackend.can_generate`, so a run that
needs states refuses an API model before it starts, not after.

The internal state of a model reading a turn is taken at one token: the last
token of the assistant's turn header, the position the model starts its reply
from. Everything the model has read about the game by then -- the rules, the
history, this observation -- and nothing of the answer is in it. It is the
header and not the token just before the action because a chat template may
put something between the two on some turns only (Qwen's empty reasoning block
on the last one), and every turn's state should sit on the same kind of token
(:mod:`.chat` finds it). Layer ``0`` is the token embeddings and layer ``i``
the residual stream after decoder block ``i``, before any final norm.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Reply(NamedTuple):
    """One generated answer, and what it cost."""

    #: the visible answer
    text: str
    #: the reasoning the model returned beside it, ``""`` if none
    reasoning: str
    input_tokens: int
    output_tokens: int
    #: what the call was billed, in USD, for an API model; ``None`` for a local one
    cost: float | None = None


class LRMBackend:
    """A language model behind one of the two calls; subclasses implement what they can."""

    #: the name :func:`~agents.lrm.backends.get_backend` builds it by
    name: str = ""
    #: :meth:`encode` returns hidden states
    can_encode: bool = False
    #: :meth:`generate` returns a reply
    can_generate: bool = False

    #: the checkpoint or API model id
    model_id: str

    @property
    def n_layers(self) -> int:
        """Hidden-state layers :meth:`encode` can return: the embeddings (0) and each block."""
        raise NotImplementedError(f"the {self.name} backend has no hidden states")

    @property
    def is_writer(self) -> bool:
        """Whether this process writes results: under ``torchrun``, rank 0 only."""
        return True

    def encode(self, chat: list[dict], targets: list[int], layers: list[int]) -> np.ndarray:
        """Read ``chat`` and return the hidden state at each target message's header.

        :param chat: ``[{"role", "content"}, ...]`` messages.
        :param targets: indices into ``chat`` of assistant messages.
        :param layers: which hidden-state layers.
        :return: ``(len(targets), len(layers), hidden_size)`` float16.
        """
        raise NotImplementedError(f"the {self.name} backend returns text, not activations")

    def generate(self, chat: list[dict]) -> Reply:
        """Answer ``chat`` (which ends on a user message) with the next assistant message."""
        raise NotImplementedError(f"the {self.name} backend does not generate")

    def record(self) -> dict:
        """What made this backend's output, for the files it is written to."""
        return {"backend": self.name, "model": self.model_id}
