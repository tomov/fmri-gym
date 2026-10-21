"""What the harness needs from a model: its internal state over a chat, or its reply.

encode() needs the weights in this process; generate() works for an API model too.
A message's content is a string, or a list of {"type": "text"/"image", ...} parts for
an image-bearing turn; can_see_images says whether a backend reads or refuses that.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Reply(NamedTuple):
    text: str
    reasoning: str
    input_tokens: int
    output_tokens: int
    cost: float | None = None


def has_images(chat: list[dict]) -> bool:
    return any(isinstance(m.get("content"), list)
               and any(part.get("type") == "image" for part in m["content"])
               for m in chat)


class VLMBackend:
    name: str = ""
    can_encode: bool = False
    can_generate: bool = False
    can_see_images: bool = False
    model_id: str

    @property
    def n_layers(self) -> int:
        raise NotImplementedError(f"the {self.name} backend has no hidden states")

    @property
    def is_writer(self) -> bool:
        return True

    def encode(self, chat: list[dict], targets: list[int], layers: list[int]) -> np.ndarray:
        raise NotImplementedError(f"the {self.name} backend returns text, not activations")

    def generate(self, chat: list[dict]) -> Reply:
        raise NotImplementedError(f"the {self.name} backend does not generate")

    def record(self) -> dict:
        return {"backend": self.name, "model": self.model_id}

    def _refuse_images_unless_supported(self, chat: list[dict]) -> None:
        if not self.can_see_images and has_images(chat):
            raise ValueError(f"the {self.name} backend ({self.model_id}) does not read "
                             "images, but this chat has an image content part")
