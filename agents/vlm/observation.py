"""One turn's chat content: the frame, the HUD lines, and the game's state as text."""

from __future__ import annotations

import numpy as np


def frame_content(frame: np.ndarray, status: list[str], state: str,
                  reply_instruction: str = "Reply with exactly one key and nothing else."
                  ) -> list[dict]:
    """The current frame as one turn's content parts.

    :param frame: the RGB frame a player would see.
    :param status: the HUD/overlay lines shown beside or over it, if any.
    :param state: the game's state as text (``""`` for none).
    :param reply_instruction: the turn's last line, saying what to reply.
    :return: ``[{"type": "image", ...}, {"type": "text", ...}]``.
    """
    from PIL import Image

    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    shown = " | ".join(status) if status else "not shown"
    lines = [f"Status shown beside the screen: {shown}", state, reply_instruction]
    return [{"type": "image", "image": image},
            {"type": "text", "text": "\n".join(line for line in lines if line)}]
