"""Policies that play a game block in place of a person.

The rig exists to compare humans against models on the same game. That needs the
model's blocks logged exactly like the subject's, so everything except the choice
of action is shared: same curriculum JSON, same EnvAdapter, same Logger, same
npz schema. The policy is the one piece that differs, and it lives here rather
than in an adapter because it is not part of any game.

Keep policies free of game knowledge. A policy sees the frame the display would
have shown, the key table the subject was taught, and whatever the subject could
read off the screen -- nothing else. The moment one is handed crafter-specific
hints it is no longer playing the game the human played, and the comparison the
rig is built for stops meaning anything.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.request
from typing import Any

import numpy as np

API_URL = "https://api.anthropic.com/v1/messages"


class Policy:
    """Chooses one action per frame from what a subject would have seen."""

    #: set by a policy that cannot always parse its own output
    invalid: int = 0

    #: set by a policy whose choice can fail to arrive at all (a network one)
    dropped: int = 0

    def reset(self) -> None:
        """Forget per-episode history; called once before each episode."""

    def act(self, frame: np.ndarray, overlay: list[str] | None) -> Any:
        """Return the action to send this frame.

        :param frame: the RGB frame the display would have shown.
        :param overlay: status lines the subject would have read beside it.
        :return: an action in whatever shape the adapter's keyspec produces.
        """
        raise NotImplementedError


class RandomPolicy(Policy):
    """Uniform over every key the subject can press, plus the noop.

    Not a baseline anyone reports: this is the pipeline test. It exercises the
    whole path (adapter, capture, savestates, npz) at full speed and with no
    network, so a schema mismatch shows up before a single model token is spent.

    :param actions: the actions to draw from, usually a keyspec's values.
    :param seed: RNG seed, so a model-free block is itself reproducible.
    """

    def __init__(self, actions: list, seed: int = 0) -> None:
        self.actions = list(actions)
        self.rng = np.random.RandomState(seed)

    def act(self, frame: np.ndarray, overlay: list[str] | None) -> Any:
        return self.actions[self.rng.randint(len(self.actions))]


class VLMPolicy(Policy):
    """Ask a vision model for the next key, given the last few frames.

    One asymmetry has to be declared rather than discovered: the subject watches
    a continuous 2.5 fps stream and feels their own key history, while the model
    gets a handful of stills and a list. ``history`` is that dial. Fix it in the
    config and report it; tuning it after seeing scores turns it into a free
    parameter that the human side does not have.

    The key is read from ``ANTHROPIC_API_KEY`` once, here, rather than per
    request: a block that plays 750 frames of noop because every call came back
    401 has spent the run and logged a model that chose to stand still.

    :param menu: ``{key name: action}``, the table the subject was taught.
    :param model: model id to query.
    :param history: how many recent frames (and past keys) the model sees.
    :param noop: action sent when the reply names no key in ``menu``.
    :param max_tokens: cap on the reply; the answer is one word.
    """

    def __init__(self, menu: dict, model: str, history: int = 4,
                 noop: Any = 0, max_tokens: int = 16) -> None:
        self.menu = dict(menu)
        self.model = model
        self.history = history
        self.noop = noop
        self.max_tokens = max_tokens
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "a vlm policy needs ANTHROPIC_API_KEY in the environment. "
                "Without it every call is refused and the block records a run "
                "of noops, which reads like a model that chose to stand still.")
        self.frames: list[np.ndarray] = []
        self.keys: list[str] = []

    def reset(self) -> None:
        self.frames = []
        self.keys = []

    def act(self, frame: np.ndarray, overlay: list[str] | None) -> Any:
        self.frames = (self.frames + [frame])[-self.history:]
        reply = self._ask(self._content(overlay))
        key = reply.strip().upper().strip(".'\"`")
        self.keys = (self.keys + [key if key in self.menu else "?"])[-self.history:]
        if key not in self.menu:
            self.invalid += 1
            return self.noop
        return self.menu[key]

    def _content(self, overlay: list[str] | None) -> list[dict]:
        """Build the user turn: the recent frames, then the instructions."""
        blocks: list[dict] = []
        for i, frame in enumerate(self.frames):
            age = len(self.frames) - 1 - i
            label = "current frame" if not age else f"{age} frame(s) ago"
            blocks.append({"type": "text", "text": label})
            blocks.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/png",
                "data": _png(frame)}})
        keys = ", ".join(self.keys) or "none yet"
        status = " | ".join(overlay) if overlay else "not shown"
        blocks.append({"type": "text", "text": (
            "You are playing an open-world survival game, one key press per "
            "turn.\nKeys you may press: " + ", ".join(sorted(self.menu))
            + f"\nYour last keys, oldest first: {keys}"
            + f"\nStatus shown beside the screen: {status}"
            + "\nReply with exactly one key from the list and nothing else.")})
        return blocks

    def _ask(self, content: list[dict]) -> str:
        """POST one turn to the messages API and return its text.

        :param content: the content blocks of the single user message.
        :return: the reply text, or ``""`` if the call or the shape failed.
        """
        body = json.dumps({"model": self.model, "max_tokens": self.max_tokens,
                           "messages": [{"role": "user", "content": content}]})
        req = urllib.request.Request(API_URL, data=body.encode(), headers={
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": self.api_key})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            return "".join(b.get("text", "") for b in data.get("content", []))
        except (OSError, ValueError) as e:
            # A dropped call must not end the block: it costs one frame, which
            # the noop below records honestly. But it must not pass for a choice
            # either, so it is counted separately from an unparsable reply and
            # said out loud the first time -- a proxy that is not there would
            # otherwise look like a model standing still on purpose.
            self.dropped += 1
            if self.dropped == 1:
                print(f"vlm policy: call failed ({e}); this frame is a noop. "
                      "Later failures are counted, not printed.")
            return ""


def _png(frame: np.ndarray) -> str:
    """Encode one RGB frame as base64 PNG.

    :param frame: ``(H, W, 3)`` uint8 array.
    :return: the base64 payload the messages API expects.
    """
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
