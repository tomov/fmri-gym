"""Replay-backed campaign checkpoints with immutable starting state for recordings."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import pickle
import secrets
import tempfile
import zlib

from replay_state_gym import ReplayState


def encode_state(state: bytes) -> str:
    """Encode a checkpoint using fmri-gym's existing savestate representation."""
    return base64.b64encode(zlib.compress(state, 1)).decode("ascii")


def decode_state(state: str) -> bytes:
    """Decode a checkpoint from a local campaign or recorded block."""
    return zlib.decompress(base64.b64decode(state, validate=True))


def save_campaign(path: Path, state: bytes) -> None:
    """Atomically save one checkpoint without replacing the prior file on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                         prefix="." + path.name + ".", delete=False) as output:
            temporary = Path(output.name)
            output.write(encode_state(state) + "\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class Campaign(ReplayState):
    """Reuse seed/action replay, binding snapshots to the exact engine and mode.

    :param env: native engine environment, with an isolated episode directory.
    :param initial_state: encoded immutable checkpoint to restore on reset.
    """

    def __init__(self, env, initial_state: str | None = None) -> None:
        super().__init__(env)
        self.initial_state = initial_state
        self._started = False
        self._save_anchor = None
        manifest = env.session.manifest
        self.fingerprint = {key: manifest[key] for key in
                            ("revision", "transport_sha256", "modern_sha256", "engine_sha256")}
        package = Path(__file__).resolve().parent
        frontend = b"".join((package / name).read_bytes() for name in
                            ("client.py", "controller.py", "rendering.py", "env.py", "campaign.py"))
        self.fingerprint["frontend_sha256"] = hashlib.sha256(frontend).hexdigest()

    def reset(self, *, seed=None, options=None):
        """Restore the configured campaign start or begin a seeded fresh episode."""
        if self.initial_state is not None:
            return self.set_state(decode_state(self.initial_state))
        if seed is None and self.env.session.player_directory is None:
            seed = secrets.randbits(32)
        result = super().reset(seed=seed, options=options)
        self._started = True
        self._save_anchor = None
        return result

    def step(self, action):
        """Record every press, retaining a continuation point before native Save."""
        result = super().step(action)
        if self.env.session.request.get("prompt") == "Really save?":
            anchor = pickle.loads(self.get_state())
            anchor["episode"]["actions"] = anchor["episode"]["actions"][:-1]
            self._save_anchor = pickle.dumps(anchor, protocol=5)
        elif self.env.session.request.get("kind") == "key":
            self._save_anchor = None
        return result

    def get_continuation_state(self) -> bytes:
        """Return the continuing campaign, retaining the position before Save/quit."""
        if self._save_anchor is not None and self.env.session.exit_text == "Be seeing you...":
            return self._save_anchor
        return self.get_state()

    def get_state(self) -> bytes:
        """Return a complete replay checkpoint, including the original seed/history."""
        if not self._started:
            raise RuntimeError("Reset before taking a campaign checkpoint")
        if self.env.session.player_directory is not None or self._episode["seed"] is None:
            raise ValueError("Replay checkpoints require an isolated, seeded episode")
        return pickle.dumps({"format": "nethack-desktop-campaign-1",
                             "fingerprint": self.fingerprint,
                             "game_mode": self.env.session.game_mode,
                             "episode": pickle.loads(super().get_state())}, protocol=5)

    def set_state(self, blob: bytes):
        """Restore a trusted checkpoint through the shared ReplayState mechanism."""
        checkpoint = pickle.loads(blob)
        if checkpoint.get("format") != "nethack-desktop-campaign-1":
            raise ValueError("Checkpoint is not from the Modern-capable native backend")
        if checkpoint["fingerprint"] != self.fingerprint:
            raise ValueError("Checkpoint engine differs; retain its matching source/build")
        if checkpoint["game_mode"] != self.env.session.game_mode:
            raise ValueError("Checkpoint game mode differs from the configured game_mode")
        if self.env.session.player_directory is not None:
            raise ValueError("Use campaign checkpoints with an isolated episode directory")
        # Replay consumes the exact input stream; only the final picture is displayed.
        self.env._replaying = True
        try:
            super().set_state(pickle.dumps(checkpoint["episode"], protocol=5))
        finally:
            self.env._replaying = False
        result = self.env._result()
        self._started = True
        self._save_anchor = None
        return result


def main() -> None:
    """Make a new session curriculum from a recorded block's final campaign state."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-block", type=Path, required=True,
                        help="recorded block directory containing events.jsonl")
    parser.add_argument("--curriculum", type=Path, required=True,
                        help="one-game native curriculum template (without includes)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = None
    for line in (args.from_block / "events.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event.get("type") == "block_end":
            extra = event.get("extra", {})
            checkpoint = extra.get("continuation_state", extra.get("state"))
    if checkpoint is None:
        raise ValueError("No block_end.extra.state found; use a completed native block")
    payload = pickle.loads(decode_state(checkpoint))
    if payload.get("format") != "nethack-desktop-campaign-1":
        raise ValueError("The block does not contain a compatible campaign checkpoint")
    curriculum = json.loads(args.curriculum.read_text())
    games = [phase for phase in curriculum["curriculum"] if phase.get("type") == "game"]
    if len(games) != 1 or games[0].get("backend") != "nethack_gamepad":
        raise ValueError("Use a template with exactly one nethack_gamepad game phase")
    settings = games[0].setdefault("env_kwargs", {})
    if settings.get("player_directory"):
        raise ValueError("A campaign curriculum cannot use a persistent native player directory")
    settings["game_mode"] = payload["game_mode"]
    settings["initial_state"] = checkpoint
    # Never overwrite the previous session's immutable starting configuration.
    with args.output.open("x") as output:
        json.dump(curriculum, output, indent=2)
        output.write("\n")
    print(f"Wrote continuing campaign curriculum: {args.output}")
