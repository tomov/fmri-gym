"""A new run's starting checkpoint, read from existing run and block logs."""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path

from .bids import subject_label, task_label
from .logging import read_events, read_state


@dataclass(frozen=True, slots=True)
class CampaignCheckpoint:
    """The saved episode plus the recorded phase it belongs to."""

    run: str
    block: str
    phase: dict
    state: bytes

    def record(self) -> dict:
        """Provenance; the episode log separately stores the initial-state bytes."""
        return {"run": self.run, "block": self.block}


def _checkpoint_candidates(root: str, subject: str, curriculum_path: str, phase_index: int):
    """Existing block logs, including one whose writer stopped before block_end."""
    task = task_label(curriculum_path)
    pattern = str(Path(root) / subject / "ses-*" / "beh"
                  / f"{subject}_ses-*_task-{task}_run-*" / "manifest.json")
    candidates = []
    for name in glob.glob(pattern):
        manifest_path = Path(name)
        manifest = json.loads(manifest_path.read_text())
        for events_path in manifest_path.parent.glob(f"block-{phase_index:02d}_*/events.jsonl"):
            with events_path.open() as source:
                first = source.readline()
            start = (json.loads(first) if first.strip() else
                     {"type": "block_start", "block_index": phase_index})
            if start.get("type") == "block_start" and start.get("block_index") == phase_index:
                candidates.append((manifest.get("start_epoch") or 0,
                                   events_path.stat().st_mtime_ns, events_path, manifest, start))
    return candidates


def latest_checkpoint(root: str, subject: str, curriculum_path: str,
                      phase_index: int) -> CampaignCheckpoint:
    """Select the latest started block for this subject, task and phase; never fall back."""
    subject_label(subject)
    task = task_label(curriculum_path)
    candidates = _checkpoint_candidates(root, subject, curriculum_path, phase_index)
    if not candidates:
        raise ValueError(f"campaign: no previous checkpoint for {subject}, task {task}, "
                         f"phase {phase_index}; use --new-campaign to start one")
    _, _, path, manifest, start = max(candidates, key=lambda candidate: candidate[:2])
    events = read_events(str(path.parent))
    if not events or events[-1].get("type") != "block_end":
        raise ValueError(f"campaign: latest block {path.parent} did not close cleanly; "
                         "recover that recording before continuing, or use --new-campaign")
    end = events[-1]
    extra = end.get("extra") or {}
    state = extra.get("continuation_state", extra.get("state"))
    if state is None:
        raise ValueError(f"campaign: {path.parent} has no saved state; cannot continue")
    endings = [event for event in events if event.get("type") == "episode_end"]
    continues_after_exit = state != extra.get("state")
    if not endings or (endings[-1]["outcome"] not in ("playing", "quit")
                       and not continues_after_exit):
        raise ValueError(f"campaign: the episode in {path.parent} ended; "
                         "use --new-campaign for a new episode")
    return CampaignCheckpoint(manifest["run"]["label"], str(path.parent),
                              start["phase"], read_state(state))
