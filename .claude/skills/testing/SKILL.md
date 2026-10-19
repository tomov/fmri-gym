---
name: testing
description: Run fmri-gym's test suite (pytest under tests/) before pushing a branch or opening a PR, pick the backends to play, read its skips and failures, and know what CI runs on a PR. Use when asked to test or verify a change, to check that a game plays or replays, or before any push.
---

# Testing a change

The suite is `tests/` (`tests/README.md` has the full map). It plays each selected game backend through the chain a scanner session depends on: the adapter steps its keys and renders, a block recorded by the random policy has the documented log shape (lines, `frames.h5`, `audio.h5` when the engine makes sound), that block replays bit for bit in a second process, and the real `fmri-play` loop runs it headless.

## Before pushing

```bash
uv run pytest --changed            # the backends your branch touches, against origin/main
uv run ruff check fmri_gym fmri_play.py agents tests gym/*/tests
```

Read the summary. A skip says what this machine lacks and how to get it (`uv sync --extra X`, `git submodule update --init`, install Chrome, fetch game files). A backend whose own files you changed does not skip: it fails until you can play it. If you cannot install it here, say so in the PR and let CI play it.

`--changed` selects by the files that differ from `origin/main`, committed or not. Name backends instead with `--backend crafter,minihack`; `--backend none` is the engine-free tests alone. Plain `pytest` is everything this machine can play (long: about 15 s of headless run per smoke config). A core change under `--changed` checks every backend's adapter, recording and replay, but plays the real loop headless only for the backends it touches directly and for `LOOP_SAMPLE` in `tests/backends.py` (gym, minihack, vizdoom); the full sweep is CI's. Tests run one at a time, so the suite leaves the rest of the machine to whoever is using it. Headless means SDL's dummy video driver: a bug only a real display shows (a window that does not open, vsync, a monitor) is not caught, here or in CI; add `--window` to run the headless runs in real windows, one at a time, when a change touches the display.

## Reading a failure

- **playable** failed: the adapter broke the contract in `fmri_gym/adapters/base.py` (frame not uint8 RGB, variables not JSON-able, sound not `(samples, channels)`, an outcome outside the vocabulary). Fix the adapter or the env under `gym/`, never the test's expectation.
- **recorded** failed: the log's shape changed. `fmri_gym/logging.py`'s docstring is the contract; the recorder is `agents/agent_play.py`.
- **replays** failed: the engine is not deterministic across processes, or `capture`/`restore`/`sound` disagree with what was logged. Run it alone for the first mismatch:
  ```bash
  uv run python -m tests.verify_replay --curriculum <config> --max-frames 200 --n-episodes 2
  ```
  The usual culprits are listed in the `add-backend` skill (hash order, a `render()` that draws from the engine's RNG, a reused buffer logged late).
- **runs** failed: the real loop could not start or logged nothing. Its stderr is in the assertion message; reproduce with `uv run python -m tests.headless --curriculum <config> --data-root /tmp/h --duration 25` and look in the run folder it prints.

## What CI runs

`.github/workflows/tests.yml` runs on a PR when it is opened, marked ready for review or pushed to; a draft runs nothing. A `core` job lints and runs the engine-free tests plus the `gym` backend; then one job per backend a Linux runner can play, with only that backend's extra synced and `--strict-backends --thorough --no-audio`: every config of the backend, and no skipping. Watch it with `gh pr checks <number> --watch`; a failed backend job keeps its run folders as an artifact.

## When you add a backend or a config

A new backend needs a `Backend` row in `tests/backends.py` (`test_configs` fails until it has one) and a line in the workflow's matrix unless it has a `ci` reason. A new config is picked up on its own: `test_configs` validates every file under `configs/`, and `--thorough` plays it.
