# Tests

The suite answers one question per backend: can a session be run on it and trusted afterwards? It plays each selected game through the same chain a scanner session depends on and refuses to vouch for a backend it could not play.

```bash
uv run pytest --changed                  # before pushing: the backends your branch touches
uv run pytest --backend crafter          # one backend, by name
uv run pytest                            # everything this machine can play (what CI does, spread over jobs)
uv run pytest --backend none             # the engine-free tests only
```

## What runs

| file | needs | checks |
|---|---|---|
| `test_configs.py` | nothing | every config under `configs/` validates as `fmri-play` would; every `get_adapter` backend has a row in `backends.py` |
| `test_core.py` | nothing | the logger's round trip (lines, pixels, savestates, sound), the keymaps, `validate_curriculum`, the change-to-backend map |
| `test_audio_stream.py` | nothing | the PortAudio callback splices queued chunks whole, once, in order, at the onset asked for |
| `test_backends.py` | the backend | per selected backend, on its smoke configs: **playable** (the adapter builds, resets, steps its keys, renders RGB, captures JSON-able variables and a savestate, sounds PCM shaped `(samples, channels)`), **recorded** (a block by the random policy writes `events.jsonl`, `frames.h5`, `audio.h5` of the documented shape), **replays** (the block rebuilt in a second process under another `PYTHONHASHSEED` matches bit for bit), **runs** (the real `fmri-play` loop plays it headless with a key script and leaves a run folder) |
| `test_crafter_cues.py` | crafter | the adapter's predicted sound cue against the engine's own outcome, press by press |
| `gym/minihack/tests/` | minihack | the `minihack-levels` package's own tests, collected when minihack is selected |

The repo root's `conftest.py` declares the suite's options (it has to be at the root: pytest reads an unknown option's value as a path, and `--backend gym` names the `gym/` directory). `backends.py` is the table: one row per backend with the extra that installs it, the import that proves it, the checkout, browser or game files it needs, its smoke configs, and why CI cannot play it, if it cannot. `changed.py` maps changed files to backends. `verify_replay.py` and `headless.py` are the two steps the backend tests drive, and both work on their own:

```bash
uv run python -m tests.verify_replay --curriculum configs/curricula/minihack/room5x5.json --max-frames 200
PYTHONHASHSEED=7 uv run python -m tests.verify_replay --check data/sub-01/ses-001/beh/<run>   # a subject's run
uv run python -m tests.headless --curriculum configs/curricula/vizdoom/basic.json --data-root /tmp/h --duration 25
uv run python -m tests.changed                                                           # which backends a branch touches
```

## Selecting backends

- `--changed [BASE]` (default base `origin/main`): the backends the branch's files touch, committed or not. An adapter, a `gym/<pkg>` package, a config or an `external/` checkout selects its backend *directly*; a change to the core (`fmri_gym/` outside the adapters, `fmri_play.py`, `agents/`, the dependencies, the tests) selects every backend.
- `--backend NAME[,NAME]` / `--backend all` / `--backend none`.
- Neither: every backend.

A selected backend this machine cannot play is **skipped, with the fix in the reason** (`-ra` prints them: `uv sync --extra X`, `git submodule update --init`, install Chrome, fetch the game files). It **fails** instead when it was named with `--backend`, when the change hit its own files, or under `--strict-backends` (CI): a suite that silently passed on a backend it never played would be the worst thing it could do (AGENTS.md, rule 4).

Other switches: `--thorough` plays every config of each selected backend through the quick check and runs the long sweeps (CI); `--no-audio` keeps the headless runs off the sound device (CI, or a machine without one; detected when the option is absent); `--frames N` and `--run-seconds S` size the recordings and the headless runs (15 s by default, 25 in CI; the game plays 10 s less, the run closing on its fixation); `--window` runs the headless runs in a real window instead of SDL's dummy driver, for a change a real display could break. A core change under `--changed` plays headless runs only for the backends it touches directly and for `LOOP_SAMPLE` (gym, minihack, vizdoom); every other selected backend still gets its adapter, recording and replay checks.

## CI

`.github/workflows/tests.yml` runs on every pull request that is not a draft, when it is opened, marked ready or pushed to: a `core` job (lint, the engine-free tests, the `gym` backend) and one job per backend a hosted Linux runner can play, each syncing only its own extra, with `--strict-backends --thorough --no-audio`. The run folders of a failed job are kept as an artifact for a week. `stk` (a GL display) and `retro` (ROMs) cannot run there and say so in `backends.py`.

## Adding a backend

Add a `Backend` row to `backends.py`: the extra, the module, one or two smoke configs (short ones; the suite plays their first game phase), the `gym_dirs` it runs on, the `external/` checkout and a file that proves it is there, and a `ci` reason if a hosted runner cannot play it. Add it to the workflow's matrix unless it has one. `test_configs` fails until the row exists.
