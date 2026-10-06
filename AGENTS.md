# Contributing to fmri-gym (humans and agents)

Please read this before writing code.

## The shape of the repo

```text
fmri_play.py                     CLI   parse the flags of one run, then play it (Run.from_config)
fmri_gym/run.py                  CORE  one run's experiment loop: trigger, phases, timing
fmri_gym/display.py              CORE  one pygame window: frames, text, fixation
fmri_gym/keys.py                 CORE  typed keys (pygame's names) to rig keys; what is held
fmri_gym/rig.py                  CORE  the rig keys, and the rig files (one per rig: what every run on it uses)
fmri_gym/logging.py              CORE  manifest.json + one crash-safe JSONL/HDF5 log per game block
fmri_gym/replay.py                     replay an episode from a block's log; frame fields as arrays
fmri_gym/menu.py                 CORE  hold-a-key pause menu: reset / forfeit / resume (opt-in per phase)
fmri_gym/adapters/base.py        CORE  EnvAdapter + FrameState: the seam
fmri_gym/adapters/keymap.py      CORE  the phase's keys, one Keymap per action space
fmri_gym/adapters/<BACKEND>.py   YOU   one small wrapper per game engine
configs/dbp_games/<GAME>.json    YOU   one curriculum per game
gym/<GAME>/                      YOU   a Gymnasium env for a game that ships none, or a rough one
external/<BACKEND>/              (gitignored) a game's own repo, cloned at the commit the README pins
```

The core is engine-agnostic: `run.py` never imports a game engine, never touches `env.unwrapped`, and never mentions a game by name. Everything engine-specific goes through an `EnvAdapter`.

Three layers, three directories, each optional until needed: `external/` is a game's own repo when its files must be read from a checkout (use only when `pip install` is not an option); `gym/` is a `gymnasium.Env` for a game that ships none (or a rough one); `fmri_gym/adapters/` is the fMRI adapter over whatever env results. Code goes in the lowest layer that owns the concern -- game rules in the game or its `gym/` env, never in the adapter.

## Rule 1: the game logic lives in the gym env, not the adapter

We evaluate AI models against the same gym environments that humans play in the scanner. If an adapter adds rules, affordances, scoring, or its own renderer, humans and models are no longer playing the same game and the comparison is void.

An adapter is lightweight glue that takes a `gymnasium.Env` and makes it fMRI-friendly. That env is the whole point: it is what the agents are trained and evaluated on, so a game that is not a `gymnasium.Env` yet is not ready for an adapter -- wrap it first. Ideally, an adapter should do only these things:

- build the env (`_make`): a `gymnasium.Env`, always. Prefer a ready-made pip package (`ale-py`, `stable-retro`, `minihack`, `rushhour-gym`, ...). If the game's own env speaks another API (old `gym`, a bare engine, a `with_img=` of its own), a thin Gymnasium env under `gym/` puts the contract in front of it (`gym/baba/`, `gym/crafter/`, `gym/vgdl/`, `gym/coom/`, `gym/baba_auto/`), and that is where any new one goes; the adapter never normalizes `reset`/`step` itself. Two rules for that env:
  - **Only the wrapper lives here.** The game's own code is a pip package, or a checkout in `external/<backend>` at the commit the README pins ("External checkouts"). The env reads that path by default and takes `repo=` for another; the pin is the README's.
  - **It is a plain RL env.** It follows the Gymnasium API and knows nothing about fMRI, scanners, phases, blocks, subjects or how it will be used; someone training an agent on it should find nothing odd. That vocabulary belongs to the adapter and the core.
- say in the module docstring what the env's action indices mean, so a config can write its `keys` (there is no default map: the config states all of it, from the rig keys)
- produce an RGB frame for the screen (`render`)
- if the engine makes sound, hand over this step's PCM (`sound`; the contract is in `EnvAdapter.sound`, and `retro.py` is a two-line example)
- if the engine has a clock of its own, say how many steps per second are real speed (`native_fps`), read from the engine where it tells; the run reports the block's speed against it
- pull out the analysis-relevant variables (`capture`)

If you find yourself writing game rules, drawing a board, tracking a selection cursor, computing legal moves, or sequencing trials inside an adapter -> pause. That likely belongs in the environment package (upstream, a fork, or a thin `gymnasium.Wrapper` shipped with the env), where the model evaluation harness gets it too.

`adapters/vizdoom.py` is a good example: one small action wrapper, then a class with a handful of short methods. `baba.py` and `minihack.py` are equally good, smaller examples.

**Budget:** a new adapter should ideally be under ~150 lines, one file (no new subpackage). Over that, ask whether the extra code belongs upstream in the env.

## Rule 2: changes to core libraries are a last resort, and stay generic

Before editing `run.py`, `display.py`, `keys.py`, `logging.py`, `base.py`, or `keymap.py`, please try to solve the problem in your adapter. If you cannot:

- Keep it **additive and default-off**, so no existing backend changes behaviour.
- Keep it **nameless**: no `if backend == "rushhour"`, no game ids, no engine imports. This holds for the prose too -- a core file's comments and docstrings state the contract, they don't cite the backend that happens to use it.
- Prefer the **opt-in capability** pattern already in use: the adapter sets a flag or defines an optional method, and core reads it defensively.
- Say so in the PR description. A core change is the part a reviewer must read closely.

## Rule 3: write for the next *human* to read

- **Each method should ideally fit on one screen** (~40 lines). If it doesn't, extract a helper with a name that says what it does. `run._episode` and `_game` are at the upper limit already; let's try to not to expand them, if possible.
- **Stay within two levels of nesting.** Use guard clauses and early `return`/`continue` instead of `else` ladders — see `_poll_keys_until` and `make_keymap`.
- **Module docstring explains *why*.** Every file here opens with the reasoning a newcomer needs: why this backend and not COOM, why `pixel_crop` and not `pixel`, what is deliberately not supported. Keep doing that; it is the most valuable text in the repo.
- **Docstrings on public methods** in the existing Sphinx style (`:param:`, `:return:`, `:raises:`). Short private helpers can get a one-liner.
- **Comments explain the trap, not the code.** The good ones here record a fact you cannot see from the source: why MiniHack needs an explicit `seed(core=, disp=)`, why classic control tears down the shared pygame window.
- **A docstring describes its own subject, not its callers.** Say what the class or function *is* and what it guarantees, in terms a reader who has only this file can check. A docstring that instead recites the order some other module calls the methods in ("construct after X is built, this one after each reset, that one once at the end") couples the two: either side can change and the text becomes a lie that nothing catches, and the reader still does not know what the thing holds. Document each method's own contract — what it opens, what it writes, what it flushes — and let the call sequence live in the caller, the one place it is actually visible.
- **Describe the present, not the path to it.** Comments, docstrings, READMEs and config `_note`s state how things are, never what they replaced or what was considered: no "no longer", "instead of the old X", "there is no env var", "not submoduled". A reader who never saw the earlier iterations should not have to. The reasoning behind a change goes in the commit message; a rejected alternative is mentioned only when a reader would otherwise try it (a documented trap), and then as a fact, not a story.
- **Paragraphs are single lines.** In Markdown, one paragraph, list item or quote is one line, however long: the viewer wraps it, and so does the editor's soft wrap. Only code blocks and tables have a layout of their own.
- `from __future__ import annotations`, type hints on signatures, `_private` for helpers, lines under ~100 chars. Otherwise, code should be self-explanatory.
- **Import heavy/optional deps lazily**, inside `_make` or inside `get_adapter`, so that installing one backend never requires the others. This is especially important when using outdated `gym` or `gym-retro` libraries, which result in dependency conflicts if imported globally.
- **No stray `print` in the frame loop.** Please make sure to remove any debugging logic before committing, to avoid code bloat and unnecessary latency.

## Rule 4: when something is wrong, stop -- never run with an invisible problem

The worst outcome this code can produce is not a crash. It is a session that runs to the end, looks fine, and turns out afterwards to have sent no triggers, opened the wrong port, ignored a config key, or quietly fallen back to something else: the participant's hour is gone and nobody knew. So:

- **Validate at start-up and raise, with the fix in the message**, before the participant screen (`fmri_play.py` builds the triggers before the window for this reason). Don't catch an error to `sys.exit` politely, and don't catch one to continue.
- **One shape per input.** No `isinstance` branches that accept two config layouts, no `x or {}` / `.get(k) or []` that turn a wrong value into an empty one. Enforce the shape and say what was expected.
- **A default that changes what the recording gets is allowed only if it is visible**: on the experimenter screen, on the console, and in the manifest (see `Triggers.status()` / `triggers.defaulted`). A silent default is a bug.
- **Test switches announce themselves** (`--dummy-trigger` prints what it skips and lands in the manifest). Degrading gracefully is for the frame loop mid-session, not for set-up.

## Adding a new backend: the checklist

0. Check that no existing backend already plays your game, possibly out of the box: `ale` takes any Atari ROM, `retro` any libretro core, `gym` any registered Gymnasium env, `vizdoom` any Doom scenario. If one does, all you need is a config.
1. `fmri_gym/adapters/<BACKEND>.py` — module docstring (what the engine is, why it was chosen, what its observation/action spaces look like, what is not supported), then a subclass of `EnvAdapter` overriding only the hooks you actually need. Everything else is inherited; do not re-state the defaults.
2. `fmri_gym/adapters/__init__.py` — one `if backend == ...:` branch with a lazy import.
3. `configs/dbp_games/<BACKEND>__<GAME>.json` — a runnable curriculum, with a `_note` field for any install step that isn't a plain `pip install`, and a `_keys_note` that says what the action indices in its `keys` mean. Lay it out like the others: one `_field` per line, the message's `text` one screen line per line, fixations as one-liners, and the game block on a few lines with `keys` (and `menu`) inline — not one key per line.
4. `pyproject.toml` — one extra under `[project.optional-dependencies]` (and a mention in `dbp` / `all` if it belongs there), then `uv lock` to refresh `uv.lock`. A game without a Gymnasium env gets one first, as its own small package under `gym/<GAME>/` (`pyproject.toml`, `README.md`, `<game>_gym/{__init__,env}.py`, a `gym.register` id), listed in `[tool.uv.sources]`; the extra installs that package. It wraps the game, it does not contain it (see Rule 1).
5. `README.md` — only if the backend needs setup beyond `pip install` (a repo checkout, a binary, an env var).

Then check your work by actually running it:

```bash
uv run fmri-play --subject sub-test --dummy-trigger --ses 1 --run 1 \
    --curriculum configs/dbp_games/<BACKEND>__<GAME>.json
ruff check fmri_gym fmri_play.py
```

and confirm the block's folder holds an `events.jsonl` whose `frame` lines (`action`, `reward`) and `episode_start` seeds look right (`fmri_gym.replay.frame_arrays`). There is no test suite yet; a run against a real config is the test.

## Things that are easy to get wrong

- **A game's `keys` are written in rig keys** (`fmri_gym/rig.py`: `UP DOWN LEFT RIGHT A B X Y LT RT`, the controller's buttons), combined with `+` when a game has more actions than buttons (not in a `turn_based` phase, which steps on single presses). Any other name is refused. Which typed key is which rig key is the machine's business (`keys.typed_keys`, the rig file, typed keys named as `pygame.key.name` names them), never a config's.
- **Every game phase writes its whole `keys` map; there is no default.** A `MultiBinary` env takes button indices (held keys combine), anything else takes the action itself and a real-time phase also needs the `""` entry, the action for no key held. One class per action space in `adapters/keymap.py`. Don't add a per-backend default or a merge step.
- **Copy observations you keep.** Several envs reuse their observation buffers, so `capture` must `.copy()` anything it stores (see `minihack.py`).
- **`render()` is not always free of side effects.** An engine that draws from the same RNG its dynamics use spends a draw on every render, so one extra call shifts everything after it and the run silently stops matching its own log. Return the frame `step` already produced rather than rendering again — crafter's night noise is the live example, and it is why `gym/crafter` caches the observation and hands that back.
- **Reproducibility is the product.** A block must be replayable from `episode_seeds` + `actions`. If the env has no savestate, leave `blob=None` and make sure `reset(seed=...)` really determines the episode. **Test it rather than assume it**: replay a recorded block in a *second process* and compare the frames bit for bit, because the usual culprit is hash order, which is fixed for the life of one interpreter and would let a replay inside the recording process pass. Crafter passes every surface check (a seed argument, a seeded `RandomState`, deterministic worldgen) and still diverges, because one creature list is built from a Python `set` and the despawn pick therefore follows object `id()`, which moves with `PYTHONHASHSEED`. Sorting that list is the whole fix, and it lives in a fork the env package pins (`gym/crafter/pyproject.toml`: `crafter @ git+https://github.com/chengfanbrain/crafter.git@deterministic`), not in the adapter: when the bug is in the engine, fix the engine and pin the build, or the rig starts owning game behaviour.
- **Turn-based games** (grid worlds, puzzles) need `"turn_based": true` in the config. 
- **Slow real-time games** (a grid world that must keep ticking, so `turn_based` is out) need `"latched_keys": true`: the default real-time path polls *held* keys, and at a few frames per second a tap that begins and ends between two frames is never seen.

## Widely accepted references

- [PEP 8 — Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 20 — The Zen of Python](https://peps.python.org/pep-0020/) ("Flat is better than nested", "Simple is better than complex", "Readability counts" — that's this document)
- [PEP 257 — Docstring Conventions](https://peps.python.org/pep-0257/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Hitchhiker's Guide to Python — Code Style](https://docs.python-guide.org/writing/style/)
- [Martin Fowler — Extract Function](https://refactoring.com/catalog/extractFunction.html)
- [Gymnasium Env API](https://gymnasium.farama.org/api/env/) — the contract every adapter presents to `run.py`
