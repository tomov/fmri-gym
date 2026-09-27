# Contributing to fmri-gym (humans and agents)

Please read this before writing code.

## The shape of the repo

```text
fmri_play.py                     CLI   parse the flags of one run, then play it (Run.from_config)
fmri_gym/run.py                  CORE  one run's experiment loop: trigger, phases, timing
fmri_gym/display.py              CORE  one pygame window: frames, text, fixation
fmri_gym/keys.py                 CORE  pygame keycode to key NAME ("LEFT", "SPACE")
fmri_gym/logging.py              CORE  manifest.json + one .npz per game block
fmri_gym/menu.py                 CORE  hold-a-key pause menu: reset / forfeit / resume (opt-in per phase)
fmri_gym/adapters/base.py        CORE  EnvAdapter + Keymap (the phase's keys) + FrameState: the seam
fmri_gym/adapters/<BACKEND>.py   YOU   one small wrapper per game engine
configs/dbp_games/<GAME>.json    YOU   one curriculum per game
```

The core is engine-agnostic: `run.py` never imports a game engine, never touches
`env.unwrapped`, and never mentions a game by name. Everything engine-specific goes
through an `EnvAdapter`.

## Rule 1: the game logic lives in the gym env, not the adapter

We evaluate AI models against the same gym environments that humans play in the scanner.
If an adapter adds rules, affordances, scoring, or its own renderer, humans and models are
no longer playing the same game and the comparison is void.

An adapter is lightweight glue that takes a gym env and makes it fMRI-friendly. Ideally, it should do only these things:

- build the env (`_make`): a `gymnasium.Env`, always. If the game's own env speaks
  another API (old `gym`, a bare engine, a `with_img=` of its own), a thin Gymnasium env
  under `vendor/` puts the contract in front of it (`vendor/baba/`, `vendor/crafter/`,
  `vendor/vgdl/`, `vendor/coom/`); the adapter never normalizes `reset`/`step` itself
- say in the module docstring what the env's action indices mean, so a config can
  write its `keys` (there is no default keyboard map: the config states all of it)
- produce an RGB frame for the screen (`render`)
- if the engine makes sound, hand over this step's PCM (`sound`; the contract is in
  `EnvAdapter.sound`, and `retro.py` is a two-line example)
- if the engine has a clock of its own, say how many steps per second are real speed
  (`native_fps`), read from the engine where it tells; the run reports the block's
  speed against it
- pull out the analysis-relevant variables (`capture`)

If you find yourself writing game rules, drawing a board, tracking a selection cursor,
computing legal moves, or sequencing trials inside an adapter -> pause. That likely belongs in the environment package (upstream, a fork, or a thin `gymnasium.Wrapper` shipped with the env),
where the model evaluation harness gets it too.

`adapters/vizdoom.py` is a good example: one small action wrapper, then a class with a
handful of short methods. `baba.py` and `minihack.py` are equally good, smaller examples.

**Budget:** a new adapter should ideally be under ~150 lines, one file (no new subpackage). Over
that, ask whether the extra code belongs upstream in the env.

## Rule 2: changes to core libraries are a last resort, and stay generic

Before editing `run.py`, `display.py`, `keys.py`, `logging.py`, or `base.py`,
please try to solve the problem in your adapter. If you cannot:

- Keep it **additive and default-off**, so no existing backend changes behaviour.
- Keep it **nameless**: no `if backend == "rushhour"`, no game ids, no engine imports.
  This holds for the prose too -- a core file's comments and docstrings state the contract,
  they don't cite the backend that happens to use it.
- Prefer the **opt-in capability** pattern already in use: the adapter sets a flag or
  defines an optional method, and core reads it defensively.
- Say so in the PR description. A core change is the part a reviewer must read closely.

## Rule 3: write for the next *human* to read

- **Each method should ideally fit on one screen** (~40 lines). If it doesn't, extract a helper with a name
  that says what it does. `run._episode` and `_game` are at the upper limit already; let's try to not to expand them, if possible. 
- **Stay within two levels of nesting.** Use guard clauses and early `return`/`continue`
  instead of `else` ladders — see `_poll_keys_until` and `Keymap.__init__`.
- **Module docstring explains *why*.** Every file here opens with the reasoning a
  newcomer needs: why this backend and not COOM, why `pixel_crop` and not `pixel`, what is
  deliberately not supported. Keep doing that; it is the most valuable text in the repo.
- **Docstrings on public methods** in the existing Sphinx style (`:param:`, `:return:`,
  `:raises:`). Short private helpers can get a one-liner.
- **Comments explain the trap, not the code.** The good ones here record a fact you cannot
  see from the source: why MiniHack needs an explicit `seed(core=, disp=)`, why classic
  control tears down the shared pygame window.
- `from __future__ import annotations`, type hints on signatures, `_private` for helpers,
  lines under ~100 chars. Otherwise, code should be self-explanatory.
- **Import heavy/optional deps lazily**, inside `_make` or inside `get_adapter`, so that
  installing one backend never requires the others. 
  This is especially important when using outdated `gym` or `gym-retro` libraries, which result in dependency conflicts if imported globally.
- **No stray `print` in the frame loop.** Please make sure to remove any debugging logic before committing, to avoid code bloat and unnecessary latency.

## Rule 4: when something is wrong, stop -- never run with an invisible problem

The worst outcome this code can produce is not a crash. It is a session that
runs to the end, looks fine, and turns out afterwards to have sent no triggers,
opened the wrong port, ignored a config key, or quietly fallen back to
something else: the participant's hour is gone and nobody knew. So:

- **Validate at start-up and raise, with the fix in the message**, before the
  participant screen (`fmri_play.py` builds the triggers before the window for
  this reason). Don't catch an error to `sys.exit` politely, and don't catch
  one to continue.
- **One shape per input.** No `isinstance` branches that accept two config
  layouts, no `x or {}` / `.get(k) or []` that turn a wrong value into an
  empty one. Enforce the shape and say what was expected.
- **A default that changes what the recording gets is allowed only if it is
  visible**: on the experimenter screen, on the console, and in the manifest
  (see `Triggers.status()` / `triggers.defaulted`). A silent default is a bug.
- **Test switches announce themselves** (`--dummy-trigger` prints what it
  skips and lands in the manifest). Degrading gracefully is for the frame
  loop mid-session, not for set-up.

## Adding a new backend: the checklist

1. `fmri_gym/adapters/<BACKEND>.py` — module docstring (what the engine is, why it was
   chosen, what its observation/action spaces look like, what is not supported), then a
   subclass of `EnvAdapter` overriding only the hooks you actually need. Everything else is
   inherited; do not re-state the defaults.
2. `fmri_gym/adapters/__init__.py` — one `if backend == ...:` branch with a lazy import.
3. `configs/dbp_games/<BACKEND>__<GAME>.json` — a runnable curriculum, with a `_note`
   field for any install step that isn't a plain `pip install`, and a `_keys_note` that
   says what the action indices in its `keys` mean.
4. `pyproject.toml` — one extra under `[project.optional-dependencies]` (and a
   mention in `dbp` / `all` if it belongs there), then `uv lock` to refresh `uv.lock`.
   A game without a Gymnasium env gets one first, as its own small package under
   `vendor/<GAME>/` (`pyproject.toml`, `README.md`, `<game>_gym/{__init__,env}.py`, a
   `gym.register` id), listed in `[tool.uv.sources]`; the extra installs that package.
5. `README.md` — only if the backend needs setup beyond `pip install` (a repo checkout, a
   binary, an env var).

Then check your work by actually running it:

```bash
uv run fmri-play --subject sub-test --dummy-trigger --ses 1 --run 1 \
    --curriculum configs/dbp_games/<BACKEND>__<GAME>.json
ruff check fmri_gym fmri_play.py
```

and confirm the block wrote a `.npz` whose `actions`, `rewards`, and `episode_seeds` look
right. There is no test suite yet; a run against a real config is the test.

## Things that are easy to get wrong

- **Key names are pygame names, upper-cased, without `K_`** (`"LEFT"`, `"SPACE"`, `"Z"`).
  Add unlisted keys to `_NAMES` in `keys.py` rather than mapping keycodes yourself;
  `validate_config` refuses a config that names a key outside that table.
- **Every game phase writes its whole `keys` map; there is no default.** A `MultiBinary`
  env takes button indices (held keys combine), anything else takes the action itself and
  a real-time phase also needs `noop`. Don't add a per-backend default or a merge step:
  the map depends on the site's input device, not on the engine.
- **Copy observations you keep.** Several envs reuse their observation buffers, so
  `capture` must `.copy()` anything it stores (see `minihack.py`).
- **Reproducibility is the product.** A block must be replayable from
  `episode_seeds` + `actions`. If the env has no savestate, leave `blob=None` and make sure
  `reset(seed=...)` really determines the episode.
- **Turn-based games** (grid worlds, puzzles) need `"turn_based": true` in the config. 

## Widely accepted references

- [PEP 8 — Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 20 — The Zen of Python](https://peps.python.org/pep-0020/) ("Flat is better than
  nested", "Simple is better than complex", "Readability counts" — that's this document)
- [PEP 257 — Docstring Conventions](https://peps.python.org/pep-0257/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Hitchhiker's Guide to Python — Code Style](https://docs.python-guide.org/writing/style/)
- [Martin Fowler — Extract Function](https://refactoring.com/catalog/extractFunction.html)
- [Gymnasium Env API](https://gymnasium.farama.org/api/env/) — the contract every adapter
  presents to `run.py`
