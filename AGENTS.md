# Contributing to fmri-gym (humans and agents)

Please read this before writing code.

## The shape of the repo

```text
fmri_play.py                     CLI   parse args, build curriculum, run a Session
fmri_gym/session.py              CORE  the experiment loop: trigger, phases, timing
fmri_gym/display.py              CORE  one pygame window: frames, text, fixation
fmri_gym/keys.py                 CORE  pygame keycode to key NAME ("LEFT", "SPACE")
fmri_gym/logging.py              CORE  manifest.json + one .npz per game block
fmri_gym/adapters/base.py        CORE  EnvAdapter + FrameState: the seam
fmri_gym/adapters/keyspec.py     CORE  keyboard to action mapping
fmri_gym/adapters/<BACKEND>.py   YOU   one small wrapper per game engine
configs/dbp_games/<GAME>.json    YOU   one curriculum per game
```

The core is engine-agnostic: `session.py` never imports a game engine, never touches
`env.unwrapped`, and never mentions a game by name. Everything engine-specific goes
through an `EnvAdapter`.

## Rule 1: the game logic lives in the gym env, not the adapter

We evaluate AI models against the same gym environments that humans play in the scanner.
If an adapter adds rules, affordances, scoring, or its own renderer, humans and models are
no longer playing the same game and the comparison is void.

An adapter is lightweight glue that takes a gym env and makes it fMRI-friendly. Ideally, it should do only these things:

- build the env (`_make`) and, if needed, normalize a non-Gymnasium API (`reset`/`step`)
- declare a default keyboard map (`_keyspec`)
- produce an RGB frame for the screen (`render`)
- pull out the analysis-relevant variables (`capture`)

If you find yourself writing game rules, drawing a board, tracking a selection cursor,
computing legal moves, or sequencing trials inside an adapter -> pause. That likely belongs in the environment package (upstream, a fork, or a thin `gymnasium.Wrapper` shipped with the env),
where the model evaluation harness gets it too.

`adapters/vizdoom.py` is a good example: three short module-level helpers that translate
the engine's button table into a `KeySpec`, then a class with four small methods. `baba.py`
and `minihack.py` are equally good, smaller examples.

**Budget:** a new adapter should ideally be under ~150 lines, one file (no new subpackage). Over
that, ask whether the extra code belongs upstream in the env.

## Rule 2: changes to core libraries are a last resort, and stay generic

Before editing `session.py`, `display.py`, `keys.py`, `logging.py`, `base.py`, or
`keyspec.py`, please try to solve the problem in your adapter. If you cannot:

- Keep it **additive and default-off**, so no existing backend changes behaviour.
- Keep it **nameless**: no `if backend == "rushhour"`, no game ids, no engine imports.
- Prefer the **opt-in capability** pattern already in use: the adapter sets a flag or
  defines an optional method, and core reads it defensively.
- Say so in the PR description. A core change is the part a reviewer must read closely.

## Rule 3: write for the next *human* to read

- **Each method should ideally fit on one screen** (~40 lines). If it doesn't, extract a helper with a name
  that says what it does. `session._episode` and `_game` are at the upper limit already; let's try to not to expand them, if possible. 
- **Stay within two levels of nesting.** Use guard clauses and early `return`/`continue`
  instead of `else` ladders — see `_get_action` and `KeySpec.maximal`.
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

## Adding a new backend: the checklist

1. `fmri_gym/adapters/<BACKEND>.py` — module docstring (what the engine is, why it was
   chosen, what its observation/action spaces look like, what is not supported), then a
   subclass of `EnvAdapter` overriding only the hooks you actually need. Everything else is
   inherited; do not re-state the defaults.
2. `fmri_gym/adapters/__init__.py` — one `if backend == ...:` branch with a lazy import.
3. `configs/dbp_games/<BACKEND>__<GAME>.json` — a runnable curriculum, with a `_note`
   field for any install step that isn't a plain `pip install`.
4. `requirements.txt` — one commented line, in the "Backends" block.
5. `README.md` — only if the backend needs setup beyond `pip install` (a repo checkout, a
   binary, an env var).

Then check your work by actually running it:

```bash
python fmri_play.py --subject sub-test --dummy-trigger \
    --curriculum configs/dbp_games/<BACKEND>__<GAME>.json
ruff check fmri_gym fmri_play.py
```

and confirm the block wrote a `.npz` whose `actions`, `rewards`, and `episode_seeds` look
right. There is no test suite yet; a run against a real config is the test.

## Things that are easy to get wrong

- **Key names come from the vocabulary in `keys.py`**, upper-cased (`"LEFT"`, `"SPACE"`,
  `"Z"`). They are ours, not the window backend's, so configs keep working if the backend
  changes. An unlisted name raises at adapter construction (with a spelling suggestion);
  to make a new key available, add a row to `_CODES` rather than mapping keycodes yourself.
- **Pick the right `KeySpec`**: `SingleKeySpec` for `Discrete`, `MultiKeySpec` when held
  keys should combine (`MultiBinary`), `PassthroughKeySpec` when the env itself consumes the
  key set. Please avoid writing a fourth one unless the three don't fit.
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
  presents to `session.py`
