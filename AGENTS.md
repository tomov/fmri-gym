# Contributing to fmri-gym (humans and agents)

Please read this before writing code.

## Where things are explained

This file is the rules: the five below, and the shape of the repo. The step-by-step procedures live as skills, one folder each under `.claude/skills/<name>/SKILL.md`, which is plain Markdown anyone can open: `testing` (run the suite, read a failure, what CI does), `add-backend` (the checklist for a new game engine and the traps), `pull-request` (branch, local checks, the description, CI). Claude Code and Cursor load them from `.claude/skills/`; Codex reads `.agents/skills/`, where each entry is a symlink to the same folder. A procedure belongs in a skill and a rule belongs here; neither belongs in both. The test suite describes itself in `tests/README.md`.

## The shape of the repo

```text
fmri_play.py                     CLI   parse the flags of one run, then play it (Run.from_config_and_curriculum)
fmri_gym/config.py               CORE  RunConfig: one run's command line (subject, session, window, switches)
fmri_gym/curriculum.py           CORE  the curriculum file: load, save, check; GamePhase, a game phase's settings
fmri_gym/run.py                  CORE  one run's experiment loop: trigger, phases, timing
fmri_gym/display.py              CORE  one pygame window: frames, text, fixation
fmri_gym/keys.py                 CORE  pygame keycode to key NAME ("LEFT", "SPACE")
fmri_gym/logging.py              CORE  manifest.json + one crash-safe JSONL/HDF5 log per game block
fmri_gym/replay.py                     replay an episode from a block's log; frame fields as arrays
fmri_gym/menu.py                 CORE  hold-a-key pause menu: reset / forfeit / resume (opt-in per phase)
fmri_gym/adapters/base.py        CORE  EnvAdapter + FrameState: the seam
fmri_gym/adapters/keymap.py      CORE  the phase's keys, one Keymap per action space
fmri_gym/adapters/<BACKEND>.py   YOU   one small wrapper per game engine
configs/dbp_games/<GAME>.json    YOU   one curriculum per game
gym/<GAME>/                      YOU   a Gymnasium env for a game that ships none, or a rough one
gym/replay_state/                      ReplayState: a savestate by replay, for any env
external/<BACKEND>/              a game's own repo, a submodule at the commit the README pins
tests/                                 the suite: every backend played, logged, replayed, run headless (tests/README.md)
.claude/skills/<NAME>/SKILL.md         the procedures (testing, add-backend, pull-request); .agents/skills links to them
.github/workflows/tests.yml            CI: the suite on every pull request that is ready for review
```

The core is engine-agnostic: `run.py` never imports a game engine, never touches `env.unwrapped`, and never mentions a game by name. Everything engine-specific goes through an `EnvAdapter`.

Three layers, three directories, each optional until needed: `external/` is a game's own repo when its files must be read from a checkout (use only when `pip install` is not an option); `gym/` is a `gymnasium.Env` for a game that ships none (or a rough one); `fmri_gym/adapters/` is the fMRI adapter over whatever env results. Code goes in the lowest layer that owns the concern -- game rules in the game or its `gym/` env, never in the adapter.

Three kinds of settings, kept apart because they are shared differently. The **curriculum** (`curriculum.py`, the `--curriculum` JSON) is what is played: the sequence of phases of one run, tied to a game and the same for every subject at every site. The **run config** (`config.py`, `RunConfig`) is one run: who plays, which session and run number, where the data goes, how this machine shows it, the test switches. The **rig config** is the site's own settings, the same for all of its subjects: today the curriculum's `triggers` section and the rig file of `checks.py`, on their way to a file of their own. A setting goes in the kind it is shared like, and keeps that kind's word all the way down the call stack: a curriculum is `curriculum` in `fmri_play`, in `Run.from_config_and_curriculum` and in `checks.prepare`, its phase list is `phases`, and nothing calls either one "the config".

## Rule 1: the game logic lives in the gym env, not the adapter

We evaluate AI models against the same gym environments that humans play in the scanner. If an adapter adds rules, affordances, scoring, or its own renderer, humans and models are no longer playing the same game and the comparison is void.

An adapter is lightweight glue that takes a `gymnasium.Env` and makes it fMRI-friendly. That env is the whole point: it is what the agents are trained and evaluated on, so a game that is not a `gymnasium.Env` yet is not ready for an adapter -- wrap it first. Ideally, an adapter should do only these things:

- build the env (`_make`): a `gymnasium.Env`, always. Prefer a ready-made pip package (`ale-py`, `stable-retro`, `minihack`, `rushhour-gym`, ...). If the game's own env speaks another API (old `gym`, a bare engine, a `with_img=` of its own), a thin Gymnasium env under `gym/` puts the contract in front of it (`gym/baba/`, `gym/crafter/`, `gym/vgdl/`, `gym/coom/`, `gym/baba_auto/`), and that is where any new one goes; the adapter never normalizes `reset`/`step` itself. Two rules for that env:
  - **Only the wrapper lives here.** The game's own code is a pip package, or a checkout in `external/<backend>` at the commit the README pins ("External checkouts"). The env reads that path by default and takes `repo=` for another; the pin is the README's.
  - **It is a plain RL env.** It follows the Gymnasium API and knows nothing about fMRI, scanners, phases, blocks, subjects or how it will be used; someone training an agent on it should find nothing odd. That vocabulary belongs to the adapter and the core.
- say in the module docstring what the env's action indices mean, so a config can write its `keys` (there is no default keyboard map: the config states all of it)
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
- **Typed contracts in memory, dicts at the boundaries.** A dict is for data that crosses a serialization boundary: the curriculum JSON on the way in, a manifest or `events.jsonl` record on the way out, a message on the writer queue. Anything handed between functions gets a frozen dataclass or a NamedTuple whose fields say what is there and what is optional (`RunConfig`, `RunProgress`, `GamePhase`, `EpisodeResult`, `AudioBlockLog`, `TriggerSettings`), with the conversion in one place on the type: a `from_dict` on the way in (`from_args` when the source is the parsed command line), a `record()` on the way out. A function that takes another module's parsed command line, or returns a dict whose keys are documented only in prose, is the thing to avoid.
- **How those types are written.** The name says what the thing is, and a type read straight off the command line or a JSON file ends in `Config` (`RunConfig`), never `Options`, `Settings` or `Args`. A boolean field starts with `is_` or `has_` (`is_turn_based`, `has_audio`), so a reader never wonders whether `audio` is a flag or the sound. A field with a closed set of string values is an `Enum` (`BlockMode`), so the options are in the code and not in a docstring. Two fields that are the same thing in two modes are one field (`duration` is the block's seconds in both of its modes). Fields that belong together sit together (`advance_on` next to `rewind_on`). The data classes go at the top of the file, under the constants, with their factory methods inside them: the shape of what a module passes around is the first thing a reader meets, and the parsing is never a loose function elsewhere.
- **One name per concept, and a name that says what it holds.** The same thing has the same name in every file, a reader meets no alias (`started_from = initial_state`) and no one-letter or abbreviated local (`fs`, `gp`, `t`, `d`) outside a one-line comprehension; a long name is fine if it reads. The vocabulary in force: `curriculum` is the file's dict, `phases` its list of phases, `phase` one of them (a dict), `phase_index` its position, `game_phase` the parsed `GamePhase`, `run_config` the `RunConfig`; `spec` only on the adapter seam (`EnvAdapter.spec`: the phase as the adapter sees it) and for a phase's sub-objects (`menu_spec`, `instructions_spec`); `run_folder` and `block_folder` are the two folders; `frame_rgb` the rendered array, `frame` the index in the block, `ep_frame` the index in the episode, `anchor_frame` the bool "this frame logs a savestate"; `savestate` the opaque bytes (`FrameState.blob` is the adapter's field for it), `initial_state` the one an episode is restored to, `rewind_savestates` the deque a rewinding block keeps; `outcome` the name of how an episode ended, `outcome_message` the line shown for it, `episode_result` an `EpisodeResult`.
- **A time says which clock it is on.** `*_perf` is a `time.perf_counter()` stamp (`flip_perf`, `deadline_perf`, `block_end_perf`, `Clock.trigger_perf`), `*_run_time` seconds since the scanner trigger, `*_epoch` `time.time()`, `*_s` / `*_ms` a duration. The log's own field names (`onset`, `offset`, `run_time`, `flip_time`, `wall_time`) stay as the record spells them, and a local that is about to become a record field is named as the field.
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

## Rule 5: build the smallest thing that reuses what is already there

- **Start from what is already recorded.** Before adding a file, a record or a module, check whether the log, the manifest, a savestate or an existing code path already holds what you need; usually the change is to read it, not to keep a second copy (resume needed no new record: the end state, elapsed time and episode count were already in `block_end`).
- **One mechanism per idea.** When features need the same thing -- resume, rewind and replay all start an episode from a saved state -- make one path (`initial_state`) serve them all, rather than a subsystem each with its own log lines and replay rules.
- **Build what was asked, then stop.** Alternative units, caps, extra menu entries and configurable timings each cost validation, docs and tests for good; ask first. A feature is cheap to add later and expensive to remove.
- **Use the seam that exists.** A new `EnvAdapter` hook or phase field is a core change (Rule 2); try `overlay()`, `block_extra()`, `info` or an env wrapper first.
- **Agree the design before writing it.** For a change that touches the core, write the plan in a few lines and ask the questions only a human can answer (is this ever interleaved? should finishing end the level?). A conversation is cheaper than rewriting a branch.
- **Treat size as a warning.** If a feature passes a few hundred lines, or its docstring needs paragraphs to defend the design, stop and look for a simpler one: a long justification usually marks a design that fights the code around it.
- **Verify against a reference.** When porting or rewriting, run the old and new side by side on the same inputs; when saving and restoring, compare bytes. Keep the result in the commit message, not a check script in the repo.

## Adding a backend, testing it, opening the PR

The checklist for a new backend and the traps that make one look fine and replay wrong are the `add-backend` skill. The suite is `uv run pytest --changed` before every push and runs on every pull request marked ready for review (the `testing` skill; `tests/README.md`). A backend is not done until its block replays bit for bit in a second process: `uv run python -m tests.verify_replay --curriculum <config>` is the check, and the suite runs it on every backend it selects. The road from a branch to a merged PR, and what the description must say, is the `pull-request` skill.

## Widely accepted references

- [PEP 8 — Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 20 — The Zen of Python](https://peps.python.org/pep-0020/) ("Flat is better than nested", "Simple is better than complex", "Readability counts" — that's this document)
- [PEP 257 — Docstring Conventions](https://peps.python.org/pep-0257/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Hitchhiker's Guide to Python — Code Style](https://docs.python-guide.org/writing/style/)
- [Martin Fowler — Extract Function](https://refactoring.com/catalog/extractFunction.html)
- [Gymnasium Env API](https://gymnasium.farama.org/api/env/) — the contract every adapter presents to `run.py`
