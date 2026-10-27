---
name: add-backend
description: Add a new game engine (backend) to fmri-gym, or wrap a game that has no Gymnasium env: the adapter, the gym/ package, the config, the pyproject extra, the tests row, and the traps that make a backend look fine and replay wrong. Use when asked to add, port or wire up a game.
---

# Adding a backend

Read AGENTS.md first: rule 1 (the game logic lives in the gym env, not the adapter) decides where every line below goes.

## The checklist

0. Check that no existing backend already plays the game, possibly out of the box: `ale` takes any Atari ROM, `retro` any libretro core, `gym` any registered Gymnasium env, `vizdoom` any Doom scenario. If one does, all you need is a config.
1. `fmri_gym/adapters/<BACKEND>.py`: module docstring (what the engine is, why it was chosen, what its observation/action spaces look like, what is not supported), then a subclass of `EnvAdapter` overriding only the hooks you actually need. Everything else is inherited; do not re-state the defaults. `adapters/vizdoom.py`, `baba.py` and `minihack.py` are the models; stay under about 150 lines, one file.
2. `fmri_gym/adapters/__init__.py`: one `if backend == ...:` branch with a lazy import.
3. `configs/curricula/<BACKEND>/levels/<GAME>.json`: a runnable curriculum of one level (a curriculum that plays several goes in `configs/curricula/<BACKEND>/` itself), and `configs/curricula/<BACKEND>/common/README.md` with any install step that is not a plain `pip install` and what the action indices in its `keys` mean. Comments in the JSON (`"## ..."` keys) are for what is particular to one file, and few. Lay it out like the others: one `##` comment per line, the message's `text` one screen line per line, fixations as one-liners, and the game block on a few lines with `keys` (and `menu`) inline, not one key per line. A game with several levels gets one curriculum per level in `levels/`, whose game phases `__include` (`"../common/<GAME>.json"`) a shared `configs/curricula/<BACKEND>/common/<GAME>.json` (the keys, the instructions, the settings) and state only the level and its title (README, "Shared phases").
4. `pyproject.toml`: one extra under `[project.optional-dependencies]` (and a mention in `dbp` / `all` if it belongs there), then `uv lock`. A game without a Gymnasium env gets one first, as its own small package under `gym/<GAME>/` (`pyproject.toml`, `README.md`, `<game>_gym/{__init__,env}.py`, a `gym.register` id), listed in `[tool.uv.sources]`; the extra installs that package. It wraps the game, it does not contain it.
5. `tests/backends.py`: a `Backend` row (the extra, the module that proves it is installed, one or two short smoke configs, the `gym/` dirs it runs on, its `external/` checkout, a `ci` reason if a hosted Linux runner cannot play it), and a line in `.github/workflows/tests.yml`'s matrix unless it has one. `test_configs` fails until the row exists.
6. `README.md`: only if the backend needs setup beyond `pip install` (a repo checkout, a binary, an env var).

Then play it, through the suite and by hand:

```bash
uv run pytest --backend <BACKEND>
uv run fmri-play --subject sub-test --dummy-trigger --ses 1 --run 1 --rig configs/examples/rigs/laptop.json \
    --curriculum configs/curricula/<BACKEND>/levels/<GAME>.json
uv run ruff check fmri_gym fmri_play.py agents tests
```

The suite's four checks (playable, recorded, replays, runs) are the acceptance test; the `testing` skill says how to read a failure. A backend that passes everything but **replays** is not done: reproducibility is the product.

## Things that are easy to get wrong

- **Key names are pygame names, upper-cased, without `K_`** (`"LEFT"`, `"SPACE"`, `"Z"`). Add unlisted keys to `_PYGAME_KEY_NAMES` in `keys.py` rather than mapping keycodes yourself; `validate_curriculum` refuses a config that names a key outside that table.
- **Every game phase writes its whole `keys` map; there is no default.** A `MultiBinary` env takes button indices (held keys combine), anything else takes the action itself and a real-time phase also needs the `""` entry, the action for no key held. One class per action space in `adapters/keymap.py`. Don't add a per-backend default or a merge step: the map depends on the site's input device, not on the engine.
- **Copy observations you keep.** Several envs reuse their observation buffers, so `capture` must `.copy()` anything it stores (see `minihack.py`). The replay check is what found nethack logging each frame's variables one frame late: a view of a reused buffer, pickled after the loop had moved on.
- **`render()` is not always free of side effects.** An engine that draws from the same RNG its dynamics use spends a draw on every render, so one extra call shifts everything after it and the run silently stops matching its own log. Return the frame `step` already produced rather than rendering again; crafter's night noise is the live example, and it is why `gym/crafter` caches the observation and hands that back.
- **Reproducibility is the product.** A block must be replayable from `episode_seeds` + `actions`. If the env has no savestate of its own, wrap it in `replay_state_gym.ReplayState` (`gym/replay_state/`: the seed and the actions since, restored by replay) and make sure `reset(seed=...)` really determines the episode. **Test it rather than assume it**: `uv run python -m tests.verify_replay --curriculum <config>` records a block with the random policy, replays it in a *second process* under another `PYTHONHASHSEED` and compares every frame, every sound chunk and every savestate anchor bit for bit. A second process because the usual culprit is hash order, which is fixed for the life of one interpreter and would let a replay inside the recording process pass. Crafter passes every surface check (a seed argument, a seeded `RandomState`, deterministic worldgen) and still diverges, because one creature list is built from a Python `set` and the despawn pick therefore follows object `id()`, which moves with `PYTHONHASHSEED`. Sorting that list is the whole fix, and it lives in a fork the env package pins (`gym/crafter/pyproject.toml`: `crafter @ git+https://github.com/chengfanbrain/crafter.git@deterministic`), not in the adapter: when the bug is in the engine, fix the engine and pin the build, or the rig starts owning game behaviour.
- **Sound is part of the record.** An engine that makes sound returns this step's PCM from `sound()` (the contract is in `EnvAdapter.sound`; `retro.py` is a two-line example); the loop plays it and logs it, and the replay check compares it chunk for chunk. ViZDoom's first episode after `init()` sounded different from one after `new_episode()`, which is why `vizdoom_gym` begins every episode the same way.
- **Turn-based games** (grid worlds, puzzles) need `"is_turn_based": true` in the config.
- **Slow real-time games** (a grid world that must keep ticking, so `is_turn_based` is out) need `"has_latched_keys": true`: the default real-time path polls *held* keys, and at a few frames per second a tap that begins and ends between two frames is never seen.
