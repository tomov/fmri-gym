# fmri-gym

**One fMRI experiment framework for (almost) any Gymnasium-compatible game.**

A proof-of-concept framework that turns games into neuroimaging tasks — fixed
uniform display, scanner-trigger sync, a declarative curriculum of phases, and
compact reconstructable per-frame logging — and works across game engines
through small pluggable **adapters**. Ships with four:

| Backend (`"backend"`) | Games | Engine |
|---|---|---|
| `ale`      | Atari 2600 (`ALE/Pong-v5`, …) | ALE / Stella |
| `retro`    | NES / SNES / Genesis / GB / … (`Airstriker-Genesis-v0`, …) | stable-retro / libretro |
| `gym`      | **any** Gymnasium env (`CartPole-v1`, MuJoCo, Box2D, toy_text, …); old-`gym` envs via shimmy | various |
| `vgdl`     | VGDL games (`aliens`, `beesAndBirds`, …) | py-vgdl / pygame |
| `crafter`  | Crafter (open-world survival) | crafter |
| `minihack` | MiniHack / NetHack tasks | minihack / NLE |

> **All backends run in ONE env and ONE process.** Verified: a single session
> with ALE + retro + gym + VGDL blocks back-to-back, and each of Crafter /
> MiniHack in turn. VGDL originally required *old* `gym` + `numpy<2`, which
> conflicted with the numpy-2 backends; that's resolved by a fork whose VGDL
> source is ported to gymnasium
> ([tomov/language_and_experience @ dbp](https://github.com/tomov/language_and_experience/tree/dbp)).
> Adapters are imported lazily, so an env only needs the backends a curriculum
> actually uses. Same code, same curriculum schema, same output format everywhere.

## Install

Everything coexists in one conda env (verified: ALE + stable-retro + plain gym
all import and run in the same process).

```bash
conda create -n fmri-gym python=3.11
conda activate fmri-gym
pip install -r requirements.txt
```

> If your default pip index is a private registry, add
> `--index-url https://pypi.org/simple`.

Atari ROMs ship with `ale-py`. For the `retro` backend you must supply and
import game ROMs once — see [Running stable-retro games](#running-stable-retro-games).
For the `vgdl` backend see [Running VGDL games](#running-vgdl-games).

## Quick start

```bash
# Built-in mixed demo: Pong, Airstriker, and Crafter, back to back
python fmri_play.py --subject sub-01 --dummy-trigger

# --- per-family demo curricula (all tested end-to-end; ~15 s per block) ---
python fmri_play.py --subject sub-01 --dummy-trigger --curriculum configs/demo_mixed.json   # Pong + Airstriker + Crafter + MiniHack
python fmri_play.py --subject sub-01 --dummy-trigger --curriculum configs/demo_atari.json    # 10 popular Atari games
python fmri_play.py --subject sub-01 --dummy-trigger --curriculum configs/demo_classic.json  # all 5 classic-control
python fmri_play.py --subject sub-01 --dummy-trigger --curriculum configs/demo_text.json     # all 5 toy_text (they DO render)
python fmri_play.py --subject sub-01 --dummy-trigger --curriculum configs/demo_box2d.json     # LunarLander, BipedalWalker, CarRacing  (pip install swig box2d-py)
MUJOCO_GL=egl python fmri_play.py --subject sub-01 --dummy-trigger --curriculum configs/demo_mujoco.json   # 10 MuJoCo tasks  (pip install "gymnasium[mujoco]")
VGDL_REPO=/path/to/language_and_experience PYTHONPATH=/path/to/language_and_experience \
  python fmri_play.py --subject sub-01 --dummy-trigger --curriculum configs/demo_vgdl_all.json   # all 10 VGDL games (see below)

# A single-game demo (one JSON per candidate game, see configs/dbp_games/):
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/atari.json
```

Drop `--dummy-trigger` for a real session (then press SPACE, then wait for the
`=` scanner trigger). For VGDL setup see [Running VGDL games](#running-vgdl-games).

Note: **MuJoCo and Box2D use continuous (`Box`) action spaces** — the default
keymap pushes arrows to each dim's limit, so they render and log fine but aren't
really human-playable without a per-game control scheme. Everything else in
these families is keyboard-playable.

### Per-game demo configs (`configs/dbp_games/`)

There is one demo curriculum per candidate game that exposes a Gymnasium API
(15 games). Each is a minimal `message → fixation → game → fixation` block and
carries `_source` / `_status` / `_note` fields (extra keys the loader ignores).
Run one with:

```bash
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/<name>.json
```

Each was actually installed and tried in the `fmri-gym` env. Results
(`_status` in each JSON):

| config | backend | status | notes |
|---|---|---|---|
| `atari` | ale | ✅ verified | any `ALE/*` id |
| `vgdl` | vgdl | ✅ verified | needs the fork checkout (see [below](#running-vgdl-games)) |
| `crafter` | crafter | ✅ verified | dedicated adapter (old-gym API wrapped); obs is the frame |
| `minihack` | minihack | ✅ verified | dedicated adapter; pixel obs + 8-way compass |
| `tile-match-gym` | gym | 🟡 display-only | runs & logs, but `Discrete(84)` swaps → no keyboard play |
| `tobutobugirldx`, `nomolos`, `anguna` | retro | ⚠️ needs ROM | backend proven (Airstriker); import the ROM + confirm integration name |
| `2048` | gym | ❌ broken upstream | package's own `reset()` crashes (numpy bug); no render |
| `pathery`, `wordle` | gym | ❌ text-only | ANSI render + placement/typed actions → not a visual real-time game |
| `mastermind` | gym | ❌ can't install | requires Python ≥3.13 (env is 3.11) |
| `rush-hour` | gym | ❌ not packaged | repo has no `setup.py`/`pyproject` |
| `coom`, `craftium` | gym | ⏸️ deferred | heavy engines (ViZDoom / Luanti); each needs a custom adapter |

Takeaway: everything that is a **real-time visual game with a pixel frame**
works (Atari, VGDL, Crafter, MiniHack, stable-retro, plus tile-match for
display). The ones that don't fit are **turn-based / text / typed-input** games
(2048-package-bug aside), which need a different input+render model than a
scanner game loop — not a framework limitation so much as a genre mismatch.

Runtime flow: experimenter screen (**SPACE**) → "Waiting for scanner..." →
scanner **trigger `=`** (anchors the session clock) → curriculum phases → done.
`ESC` quits early but still saves. Flags: `--size 1280x1024`, `--fullscreen`,
`--save-pixels` (ALE only; see below).

## Running stable-retro games

stable-retro only exposes a game once it has an **integration** and the game's
**ROM** has been imported. ROMs are matched by their SHA-1 checksum and copied
into a data directory *inside the installed `stable_retro` package* — that's
where they must live; there is no ROM folder in this repo, and ROM binaries
should never be committed.

- **Where ROMs go.** Import them into the package's `stable/` data dir with:

  ```bash
  python -m retro.import /path/to/dir_of_roms/
  ```

  This scans the directory, checksums each ROM, and installs the ones that
  match a known integration into
  `…/site-packages/stable_retro/data/stable/<Game>/rom.<ext>`. Airstriker
  (used by the demos) ships with stable-retro, so it needs no import.

- **Games without a built-in integration** (the homebrew titles on the DBP
  list — Tobu Tobu Girl DX, Nomolos, Anguna) need an integration created first.
  The companion `stable-retro-examples` repo has an `add_game.py` that, given a
  ROM, picks the platform from the extension (`.gb`→GameBoy, `.gbc`→GbColor,
  `.nes`→Nes, `.md`→Genesis, `.sfc`→Snes, …), copies the ROM into the package
  data dir, and writes a minimal `data.json`/`metadata.json` so the env can be
  created. It prints the exact `<Name>-<Platform>` id to use.

Once imported, reference the env id in a game phase and it plays like any other
backend:

```jsonc
{"type": "game", "backend": "retro", "game": "TobuTobuGirlDX-GameBoy",
 "mode": "duration", "duration": 30.0, "fps": 60, "state_stride": 15}
```

The `configs/dbp_games/{tobutobugirldx,nomolos,anguna}.json` demos are marked
`needs-ROM` for exactly this reason: the `retro` backend itself is verified
(with Airstriker), but those titles won't run until you import their ROMs and
confirm the integration name.

## Running VGDL games

The `vgdl` backend drives the VGDL games from a gymnasium-ported fork:
**[tomov/language_and_experience @ dbp](https://github.com/tomov/language_and_experience/tree/dbp)**.
Because it runs under gymnasium + numpy 2, no separate conda env is needed — the
same `fmri-gym` env works.

1. Clone the fork (the `dbp` branch has the gymnasium port):

   ```bash
   git clone -b dbp https://github.com/tomov/language_and_experience.git
   ```

2. Point the framework at the checkout and add it to `PYTHONPATH` (so
   `src.vgdl...` is importable), then run a VGDL curriculum:

   ```bash
   VGDL_REPO=/path/to/language_and_experience \
   PYTHONPATH=/path/to/language_and_experience \
     python fmri_play.py --subject sub-01 --curriculum configs/demo_vgdl_all.json
   ```

   `VGDL_REPO` locates the game/level/sprite files; a phase can also override it
   per block with a `"repo"` field. Game files live at
   `<repo>/games/<game>_v0/<game>.txt` and `<game>_lvl<level>.txt`. Available
   games include `aliens`, `beesAndBirds`, `avoidGeorge`, `jaws`,
   `missile_command`, `plaqueAttack`, `portals`, `preconditions`,
   `pushBoulders`, `relational`.

VGDL blocks log a symbolic per-cell object grid (`symbolic_state`) and collision
`events` as analysis variables, plus a per-frame exact savestate (get/set_state)
for determinism-free reconstruction.

## Design: the experiment loop never knows the engine

```
fmri_gym/
  session.py        # trigger, clock, curriculum loop, phases  — 100% engine-agnostic
  display.py        # pygame: fixed window, aspect-fit frame, fixation, text, survey
  logging.py        # manifest.json + one compressed .npz per game block
  adapters/
    base.py         # EnvAdapter + KeySpec + FrameState (the seam)
    ale.py          # clone_state, getRAM, lossless indexed pixels
    retro.py        # em.get_state, get_ram, decoded info vars, console-button keymap
    default.py      # ANY gym env: rgb frames, seed+replay, obs-as-state
    vgdl.py         # VGDLEnv: get_state/set_state, symbolic grid + events
    crafter.py      # old-gym-API wrapper; obs is the frame; achievements
    minihack.py     # pixel obs + compass keymap; blstats/glyphs/message
fmri_play.py        # CLI entry point
configs/            # example curricula
```

The loop (`session.py`) only ever calls the adapter — never `env.unwrapped`, an
emulator, or an engine module. Each engine-specific concern lives behind
**`EnvAdapter`**:

```python
class EnvAdapter:
    def make(self, spec)          -> gym.Env       # build the env for a block
    def keymap(self, env)         -> KeySpec       # held keys -> action
    def reset(self, env, seed, spec) -> (obs, info)
    def capture(self, env, obs, info) -> FrameState  # per-frame state to log
    def restore(self, env, blob)  -> None          # inverse of capture().blob
```

`FrameState` carries a standard shape for **every** backend:
- `blob`: opaque bytes that `restore()` turns back into this exact state
  (ALE `clone_state`, retro `em.get_state()`), or `None` if the engine has no
  savestate — then reconstruction falls back to seed + action replay.
- `variables`: named analysis fields surfaced uniformly (`ram`, retro's decoded
  `info_score`/`info_lives`, the raw `obs`, …), so the loop and downstream
  analysis code are identical across engines.

**Adding a new engine = writing one adapter** (~40–80 lines). Nothing else changes.

## Curriculum format

An ordered JSON list of **phases** (bare list or `{"curriculum": [...]}`):

```jsonc
{"type": "fixation", "duration": 2.0}                 // "+" for N seconds
{"type": "message", "text": "Get ready", "duration": 2.0}  // text; omit duration to wait for a key
{"type": "survey", "n_points": 7, "questions": ["...","..."]}

{"type": "game",
 "backend": "ale",              // "ale" | "retro" | "gym"
 "game": "ALE/Pong-v5",         // env id for that backend
 "mode": "duration",            // "duration" = replay until time up; "episode" = play N episodes
 "duration": 30.0,              // seconds (duration mode)
 "n_episodes": 1,               // episodes (episode mode)
 "max_duration": 300.0,         // hard wall-clock safety cap (episode mode)
 "fps": 30,                     // target game frames/second
 "seed": 1234,                  // base RNG seed (optional)
 "state_stride": 1,             // save a full savestate every K frames (see below)
 "state": "Level1",             // retro: named savestate/level (optional)
 "scenario": null,              // retro: scenario name (optional)
 "level": 0,                    // vgdl: level index; also uses "game","block_size"
 "keys": {"LEFT": 0, "RIGHT": 1}, // override keyboard->action map (see below)
 "save_pixels": false}          // ALE: also store lossless pixels (see warning)
```

### Keymaps

Each backend builds a default keyboard→action map:

- **ale**: built from the game's action meanings (arrows move, SPACE fires).
- **retro**: keyboard → console buttons (arrows move; Z/X/C = A/B/C; ENTER =
  start); multiple held keys combine (e.g. RIGHT+Z).
- **gym**: a generic default (arrows → first Discrete actions, or ±limits on
  Box dims). Because a bare `Discrete(n)` has no inherent meaning, **specify
  `keys` per game** for anything non-obvious.

### Remapping keys (the `keys` field)

Any game phase can override the mapping with a `keys` dict of
`"<key(s)>": <action>`. The keys are pygame names (`UP`, `DOWN`, `LEFT`,
`RIGHT`, `SPACE`, `RETURN`, letters `A`–`Z`, digits) and `<action>` is the
action the env expects — an **integer** for a `Discrete` space (ale, gym,
vgdl, …). Combine keys with `+` (e.g. `"UP+SPACE"`). The most specific fully-held
combo wins, so a combo overrides its parts.

To find the action indices for an Atari game, read its meanings:

```python
import gymnasium as gym, ale_py; gym.register_envs(ale_py)
gym.make("ALE/Pong-v5").unwrapped.get_action_meanings()
# ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']  -> RIGHT=2, LEFT=3
```

**Example — Pong on up/down arrows** (its paddle is `RIGHT`=2 / `LEFT`=3):

```jsonc
{"type": "game", "backend": "ale", "game": "ALE/Pong-v5",
 "mode": "duration", "duration": 30.0,
 "keys": {"UP": 2, "DOWN": 3}}      // UP = paddle up, DOWN = paddle down; SPACE still serves (FIRE=1)
```

`configs/dbp_games/atari.json` and the built-in demo both use this Pong mapping.
CartPole similarly uses `{"LEFT": 0, "RIGHT": 1}`.

## Output & data format

Each session writes `data/<subject>_<timestamp>/`:

- **`manifest.json`** — subject, curriculum, trigger epoch, and per-phase
  onsets/offsets (+ survey responses).
- **`block-NN_<backend>_<game>.npz`** — one per game block, uniform schema:

  | key | meaning |
  |-----|---------|
  | `actions`, `rewards`, `terminal`, `episode_id` | per frame |
  | `t_rel`, `t_epoch` | seconds since trigger; wall-clock epoch |
  | `states` | per-frame savestate blob (object array; `None` if engine has none) |
  | `episode_seeds` | RNG seed per episode |
  | `backend`, `game` | provenance |
  | *backend vars* | `ram` (ale/retro), `info_*` (retro decoded score/lives/…), `obs` (gym), `screen_index` (ale `--save-pixels`) |

### Reconstruction (all verified bit-exact)

1. **Per-frame state** (ale, retro): `restore(states[i])` → exact frame `i`, no
   determinism assumption.
2. **Seed + action replay** (any deterministic env, incl. gym): `episode_seeds`
   + `actions` reproduce an episode frame-for-frame.
3. **Stored pixels** (ale opt-in): `palette[screen_index]` *is* the RGB frame.

```python
import numpy as np, pickle, gymnasium as gym, ale_py, stable_retro as retro
gym.register_envs(ale_py)

# ALE: restore any frame's exact state
d = np.load("block-00_ale_Pong-v5.npz", allow_pickle=True)
env = gym.make(str(d["game"]), render_mode="rgb_array",
               frameskip=1, repeat_action_probability=0.0); env.reset()
env.unwrapped.restore_state(pickle.loads(d["states"][10]))
frame10 = env.unwrapped.ale.getScreenRGB()

# retro: restore any frame's exact state
d = np.load("block-01_retro_Airstriker-Genesis-v0.npz", allow_pickle=True)
r = retro.make(str(d["game"]), render_mode="rgb_array"); r.reset()
r.unwrapped.em.set_state(d["states"][10]); r.unwrapped.data.update_ram()
```

> ⚠️ **Storage note & `state_stride`.** Per-frame savestates are cheap for ALE
> (~0.4 KB/frame) but large for retro consoles: a Genesis state is ~1 MB/frame.
> Set **`"state_stride": K`** on a game phase to snapshot a full savestate only
> every K frames (always including each episode's first frame, the replay
> anchor); frames between anchors stay reconstructable by restoring the last
> anchor and replaying the logged actions (retro/ALE/VGDL are deterministic).
> Measured on Airstriker-Genesis: a 1.5 s @60 fps block drops from **780 KB →
> 86 KB with `state_stride: 15`** (~9×). Analysis variables (RAM, `info_*`) are
> always logged every frame regardless of stride.
>
> ⚠️ **`--save-pixels` (ALE)** stores the screen every frame. It's lossless
> (indexed palette; `palette[screen_index] == RGB`) and zlib-friendly
> (~0.25 KB/frame) — but unnecessary, since per-frame state already
> reconstructs pixels. Prints a loud warning when enabled.

## Migrating your game list

Many games already expose a Gymnasium API and drop straight into the `gym`
backend; stable-retro titles use the `retro` backend; and the VGDL games use the
`vgdl` backend (their source was ported from old `gym` to gymnasium so they run
in the same numpy-2 env).

**old-`gym` games (e.g. chess, hanoi, Sokoban, Baba, NetHack).** Two options:
(a) **port the source to gymnasium**, as done for VGDL — usually a small
mechanical diff (swap `gym`→`gymnasium`, fix removed `np.*` aliases and
`pkg_resources`); or (b) run them via **shimmy** with `"legacy_gym": true`
(routed via `GymV21Environment-v0`) in a dedicated `numpy<2` env. Note shimmy's
v0.21 compat calls the removed `.seed()` and `gym==0.26` is incompatible with
`numpy>=2`, so (a) is usually cleaner. The `legacy_gym` code path exists in
`default.py`.

**Porting an old-`gym` game to gymnasium (the VGDL recipe).** The whole change
was: `import gym`→`import gymnasium as gym` across the env/registration/play
files; skip `Space.__init__` in a custom variable-length space; replace removed
`np.float` with `float`; and drop `pkg_resources` (gone in setuptools≥81) by
resolving data dirs relative to `__file__`. Result: VGDL runs under gymnasium
1.3 + numpy 2.4 in the same env as every other backend.

## Future work / TODO

- [x] Configurable **state stride** (`"state_stride": K`) — done.
- [x] **VGDL backend** ([tomov/language_and_experience @ dbp](https://github.com/tomov/language_and_experience/tree/dbp)),
      ported to gymnasium so it runs in the same env as the other backends — done.
- [ ] Finish the **old-`gym` / shimmy** path against a real game (Sokoban,
      chess) — either port its source (VGDL recipe above) or run via shimmy in a
      `numpy<2` env; code path exists but is untested end-to-end.
- [ ] **Photodiode sync square** and **LSL / parallel-port markers** for
      MEG/EEG-grade timing; fMRI's slow HRF makes the `=`-anchored software
      clock adequate.
- [ ] **retro `.bk2` movie logging** as an alternative to per-frame states
      (frame-exact, tiny).
- [ ] Per-subject deterministic curriculum generation; multi-run structure with
      one trigger per fMRI run.
- [ ] More adapters: ViZDoom/COOM, MiniHack, crafter, MuJoCo (`qpos/qvel` as
      state vars) — each a small `EnvAdapter`.
- [ ] Button-box / MRI-safe response device key remapping.
- [ ] BIDS-style output layout + `events.tsv` per run.
- [ ] A replay/QC utility to render any block to video from its states.
- [ ] Crash-safe incremental logging (stream frames to disk) for long runs.
