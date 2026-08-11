# fmri-gym

**One fMRI experiment framework for (almost) any Gymnasium-compatible game.**

A proof-of-concept framework that turns games into neuroimaging tasks — fixed
uniform display, scanner-trigger sync, a declarative curriculum of phases, and
compact reconstructable per-frame logging — and works across game engines
through small pluggable **adapters**. Ships with three:

| Backend (`"backend"`) | Games | Engine | conda env |
|---|---|---|---|
| `ale`   | Atari 2600 (`ALE/Pong-v5`, …) | ALE / Stella | `fmri-gym` |
| `retro` | NES / SNES / Genesis / GB / … (`Airstriker-Genesis-v0`, …) | stable-retro / libretro | `fmri-gym` |
| `gym`   | **any** Gymnasium env (`CartPole-v1`, MuJoCo, Box2D, toy_text, …); old-`gym` envs via shimmy | various | `fmri-gym` |
| `vgdl`  | VGDL games (`aliens`, `beesAndBirds`, …) from ccolas/language_and_experience | py-vgdl / pygame | `language_and_experience` |

> **One framework, sometimes several conda envs.** The `ale`/`retro`/`gym`
> backends coexist in one env (`fmri-gym`, numpy 2). The `vgdl` backend is built
> on *old* `gym` and requires `numpy<2` + `setuptools<81`, which is
> irreconcilable with that — so VGDL curricula run in the separate
> `language_and_experience` env. This is a dependency reality, not a design
> split: the framework *core* needs only `pygame`+`numpy` and imports cleanly in
> both envs; adapters are imported lazily, so each env only needs the backends
> it uses. Same code, same curriculum schema, same output format everywhere.

It is the generalization of two single-engine siblings — `../ALE-examples`
(Atari) and `../stable-retro-examples` (retro) — and mirrors the fMRI/MEG tasks
in `../vgdl` and `../mario_task`, stripped to a POC and made engine-agnostic.

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

stable-retro needs game ROMs imported once (see stable-retro docs / the sibling
`stable-retro-examples`); Atari ROMs ship with `ale-py`.

## Quick start

```bash
# Built-in mixed demo: an Atari game, a Genesis game, and CartPole, back to back
python fmri_play.py --subject sub-01 --dummy-trigger

# Your own curriculum:
python fmri_play.py --subject sub-01 --curriculum configs/demo_mixed.json

# VGDL games run in their own conda env (numpy<2):
conda activate language_and_experience
VGDL_REPO=~/Documents/projects/DBP/language_and_experience \
  python fmri_play.py --subject sub-01 --curriculum configs/demo_vgdl.json
```

Runtime flow: experimenter screen (**SPACE**) → "Waiting for scanner..." →
scanner **trigger `=`** (anchors the session clock) → curriculum phases → done.
`ESC` quits early but still saves. Flags: `--size 1280x1024`, `--fullscreen`,
`--save-pixels` (ALE only; see below).

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

- **ale**: built from the game's action meanings (arrows move, SPACE fires).
- **retro**: keyboard → console buttons (arrows move; Z/X/C = A/B/C; ENTER =
  start); multiple held keys combine (e.g. RIGHT+Z).
- **gym**: a generic default (arrows → first Discrete actions, or ±limits on
  Box dims). Because a bare `Discrete(n)` has no inherent meaning, **specify
  `keys` per game** for anything non-obvious, e.g. `{"LEFT": 0, "RIGHT": 1}` for
  CartPole. Combos use `"LEFT+SPACE"`.

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

Of the games in the DBP survey, ~15 already expose a Gymnasium API and drop
straight into the `gym` backend; stable-retro titles use the `retro` backend;
and the VGDL games use the `vgdl` backend. Each new game typically needs only a
per-game **keymap** and, if its `Discrete` actions aren't self-evident, a `keys`
override.

**old-`gym` games (chess, hanoi, Sokoban, Baba, NetHack).** These load through
**shimmy** with `"legacy_gym": true` (routed via `GymV21Environment-v0`). One
caveat learned the hard way: shimmy's v0.21 compat calls the removed `.seed()`
method, and `gym==0.26` itself is incompatible with `numpy>=2`. So, like VGDL,
these are best run in a dedicated `numpy<2` env with the matching `gym`/game
package installed. The `legacy_gym` code path exists in `default.py`; wire up a
specific game once its package is installed in such an env.

## Future work / TODO

- [x] Configurable **state stride** (`"state_stride": K`) — done.
- [x] **VGDL backend** (ccolas/language_and_experience) — done.
- [ ] Finish the **old-`gym` / shimmy** path against a real game (Sokoban,
      chess) in a `numpy<2` env; currently the code path exists but is untested
      end-to-end (blocked on installing a game package).
- [ ] **Photodiode sync square** and **LSL / parallel-port markers** for
      MEG/EEG-grade timing (see `../mario_task/markers.py`); fMRI's slow HRF
      makes the `=`-anchored software clock adequate.
- [ ] **retro `.bk2` movie logging** as an alternative to per-frame states
      (frame-exact, tiny), folding in `stable-retro-examples`.
- [ ] Per-subject deterministic curriculum generation; multi-run structure with
      one trigger per fMRI run.
- [ ] More adapters: ViZDoom/COOM, MiniHack, crafter, MuJoCo (`qpos/qvel` as
      state vars) — each a small `EnvAdapter`.
- [ ] Button-box / MRI-safe response device key remapping.
- [ ] BIDS-style output layout + `events.tsv` per run.
- [ ] A replay/QC utility to render any block to video from its states.
- [ ] Crash-safe incremental logging (stream frames to disk) for long runs.
