# DBP games — fmri-gym integration status

Status of every game in the DBP survey that exposes a **Gymnasium** API (15
games), after installing each in the `fmri-gym` conda env and actually running
it through the framework. One demo curriculum per game lives in
`configs/dbp_games/<name>.json`; each carries a matching `_status` / `_verified` /
`_note`.

Legend: ✅ verified · 🟡 runs, display-only · ⚠️ needs assets · ❌ doesn't fit · ⏸️ deferred

## Summary table

| Game | config | backend | status | one-liner |
|---|---|---|---|---|
| Atari | `atari` | ale | ✅ verified | any `ALE/*` id; keymap auto-derived |
| VGDL | `vgdl` | vgdl | ✅ verified | needs the gymnasium-ported fork checkout |
| Crafter | `crafter` | crafter | ✅ verified | new adapter (old-gym API); obs is the frame; logs achievements |
| MiniHack | `minihack` | minihack | ✅ verified | new adapter; pixel obs + 8-way compass; logs blstats/glyphs/message |
| Tile Match | `tile-match-gym` | gym | 🟡 display-only | runs & logs, but `Discrete(84)` swaps → no keyboard play |
| Tobu Tobu Girl DX | `tobutobugirldx` | retro | ⚠️ needs ROM | backend proven (Airstriker); import ROM + confirm integration name |
| Nomolos | `nomolos` | retro | ⚠️ needs ROM | ditto |
| Anguna | `anguna` | retro | ⚠️ needs ROM | ditto |
| 2048 | `2048` | gym | ❌ broken upstream | package's own `reset()` crashes (numpy bug); no render |
| Pathery | `pathery` | gym | ❌ text-only | ANSI render + `MultiDiscrete` placement → not real-time visual |
| Wordle | `wordle` | gym | ❌ text-only | `Text(5)` typed-word action + text render |
| Mastermind | `mastermind` | gym | ❌ can't install | requires Python ≥3.13 (env is 3.11) |
| Rush Hour | `rush-hour` | gym | ❌ not packaged | repo has no `setup.py`/`pyproject` |
| COOM | `coom` | gym | ⏸️ deferred | ViZDoom; own env factory → needs a custom adapter |
| Craftium | `craftium` | gym | ⏸️ deferred | needs the Luanti/Minetest engine built |

**Takeaway:** every game that is a **real-time visual game with a pixel frame**
works (Atari, VGDL, Crafter, MiniHack, stable-retro, plus tile-match for
display). The ones that don't fit are **turn-based / text / typed-input** games
— a genre mismatch with a scanner game loop, not a framework limitation (the
2048 case is a separate upstream bug).

## Per-game detail

### ✅ atari (ale)
Any `ALE/*` env id. Keymap auto-derived from action meanings. Per-frame
`clone_state` savestate; RAM logged; optional lossless indexed pixels.

### ✅ vgdl (vgdl)
Runs from the gymnasium-ported fork
<https://github.com/tomov/language_and_experience/tree/dbp>. Set `VGDL_REPO` +
`PYTHONPATH` to the checkout. Per-frame `get_state`/`set_state` savestate;
symbolic object grid + collision events logged. Games: aliens, beesAndBirds,
avoidGeorge, jaws, missile_command, plaqueAttack, portals, preconditions,
pushBoulders, relational.

### ✅ crafter (crafter)  — new backend
`pip install crafter`. Uses the **old-gym API shape** (reset()→obs only,
step()→4-tuple), so a dedicated `crafter` adapter normalizes it to gymnasium.
The observation IS the 64×64×3 RGB frame; `Discrete(17)` actions (arrows move,
SPACE=interact, S=sleep; place/make via a `keys` override). No savestate →
seed+replay. Achievements logged as an analysis variable.

### ✅ minihack (minihack)  — new backend
`pip install minihack` (needs `setuptools<81` for `pkg_resources`; pulls
`nle`, downgrades gymnasium to 1.2). Default `render()` is None and obs is
ASCII/tty, so the adapter requests a `pixel` observation (336×1264×3) and
displays that. `Discrete(8)` compass (arrows = N/E/S/W; diagonals via `keys`
override 4–7). No savestate → seed+replay. blstats/glyphs/message logged.

### 🟡 tile-match-gym (gym, display-only)
`pip install tile-match-gym` (pulls numba → numpy 1.26). Import the package to
register `TileMatch-v0`; constructor needs board kwargs (via `env_kwargs`).
5-tuple, `render_mode="rgb_array"` → (324,273,3). Runs & logs fine, but
`action_space = Discrete(84)` (choose a tile swap) has no natural arrow-key
mapping — a human would need a pointer/selection input mode.

### ⚠️ tobutobugirldx / nomolos / anguna (retro)
stable-retro homebrew integrations. The `retro` backend is proven end-to-end
(Airstriker-Genesis). These need their specific ROMs imported
(`python -m retro.import`) and the exact integration name confirmed; no ROMs
were available here.

### ❌ 2048 (gym) — broken upstream
`pip install --no-deps gym-2048` (it pins numpy~=1.14). Its own
`_place_random_tiles` crashes on reset with a numpy shape-mismatch, under both
gym and gymnasium. No `render()` either. Would need an upstream patch + a
custom renderer; better to use a different 2048 env.

### ❌ pathery (gym) — text-only
`pip install git+.../PatheryEnv`. 3 ids (`pathery_env/Pathery-*`).
`render_mode="rgb_array"` raises `AssertionError` — only `ansi` (str).
`action_space = MultiDiscrete([9,17])` (place a wall). No pixel frame +
placement actions → needs a text→image renderer and a selection input.

### ❌ wordle (gym) — text-only
`pip install git+.../wordle-gym-environment` (installs top-level `envs`/`utils`;
use `WordleEnv` directly — its gym registration is fragile). `action_space =
Text(5)` (type a 5-letter word); ANSI/text render only, no rgb_array. Typed-word
game → needs a bespoke text input+render.

### ❌ mastermind (gym) — can't install
`MastermindGymnasiumEnvironment` requires **Python ≥3.13**; the env is 3.11, so
pip refuses. Also turn-based (would need custom input). Revisit in a py3.13 env.

### ❌ rush-hour (gym) — not packaged
`chrplr/Rush-Hour` has no `setup.py`/`pyproject.toml` → not pip-installable.
Sliding-block puzzle; would need vendoring + a custom adapter and selection
input.

### ⏸️ coom (gym) — deferred
`TTomilin/COOM`, a ViZDoom-based continual-RL suite with its **own env factory**
(not `gymnasium.make`). Needs a dedicated adapter + ViZDoom + Doom WADs.

### ⏸️ craftium (gym) — deferred
`mikelma/craftium` needs the **Luanti/Minetest** engine built. Env ids
`Craftium/<Task>-v0`; Box/dict actions likely need a custom adapter.

## Environment notes

All backends were verified to run **together in one `fmri-gym` conda env**
despite third-party pins:

- `minihack` pins **gymnasium == 1.2**
- `tile-match-gym` (numba) pins **numpy == 1.26**
- `minihack` imports `pkg_resources` → needs **setuptools < 81**

ale / retro / gym / vgdl / crafter / minihack were all re-verified end-to-end
under these versions. If you only need a subset of backends, install fewer and
the pins relax.

## Framework changes made during this pass

- New `crafter` and `minihack` backends (`fmri_gym/adapters/{crafter,minihack}.py`).
- `import_module` option on the default gym adapter (many envs only register
  their ids as a side effect of importing their package).
