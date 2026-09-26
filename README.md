# fmri-gym

**One fMRI experiment framework for (almost) any Gymnasium-compatible game.**

A proof-of-concept framework that turns games into neuroimaging tasks — fixed
uniform display, scanner-trigger sync, a declarative curriculum of phases, and
compact reconstructable per-frame logging — and works across game engines
through small pluggable **adapters**:

| Backend (`"backend"`) | Games | Engine |
|---|---|---|
| `ale`         | Atari 2600 (`ALE/Pong-v5`, …) | ALE / Stella |
| `retro`       | NES / SNES / Genesis / GB / … (`Airstriker-Genesis-v0`, …) | stable-retro / libretro |
| `gym`         | **any** Gymnasium env (`CartPole-v1`, MuJoCo, Box2D, toy_text, …); old-`gym` envs via shimmy | various |
| `vgdl`        | VGDL games (`aliens`, `beesAndBirds`, …) | py-vgdl / pygame |
| `crafter`     | Crafter (open-world survival) | crafter |
| `minihack`    | MiniHack tasks (pixel obs) | minihack / NLE |
| `nethack`     | NetHack (`NetHack*-v0`; TTY rendered to pixels) | nle |
| `aigamestore` | AI GameStore p5.js/browser games (`game1`…`game10`) | p5.js via headless browser |
| `vizdoom`     | Doom action-shooter scenarios (COOM's engine) | ViZDoom |
| `coom`        | COOM's own continual-RL scenarios (`pitfall`, `chainsaw`, …), read as ViZDoom scenario assets from a `COOM_REPO` checkout (COOM package itself not installed -- conflicting `gymnasium` pin) | ViZDoom |
| `overcooked`  | Overcooked co-op cooking (social) | overcooked_ai |
| `baba`        | Baba Is You (rule-manipulation puzzle) | baba-is-ai |
| `rushhour`    | Rush Hour sliding-block puzzle | `rushhour-gym` (PyPI; fetches its Go engine) |
| `supertuxkart`| SuperTuxKart 3D racing (needs a real GL display) | pystk2 |
| `stk_gym`     | SuperTuxKart, the current game: frames from the game's gym server, keys to its player controller (needs a real GL display) | [chrplr/stk-code](https://github.com/chrplr/stk-code) fork |

> **All backends run in ONE env and ONE process.** Verified: a single session
> with ALE + retro + gym + VGDL blocks back-to-back, and each of Crafter /
> MiniHack in turn. VGDL originally required *old* `gym` + `numpy<2`, which
> conflicted with the numpy-2 backends; that's resolved by a fork whose VGDL
> source is ported to gymnasium
> ([tomov/language_and_experience @ dbp](https://github.com/tomov/language_and_experience/tree/dbp)).
> Adapters are imported lazily, so an env only needs the backends a curriculum
> actually uses. Same code, same curriculum schema, same output format everywhere.

## Install

See [Machine requirements](MACHINE_REQUIREMENTS.md) for minimum and recommended
hardware, and the [local test log](docs/local-testing/2026-09-14.md) for measurements
and their scope.

With [uv](https://docs.astral.sh/uv/):

```bash
sudo apt install libportaudio2               # PortAudio; every backend needs it
uv sync --extra dbp                          # .venv/ with the nine DBP backends, pinned by uv.lock
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/dbp_games/atari__pong.json --ses 1 --run 1
```

`dbp` is the nine DBP games. Each backend is also its own extra (`ale`,
`retro`, `vizdoom`, `minihack`, `rushhour`, …), and `--extra all` installs
every backend.

Without uv: pip into a venv of your own, and `python fmri_play.py` in place of
`fmri-play`:

```bash
pip install -e ".[dbp]"            # private default index? add --index-url https://pypi.org/simple
python fmri_play.py --subject sub-01 --dummy-trigger --curriculum configs/dbp_games/atari__pong.json --ses 1 --run 1
```

Atari ROMs ship with `ale-py`. For the `retro` backend you must supply and
import game ROMs once — see [Running stable-retro games](#running-stable-retro-games).
For the `vgdl` backend see [Running VGDL games](#running-vgdl-games);
for Rush Hour, [Running Rush-Hour](#running-rush-hour); for the current
SuperTuxKart, [Running SuperTuxKart from the stk-code fork](#running-supertuxkart-from-the-stk-code-fork-stk_gym).

## Quick start

```bash
# --- per-family demo curricula (all tested end-to-end; ~15 s per block) ---
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/demo_atari.json --ses 1 --run 1    # 10 popular Atari games
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/demo_classic.json --ses 1 --run 1  # all 5 classic-control
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/demo_text.json --ses 1 --run 1     # all 5 toy_text (render RGB; turn-based, arrow keys)
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/demo_box2d.json --ses 1 --run 1     # LunarLander, BipedalWalker, CarRacing  (`box2d` extra)
MUJOCO_GL=egl uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/demo_mujoco.json --ses 1 --run 1   # 10 MuJoCo tasks  (`mujoco` extra)
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/demo_aigamestore.json --ses 1 --run 1  # 10 AI GameStore p5.js games (`aigamestore` extra; see below)
VGDL_REPO=../language_and_experience PYTHONPATH=../language_and_experience \
  uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/demo_vgdl_all.json --ses 1 --run 1   # all 10 VGDL games (see below)

# demo_mixed spans EVERY backend in one session (Pong/ale, Airstriker/retro,
# Crafter, MiniHack, Aliens/vgdl, MountainCar/classic, FrozenLake/toy_text,
# CarRacing/box2d, WaterSort/aigamestore) -- needs the VGDL repo + box2d-py +
# crafter + minihack + playwright:
VGDL_REPO=../language_and_experience PYTHONPATH=../language_and_experience \
  uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/demo_mixed.json --ses 1 --run 1

# Play ONE game on its own, for a long stretch (see configs/dbp_games/):
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/dbp_games/atari__pong.json --ses 1 --run 1
```

A run is a JSON file and a session is a `.sh` script (below). Write them by
hand, or design them in `fmri-edit` -- either way `fmri-play` reads the same
files, and refuses one it cannot play. `fmri-play` plays one run and nothing
else; the editor is a command of its own, which opens on a run, a session, or
a new run. **Play** there saves what it shows and starts it: an `fmri-play`
command for a run, the script itself for a session. Its Launch tab holds the
flags of that launch (subject, session, monitor, window, the test switches),
which belong to the launch, not to the files. The editor is the `gui` extra:

```bash
uv sync --extra dbp --extra gui
uv run fmri-edit --curriculum configs/demo_meg.json
uv run fmri-edit --session configs/ses1.sh
uv run fmri-edit                                    # start from a new run
```

Drop `--dummy-trigger` for a real session (then press SPACE, then wait for the
`=` scanner trigger). For VGDL setup see [Running VGDL games](#running-vgdl-games).

stable-retro games play their native audio, and ViZDoom does when its config sets
`env_kwargs.audio_buffer_enabled`. Use `--no-audio` or a game phase's
`"audio": false` to mute playback; logged audio is unchanged. Each frame's sound
starts a constant delay after the flip that shows it, measured from the system's
default output at start-up and logged. A game with sound must run at its engine's
own frame rate (ViZDoom: `fps * frame_skip == 35`; Genesis cores: 59.92), or the
block stops and names the fps that fits.

Note: **MuJoCo and Box2D use continuous (`Box`) action spaces** — the default
keymap pushes arrows to each dim's limit, so they render and log fine but aren't
really human-playable without a per-game control scheme. Everything else in
these families is keyboard-playable.

### Per-game configs (`configs/dbp_games/`)

There is **one config per individual supported game** (63 of them), sourced from
the DBP game spreadsheet, so you can play any single game on its own for a long
stretch with a one-line command. Filenames are `<class>__<game>.json`:

```bash
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/dbp_games/atari__pong.json --ses 1 --run 1
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/dbp_games/text__frozenlake.json --ses 1 --run 1
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/dbp_games/aigamestore__game1.json --ses 1 --run 1
```

Each is a minimal `message → fixation → game (300 s) → fixation` curriculum with
the right per-game keymap/settings baked in. Coverage by class:

| class prefix | count | games |
|---|---|---|
| `atari__` | 10 | pong, breakout, spaceinvaders, mspacman, seaquest, qbert, asterix, beamrider, enduro, boxing |
| `vgdl__` | 10 | aliens, beesAndBirds, … (needs the VGDL fork checkout) |
| `minihack__` | 6 | room5x5/15x15, mazewalk9x9, river, corridor, eat |
| `nethack__` | 1 | score (base NetHack; TTY rendered to a pixel frame) |
| `aigamestore__` | 10 | game1…game10 (p5.js; needs playwright) |
| `classic__` | 5 | cartpole, mountaincar, acrobot, pendulum, mountaincarcontinuous |
| `box2d__` | 3 | lunarlander, bipedalwalker, carracing (needs box2d-py) |
| `mujoco__` | 10 | ant, halfcheetah, hopper, humanoid, … (`MUJOCO_GL=egl`) |
| `text__` | 5 | frozenlake, frozenlake8x8, cliffwalking, taxi, blackjack (turn-based) |
| `crafter__` | 1 | crafter |
| `craftium__` | 1 | choptree (Luanti voxel; other ids: Room/Speleo/OpenWorld/…) |
| `vizdoom__` | 10 | basic, deadly_corridor, defend_center, defend_line, health_gathering_supreme, my_way_home, predict_position, take_cover, deathmatch (Doom; COOM's engine; other `Vizdoom*-v1` scenarios work too), plus `take_cover_defend_line` running two of them back to back in one session |
| `coom__` | 9 | pitfall, chainsaw, hide_and_seek, health_gathering, arms_dealer, parkour, raise_the_roof, run_and_gun, floor_is_lava (needs the COOM repo checkout) |
| `overcooked__` | 1 | cramped_room (co-op cooking; other layouts) |
| `baba__` | 1 | make_win (rule-manipulation puzzle; other ids) |
| `rushhour__` | 1 | easy (sliding-block puzzle). `rushhour__complete.json` is the full self-paced session of Rush-Hour's own program, then the rest of the library: all 49 puzzles, the first 12 easiest-first and the other 37 in a fixed shuffled order, one game phase each, with ready screens and solved feedback as message phases |
| `supertuxkart__` | 1 | race (3D racing; needs a real GL display) |
| `stk_gym__` | 1 | race (the current SuperTuxKart via its gym server; needs the fork built and a real GL display) |
| `retro__` | 3 | tobutobugirldx, nomolos, anguna (need ROMs imported) |

Each config carries `_game` / `_note` (per-game setup reminders). Games use
`mode: "duration"` (300 s) so they auto-restart on game-over for continuous
play; `ESC` quits. The `_note` flags class-specific requirements (VGDL repo,
playwright, box2d-py, MuJoCo GL, ROM import).

> **Not covered** — configs live under `configs/dbp_games/unsupported/`, each
> with a `_status`/`_note` explaining why: games with no real-time pixel
> interface — `2048` (upstream reset bug), `pathery`/`wordle` (text/placement),
> `tile-match-gym` (display-only, `Discrete(84)` swaps → no keyboard play),
> `mastermind` (needs Python ≥3.13), and `craftium` (needs the Luanti engine
> built).

Runtime flow: experimenter screen (**SPACE**) → "Waiting for scanner..." →
scanner **trigger `=`** (anchors the session clock) → curriculum phases → done.
`ESC` quits early but still saves. Flags: `--size 1280x1024`, `--fullscreen`,
`--monitor 1` (which screen, when there are several), `--no-vsync` (see
[Timing](#timing-what-is-stamped-when)). The editor opens on its Launch tab,
fullscreen ticked and the monitor picked there.

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

The `configs/dbp_games/retro__{tobutobugirldx,nomolos,anguna}.json` configs
need ROMs for exactly this reason: the `retro` backend itself is verified
(with Airstriker), but those titles won't run until you import their ROMs and
confirm the integration name.

## Running VGDL games

The `vgdl` backend drives the VGDL games from a gymnasium-ported fork:
**[tomov/language_and_experience @ dbp](https://github.com/tomov/language_and_experience/tree/dbp)**.
Because it runs under gymnasium + numpy 2, no separate env is needed — the
same `fmri-gym` env works.

1. Clone the fork (the `dbp` branch has the gymnasium port) **as an adjacent
   repo** — the commands below assume it sits next to `fmri-gym`:

   ```bash
   git clone -b dbp https://github.com/tomov/language_and_experience.git ../language_and_experience
   ```

2. Point the framework at the checkout and add it to `PYTHONPATH` (so
   `src.vgdl...` is importable), then run a VGDL curriculum:

   ```bash
   VGDL_REPO=../language_and_experience \
   PYTHONPATH=../language_and_experience \
     uv run fmri-play --subject sub-01 --curriculum configs/demo_vgdl_all.json --ses 1 --run 1
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

## Running COOM games

The `coom` backend plays [TTomilin/COOM](https://github.com/TTomilin/COOM)'s
own continual-RL Doom scenarios (`pitfall`, `chainsaw`, `hide_and_seek`,
`health_gathering`, `arms_dealer`, `parkour`, `raise_the_roof`, `run_and_gun`,
`floor_is_lava`) -- distinct from the stock ViZDoom scenarios the `vizdoom`
backend already covers (DeadlyCorridor, DefendCenter, ...).

COOM's own Python package pins `gymnasium==0.28.1`, which conflicts with
minihack's `gymnasium==1.2` pin in this shared env, so **the COOM package is
never installed or imported**. Instead the `coom` backend drives
`vizdoom.DoomGame` (the `vizdoom` extra) directly against COOM's own scenario
config/WAD files, read straight off disk from a checkout:

1. Clone COOM as an adjacent repo:

   ```bash
   git clone https://github.com/TTomilin/COOM.git ../COOM
   ```

2. Point the framework at the checkout (no install, no `PYTHONPATH` needed --
   only the scenario asset files under `COOM/env/scenarios/` are read) and run
   a COOM curriculum:

   ```bash
   COOM_REPO=../COOM \
     uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/coom__pitfall.json --ses 1 --run 1
   ```

   `COOM_REPO` locates `<repo>/COOM/env/scenarios/<scenario>/conf.cfg` and
   `<task>.wad` (`env_kwargs.task`, default `"default"`; some scenarios like
   `run_and_gun` ship extra task variants -- `blue`, `red`, `hard`, ...); a
   phase can also override it per block with a `"repo"` field.

Every scenario always exposes exactly 4 buttons (`TURN_LEFT`, `TURN_RIGHT`,
`MOVE_FORWARD`, plus one of `JUMP`/`ATTACK`/`SPEED`/`USE`), so the backend
derives a sensible default keymap automatically (arrows to turn/move, that
4th button on SPACE/LSHIFT/E) -- no curriculum `keys` override needed unless
you want to remap it. COOM blocks log the raw ViZDoom `game_variables`
(health, ammo, position, ...) as an analysis variable; there's no in-memory
savestate, so reconstruction is via seed + action replay like most backends.
No native audio yet, unlike the `vizdoom` backend's `sound()`.

## Running AI GameStore games

[AI GameStore](https://aigamestore.org) is a benchmark of LLM-generated
**browser games** (plain HTML + JavaScript + p5.js). There's no standard
p5.js↔Gymnasium bridge, so the `aigamestore` backend builds one with a headless
(or headed) browser via **Playwright**:

- a tiny local HTTP server serves the vendored games (`vendor/aigamestore/`; ES
  modules need `http://`, not `file://`);
- currently-held keyboard keys are pressed/released in the page each step;
- the game's `<canvas>` is screenshotted → the RGB frame for display;
- each game exposes `window.getGameState()` with a `score` and a `gamePhase`
  (START / PLAYING / GAMEOVER …), mapped to reward (score delta) and done, and
  logged (scalar fields as `state_*` analysis variables).

Setup — the `aigamestore` extra (Playwright + Pillow) and a browser (uses the
**system Chrome** by default):

```bash
uv sync --extra aigamestore        # or: pip install -e ".[aigamestore]"
# then either rely on system Chrome (default), or install the bundled one:
# playwright install chromium   # and set "browser_channel": null in the phase
```

Run the 10 vendored public games (each keyboard-controlled — arrows + SPACE/Z/
ENTER; `game1` = Water Sort, `game2` ≈ Angry Birds, …):

```bash
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/demo_aigamestore.json --ses 1 --run 1
```

Phase fields: `game` (`"game1"`…`"game10"`, or an `http(s)://…/index.html`
URL), `games_dir` (override the vendored dir), `headed` (show the window),
`browser_channel` (`"chrome"` default, or `null` for bundled Chromium),
`start_key` (default `Enter`, pressed once to leave the START screen).

> Note: a browser step (screenshot + `getGameState`) costs ~0.1–0.5 s, so
> effective fps is lower than the emulator backends — fine for these
> puzzle/casual games, and the framework paces to whatever it can sustain.

## Running Rush-Hour

[Rush-Hour](https://github.com/chrplr/Rush-Hour) is a sliding-block puzzle
written as a psychophysics experiment in Go, with its rules in a small engine
binary that both an agent and a participant play through `rushhour-gym`. The
`rushhour` backend drives the package's `RushHourHuman-v0` — the experiment
program's own interface as an env: car selection on four buttons, its picture,
its results columns in `info` — so the adapter is a keymap plus the fields to
log. One game phase is one puzzle (`"puzzle": "p07"`); the program's ready
screens, blank intervals and solved feedback are `message` phases the
curriculum lists around each puzzle, so every puzzle is its own block in the
manifest and its own `.npz`. Nothing to install beyond the `rushhour` extra: on first use the
package downloads the engine of its matching release into
`~/.cache/rushhour-gym/` (checksum-verified; Linux x86-64, macOS arm64, Windows
x86-64). On a machine without network, run a config once while online or copy
that directory; `RUSHHOUR_ENV_BIN` names a binary of your own.

```bash
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/dbp_games/rushhour__easy.json --ses 1 --run 1      # 5 min of random easy puzzles
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/dbp_games/rushhour__complete.json --ses 1 --run 1   # the program's session then the rest of the library: 49 puzzles, one block each
```

Controls, phase fields and the logged columns are documented in the configs'
`_note`s and in the package's README ("A person at the board").

## Running SuperTuxKart from the stk-code fork (`stk_gym`)

The [chrplr/stk-code](https://github.com/chrplr/stk-code) fork is the current
SuperTuxKart with a gym server built in (`--gym`), and `stk_gym` is its Python
client. The `stk_gym` backend drives `stk_gym.StkEnv` with
`render_mode="rgb_array"` and `action_mode="keys"`: the game renders into a
window that is created hidden, every step brings the frame back and fmri-gym
shows it; the held keys go to the game's own player controller, so steering
ramps and skids latch as they do for a keyboard. Participant and model are in
front of the same env object, and a block replays from `episode_seeds` +
`actions`. The adapter is a keymap and the fields to log. The older `supertuxkart`
backend (pystk2) is untouched; the two coexist.

```bash
uv pip install "fmri-gym[stk_gym]"      # or: pip install supertuxkart-gym
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/dbp_games/stk_gym__race.json --ses 1 --run 1
```

No checkout and no build: the wheel is pure Python and fetches the game with a
trimmed asset pack (254 MiB, five tracks) from its GitHub release the first time
an env is made, into `~/.cache/supertuxkart-gym`. It says so while it downloads,
and never does it twice. Linux x86_64 only for now; on anything else it says
which platform it has no pack for.

To work on the engine itself, build the fork and install its client instead --
a checkout is preferred over the downloaded pack, so nothing else changes:

```bash
git clone https://github.com/chrplr/stk-code.git ../stk-code
cmake -S ../stk-code -B ../stk-code/build -DCMAKE_BUILD_TYPE=Release && cmake --build ../stk-code/build -j
uv pip install -e ../stk-code/python    # or pip install -e, in the same env
```

Either way the binary can be overridden with `STK_ENV_BIN`, and
`STK_ENV_OFFLINE=1` forbids the download outright. It needs a real OpenGL
display (the frame is the game's rendering). `fps` must equal the game's physics
rate over `frame_skip` (120 / 2 = 60 in the config); the config's `_note`s list
the keys, the phase fields and the logged columns, and the fork's
`python/README.md` ("Frames", "Reproducibility") the details and measured cost.

## Design: the experiment loop never knows the engine

```
fmri_gym/
  run.py        # trigger, clock, curriculum loop, phases  — 100% engine-agnostic
  display.py        # pygame: fixed window, aspect-fit frame, fixation, text; vsync-locked flip + call_on_flip
  logging.py        # manifest.json + one compressed .npz per game block
  triggers.py       # run-start sync (wait/send/none) + MEG/EEG trigger codes over lsl/serial/parallel
  photodiode.py     # `python -m fmri_gym.photodiode`: flash a patch to measure the flip-to-photon offset
  checks.py         # the rig check: its phases, the rig file, report.html/.md, rigchecks.tsv, `pool`
  adapters/
    base.py         # EnvAdapter + KeySpec flavors + FrameState (the seam)
    ale.py          # clone_state, getRAM, lossless indexed pixels
    retro.py        # em.get_state, get_ram, decoded info vars, console-button keymap
    default.py      # ANY gym env: rgb frames, seed+replay, obs-as-state
    vgdl.py         # VGDLEnv: get_state/set_state, symbolic grid + events
    crafter.py      # old-gym-API wrapper; obs is the frame; achievements
    minihack.py     # pixel obs + compass keymap; blstats/glyphs/message
    nethack.py      # base NLE: TTY grid -> RGB; vi-key movement; blstats
    aigamestore.py  # p5.js browser games via Playwright: canvas->RGB, getGameState
    rushhour.py     # Go engine via rushhour-gym; select+slide UI, rushui look, Rush-Hour's log columns; one puzzle per block
    stk_gym.py      # the current SuperTuxKart via stk_gym: frames from the game's hidden window, held keys as the env's action
fmri_play.py        # CLI entry point
configs/            # example curricula
vendor/aigamestore/ # the 10 public AI GameStore games (p5.js/HTML/JS)
```

The loop (`run.py`) only ever calls the adapter — never `env.unwrapped`, an
emulator, or an engine module. Each engine-specific concern lives behind
**`EnvAdapter`**:

```python
class EnvAdapter:
    def _make(self, spec)         -> gym.Env       # build the env for a block
    def _keyspec(self)            -> KeySpec       # held keys -> action
    def reset(self, seed)         -> (obs, info)
    def capture(self, obs, info)  -> FrameState    # per-frame state to log
    def restore(self, blob)       -> None          # inverse of capture().blob
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

A **run** is one JSON file: a `"curriculum"` of phases, an optional
`"triggers"` section (below), and `_`-prefixed notes. A bare list, an unknown
top-level key or an unknown phase `type` stops the run at start-up with the
reason.

One config is one run. A whole scanning session is a plain shell script with
one line per run, in order -- so every run is a process of its own, with a
fresh interpreter, display and trigger port:

```sh
#!/bin/sh
# fmri-gym session: one line per run, in order.
set -e
SES=${1:-$(uv run fmri-ses --subject sub-01)}
uv run fmri-play --curriculum configs/pong.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
./scripts/localizer.sh "$SES"
# uv run fmri-play --curriculum configs/mario.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
```

**The numbers come from the script, not from the disk.** The `SES=` line picks
the session once, so every run lands in it: the script's own argument if it was
given one (`sh ses1.sh 003` resumes session 3), else the subject's next free
session. Each run states its `--run`, which is its place among the lines that
play that task -- so it is the same run number however the session went, and
skipping a line renumbers nothing after it. Both flags are required of
`fmri-play`: it never picks a number itself.

A run whose folder already has data is **re-acquired, never overwritten**: it
writes to `..._02` (then `_03`) beside the attempt that stopped. The suffix
names the folder only, so the re-acquisition replays the same episodes as the
run it replaces.

`set -e` stops the script at the first run that fails or is quit with ESC
(`fmri-play` then exits with status 3). A line that is not an `fmri-play` run
is any command of yours, as typed; a commented line is a skipped run, which is
how a stopped session is resumed. Write it by hand, or in `fmri-edit`, whose Session manager tab
has two panels, each a form and the text it stands for: Session design (the
list of lines -- add an existing or a new config, an external script; repeat,
reorder, skip, "Start here" -- or the script itself) and Run design (the
selected run's phases, or its JSON). Each panel starts with a drop-down of what
`configs/` holds: the Session one opens a script, the Run one opens a config
(or, in a session, adds it after the selected line); File > Open takes a `.sh`
like a `.json` from anywhere. Save writes the script and the configs you
edited. Run it from the repo root: `sh configs/ses1.sh`.

Three example sessions of about an hour each ship in `configs/` -- eleven
runs of one DBP game apiece (5 min of play, plus the instructions screen and
the start-up between runs):

```bash
sh configs/ses_dbp_mix.sh      # one run per genre: Crafter, COOM, MiniHack, Rush Hour, Baba, ViZDoom, AI GameStore...
sh configs/ses_dbp_doom.sh     # the nine COOM scenarios, then two ViZDoom ones
sh configs/ses_dbp_puzzle.sh   # nine AI GameStore puzzles, Rush Hour, Baba Is You
sh configs/ses_dbp_mix.sh 003  # ... into session 3: how one that stopped is resumed
```

They assume `sub-01` and a 1024x768 window, and take the subject's next free
session unless given one: open one in `fmri-edit --session
configs/ses_dbp_mix.sh` to change any of that, or `--dummy-trigger` a run of
it at the desk.

```jsonc
{"type": "fixation", "duration": 2.0}                 // "+" for N seconds
{"type": "message", "text": "Get ready", "duration": 2.0}  // text: string or list of lines; omit duration to wait for a key
{"type": "survey", "n_points": 7, "questions": ["...","..."]}

{"type": "game",
 "backend": "ale",              // "ale" | "retro" | "gym"
 "game": "ALE/Pong-v5",         // env id for that backend
 "mode": "duration",            // "duration" = replay until time up; "episode" = play N episodes
 "duration": 30.0,              // seconds (duration mode)
 "n_episodes": 1,               // episodes (episode mode)
 "max_duration": 300.0,         // hard wall-clock safety cap (episode mode)
 "fps": 30,                     // required: steps (and frames) per second. The engine's own rate
                                // (console cores and Atari ~60, Doom 35 / frame_skip) plays the game
                                // at its real speed and fits its sound; the editor's Controls tab
                                // shows it. Any other value plays the game slower or faster: the
                                // manifest logs "speed" and the console says so when it is not 1
 "turn_based": false,           // step only on a key PRESS, not per frame (grid/toy_text games)
 "seed": 1234,                  // optional base seed: episodes play with seed, seed+1, ...
                                // Pinned, every participant and run gets the same episodes.
                                // Left out, it is derived from the run (sub/ses/task/run) and
                                // the phase, so no two runs replay each other's; fmri-play
                                // prints each phase's seed, the editor shows it (and Pin
                                // copies it in), the manifest logs it

 "state_stride": 1,             // save a full savestate every K frames (see below)
 "state": "Level1",             // retro: named savestate/level (optional)
 "scenario": null,              // retro: scenario name (optional)
 "level": 0,                    // vgdl: level index; also uses "game","block_size"
 "keys": {"LEFT": 0, "RIGHT": 1}, // override keyboard->action map (see below)
 "save_pixels": false}          // also store lossless pixels, where the backend can
```

### Keymaps

Each backend builds a default keyboard→action map:

- **ale**: built from the game's action meanings (arrows move, SPACE fires).
- **retro**: keyboard → console buttons (arrows move; Z/X/C = A/B/C; ENTER =
  start); multiple held keys combine (e.g. RIGHT+Z).
- **vizdoom**: arrows move/turn, Z/X strafe, SPACE shoots. Held keys combine
  when the scenario is made with `"env_kwargs": {"max_buttons_pressed": 0}`
  (a `MultiBinary` space — walk forward while turning); `keys` are the
  scenario's `Discrete` action indices either way. Scenarios that also declare
  the mouse axes (Deathmatch, the full-game maps) get those axes dropped, since
  the scanner has no mouse: turning is `TURN_LEFT`/`TURN_RIGHT`.
- **gym**: a generic default (arrows → first Discrete actions, or ±limits on
  Box dims). Because a bare `Discrete(n)` has no inherent meaning, **specify
  `keys` per game** for anything non-obvious.

A backend picks one of three keymap flavors (`fmri_gym/adapters/base.py`),
which differ only in how they combine the matching combos:

| Flavor | Action sent | Used by |
| --- | --- | --- |
| `SingleKeySpec` | the most specific held combo | ale, gym, vgdl, crafter, nethack, … |
| `MultiKeySpec` | OR of every held combo's buttons | retro, vizdoom, stk_gym (MultiBinary) |
| `PassthroughKeySpec` | the held key names, `"+"`-joined | aigamestore, supertuxkart |

### Remapping keys (the `keys` field)

Any game phase can override the mapping with a `keys` dict of
`"<key(s)>": <action>`. The keys are pygame names (`UP`, `DOWN`, `LEFT`,
`RIGHT`, `SPACE`, `RETURN`, letters `A`–`Z`, digits) and `<action>` is the
action the env expects — an **integer** for a `Discrete` space (ale, gym,
vgdl, …). Combine keys with `+` (e.g. `"UP+SPACE"`). A combo overrides its
parts: with `SingleKeySpec` the most specific fully-held combo wins, and with
`MultiKeySpec` a held combo replaces (rather than ORs with) its own parts.

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

`configs/dbp_games/atari__pong.json` and `configs/demo_mixed.json` both use this mapping.
CartPole similarly uses `{"LEFT": 0, "RIGHT": 1}`.

## Triggers: fMRI vs MEG/EEG

The `"triggers"` section next to `"curriculum"` says how a run starts and what
the recording gets (full example: `configs/demo_meg.json`):

```jsonc
"triggers": {
  "sync": {"mode": "send", "delay": 0.0},
  "backend": "serial", "port": "/dev/ttyUSB0"
}
```

| `sync.mode` | after the experimenter's SPACE… |
|---|---|
| `wait` | wait for the key the trigger box types (`key`), then start |
| `send` | send the start code on the trigger line, wait `delay` s, then start |
| `none` | start immediately |

`backend`: `null`, `lsl`, `serial` or `parallel` — `uv sync --extra triggers`
(pylsl / pyserial / pyparallel); `port` for serial/parallel, `lsl_stream_name`
for LSL.

Which of these a rig needs varies: the scanner may type a key at every volume,
or start its recording when the stimulus PC sends a code, or neither. So the
config says what happens rather than naming a modality, and the editor offers
a template per common setup:

| setup | `sync.mode` | `backend` |
|---|---|---|
| the scanner types a key at every volume | `wait` | `null` (no trigger line) |
| the recording starts from the trigger input | `send` | `serial` / `parallel` / `lsl` |
| the recording is started by hand, the PC gets the scanner's key | `wait` | `serial` / `parallel` / `lsl` |
| bench test, nothing connected | `none` | `null` |

Leaving `sync.mode` or `backend` out defaults to `wait` / `null`, and says so:
the experimenter screen, the console and the manifest all report it
(`NOT SET in config: sync.mode, backend`), as they do for `--dummy-trigger`.

What is sent: `task_start` when the clock anchors, `episode_start` at each
reset, one code per frame (`"frame_every": N` to thin, `"on_frame": false` to
drop), `task_stop` at the end. Codes never share bits, so two triggers on the
same sample still decode: frames cycle 1–7 in the low 3 bits, `task_start`=8,
`task_stop`=16, `episode_start`=32, `scanner_start`=64, and a lifecycle code
is OR'd with the current frame code (all under `"codes"`; overlaps are
refused). Every value sent is logged: per frame as `trigger` in the block
`.npz`, lifecycle events with their `run_time` under `triggers` in
`manifest.json`.

## Timing

Frames are shown with a vsync-locked flip and each frame's onset is logged as
`flip_time`; message/fixation onsets in the manifest are flip times too. Key
presses and releases are logged as they arrive (`key_time`, `key_name`,
`key_down`), independent of the frame grid. The manifest records the display
actually obtained (`vsync`, measured at start-up; `refresh_rate`).

- A frame is shown at the next refresh after its step, so an `fps` that
  divides the refresh rate (30 or 60 on a 60 Hz screen) shows every frame for
  the same number of refreshes; otherwise frames alternate between one and two
  and each onset can be up to one refresh late. `flip_time` records what
  happened either way. Some cores' own rate is 59.92: close enough to 60 Hz
  that one frame in ~800 is shown twice.
- Check that the rig locks to the refresh before a session:
  `python -m fmri_gym.display --fullscreen` (verdict LOCKED / NOT locked; if
  not, use fullscreen and disable the desktop compositor). `--no-vsync` turns
  the request off. Pass the session's `--monitor` here and to the photodiode:
  refresh, vsync and the photon offset belong to the monitor.
- Once per rig, measure the constant flip-to-photon offset with a photodiode on
  the screen, then subtract it from `flip_time` and the frame triggers:

  ```bash
  python -m fmri_gym.photodiode --fullscreen --config configs/demo_meg.json   # diode into the MEG/EEG amp
  python -m fmri_gym.photodiode --fullscreen --audio                          # diode into this PC's sound card
  python -m fmri_gym.photodiode --fullscreen --audio --audio-click --mic       # + mic on input 1: when sound is heard
  ```

  The first flashes a patch with the frame trigger on each white flip; match
  the triggers to the diode edges in your recording with
  `fmri_gym.photodiode.match_edges(trigger_times, edge_times)`. The second
  records the diode on the sound-card input and prints the offsets itself
  (`--list-audio-devices` to pick the input). `--audio-click` also plays a tone
  burst on each white flip through the session's audio output and reports when
  it reaches the DAC; with `--mic`, when a microphone on input channel 1 hears it.
- **The rig check** runs all of these as one run: `configs/rig-check.json`, whose
  curriculum holds check phases instead of games, played by `fmri-play` with the
  window, trigger line and audio output a session opens. So the editor edits it
  like any config -- its **Triggers** tab sets the line it tests (make it the
  session's), its **Controls** tab the buttons it asks for (Use a device layout
  to fill them in) -- and it can stand first in any session: add
  `configs/rig-check.json` as its first run. A failed test does not stop the
  check or the session: it is listed, with why, on screen, on the console and
  in the report. On its own, the session `configs/rig-check.sh` plays the long
  check (filed under `sub-rig`, as MEG-BIDS files empty-room recordings under
  `sub-emptyroom`; its first line, the quick check, is skipped: un-skip it to
  play that one): open it in the editor, `fmri-edit --session
  configs/rig-check.sh`, and press Play, or run `sh configs/rig-check.sh`.

  | phase | test | fails when |
  |---|---|---|
  | `check_display` | display | flips are not locked to the refresh (missed refreshes are counted) |
  | `check_frames` | frames | a built-in test pattern, played as a game through the session's own game loop at each of `rates` fps and under each of `loads` (`cpu`: every core busy), loses a frame, shows one a refresh later than its rate allows, resets its pacing, or its frame triggers do not mark every frame in their cycle (and, given `recording_hz`, for 2 samples each) |
  | `check_triggers` | triggers | the line does not open, LSL does not read back what was sent, or the scanner's pulses do not come or come irregularly (counted when the run waits for the scanner). This PC's ports and LSL are listed first, with a warning when none is usable. Codes sent over serial/parallel are printed: check them in the recording |
  | `check_controls` | controls | a key is not read back through the event queue as itself (automatic), or is not pressed when asked for on screen (the input devices plugged in are listed) |
  | `check_photodiode` | photodiode | the diode misses a flash, or sees only the flashes with a click (it hears sound). `"readout": "recording"`: matched offline to the frame triggers |
  | | audio | a tone burst (every other flash, through the session's output) never reaches the DAC, or with `"mic": true` a microphone at the ear on input 1 misses one |

  The rig-check configs ship with `"triggers": null`: no setup is anyone's
  default. The first rig check asks for one -- a preset (fMRI, MEG, EEG,
  behavioural) or your own -- and saves it in the config; the Triggers tab
  changes it later. With no screen for the dialog it stops and says so.

  Every check runs whichever fails, and each test gets its verdict. The screen
  says what each check is doing -- which code goes out on which line, which
  flash (and whether it carries a click), which rate under which load -- and
  then its verdict. A session's runs must share one triggers section: the
  editor's Check says so when they differ (each run keeps its own, so a change
  on the Triggers tab applies to the selected run only).
  `configs/rig-check.json` is the **quick** check, before every session: under a
  minute, most of it pressing the buttons -- enough to show today's rig is the
  one that was measured. `configs/rig-check-long.json` **measures** the rig,
  once per rig and after any hardware, driver or OS change (about 25 min; the
  frame test alone plays 60, 30, 20 and 50 fps for a minute each, idle and
  under CPU load):
  offsets to a fraction of a ms, their drift in ms/min (a sound card's clock
  runs apart from the PC's), missed refreshes and pulses, the TR on this PC's
  clock. With `sync.mode` `wait`, the check waits for the scanner like any
  run: start a sequence, or the trigger box's test mode.

  Each machine needs a **rig file**, `rig.json` (not in git): site, rig, PI's
  initials, modality, and what software cannot see -- monitor or projector,
  photodiode, sound path to the ear, trigger hardware. When it is missing or
  invalid, the check opens a form to fill it in (the `gui` extra; Save stays
  disabled until every field is valid) before the window; cancelled, or with no
  screen, it stops. `python -m fmri_gym.checks rig` reopens the form when
  the hardware changes; `RIG=<path>` points elsewhere.

  The check's run folder holds, beside the run's own `manifest.json` (with the
  rig, each check's numbers and each test's verdict) and one
  `block-NN_check_<name>.npz` per check:

  ```
  report.html      for the people who run the rig: a verdict per test, the numbers,
                   charts (flip intervals, each frame rate's intervals, every flash's
                   offsets), and this rig's previous checks; one offline file
  report.md        the same verdicts and numbers, as text
  rigcheck.tsv     for pooling: one row, fixed columns, n/a where a test did not run
  rigcheck.json    the columns' descriptions and units (BIDS sidecar)
  ```

  and at the data root, rebuilt after every check, **`data/rigchecks.tsv`** (+
  its `.json` sidecar): every rig check filed there, one row each, oldest first.
  A failed check is filed too, with each failed test and why. The rows are read
  from each check's `manifest.json`, so checks filed by an earlier version line
  up, with `n/a` in the columns they predate. Across sites:
  `python -m fmri_gym.checks pool data/ /mnt/siteB/data/ --out rigchecks.tsv`.
  `python -m fmri_gym.checks report <run folder>...` re-files checks already
  run (their reports and the table) with this version.

## Output & data format

Each run writes one folder, named and numbered as BIDS does:

```
data/sub-01/ses-001/beh/sub-01_ses-001_task-pong_run-001/
data/sub-01/ses-001/beh/sub-01_ses-001_task-pong_run-002/        the same task again
data/sub-01/ses-001/beh/sub-01_ses-001_task-pong_run-002_02/     ... re-acquired
data/sub-01/ses-001/beh/sub-01_ses-001_task-crafter_run-001/
```

The task is the config's file name (letters and digits); `--subject` must be
`sub-<letters/digits>`. `--ses` and `--run` are **required**: the numbers come
from the session design (the script's `SES=` line and each run's place in it),
never from what is on disk, so they survive a session that was interrupted,
resumed or re-acquired. `--data-root` moves the tree (default `data`).

Data is never overwritten. A run whose folder is already there writes to
`..._02`, then `_03`; the attempt that stopped stays where it is. That suffix
names the folder only — the run's label, which the manifest records with the
`attempt` number and which keys the seeds, stays canonical, so every attempt
at a run plays the same episodes.

The names follow BIDS apart from that suffix, the contents not yet (no
`_beh.tsv` / `_events.tsv`). Each folder holds:

- **`manifest.json`** — subject, curriculum, trigger epoch, per-phase
  onsets/offsets (+ survey responses; onsets are flip times), the `display`
  actually opened (size, `vsync`, `refresh_rate`, driver), the `triggers`
  settings + lifecycle triggers sent (+ what the config left `defaulted`), the
  `audio` output (device, measured device delay, chosen delay),
  `dummy_trigger`, the `run` it is (label and `attempt`), the `seeds` (each game phase derived or pinned) and the `versions` of pygame and SDL.
- **`block-NN_<backend>_<game>.npz`** — one per game block, uniform schema:

  | key | meaning |
  |-----|---------|
  | `actions`, `rewards`, `terminal`, `episode_id` | per frame |
  | `run_time`, `wall_time` | seconds since this run's trigger (after the step); wall-clock Unix time |
  | `flip_time` | seconds since trigger of the **flip that showed the frame** (its onset; vsync-locked when the display reports `vsync: true`) |
  | `pacing_reset_time`, `pacing_reset_late` | flips that ended a stall of more than a frame, and how many seconds late each was: the frame schedule restarted there instead of catching up with a burst of short frames. Empty in a clean block; the manifest counts them per phase (`n_pacing_resets`). Why frames fall behind is open (issue #43) |
  | `key_time`, `key_name`, `key_down` | every key press/release during the block, stamped on arrival (~1 ms), independent of the frame grid |
  | `trigger` | the code sent on that frame's flip (only when a trigger backend is active) |
  | `audio_onset` | seconds since trigger that the frame's sound reached the DAC, NaN if none (only when the block played sound; with `audio_delay_ms`, `audio_resyncs`, `audio_trimmed_samples`) |
  | `states` | per-frame savestate blob (object array; `None` if engine has none) |
  | `episode_seeds` | RNG seed per episode |
  | `backend`, `game` | provenance |
  | *backend vars* | `ram` (ale/retro), `info_*` (retro decoded score/lives/…), `obs` (gym), `screen_index` (ale, with `"save_pixels"`) |

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
> ⚠️ **`"save_pixels": true`** stores the screen every frame. It's lossless
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
- [x] **LSL / serial / parallel-port triggers** and a send-mode start signal
      for MEG/EEG (`"triggers"` section) -- done.
- [x] **Photodiode calibration task** (`python -m fmri_gym.photodiode`) to measure
      the flip-to-photon offset of a rig -- done; an always-on sync square in
      the corner of every frame remains an option if a lab wants per-frame
      verification.
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
