# fmri-gym

**One fMRI experiment framework for (almost) any Gymnasium-compatible game.**

A proof-of-concept framework that turns games into neuroimaging tasks — fixed uniform display, scanner-trigger sync, a declarative curriculum of phases, and compact reconstructable per-frame logging — and works across game engines through small pluggable **adapters**:

| Backend (`"backend"`) | Games | Engine |
|---|---|---|
| `ale`         | Atari 2600 (`ALE/Pong-v5`, …) | ALE / Stella |
| `retro`       | NES / SNES / Genesis / GB / … (`Airstriker-Genesis-v0`, …) | stable-retro / libretro |
| `gym`         | **any** Gymnasium env (`CartPole-v1`, MuJoCo, Box2D, toy_text, …); old-`gym` envs via shimmy | various |
| `vgdl`        | VGDL games (`aliens`, `beesAndBirds`, …), from a pinned checkout in `external/vgdl` | vgdl-gym (`gym/vgdl/`) over the language_and_experience fork |
| `crafter`     | Crafter (open-world survival) | crafter-gym (`gym/crafter/`) over crafter |
| `minihack`    | MiniHack tasks (pixel obs) | minihack / NLE |
| `nethack`     | NetHack (`NetHack*-v0`; TTY rendered to pixels) | nle |
| `aigamestore` | AI GameStore browser games (`game1`…`game10`), lock-stepped | aigamestore-gym (`gym/aigamestore/`) |
| `vizdoom`     | Doom action-shooter scenarios (COOM's engine) | ViZDoom |
| `coom`        | COOM's own continual-RL scenarios (`pitfall`, `chainsaw`, …), from a pinned checkout in `external/coom` (COOM package itself not installed -- conflicting `gymnasium` pin) | coom-gym (`gym/coom/`) |
| `baba`        | Baba Is AI: small generated rule-manipulation puzzles in the style of Baba Is You | baba-gym (`gym/baba/`) over baba-is-ai |
| `baba_auto`   | Baba Is Auto: Baba Is You's original levels (`baba_is_you`, `out_of_reach`, …) under the full ruleset; a C++ engine compiled from a pinned checkout in `external/baba_auto` | baba-auto-gym (`gym/baba_auto/`) over utilForever/baba-is-auto |
| `rushhour`    | Rush Hour sliding-block puzzle | `rushhour-gym` (PyPI; fetches its Go engine) |
| `stk`         | SuperTuxKart 3D racing: frames from the game's gym server, keys to its player controller (needs a real GL display) | [supertuxkart-gym](https://pypi.org/project/supertuxkart-gym/) (brings the game itself) |

> **All backends run in ONE env and ONE process.** Verified: a single session with ALE + retro + gym + VGDL blocks back-to-back, and each of Crafter / MiniHack in turn. VGDL originally required *old* `gym` + `numpy<2`, which conflicted with the numpy-2 backends; that's resolved by a fork whose VGDL source is ported to gymnasium ([tomov/language_and_experience @ dbp](https://github.com/tomov/language_and_experience/tree/dbp)). Adapters are imported lazily, so an env only needs the backends a curriculum actually uses. Same code, same curriculum schema, same output format everywhere.

## Install

See [Machine requirements](docs/MACHINE_REQUIREMENTS.md) for minimum and recommended hardware, and the [local test log](docs/local-testing/2026-09-14.md) for measurements and their scope.

Clone with `--recursive`: the vgdl / coom / baba_auto game repos are git submodules under `external/` (see [External checkouts](#external-checkouts-external)). In a clone made without it, run `git submodule update --init` once.

With [uv](https://docs.astral.sh/uv/):

```bash
sudo apt install libportaudio2               # PortAudio; every backend needs it
uv sync --extra dbp                          # .venv/ with the DBP backends, pinned by uv.lock
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/dbp_games/atari__pong.json --ses 1 --run 1
```

`dbp` is the DBP games. Each backend is also its own extra (`ale`, `retro`, `vizdoom`, `minihack`, `rushhour`, …), and `--extra all` installs every backend.

Without uv: pip into a venv of your own, and `python fmri_play.py` in place of `fmri-play`:

```bash
pip install -e ".[dbp]"            # private default index? add --index-url https://pypi.org/simple
python fmri_play.py --subject sub-01 --dummy-trigger --curriculum configs/dbp_games/atari__pong.json --ses 1 --run 1
```

Atari ROMs ship with `ale-py`. For the `retro` backend you must supply and import game ROMs once — see [Running stable-retro games](#running-stable-retro-games). For the `vgdl` backend see [Running VGDL games](#running-vgdl-games); for Rush Hour, [Running Rush-Hour](#running-rush-hour); for SuperTuxKart, [Running SuperTuxKart](#running-supertuxkart-stk).

## Quick start

```bash
# --- per-family demo curricula (all tested end-to-end; ~15 s per block) ---
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/demo_atari.json --ses 1 --run 1    # 10 popular Atari games
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/demo_classic.json --ses 1 --run 1  # all 5 classic-control
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/demo_text.json --ses 1 --run 1     # all 5 toy_text (render RGB; turn-based, arrow keys)
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/demo_box2d.json --ses 1 --run 1     # LunarLander, BipedalWalker, CarRacing  (`box2d` extra)
MUJOCO_GL=egl uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/demo_mujoco.json --ses 1 --run 1   # 10 MuJoCo tasks  (`mujoco` extra)
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/demo_aigamestore.json --ses 1 --run 1  # 10 AI GameStore p5.js games (`aigamestore` extra; see below)
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/demo_vgdl_all.json --ses 1 --run 1   # all 10 VGDL games (`vgdl` extra + external/vgdl; see below)

# demo_mixed spans EVERY backend in one session (Pong/ale, Airstriker/retro,
# Crafter, MiniHack, Aliens/vgdl, MountainCar/classic, FrozenLake/toy_text,
# CarRacing/box2d, WaterSort/aigamestore) -- needs the VGDL checkout + box2d-py +
# crafter + minihack + playwright:
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/demo_mixed.json --ses 1 --run 1

# Play ONE game on its own, for a long stretch (see configs/dbp_games/):
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/dbp_games/atari__pong.json --ses 1 --run 1
```

A run is a JSON file and a session is a `.sh` script (below). Write them by hand, or design them in `fmri-edit` -- either way `fmri-play` reads the same files, and refuses one it cannot play. `fmri-play` plays one run and nothing else; the editor is a command of its own, which opens on a run, a session, or a new run. **Play** there saves what it shows and starts it: an `fmri-play` command for a run, the script itself for a session. Its Launch tab holds the flags of that launch (subject, session, monitor, window, the test switches), which belong to the launch, not to the files. The editor is the `gui` extra:

```bash
uv sync --extra dbp --extra gui
uv run fmri-edit --curriculum configs/demo_meg.json
uv run fmri-edit --session configs/ses1.sh
uv run fmri-edit                                    # start from a new run
```

Drop `--dummy-trigger` for a real session (then press SPACE, then wait for the `=` scanner trigger). For VGDL setup see [Running VGDL games](#running-vgdl-games).

stable-retro games play their native audio; ViZDoom and COOM do when their config sets `env_kwargs.audio_buffer_enabled`. Use `--no-audio` or a game phase's `"audio": false` to mute playback; logged audio is unchanged. Each frame's sound starts a constant delay after the flip that shows it, measured from the system's default output at start-up and logged. A game with sound must run at its engine's own frame rate (ViZDoom: `fps * frame_skip == 35`; Genesis cores: 59.92), or the block stops and names the fps that fits.

Note: **MuJoCo and Box2D use continuous (`Box`) action spaces** — their configs write list actions that push the first dims to their limits (`"LEFT": [-1.0, 0.0, ...]`), so they render and log fine but aren't really human-playable without a per-game control scheme. Everything else in these families is keyboard-playable.

### Per-game configs (`configs/dbp_games/`)

There is **one config per individual supported game** (63 of them), sourced from the DBP game spreadsheet, so you can play any single game on its own for a long stretch with a one-line command. Filenames are `<class>__<game>.json`:

```bash
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/dbp_games/atari__pong.json --ses 1 --run 1
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/dbp_games/text__frozenlake.json --ses 1 --run 1
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/dbp_games/aigamestore__game1.json --ses 1 --run 1
```

Each is a minimal `message → fixation → game (300 s) → fixation` curriculum with the right per-game keymap/settings baked in. Coverage by class:

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
| `vizdoom__` | 78 | basic, deadly_corridor, defend_center, defend_line, health_gathering_supreme, my_way_home, predict_position, take_cover, deathmatch (Doom; COOM's engine; other `Vizdoom*-v1` scenarios work too), freedoom1_e1m1 … freedoom1_e4m9 and freedoom2_map01 … freedoom2_map32 (every map of [Freedoom](https://freedoom.github.io/) Phase 1 and Phase 2, whose WADs ship with `vizdoom`; `VizdoomDoom*` plays the original WADs, given them), plus `take_cover_defend_line` running two of them back to back in one session |
| `coom__` | 9 | pitfall, chainsaw, hide_and_seek, health_gathering, arms_dealer, parkour, raise_the_roof, run_and_gun, floor_is_lava (needs the COOM repo checkout) |
| `baba__` | 1 | make_win (rule-manipulation puzzle; other ids) |
| `baba_auto__` | 12 | baba_is_you, out_of_reach, volcano, off_limits, grass_yard, pillar_yard, brick_wall, icy_waters, novice_locksmith, lock, affection, turns (the levels the baba-is-auto checkout ships; its other maps are rule-engine fixtures; `game` takes any map file) |
| `rushhour__` | 1 | easy (sliding-block puzzle). `rushhour__complete.json` is the full self-paced session of Rush-Hour's own program, then the rest of the library: all 49 puzzles, the first 12 easiest-first and the other 37 in a fixed shuffled order, one game phase each, with ready screens and solved feedback as message phases |
| `stk__` | 5 | hacienda, cornfield_crossing, lighthouse, scotland, snowmountain (SuperTuxKart via its gym server: the race tracks its asset pack ships; needs a real GL display) |
| `retro__` | 3 | tobutobugirldx, nomolos, anguna (need ROMs imported) |

Each config carries `_game` / `_note` (per-game setup reminders). Games use `mode: "duration"` (300 s) so they auto-restart on game-over for continuous play; `ESC` quits. The `_note` flags class-specific requirements (VGDL repo, playwright, box2d-py, MuJoCo GL, ROM import).

> **Not covered** — configs live under `configs/dbp_games/unsupported/`, each with a `_status`/`_note` explaining why: games with no real-time pixel interface — `2048` (upstream reset bug), `pathery`/`wordle` (text/placement), `tile-match-gym` (display-only, `Discrete(84)` swaps → no keyboard play), `mastermind` (needs Python ≥3.13), and `craftium` (needs the Luanti engine built).

Runtime flow: experimenter screen (**SPACE**) → "Waiting for scanner..." → scanner **trigger `=`** (anchors the session clock) → curriculum phases → done. `ESC` quits early but still saves. Flags: `--size 1280x1024`, `--fullscreen`, `--monitor 1` (which screen, when there are several), `--no-vsync` (see [Timing](#timing-what-is-stamped-when)). The editor opens on its Launch tab, fullscreen ticked and the monitor picked there.

**A response device that is not a keyboard.** Some button boxes and gamepads (a Current Designs interface with a controller on it, say) report HID *joystick* buttons rather than keystrokes: nothing a phase's `keys` names is ever pressed, the game does not move, and the run records no responses with nothing on screen to say why. A plugged-in controller is therefore read directly, and its controls press the keys a phase's `keys` can already map:

| control | y | b | x | a | LT | RT | stick |
|---|---|---|---|---|---|---|---|
| presses | `W` | `D` | `A` | `S` | `Q` | `E` | `UP` / `DOWN` / `LEFT` / `RIGHT` |

The map is fixed (`fmri_gym/pad.py`), so every site with this controller expects the same keys and one curriculum plays at all of them; a curriculum keeps saying which *key* does what, and nothing else in the run can tell a pad from a keyboard. Press a button no key is mapped to and its number is printed on stderr; run `python -m fmri_gym.pad` to press each control and see the key it sends. `--no-pad` ignores the controller.

**Calibrate the stick once per rig.** A stick neither rests at 0 nor reaches 1 both ways -- one pad measured -0.03 at rest and travelled about twice as far right as left -- so a single symmetric threshold asks for twice the movement one way as the other. `fmri-pad-calibrate` measures this machine's pad (roll the stick around its edge, hands off, then ~60 cued presses) and writes `~/.config/fmri-gym/pad.json`, which every run then reads: each axis' rest and its two extremes, so a reading becomes a fraction of the travel *available in that direction*, plus the thresholds fitted from how hard the stick was actually pushed. It lives outside the checkout because it describes the hardware here, not the task; `FMRI_GYM_PAD_CONFIG` points elsewhere, and the raw samples are saved beside it. Without one the raw readings are used as-is and the run's banner says `UNCALIBRATED`. It cues one control at a time in a fixed order, announced before each run of trials, so a press aimed at the wrong button never has to be measured. `fmri-pad-scope` opens on its own when it finishes -- push each control and watch the thresholds it just wrote act on it -- and can be run any time after; its arrow keys move `axis_on`/`axis_off` live and `S` saves them back into the calibration.

**One direction at a time.** A thumb does not move one axis at a time, and a stick does not let it: on the pad this was written for, a push meant for *down* carried the horizontal axis to 0.87-0.98 of its own travel on half the trials. Thresholding the two axes separately therefore delivers diagonals nobody gave, which a game with four discrete actions -- nearly every curriculum here -- will act on. So the stick is read as one thing rather than two: how far it is from rest says *whether* a direction is held, and the angle of the push says *which*, by whichever of UP/DOWN/LEFT/RIGHT it points nearest to. Exactly one is pressed. A game that steers can set `axis_diagonal` below 1.0 in `pad.json` (or press `D` in the scope), which reads the axes independently again and allows two at once.

Those four directions are the obvious ones -- 0, 90, 180, 270 -- and the calibration does not move them. It does *measure* where each cue really pointed and report it: on the measured pad "down" came out 32 degrees round towards the left and "right" 29 degrees down, consistently. That skew is as much a property of the hand that was measured as of the stick, and the hand in the bore is a different one, so applying it would bake one person's grip into the site's config, and it would move under a participant who is learning the device. The summary prints both margins -- what the shipped directions give and what fitting would -- and `--fit-sectors` applies the measured ones for a stick that is genuinely skewed. `fmri-pad-calibrate --refit <samples.csv>` re-derives a calibration from an earlier recording, for when the fit changed rather than the hardware.

**A direction has to mean it.** A thumb does not leave rest along the line it is aimed at: on the way out to *down* the stick crosses the left sector, and for a frame it classifies as LEFT. Over 24 cued pushes that happened 5 times, always LEFT -- that axis' left travel is a third of its right, so any leftward component of a push normalises large -- and always for one 17 ms frame, against pushes that were held for 254 ms at the shortest. So a new direction must hold for `stick_dwell_s` before it is pressed, which the calibration fits from those transients (33 ms, two frames, on the measured pad) and `[` / `]` tune live in the scope. Releases are never delayed: a key must not outlast the stick coming back. The cost is that a stick press is reported that much late, the same amount every time, which a reaction time measured off the stick has to subtract. Buttons are unaffected -- they are already clean.

**The centre is not a constant, so it is measured as one movement, not stored as one place.** Cold, this pad read -0.003 one session and -0.019 the next; after it had been deflected once it sat at -0.10 in the first and -0.165 in the second and stayed there. The gimbal does not return where it started. So the calibration measures *how far* the centre moves -- where the stick sits before it has been touched against where it settles after the sweep -- and a run takes its centre once, the first time the stick is used and then let go still for a second, allowed to move only as far as that measurement says it can. It never re-centres while a direction is pressed, which would release the key under a subject leaning on the stick, and a stick that settles further out than the measured movement is refused with a message rather than believed: that is a stick that has drifted past its calibration, and the stops have not moved with it, so one side really has lost travel. With a 0.20 drift injected into a recorded session, a fixed centre held a key down on 43% of the frames where nobody was pushing; taking it once, 16%.

**The rig check measures it.** `check_controls` looks for a controller first. If one is plugged in it runs the calibration there and then, on the window the check already has open, and writes this rig's `pad.json` -- so the measurement is part of setting the rig up rather than a separate thing to remember, and every later run on the machine reads what it wrote. The check then asks for its keys as usual, now through the calibration it just made, and fails if the fit is not usable: a cued trial with no push, a cue read as another control, or under 10 degrees between a push and the next direction. With no controller plugged in there is nothing to measure and whatever device is in use is expected to send the keys itself, which is what the key prompts test. `"pad": false` in the phase skips the measurement and leaves any existing calibration alone; `"pad_reps"` sets how many times each control is cued (3, so 30 trials).

## Running stable-retro games

stable-retro only exposes a game once it has an **integration** and the game's **ROM** has been imported. ROMs are matched by their SHA-1 checksum and copied into a data directory *inside the installed `stable_retro` package* — that's where they must live; there is no ROM folder in this repo, and ROM binaries should never be committed.

- **Where ROMs go.** Import them into the package's `stable/` data dir with:

  ```bash
  python -m retro.import /path/to/dir_of_roms/
  ```

  This scans the directory, checksums each ROM, and installs the ones that match a known integration into `…/site-packages/stable_retro/data/stable/<Game>/rom.<ext>`. Airstriker (used by the demos) ships with stable-retro, so it needs no import.

- **Games without a built-in integration** (the homebrew titles on the DBP list — Tobu Tobu Girl DX, Nomolos, Anguna) need an integration created first. The companion `stable-retro-examples` repo has an `add_game.py` that, given a ROM, picks the platform from the extension (`.gb`→GameBoy, `.gbc`→GbColor, `.nes`→Nes, `.md`→Genesis, `.sfc`→Snes, …), copies the ROM into the package data dir, and writes a minimal `data.json`/`metadata.json` so the env can be created. It prints the exact `<Name>-<Platform>` id to use.

Once imported, reference the env id in a game phase and it plays like any other backend:

```jsonc
{"type": "game", "backend": "retro", "game": "TobuTobuGirlDX-GameBoy",
 "mode": "duration", "duration": 30.0, "fps": 60, "state_stride": 15}
```

The `configs/dbp_games/retro__{tobutobugirldx,nomolos,anguna}.json` configs need ROMs for exactly this reason: the `retro` backend itself is verified (with Airstriker), but those titles won't run until you import their ROMs and confirm the integration name.

## External checkouts (`external/`)

Three backends read their game files from a checkout of the game's own repo: `vgdl` (game, level and sprite files), `coom` (`conf.cfg`/`.wad` scenarios) and `baba_auto` (the C++ sources of `pyBaba`, plus maps and sprites). The checkout is a git submodule at `external/<backend>`, pinned to a commit, so every session plays the same tree; `git clone --recursive` (or `git submodule update --init`) fetches all three:

| backend | submodule `external/<backend>` of | commit |
|---|---|---|
| `vgdl` | [tomov/language_and_experience](https://github.com/tomov/language_and_experience) `dbp` | `d70c3248c504992068c8fb318238f1c3fbe956bf` |
| `coom` | [TTomilin/COOM](https://github.com/TTomilin/COOM) | `7929801176c6e2e036c7c1c7dd6ce9b84a9d1f3e` |
| `baba_auto` | [utilForever/baba-is-auto](https://github.com/utilForever/baba-is-auto) | `24cefb48d47ae6a6f5c0d936310d8bceb9c4279d` |

The env reads `external/<backend>` by default; a phase's `"repo"` field names another checkout for that block. To bump a pin, check the new commit out in the submodule, commit the submodule and this table together (for `baba_auto`, also `uv sync --reinstall-package baba-auto-gym`: the compiled engine is tied to the checkout it was built from).

## Running VGDL games

The `vgdl` backend drives the VGDL games from a gymnasium-ported fork: **[tomov/language_and_experience @ dbp](https://github.com/tomov/language_and_experience/tree/dbp)**. Because it runs under gymnasium + numpy 2, no separate env is needed — the same `fmri-gym` env works.

1. The fork is the `external/vgdl` submodule, at its pinned commit on the `dbp` branch, which has the gymnasium port (see [External checkouts](#external-checkouts-external)).

2. Install the `vgdl` extra -- **`vgdl-gym`** (`gym/vgdl/`), the standard Gymnasium env over the fork's `VGDLEnv` (a game name and level in, `reset(seed=)`, an offscreen `render()`; it puts the checkout on `sys.path`) -- and run a VGDL curriculum:

   ```bash
   uv sync --extra vgdl               # or: pip install -e gym/vgdl
   uv run fmri-play --subject sub-01 --curriculum configs/demo_vgdl_all.json --ses 1 --run 1
   ```

   Game files live at `<repo>/games/<game>_v0/<game>.txt` and `<game>_lvl<level>.txt`. Available games include `aliens`, `beesAndBirds`, `avoidGeorge`, `jaws`, `missile_command`, `plaqueAttack`, `portals`, `preconditions`, `pushBoulders`, `relational`.

VGDL blocks log a symbolic per-cell object grid (`symbolic_state`) and collision `events` as analysis variables, plus a per-frame exact savestate (get/set_state) for determinism-free reconstruction.

## Running COOM games

The `coom` backend plays [TTomilin/COOM](https://github.com/TTomilin/COOM)'s own continual-RL Doom scenarios (`pitfall`, `chainsaw`, `hide_and_seek`, `health_gathering`, `arms_dealer`, `parkour`, `raise_the_roof`, `run_and_gun`, `floor_is_lava`) -- distinct from the stock ViZDoom scenarios the `vizdoom` backend already covers (DeadlyCorridor, DefendCenter, ...).

COOM's own Python package pins `gymnasium==0.28.1`, which conflicts with minihack's `gymnasium==1.2` pin in this shared env, so **the COOM package is never installed or imported**. The gym env is **`coom-gym`** (`gym/coom/`, the `coom` extra; also in `dbp`), which drives `vizdoom.DoomGame` on the scenario files of a COOM checkout at its pinned commit (the `external/coom` submodule; see [External checkouts](#external-checkouts-external)):

```bash
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/coom__pitfall.json --ses 1 --run 1
```

The env reads `<repo>/COOM/env/scenarios/<scenario>/conf.cfg` and `<task>.wad` (`env_kwargs.task`, default `"default"`; `run_and_gun` also ships `blue`, `red`, `hard`, ...). Without the checkout the run stops at start-up.

Every scenario always exposes exactly 4 buttons (`TURN_LEFT`, `TURN_RIGHT`, `MOVE_FORWARD`, plus one of `JUMP`/`ATTACK`/`SPEED`/`USE`), driven through COOM's own 12-action table (turn x move x execute), so a phase's `keys` are indices into it -- 8 = turn left, 4 = turn right, 2 = forward, 1 = the 4th button, sums for combos, `"": 0` for no key; each `coom__*.json` spells its map out. COOM blocks log the raw ViZDoom `game_variables` (health, ammo, position, ...) as an analysis variable; there's no in-memory savestate, so reconstruction is via seed + action replay like most backends. The nine COOM curricula enable audio via `env_kwargs.audio_buffer_enabled`. On Ubuntu, install OpenAL with `sudo apt install libopenal1`. Keep `fps: 35` for one Doom tic of 44.1 kHz stereo PCM per step. `--no-audio` mutes playback but preserves recorded PCM; terminal frames are marked `audio_valid=false`. `audio_efx: false` disables reverb to avoid an older OpenAL crash. MIDI music requires a working engine MIDI renderer; check the startup console for errors. Queued sounds can be cut at episode end. Logged `audio_onset` estimates DAC timing, not physical speaker latency.

## Running AI GameStore games

[AI GameStore](https://aigamestore.org) is a benchmark of ten LLM-generated **browser games** (plain HTML + JavaScript; p5.js, and three.js for `game3`). The paper's own model harness drives the page in real time and pauses the game while the model thinks, which is neither steppable nor replayable. So the games ship here with a small Gymnasium env, **`aigamestore-gym`** (`gym/aigamestore/aigamestore_gym/`), that the `aigamestore` backend wraps like any other env:

- the page is still the game -- Playwright opens it in Chrome (headless by default) from a tiny local server (ES modules need `http://`);
- an init script, `lockstep.js`, takes over the page's clock (`requestAnimationFrame`, `setTimeout`, `performance.now`, `Date`) and seeds `Math.random`, so one `step()` holds a set of keys and advances the game by exactly `frame_skip` of its 60 Hz frames, then returns the canvas pixels and the game's own `getGameState()`. Nothing happens between steps: a model that deliberates for a minute and a subject at 10 Hz meet the same game, and `reset(seed)` + the actions replay an episode (the game state exactly; the pixels up to GPU rasterisation of gradients);
- action = `MultiBinary` over the game's keys (`env.keys`), reward = score delta, `info["state"]` = the scalar game state, logged as `state_*`;
- **an episode is one level**: `reset()` starts the env's level afresh (score 0, full lives) and the episode ends on a win, a loss or a level clear, the same way in every game. game4 is an endless runner without levels; game6 is one long level; game7's levels are procedural and unbounded (`aigamestore_gym.GAME_LEVELS` has the counts).

```python
import gymnasium as gym, aigamestore_gym
env = gym.make("AIGameStore/game6-level3-v0")  # or AIGameStoreEnv("game6", level=3, frame_skip=6)
obs, info = env.reset(seed=1)                   # obs: the 600x400 canvas, RGB, level 3 just started
obs, r, term, trunc, info = env.step([1, 0, 0, 0, 0])  # hold LEFT (env.keys[0])
```

Setup — the `aigamestore` extra (an editable install of `gym/aigamestore`) and a Chromium browser (the **system Chrome** by default):

```bash
uv sync --extra aigamestore        # or: pip install -e gym/aigamestore
# or the bundled browser: playwright install chromium, then "browser_channel": null
```

Run the 10 vendored public games (`game1` = Water Sort, `game2` ≈ Angry Birds, …):

```bash
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/demo_aigamestore.json --ses 1 --run 1
```

Phase fields: `game` (`"game1"`…`"game10"`, plus `/level<N>` to name the level -- `"game6/level3"`; bare `"game6"` is level 1 -- or an `http(s)://…/index.html` URL, then `game_keys` lists the keys that game listens for), `frame_skip` (default 6; `fps` must equal `60 / frame_skip`, and the block refuses to start otherwise), `headed` (show the browser window), `browser_channel` (`"chrome"` default, or `null` for the bundled Chromium), `games_dir` (override the vendored dir). A curriculum lists one game phase per level it wants played (`configs/dbp_games/aigamestore__game1.json` has all nine of Water Sort's). `keys` values are indices into the game's key list (`aigamestore_gym.GAME_KEYS`; game1: LEFT, RIGHT, SPACE, Z), so `{"B": 2}` puts the game's SPACE on the B key; the env relabels the on-canvas hints to match.

## Running Rush-Hour

[Rush-Hour](https://github.com/chrplr/Rush-Hour) is a sliding-block puzzle written as a psychophysics experiment in Go, with its rules in a small engine binary that both an agent and a participant play through `rushhour-gym`. The `rushhour` backend drives the package's `RushHourHuman-v0` — the experiment program's own interface as an env: car selection on four buttons, its picture, its results columns in `info` — so the adapter is a keymap plus the fields to log. One game phase is one puzzle (`"puzzle": "p07"`); the program's ready screens, blank intervals and solved feedback are `message` phases the curriculum lists around each puzzle, so every puzzle is its own block in the manifest and its own log folder. Nothing to install beyond the `rushhour` extra: on first use the package downloads the engine of its matching release into `~/.cache/rushhour-gym/` (checksum-verified; Linux x86-64, macOS arm64, Windows x86-64). On a machine without network, run a config once while online or copy that directory; `RUSHHOUR_ENV_BIN` names a binary of your own.

```bash
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/dbp_games/rushhour__easy.json --ses 1 --run 1      # 5 min of random easy puzzles
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/dbp_games/rushhour__complete.json --ses 1 --run 1   # the program's session then the rest of the library: 49 puzzles, one block each
```

Controls, phase fields and the logged columns are documented in the configs' `_note`s and in the package's README ("A person at the board").

## Running Baba Is Auto (`baba_auto`)

[utilForever/baba-is-auto](https://github.com/utilForever/baba-is-auto) is a C++ simulator of Baba Is You with the game's real ruleset and the original levels' maps (`baba_is_you`, `out_of_reach`, `off_limits`, ...), where the `baba` backend plays baba-is-ai's small generated puzzles. Its Python module exposes only the `Game`, so the Gymnasium env is **`baba-auto-gym`** (`gym/baba_auto/`, the `baba_auto` extra), one `baba_auto__<level>.json` per level the checkout ships. Installing the extra compiles the engine from the pinned checkout (see [External checkouts](#external-checkouts-external); a C++17 compiler and `python3-dev` are needed; about 30 s), and the env reads the maps and the GUI's sprites from the same checkout:

```bash
uv sync --extra dbp --extra baba_auto
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/baba_auto__baba_is_you.json --ses 1 --run 1
```

A phase's `game` is a map name under the checkout's `Resources/Maps` or a map file's path; `keys` index `Discrete(5)`: 0 = wait, 1 = up, 2 = down, 3 = left, 4 = right (turn-based). Each turn logs `play_state` (0 playing, 1 won, 2 lost) and the engine's `(16, H, W)` state tensor. No savestate: seed + action replay.

## Running SuperTuxKart (`stk`)

`supertuxkart-gym` is the current SuperTuxKart with a gym server built in (`--gym`), plus the `stk_gym` Python client that drives it. The backend is called `stk` and the client package `stk_gym`; the backend drives `stk_gym.StkEnv` with `render_mode="rgb_array"` and `action_mode="keys"`: the game renders into a window that is created hidden, every step brings the frame back and fmri-gym shows it; the held keys go to the game's own player controller, so steering ramps and skids latch as they do for a keyboard. Participant and model are in front of the same env object, and a block replays from `episode_seeds` + `actions`. The adapter is a keymap and the fields to log. (This replaced an earlier pystk2-gymnasium backend, which had no pixel obs and needed its own control mapping, so human and model play were not the same interface.)

```bash
uv pip install "fmri-gym[stk]"      # or: pip install supertuxkart-gym
uv run fmri-play --subject sub-01 --dummy-trigger --curriculum configs/dbp_games/stk__hacienda.json --ses 1 --run 1
```

There is one config per race track the asset pack ships with geometry — `stk__{hacienda,cornfield_crossing,lighthouse,scotland,snowmountain}.json`, 880–1330 m per lap, three laps, four karts. The pack's other track directories (`abyss`, `mines`, `zengarden`, … and the battle arenas and soccer fields) are metadata only: naming one kills the game process, so more tracks means a pack that carries them, not more configs. A track is fixed for the life of a process (`reset(options={"track": ...})` is refused by name), so race several by giving a curriculum one game phase per track.

No checkout and no build: the wheel is pure Python and fetches the game with a trimmed asset pack (270 MiB, five tracks) from its GitHub release the first time an env is made, into `~/.cache/supertuxkart-gym`. It says so while it downloads, and never does it twice. The pack is tied to the package version, so a wheel and its engine cannot drift apart: upgrading `supertuxkart-gym` is the whole update procedure. Linux x86_64 only for now; on anything else it says which platform it has no pack for.

The binary can be overridden with `STK_ENV_BIN`, and `STK_ENV_OFFLINE=1` forbids the download outright. It needs a real OpenGL display (the frame is the game's rendering). `fps` must equal the game's physics rate over `frame_skip` (120 / 2 = 60 in the config); the config's `_note`s list the keys, the phase fields and the logged columns, and the package's own README on [PyPI](https://pypi.org/project/supertuxkart-gym/) ("Frames", "Reproducibility") the details and measured cost.

## Design: the experiment loop never knows the engine

A game reaches the scanner through up to three layers, each its own directory:

| | holds | needed when |
|---|---|---|
| `external/<backend>` | the game's own repo, a submodule pinned to a commit | its files or sources are read from a checkout (VGDL, COOM, Baba Is Auto) |
| `gym/<game>` | a small package with a `gymnasium.Env` for the game | the game ships no Gymnasium env, or a rough one |
| `fmri_gym/adapters/<backend>.py` | the fMRI adapter: build the env, map keys, log frames | always, one per engine |

Most games need only the last: a pip package already provides the env. Nothing about fMRI leaks below the adapter, so the env an agent is trained on is the one the subject plays.

```
fmri_gym/
  run.py        # trigger, clock, curriculum loop, phases  — 100% engine-agnostic
  display.py        # pygame: fixed window, aspect-fit frame, fixation, text; vsync-locked flip + call_on_flip
  logging.py        # manifest.json + one crash-safe JSONL/HDF5 log per game block
  replay.py         # replay an episode from a block's log; its frame fields as arrays
  menu.py           # the hold-a-key pause menu (reset / forfeit / resume), opt-in per game phase
  triggers.py       # run-start sync (wait/send/none) + MEG/EEG trigger codes over lsl/serial/parallel
  photodiode.py     # `python -m fmri_gym.photodiode`: flash a patch to measure the flip-to-photon offset
  checks.py         # the rig check: its phases, the rig file, report.html/.md, rigchecks.tsv, `pool`
  adapters/
    base.py         # EnvAdapter + FrameState (the seam)
    keymap.py       # the phase's keys -> actions, one Keymap per action space
    ale.py          # clone_state, getRAM, lossless indexed pixels
    retro.py        # em.get_state, get_ram, decoded info vars, console-button keymap
    default.py      # ANY gym env: rgb frames, seed+replay, obs-as-state
    vgdl.py         # VGDL via vgdl-gym: get_state/set_state savestate, symbolic grid + events
    crafter.py      # Crafter via crafter-gym: obs is the frame; achievements
    minihack.py     # pixel obs + compass keymap; blstats/glyphs/message
    nethack.py      # base NLE: TTY grid -> RGB; vi-key movement; blstats
    aigamestore.py  # AI GameStore via aigamestore-gym: held keys as the env's action, state_* from getGameState
    coom.py         # COOM via coom-gym: Discrete(12) actions, game variables and PCM logged
    baba_auto.py    # Baba Is Auto (Baba Is You's levels on baba-is-auto) via baba-auto-gym: Discrete(5) turns, play_state + state tensor logged
    rushhour.py     # Go engine via rushhour-gym; select+slide UI, rushui look, Rush-Hour's log columns; one puzzle per block
    stk.py          # SuperTuxKart via stk_gym: frames from the game's hidden window, held keys as the env's action
fmri_play.py        # CLI entry point: a person in the scanner
agents/             # CLI entry point: a policy, on the same config, seeds and log schema
    policies.py     # what chooses the action when nobody is at the keyboard
    agent_play.py   # the block loop a policy needs: no window, no clock, no key queue
configs/            # example curricula
gym/                # Gymnasium envs for games that ship none, or a rough one: one small package each
    aigamestore/    # the 10 public AI GameStore games (HTML/JS) + aigamestore_gym, their lock-stepped env
    coom/           # coom_gym: COOM scenarios, on a COOM checkout's WADs
    baba/           # baba_gym: the Gymnasium contract in front of baba-is-ai's old-gym env
    baba_auto/      # baba_auto_gym: over baba-is-auto's pyBaba (C++, built from the baba-is-auto checkout)
    crafter/        # crafter_gym: the same for crafter.Env, with reset(seed=)
    vgdl/           # vgdl_gym: a standard env over the language_and_experience fork's VGDLEnv
external/           # the vgdl / coom / baba_auto game repos, submodules at the commits pinned above
```

The loop (`run.py`) only ever calls the adapter — never `env.unwrapped`, an emulator, or an engine module. Each engine-specific concern lives behind **`EnvAdapter`**:

```python
class EnvAdapter:
    def _make(self, spec)         -> gym.Env       # build the env for a block
    def reset(self, seed)         -> (obs, info)
    def capture(self, obs, info)  -> FrameState    # per-frame state to log
    def restore(self, blob)       -> None          # inverse of capture().blob
    def autoplay(self, info)      -> action | None # step with this, don't wait
```

`autoplay` is how a `turn_based` block gets through a stretch the env takes no action in. Crafter is the case it exists for: a sleeping player's action is overwritten with `sleep` until their energy fills, so the sixty-six presses a full sleep would cost under `turn_based` are all discarded by the engine, and how many there are is a fact about the game rather than a decision the subject is making. Returning an action says "step with this and do not wait for a press"; the loop keeps doing so, at the phase's fps and logging every frame like any other, until it returns `None` again. Nothing is skipped and nothing is hidden: the subject watches the night pass, and because waking from a blow ends the stretch, the picture stops on the frame with the zombie beside them. The default is `None`, so every other backend and every real-time phase is unaffected.

`self.env` is always a `gymnasium.Env`. A game that has no such env -- a bare engine, an old-`gym` env, a gymnasium env with habits of its own -- gets a thin one under `gym/` (see `gym/coom/`, `gym/baba/`, `gym/vgdl/`), not in its adapter; the adapter is then `_make` plus what to log.

`FrameState` carries a standard shape for **every** backend:
- `blob`: opaque bytes that `restore()` turns back into this exact state (ALE `clone_state`, retro `em.get_state()`), or `None` if the engine has no savestate — then reconstruction falls back to seed + action replay.
- `variables`: named analysis fields surfaced uniformly (`ram`, retro's decoded `info_score`/`info_lives`, the raw `obs`, …), so the loop and downstream analysis code are identical across engines.

**Adding a new engine = writing one adapter** (~40–80 lines). Nothing else changes.

## Curriculum format

A **run** is one JSON file: a `"curriculum"` of phases, an optional `"triggers"` section (below), and `_`-prefixed notes. A bare list, an unknown top-level key or an unknown phase `type` stops the run at start-up with the reason.

One config is one run. A whole scanning session is a plain shell script with one line per run, in order -- so every run is a process of its own, with a fresh interpreter, display and trigger port:

```sh
#!/bin/sh
# fmri-gym session: one line per run, in order.
set -e
SES=${1:-$(uv run fmri-ses --subject sub-01)}
uv run fmri-play --curriculum configs/pong.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
./scripts/localizer.sh "$SES"
# uv run fmri-play --curriculum configs/mario.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
```

**The numbers come from the script, not from the disk.** The `SES=` line picks the session once, so every run lands in it: the script's own argument if it was given one (`sh ses1.sh 003` resumes session 3), else the subject's next free session. Each run states its `--run`, which is its place among the lines that play that task -- so it is the same run number however the session went, and skipping a line renumbers nothing after it. Both flags are required of `fmri-play`: it never picks a number itself.

A run whose folder already has data is **re-acquired, never overwritten**: it writes to `..._02` (then `_03`) beside the attempt that stopped. The suffix names the folder only, so the re-acquisition replays the same episodes as the run it replaces.

`set -e` stops the script at the first run that fails or is quit with ESC (`fmri-play` then exits with status 3). A line that is not an `fmri-play` run is any command of yours, as typed; a commented line is a skipped run, which is how a stopped session is resumed. Write it by hand, or in `fmri-edit`, whose Session manager tab has two panels, each a form and the text it stands for: Session design (the list of lines -- add an existing or a new config, an external script; repeat, reorder, skip, "Start here" -- or the script itself) and Run design (the selected run's phases, or its JSON). Each panel starts with a drop-down of what `configs/` holds: the Session one opens a script, the Run one opens a config (or, in a session, adds it after the selected line); File > Open takes a `.sh` like a `.json` from anywhere. Save writes the script and the configs you edited. Run it from the repo root: `sh configs/ses1.sh`.

Three example sessions of about an hour each ship in `configs/` -- eleven runs of one DBP game apiece (5 min of play, plus the instructions screen and the start-up between runs):

```bash
sh configs/ses_dbp_mix.sh      # one run per genre: Crafter, COOM, MiniHack, Rush Hour, Baba, ViZDoom, AI GameStore...
sh configs/ses_dbp_doom.sh     # the nine COOM scenarios, then two ViZDoom ones
sh configs/ses_dbp_puzzle.sh   # nine AI GameStore puzzles, Rush Hour, Baba Is AI
sh configs/ses_dbp_mix.sh 003  # ... into session 3: how one that stopped is resumed
```

They assume `sub-01` and a 1024x768 window, and take the subject's next free session unless given one: open one in `fmri-edit --session configs/ses_dbp_mix.sh` to change any of that, or `--dummy-trigger` a run of it at the desk.

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
 "advancing_outcomes": ["won"], // episode mode, optional: only an episode that ends this way counts
                                // towards n_episodes; any other replays the same instance (a level
                                // is repeated until cleared). Names: won, lost, terminated,
                                // truncated, playing (the block's clock) -- see EnvAdapter.outcome
 "max_duration": 300.0,         // hard wall-clock safety cap (episode mode)
 "outcome_duration": 2.0,       // seconds "Final score: N" + the outcome show after each episode
 "fps": 30,                     // required: steps (and frames) per second. The engine's own rate
                                // (console cores and Atari ~60, Doom 35 / frame_skip) plays the game
                                // at its real speed and fits its sound; the editor's Controls tab
                                // (Check with the engine) shows it. Any other value plays the game slower or faster: the
                                // manifest logs "speed" and the console says so when it is not 1
 "turn_based": false,           // step only on a key PRESS, not per frame (grid/toy_text games)
 "latched_keys": false,         // real-time: a fresh key PRESS beats the held-key poll, so a tap
                                // shorter than one frame is not dropped (games below ~10 fps)
 "live_hud": false,             // turn_based only: redraw the standing frame while the loop waits
                                // for the press, so the clock in the HUD keeps up with the block's.
                                // Draws pixels and nothing else -- no step, no log row, no trigger,
                                // no sound. A duration block ends on the wall clock whether anyone
                                // presses or not, so without this a subject who is thinking reads
                                // the time their last press left behind
 "seed": 1234,                  // optional base seed: episodes play with seed, seed+1, ...
                                // Pinned, every participant and run gets the same episodes.
                                // Left out, it is derived from the run (sub/ses/task/run) and
                                // the phase, so no two runs replay each other's; fmri-play
                                // prints each phase's seed, the editor shows it (and Pin
                                // copies it in), the manifest logs it

 "state_stride": 1,             // save a full savestate every K frames (see below)
 "frame_stride": 1,             // save the rendered frame every K frames (0 = none)
 "state": "Level1",             // retro: named savestate/level (optional)
 "scenario": null,              // retro: scenario name (optional)
 "level": 0,                    // vgdl: level index; also uses "game","block_size"
 "keys": {"": 0, "LEFT": 0, "RIGHT": 1}, // REQUIRED: key -> env action, the whole map;
                                //   "" is the action sent with no key held (see below)
 "save_pixels": false,          // also store lossless pixels, where the backend can
 "show_score": false}           // crafter: draw the achievement count beside the frame
```

### Keys (the `keys` field)

Every game phase states its whole keyboard map. **There is no default**: which key does what differs between sites (a keyboard at the desk, a gamepad that types keys in the scanner), so the file says it, and a phase without `keys` is refused before the window opens. `keys` is a dict of `"<key(s)>": <action>`:

- The key is a name from `fmri_gym/keys.py` (`UP`, `DOWN`, `LEFT`, `RIGHT`, `SPACE`, `RETURN`, `LSHIFT`, `A`–`Z`, `0`–`9`, `F1`–`F12`, `KP0`–`KP9`, …); a name not in that table is refused, since it could never be pressed. Join keys with `+` for a combo (`"UP+SPACE"`).
- The action is what the env's `step` takes, as JSON, and the rule is the env's action space (`fmri_gym/adapters/keymap.py`, one class per space):
  - **`MultiBinary`** (retro, vizdoom, stk, aigamestore): the value is the **index of the button** the key holds down; every held key sets its bit, so keys combine as on a controller, and nothing held is every button up. An index outside the space is refused when the env is built, and so is a `""` entry (nothing held already means no button).
  - **`Discrete`** (ale, coom, vgdl, crafter, minihack, baba, rushhour, gym) or **`Box`** (gym): the value is the action itself, an index or a list (`"LEFT": [-1.0, 0.0]`). The most specific combo whose keys are all held wins (`"UP+SPACE"` over `"UP"`), and the **`""`** entry is what a frame with no key held sends. `""` is required unless the phase is `turn_based`, where nothing is sent between presses: these spaces have no action that means "do nothing" everywhere (FrozenLake's 0 is LEFT, MiniHack's is "move N"). A value outside the space is refused when the env is built.
  - Any other action space is refused when the env is built.

Each adapter's module docstring says what its indices mean, and each config's `_keys_note` spells out the map it uses. For an Atari game, read its meanings:

```python
import gymnasium as gym, ale_py; gym.register_envs(ale_py)
gym.make("ALE/Pong-v5").unwrapped.get_action_meanings()
# ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']  -> RIGHT=2, LEFT=3
```

**Example — Pong on up/down arrows** (its paddle is `RIGHT`=2 / `LEFT`=3):

```jsonc
{"type": "game", "backend": "ale", "game": "ALE/Pong-v5",
 "mode": "duration", "duration": 30.0,
 "keys": {"": 0, "UP": 2, "DOWN": 3, "SPACE": 1, "UP+SPACE": 4, "DOWN+SPACE": 5}}
```

The editor's Controls tab edits the table and, with *Check with the engine*, builds the env to confirm the keys fit its action space (and shows the engine's own rate for `fps`).

### The pause menu (the `menu` field)

A subject can get stuck (a Baba Is You puzzle with its rules pushed into a corner) or want out of a level. A game phase can opt into a pause menu:

```jsonc
"menu": {"key": "X", "hold": 5.0, "after": 15.0,
         "options": ["reset", "forfeit", "resume"],
         "move": ["UP", "DOWN"], "confirm": "SPACE"}
```

Holding `key` for `hold` seconds -- even if that key does something in the game -- pauses the game and shows the options; `move` steps through them and `confirm` picks one. `reset` starts the episode over -- a new `reset()` of the env with the same seed, so the very same level instance -- `forfeit` ends the block and moves on to the next phase, `resume` continues. The menu is unavailable for the first `after` seconds of the block (0 = always), and a hold plus a two-key choice is hard to do by accident -- the point is that this is possible but never done lightly. Only `key` is required; the rest default to the values shown, and `options` may list any subset, in the order to show them. The manifest's phase entry records the settings and every pop-up (`menu.events`: when, and what was chosen); a reset or forfeited episode's `episode_outcome` is `reset` / `forfeit` (the env's own `terminated` / `truncated` flags are never touched), and a reset one does not count towards `n_episodes`. `configs/dbp_games/baba__make_win.json` uses it.

## Triggers: fMRI vs MEG/EEG

The `"triggers"` section next to `"curriculum"` says how a run starts and what the recording gets (full example: `configs/demo_meg.json`):

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

`backend`: `null`, `lsl`, `serial` or `parallel` — `uv sync --extra triggers` (pylsl / pyserial / pyparallel); `port` for serial/parallel, `lsl_stream_name` for LSL.

Which of these a rig needs varies: the scanner may type a key at every volume, or start its recording when the stimulus PC sends a code, or neither. So the config says what happens rather than naming a modality, and the editor offers a template per common setup:

| setup | `sync.mode` | `backend` |
|---|---|---|
| the scanner types a key at every volume | `wait` | `null` (no trigger line) |
| the recording starts from the trigger input | `send` | `serial` / `parallel` / `lsl` |
| the recording is started by hand, the PC gets the scanner's key | `wait` | `serial` / `parallel` / `lsl` |
| bench test, nothing connected | `none` | `null` |

Leaving `sync.mode` or `backend` out defaults to `wait` / `null`, and says so: the experimenter screen, the console and the manifest all report it (`NOT SET in config: sync.mode, backend`), as they do for `--dummy-trigger`.

What is sent: `task_start` when the clock anchors, `episode_start` at each reset, one code per frame (`"frame_every": N` to thin, `"on_frame": false` to drop), `task_stop` at the end. Codes never share bits, so two triggers on the same sample still decode: frames cycle 1–7 in the low 3 bits, `task_start`=8, `task_stop`=16, `episode_start`=32, `scanner_start`=64, and a lifecycle code is OR'd with the current frame code (all under `"codes"`; overlaps are refused). Every value sent is logged: per frame as `trigger` on the block's `frame` lines, lifecycle events with their `run_time` under `triggers` in `manifest.json`.

## Timing

Frames are shown with a vsync-locked flip and each frame's onset is logged as `flip_time`; message/fixation onsets in the manifest are flip times too. Key presses and releases are logged as they arrive (`key_time`, `key_name`, `key_down`), independent of the frame grid. The manifest records the display actually obtained (`vsync`, measured at start-up; `refresh_rate`).

- A frame is shown at the next refresh after its step, so an `fps` that divides the refresh rate (30 or 60 on a 60 Hz screen) shows every frame for the same number of refreshes; otherwise frames alternate between one and two and each onset can be up to one refresh late. `flip_time` records what happened either way. Some cores' own rate is 59.92: close enough to 60 Hz that one frame in ~800 is shown twice.
- Check that the rig locks to the refresh before a session: `python -m fmri_gym.display --fullscreen` (verdict LOCKED / NOT locked; if not, use fullscreen and disable the desktop compositor). `--no-vsync` turns the request off. Pass the session's `--monitor` here and to the photodiode: refresh, vsync and the photon offset belong to the monitor.
- Once per rig, measure the constant flip-to-photon offset with a photodiode on the screen, then subtract it from `flip_time` and the frame triggers:

  ```bash
  python -m fmri_gym.photodiode --fullscreen --config configs/demo_meg.json   # diode into the MEG/EEG amp
  python -m fmri_gym.photodiode --fullscreen --audio                          # diode into this PC's sound card
  python -m fmri_gym.photodiode --fullscreen --audio --audio-click --mic       # + mic on input 1: when sound is heard
  ```

  The first flashes a patch with the frame trigger on each white flip; match the triggers to the diode edges in your recording with `fmri_gym.photodiode.match_edges(trigger_times, edge_times)`. The second records the diode on the sound-card input and prints the offsets itself (`--list-audio-devices` to pick the input). `--audio-click` also plays a tone burst on each white flip through the session's audio output and reports when it reaches the DAC; with `--mic`, when a microphone on input channel 1 hears it.
- **The rig check** runs all of these as one run: `configs/rig-check.json`, whose curriculum holds check phases instead of games, played by `fmri-play` with the window, trigger line and audio output a session opens. So the editor edits it like any config -- its **Triggers** tab sets the line it tests (make it the session's), its **Controls** tab the buttons it asks for (Use a device layout to fill them in) -- and it can stand first in any session: add `configs/rig-check.json` as its first run. A failed test does not stop the check or the session: it is listed, with why, on screen, on the console and in the report. On its own, the session `configs/rig-check.sh` plays the long check (filed under `sub-rig`, as MEG-BIDS files empty-room recordings under `sub-emptyroom`; its first line, the quick check, is skipped: un-skip it to play that one): open it in the editor, `fmri-edit --session configs/rig-check.sh`, and press Play, or run `sh configs/rig-check.sh`.

  | phase | test | fails when |
  |---|---|---|
  | `check_display` | display | flips are not locked to the refresh (missed refreshes are counted) |
  | `check_frames` | frames | a built-in test pattern, played as a game through the session's own game loop at each of `rates` fps and under each of `loads` (`cpu`: every core busy), loses a frame, shows one a refresh later than its rate allows, resets its pacing, or its frame triggers do not mark every frame in their cycle (and, given `recording_hz`, for 2 samples each) |
  | `check_triggers` | triggers | the line does not open, LSL does not read back what was sent, or the scanner's pulses do not come or come irregularly (counted when the run waits for the scanner). This PC's ports and LSL are listed first, with a warning when none is usable. Codes sent over serial/parallel are printed: check them in the recording |
  | `check_controls` | controls | a key is not read back through the event queue as itself (automatic), or is not pressed when asked for on screen (the input devices plugged in are listed) |
  | `check_photodiode` | photodiode | the diode misses a flash, or sees only the flashes with a click (it hears sound). `"readout": "recording"`: matched offline to the frame triggers |
  | | audio | a tone burst (every other flash, through the session's output) never reaches the DAC, or with `"mic": true` a microphone at the ear on input 1 misses one |

  The rig-check configs ship with `"triggers": null`: no setup is anyone's default. The first rig check asks for one -- a preset (fMRI, MEG, EEG, behavioural) or your own -- and saves it in the config; the Triggers tab changes it later. With no screen for the dialog it stops and says so.

  Every check runs whichever fails, and each test gets its verdict. The screen says what each check is doing -- which code goes out on which line, which flash (and whether it carries a click), which rate under which load -- and then its verdict. A session's runs must share one triggers section: the editor's Check says so when they differ (each run keeps its own, so a change on the Triggers tab applies to the selected run only). `configs/rig-check.json` is the **quick** check, before every session: under a minute, most of it pressing the buttons -- enough to show today's rig is the one that was measured. `configs/rig-check-long.json` **measures** the rig, once per rig and after any hardware, driver or OS change (about 25 min; the frame test alone plays 60, 30, 20 and 50 fps for a minute each, idle and under CPU load): offsets to a fraction of a ms, their drift in ms/min (a sound card's clock runs apart from the PC's), missed refreshes and pulses, the TR on this PC's clock. With `sync.mode` `wait`, the check waits for the scanner like any run: start a sequence, or the trigger box's test mode.

  Each machine needs a **rig file**, `rig.json` (not in git): site, rig, PI's initials, modality, and what software cannot see -- monitor or projector, photodiode, sound path to the ear, trigger hardware. When it is missing or invalid, the check opens a form to fill it in (the `gui` extra; Save stays disabled until every field is valid) before the window; cancelled, or with no screen, it stops. `python -m fmri_gym.checks rig` reopens the form when the hardware changes; `RIG=<path>` points elsewhere.

  The check's run folder holds, beside the run's own `manifest.json` (with the rig, each check's numbers and each test's verdict) and one `block-NN_check_<name>.npz` per check:

  ```
  report.html      for the people who run the rig: a verdict per test, the numbers,
                   charts (flip intervals, each frame rate's intervals, every flash's
                   offsets), and this rig's previous checks; one offline file
  report.md        the same verdicts and numbers, as text
  rigcheck.tsv     for pooling: one row, fixed columns, n/a where a test did not run
  rigcheck.json    the columns' descriptions and units (BIDS sidecar)
  ```

  and at the data root, rebuilt after every check, **`data/rigchecks.tsv`** (+ its `.json` sidecar): every rig check filed there, one row each, oldest first. A failed check is filed too, with each failed test and why. The rows are read from each check's `manifest.json`, so checks filed by an earlier version line up, with `n/a` in the columns they predate. Across sites: `python -m fmri_gym.checks pool data/ /mnt/siteB/data/ --out rigchecks.tsv`. `python -m fmri_gym.checks report <run folder>...` re-files checks already run (their reports and the table) with this version.

## Output & data format

Each run writes one folder, named and numbered as BIDS does:

```
data/sub-01/ses-001/beh/sub-01_ses-001_task-pong_run-001/
data/sub-01/ses-001/beh/sub-01_ses-001_task-pong_run-002/        the same task again
data/sub-01/ses-001/beh/sub-01_ses-001_task-pong_run-002_02/     ... re-acquired
data/sub-01/ses-001/beh/sub-01_ses-001_task-crafter_run-001/
```

The task is the config's file name (letters and digits); `--subject` must be `sub-<letters/digits>`. `--ses` and `--run` are **required**: the numbers come from the session design (the script's `SES=` line and each run's place in it), never from what is on disk, so they survive a session that was interrupted, resumed or re-acquired. `--data-root` moves the tree (default `data`).

Data is never overwritten. A run whose folder is already there writes to `..._02`, then `_03`; the attempt that stopped stays where it is. That suffix names the folder only — the run's label, which the manifest records with the `attempt` number and which keys the seeds, stays canonical, so every attempt at a run plays the same episodes.

The names follow BIDS apart from that suffix, the contents not yet (no `_beh.tsv` / `_events.tsv`). Each folder holds:

- **`manifest.json`** — subject, curriculum, trigger epoch, per-phase onsets/offsets (+ survey responses; onsets are flip times), the `display` actually opened (size, `vsync`, `refresh_rate`, driver), the `triggers` settings + lifecycle triggers sent (+ what the config left `defaulted`), the `audio` output (device, measured device delay, chosen delay), `dummy_trigger`, the `run` it is (label and `attempt`), the `seeds` (each game phase derived or pinned) and the `versions` of pygame and SDL.
- **`block-NN_<backend>_<game>/`** — one folder per game block, written as the block plays (so a crash loses at most the last second), holding:
  - **`events.jsonl`** — one JSON object per line, in order; `type` says which:

    | `type` | fields |
    |--------|--------|
    | `block_start` | first line: `format`, `version`, `subject`, `block_index`, `backend`, `game`, `phase` (the config), `base_seed` |
    | `episode_start` | `episode_id`, `seed` |
    | `frame` | `frame` (index in the block), `episode_id`, `ep_frame` (index in the episode), `action`, `reward`, `terminated`, `truncated`, `run_time` (seconds since the trigger, after the step), `flip_time` (of the **flip that showed the frame**: its onset; vsync-locked when the display reports `vsync: true`), `wall_time` (Unix), `variables` (the backend's: `ram` for ale/retro, `info_*` for retro, `obs` for gym, …), and when present `env_action` (an adapter's own reading of the keys, e.g. Rush Hour), `trigger` (the code sent on that flip), `audio_chunk` (the chunk this frame's sound was queued as, -1 none; in blocks that play sound), `state` (a savestate, zlib'd then base64, every `state_stride` frames) |
    | `input_event` | `run_time`, `name`, `down`: every key press/release during the block, stamped on arrival (~1 ms), independent of the frame grid; the pause menu's included |
    | `pacing_reset` | `flip_time`, `late`: a flip that ended a stall of more than a frame, and how many seconds late it was; the frame schedule restarted there instead of catching up with a burst of short frames (none in a clean block; why frames fall behind is open, issue #43) |
    | `episode_end` | `episode_id`, `outcome` (`won`, `lost`, `terminated`, `truncated`, `playing` for the block's clock; `quit`, `reset`, `forfeit` for the subject's) , `terminated`, `truncated`, `score`, `n_pacing_resets` |
    | `block_end` | last line: `n_episodes`, `n_frames`, `outcomes`, `n_pacing_resets`, `total_reward` (the manifest entry's summary), `audio` (when the block played sound: `onsets`, `[chunk, run_time]` when each chunk reached the DAC, so a frame's sound onset is the entry for its `audio_chunk` and `onset - flip_time` the audio delay achieved; `delay_ms`, `resyncs`, `trimmed_samples`), `extra` (the adapter's block extras, e.g. an ALE palette) |

  - **`frames.h5`** — the rendered frames, `frames` `(N, H, W, C)` gzip'd one chunk per frame plus `frame_index` `(N,)`, the `frame` of each row: every `frame_stride`-th frame (default every one; `0` keeps none). `fmri_gym.logging.read_frames` / `read_frame` read them.

  The manifest's phase entry carries the same summary and names the folder (`data_dir`).

### Reconstruction (all verified bit-exact)

1. **Per-frame state** (ale, retro): restore the `state` of a `frame` line → exact frame, no determinism assumption.
2. **Seed + action replay** (any deterministic env, incl. gym): an episode's `seed` + its `action`s reproduce it frame-for-frame: `fmri_gym.replay.reconstruct_episode(block)`.
3. **Stored pixels**: `frames.h5` *is* what was on screen, before the HUD.

```python
import base64, pickle, zlib, gymnasium as gym, ale_py, stable_retro as retro
from fmri_gym.logging import read_events
from fmri_gym.replay import frame_arrays, reconstruct_episode
gym.register_envs(ale_py)

block = "data/sub-01/ses-001/beh/sub-01_ses-001_task-pong_run-001/block-00_ale_Pong-v5"
a = frame_arrays(block)                    # a["action"], a["reward"], a["ram"], ... one row per frame
states = [zlib.decompress(base64.b64decode(e["state"])) if "state" in e else None
          for e in read_events(block) if e["type"] == "frame"]

# ALE: restore any frame's exact state
env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=1, repeat_action_probability=0.0); env.reset()
env.unwrapped.restore_state(pickle.loads(states[10]))
frame10 = env.unwrapped.ale.getScreenRGB()

# retro: restore any frame's exact state
r = retro.make("Airstriker-Genesis", render_mode="rgb_array"); r.reset()
r.unwrapped.em.set_state(states[10]); r.unwrapped.data.update_ram()

# any deterministic env: replay episode 0 from its seed and actions
adapter, plan = reconstruct_episode(block, episode_id=0)
```

> ⚠️ **Storage note & `state_stride`.** Per-frame savestates are cheap for ALE (~0.4 KB/frame) but large for retro consoles: a Genesis state is ~1 MB/frame. Set **`"state_stride": K`** on a game phase to snapshot a full savestate only every K frames (always including each episode's first frame, the replay anchor); frames between anchors stay reconstructable by restoring the last anchor and replaying the logged actions (retro/ALE/VGDL are deterministic). Analysis variables (RAM, `info_*`) are always logged every frame regardless of stride. The rendered frames have their own **`"frame_stride"`**; ⚠️ **`"save_pixels": true`** (ale) additionally logs the indexed screen as a variable (`palette[screen_index] == RGB`), which `frames.h5` makes unnecessary. Prints a loud warning when enabled.

> ⚠️ **Crafter replays only on a fork.** Every tenth step stock crafter rebalances creatures per chunk by iterating a Python *set* of objects, so which animal is despawned follows object `id()` and two runs of the same seed and the same action list diverge: terrain is identical, creatures are not. Measured 2026-09-28 on stock 1.8.3, replaying one episode's 225 actions in a second process that differed only in `PYTHONHASHSEED`: the two left each other at step 30, and the episode ended a step apart. Ordering that list by position is the whole fix, and it lives in [`chengfanbrain/crafter@deterministic`](https://github.com/chengfanbrain/crafter/tree/deterministic), which `gym/crafter/pyproject.toml` pins as a direct reference resolved to a commit in `uv.lock` (nothing to clone into `external/`). The same measurement on that build: a 750-frame block of 5 episodes replayed from `episode_seeds` + `actions` into every logged pixel and the whole logged symbolic state, 750 frames of 750, and all 32 savestate anchors restored and then played their episode out identically, which is what a model rollout from a subject's own state needs. The block's `frames.h5` keeps the displayed frames regardless, and those pixels are the record that does not depend on whoever opens the block later having the fork installed. What that block cost at size 384 and 2.5 fps, measured with per-frame zlib: a median frame is 7.4 KB, but crafter mixes per-pixel noise into the view at night, so its 107 night frames ran to 212 KB and the pixels came to 22.5 MB; the 32 anchors pickle to 2.3 MB each, nearly all of it the observation space's constant bounds and the cached frame, and compress to 1.8 MB. That night noise is drawn from the RNG the creatures use, so a `render()` outside the step loop would shift every later draw; nothing renders out of band, because `CrafterEnv` hands back the frame `step` already produced.

## Playing a block with a model

`agents/agent_play.py` runs the same curriculum with a policy where `fmri_play.py` puts a person. It shares everything that defines the task (the config, the adapter, the episode seeds, the Logger and its log schema) and none of the session loop, which exists for a scanner: trigger wait, fps pacing, a window, a key queue.

```sh
python agents/agent_play.py --curriculum configs/dbp_games/crafter__crafter_L4.json \
    --policy random --outdir data/model-random --n-episodes 2 --max-frames 60
export ANTHROPIC_API_KEY=...      # `--policy vlm` refuses to start without it
python agents/agent_play.py --curriculum configs/dbp_games/crafter__crafter_L4.json \
    --policy vlm --model claude-sonnet-5 --history 4 --max-frames 60
```

The key is read before the first frame rather than at the first request: a block whose every call came back 401 would otherwise run to the end and log a model that chose to stand still. A call that fails later costs one frame, recorded as a noop and counted in the phase log as `dropped_calls`, beside `invalid_replies` (a reply that named no key) and `skipped_frames` (a turn-based block, where pressing nothing steps nothing).

The block's log carries the fields a human block carries, plus `policy` and `policy_model` in its `block_end` record and manifest entry, so one analysis reads both. Episode seeds are the block's own (`seed` + episode index), which is what makes this the same-worlds condition rather than a fresh sample of the game.

A policy is given the frame the display would have shown, the key table the subject was taught, and the lines the subject could read beside the frame. Nothing else: a policy handed engine state is no longer playing the game the human played. Where a block's feedback is a sound, the harness turns on the adapter's `cue_overlay` so the same thing is said in text, a model having no ears. `--history` fixes how many recent frames and past keys a VLM policy sees, and it is the one asymmetry against a subject watching a continuous stream, so pin it and report it rather than tuning it against scores.

Rolling a model out from a *human's* savestate anchor, and reading a VLM's forward pass over a human's frames, both consume these logs from outside and are deliberately not part of this repo.

## Migrating your game list

Many games already expose a Gymnasium API and drop straight into the `gym` backend; stable-retro titles use the `retro` backend; and the VGDL games use the `vgdl` backend (their source was ported from old `gym` to gymnasium so they run in the same numpy-2 env).

**old-`gym` games (e.g. chess, hanoi, Sokoban).** Three options: (a) **a thin Gymnasium env under `gym/`** that holds the old env and presents the contract in front of it -- `reset(seed=)` returning `(obs, info)`, a 5-tuple `step`, a `render()` without a mode -- as done for Baba (`gym/baba/`, 70 lines) and Crafter (`gym/crafter/`); the game's own package is untouched and the adapter stays `_make` plus what to log. This is the usual answer. (b) **port the source to gymnasium**, as done for VGDL — usually a small mechanical diff (swap `gym`→`gymnasium`, fix removed `np.*` aliases and `pkg_resources`); or (c) run them via **shimmy** with `"legacy_gym": true` (routed via `GymV21Environment-v0`) in a dedicated `numpy<2` env. Note shimmy's v0.21 compat calls the removed `.seed()` and `gym==0.26` is incompatible with `numpy>=2`, so (a) or (b) is usually cleaner. The `legacy_gym` code path exists in `default.py`.

**Porting an old-`gym` game to gymnasium (the VGDL recipe).** The whole change was: `import gym`→`import gymnasium as gym` across the env/registration/play files; skip `Space.__init__` in a custom variable-length space; replace removed `np.float` with `float`; and drop `pkg_resources` (gone in setuptools≥81) by resolving data dirs relative to `__file__`. Result: VGDL runs under gymnasium 1.3 + numpy 2.4 in the same env as every other backend.

## Future work / TODO

- [x] Configurable **state stride** (`"state_stride": K`) — done.
- [x] **VGDL backend** ([tomov/language_and_experience @ dbp](https://github.com/tomov/language_and_experience/tree/dbp)), ported to gymnasium so it runs in the same env as the other backends — done.
- [ ] Finish the **old-`gym` / shimmy** path against a real game (Sokoban, chess) — either port its source (VGDL recipe above) or run via shimmy in a `numpy<2` env; code path exists but is untested end-to-end.
- [x] **LSL / serial / parallel-port triggers** and a send-mode start signal for MEG/EEG (`"triggers"` section) -- done.
- [x] **Photodiode calibration task** (`python -m fmri_gym.photodiode`) to measure the flip-to-photon offset of a rig -- done; an always-on sync square in the corner of every frame remains an option if a lab wants per-frame verification.
- [ ] **retro `.bk2` movie logging** as an alternative to per-frame states (frame-exact, tiny).
- [ ] Per-subject deterministic curriculum generation; multi-run structure with one trigger per fMRI run.
- [ ] More adapters: ViZDoom/COOM, MiniHack, crafter, MuJoCo (`qpos/qvel` as state vars) — each a small `EnvAdapter`.
- [ ] Button-box / MRI-safe response device key remapping.
- [ ] BIDS-style output layout + `events.tsv` per run.
- [ ] A replay/QC utility to render any block to video from its states.
- [ ] Crash-safe incremental logging (stream frames to disk) for long runs.
