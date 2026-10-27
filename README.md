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
sudo apt install libportaudio2 libopenal1 libsdl2-2.0-0 libxcb-cursor0   # Linux only (see below)
uv sync --extra dbp                          # .venv/ with the DBP backends, pinned by uv.lock
uv run fmri-play --subject sub-01 --dummy-trigger --ses 1 --run 1 --rig configs/rigs/example/laptop.json --curriculum configs/curricula/minihack/levels/room5x5.json
```

Nothing is installed by default: a bare `uv sync` gets the engine-agnostic core alone, and every game arrives as an extra. `dbp` is the DBP games; each backend is also its own extra (`ale`, `retro`, `vizdoom`, `minihack`, `rushhour`, …), so `--extra baba` is the core plus that one game. `gui` adds the `fmri-edit` editor, and `--extra all` installs every backend and the editor -- enough to run every game here without an import error. Extras combine (`--extra dbp --extra gui`), and each `uv sync` makes `.venv/` match exactly the extras it names -- one you leave out is one uv uninstalls, so repeat the whole list every time. `uv run` does not: it adds what the lockfile requires and leaves the rest, so the games survive every `uv run`; only the next bare `uv sync` strips them. `baba_auto` (hence `all`) compiles a C++ engine at install, so it needs a compiler: `build-essential` on Linux, `xcode-select --install` on macOS. The `apt` line is the libraries the wheels load from the system rather than bundle, none of which a stock Ubuntu 22.04 desktop has: PortAudio (`sounddevice`; macOS/Windows wheels bundle it), OpenAL and SDL2 (the SuperTuxKart engine, and OpenAL for ViZDoom/COOM sound), and xcb-cursor (the Qt window of `fmri-edit`). `.python-version` asks uv for Python 3.12, the version CI plays: uv takes an installed interpreter that fits `requires-python` before it downloads one, so without it Ubuntu 22.04's own Python 3.10 becomes `.venv/`'s, and `arc3` (3.12 or later) is left out of `--extra all` without a word. `requires-python` caps the interpreter at `<3.14` (nle and box2d have no 3.14 wheels), which is also what pip reads.

Without uv: pip into a venv of your own, and `python fmri_play.py` in place of `fmri-play`:

```bash
pip install -e ".[dbp]"            # private default index? add --index-url https://pypi.org/simple
python fmri_play.py --subject sub-01 --dummy-trigger --ses 1 --run 1 --rig configs/rigs/example/laptop.json --curriculum configs/curricula/minihack/levels/room5x5.json
```

To check an install, or a change, play the suite: `uv run pytest --changed` plays the backends your branch touches through the adapter, a recorded block, a bit-exact replay and a headless run of the real loop, and `uv run pytest --backend crafter` plays one by name; a backend this machine cannot play is skipped with the fix in the reason. CI runs the same on every pull request marked ready for review. `tests/README.md` has the map.

Atari ROMs ship with `ale-py`. For the `retro` backend you must supply and import game ROMs once — see [Running stable-retro games](#running-stable-retro-games). For the `vgdl` backend see [Running VGDL games](#running-vgdl-games); for Rush Hour, [Running Rush-Hour](#running-rush-hour); for SuperTuxKart, [Running SuperTuxKart](#running-supertuxkart-stk).

## Quick start

Every command here plays on a laptop as it is: `--rig configs/rigs/example/laptop.json` (`configs/rigs/example/laptop.json`) is a 1024x768 window with the game's sound on the laptop's speakers, the keyboard, and no scanner, so the run starts at the experimenter screen's SPACE. `--dummy-trigger` skips that screen too, and every key wait after it: drop it to read the instructions and play at your own pace.

```bash
# --- per-family demo curricula (all tested end-to-end; ~15 s per block) ---
uv run fmri-play --subject sub-01 --dummy-trigger --ses 1 --run 1 --rig configs/rigs/example/laptop.json --curriculum configs/curricula/ale/all.json    # 10 popular Atari games
uv run fmri-play --subject sub-01 --dummy-trigger --ses 1 --run 1 --rig configs/rigs/example/laptop.json --curriculum configs/curricula/gym/classic_control.json  # all 5 classic-control
uv run fmri-play --subject sub-01 --dummy-trigger --ses 1 --run 1 --rig configs/rigs/example/laptop.json --curriculum configs/curricula/gym/toy_text.json     # all 5 toy_text (render RGB; turn-based, arrow keys)
uv run fmri-play --subject sub-01 --dummy-trigger --ses 1 --run 1 --rig configs/rigs/example/laptop.json --curriculum configs/curricula/gym/box2d.json     # LunarLander, BipedalWalker, CarRacing  (`box2d` extra)
uv run fmri-play --subject sub-01 --dummy-trigger --ses 1 --run 1 --rig configs/rigs/example/laptop.json --curriculum configs/curricula/aigamestore/all.json  # 10 AI GameStore p5.js games (`aigamestore` extra; see below)
uv run fmri-play --subject sub-01 --dummy-trigger --ses 1 --run 1 --rig configs/rigs/example/laptop.json --curriculum configs/curricula/vgdl/all.json   # all 10 VGDL games (`vgdl` extra + external/vgdl; see below)

# demos/mixed spans seven backends in one session (Pong/ale, Airstriker/retro,
# Crafter, MiniHack, Aliens/vgdl, MountainCar/classic, FrozenLake/toy_text,
# CarRacing/box2d, WaterSort/aigamestore) -- needs the VGDL checkout + box2d-py +
# crafter + minihack + playwright:
uv run fmri-play --subject sub-01 --dummy-trigger --ses 1 --run 1 --rig configs/rigs/example/laptop.json --curriculum configs/curricula/demos/mixed.json

# Play ONE game on its own, for a long stretch (see configs/curricula/):
uv run fmri-play --subject sub-01 --dummy-trigger --ses 1 --run 1 --rig configs/rigs/example/laptop.json --curriculum configs/curricula/ale/levels/pong.json
```

A run is a JSON file and a session is a `.sh` script (below). Write them by hand, or design them in `fmri-edit` -- either way `fmri-play` reads the same files, and refuses one it cannot play. `fmri-play` plays one run and nothing else; the editor is a command of its own, which opens on a run, a session, or a new run. **Play** there saves what it shows and starts it: an `fmri-play` command for a run, the script itself for a session. Its Launch tab holds the flags of that launch (subject, session, rig, the test switch), which belong to the launch, not to the files; its **Edit rig...** opens the rig file (see *Rigs* below). The editor is the `gui` extra:

```bash
uv sync --extra dbp --extra gui
uv run fmri-edit --curriculum configs/curricula/ale/levels/pong.json
uv run fmri-edit --session configs/subjects/boston-college/sub-01/ses-001/dbp_mix.sh
uv run fmri-edit                                    # start from a new run
```

Drop `--dummy-trigger` for a real session (then press SPACE, then wait for the `=` scanner trigger). For VGDL setup see [Running VGDL games](#running-vgdl-games).

stable-retro games play their native audio; ViZDoom and COOM do when their config sets `env_kwargs.audio_buffer_enabled`. Use the rig file's `"has_audio": false` (every run on that rig) or a game phase's `"has_audio": false` to mute playback; logged audio is unchanged. Each frame's sound starts a constant delay after the flip that shows it, measured from the system's default output at start-up and logged. A game with sound must run at its engine's own frame rate (ViZDoom: `fps * frame_skip == 35`; Genesis cores: 59.92), or the block stops and names the fps that fits.

Note: **Pendulum, MountainCarContinuous and BipedalWalker use continuous (`Box`) action spaces** — their configs write list actions that push the first dims to their limits (`"LEFT": [-1.0, 0.0, ...]`), so they render and log fine but aren't really human-playable without a per-game control scheme. Everything else in these families is keyboard-playable.

### Curricula, rigs and sessions (`configs/`)

`configs/` holds three kinds of file, each changing on its own schedule (`configs/README.md` has the details):

| folder | what it is | specific to |
|---|---|---|
| `configs/curricula/<backend>/levels/<name>.json`, `configs/curricula/<backend>/<name>.json` | a **curriculum**: what one run plays -- one level (in `levels/`), or a sequence of levels (in the backend's folder) | the game; the same for every subject and site |
| `configs/rigs/<site>/<rig>.json` | a **rig**: the machine a session runs on (`configs/rigs/README.md`) | the site and its machine |
| `configs/subjects/<site>/sub-<label>/ses-<NNN>/<name>.sh` | a **session**: one `fmri-play` line per run, in order | one subject's one session |

Every game that plays has a curriculum for each of its supported levels, so any one of them is a one-line command:

```bash
uv run fmri-play --subject sub-01 --dummy-trigger --ses 1 --run 1 --rig configs/rigs/example/laptop.json --curriculum configs/curricula/minihack/levels/room5x5.json
uv run fmri-play --subject sub-01 --dummy-trigger --ses 1 --run 1 --rig configs/rigs/example/laptop.json --curriculum configs/curricula/vizdoom/levels/freedoom1_e1m1.json
uv run fmri-play --subject sub-01 --dummy-trigger --ses 1 --run 1 --rig configs/rigs/example/laptop.json --curriculum configs/curricula/rushhour/complete.json
```

A curriculum's path after `curricula/`, in camelCase (BIDS labels are letters and digits only), is the run's BIDS task label (`curricula/minihack/levels/room5x5.json` is `task-minihackLevelsRoom5x5`), so a level keeps its label whichever session plays it. What ships, by backend (`levels/` holds the curricula of one level each, the backend's folder those that play several):

| folder | count | curricula |
|---|---|---|
| `aigamestore/` | 11 | game1…game10 (p5.js; needs playwright), one file each in `aigamestore/` with its phase in `common/` (no `levels/`: game4 and game6 have one level, the others several); and `all`, the ten in a row |
| `ale/` | 11 | `levels/`: pong, breakout, spaceinvaders, mspacman, seaquest, qbert, asterix, beamrider, enduro, boxing, and `all` |
| `arc3/` | 6 | `levels/`: g50t, ls20, re86, tr87, tu93, wa30 (the game files are fetched once; see [ARC-AGI-3 games](#arc-agi-3-games)) |
| `baba/` | 1 | `levels/`: make_win (rule-manipulation puzzle; other ids) |
| `baba_auto/` | 13 | `levels/`: baba_is_you, out_of_reach, volcano, off_limits, grass_yard, pillar_yard, brick_wall, icy_waters, novice_locksmith, lock, affection, turns (the levels the baba-is-auto checkout ships; its other maps are rule-engine fixtures; `game` takes any map file); and `all`, those in a row |
| `checks/` | 2 | rig-check, rig-check-long: the rig check's phases instead of games (see [Timing](#timing)) |
| `coom/` | 10 | `levels/`: pitfall, chainsaw, hide_and_seek, health_gathering, arms_dealer, parkour, raise_the_roof, run_and_gun, floor_is_lava (needs the COOM checkout); and `all`, those in a row |
| `crafter/` | 4 | `levels/`: L1…L4, the paradigm's four levels |
| `demos/` | 2 | mixed (seven backends in one run), meg (Pong with a full MEG `triggers` section) |
| `gym/` | 16 | `levels/`: classic control (cartpole, mountaincar, acrobot, pendulum, mountaincarcontinuous), toy text (frozenlake, frozenlake8x8, cliffwalking, taxi, blackjack; turn-based), Box2D (lunarlander, bipedalwalker, carracing; the `box2d` extra); and `classic_control`, `toy_text`, `box2d`, each family in a row |
| `minihack/` | 22 | `levels/`: room5x5/15x15, mazewalk9x9, river, corridor, eat: the `-Allocentric-v0` twin from `minihack_levels` (view fixed on the map), each with an `_egocentric` sibling on MiniHack's own id (view centred on the agent); and the designed levels room_*, mazewalk_*, river_*; and `all`, the six standard tasks (allocentric) in a row |
| `nethack/` | 1 | `levels/`: score (base NetHack; TTY rendered to a pixel frame) |
| `retro/` | 1 | `levels/`: airstriker (ships with stable-retro; sound) |
| `rushhour/` | 2 | `levels/`: easy (5 min of random easy puzzles); complete (Rush-Hour's own program, then the rest of the library: all 49 puzzles, one game phase each, with ready screens and solved feedback as message phases) |
| `stk/` | 6 | `levels/`: hacienda, cornfield_crossing, lighthouse, scotland, snowmountain (the race tracks its asset pack ships; needs a real GL display); and `all`, those in a row |
| `vgdl/` | 11 | `levels/`: aliens, avoidGeorge, beesAndBirds, jaws, missile_command, plaqueAttack, portals, preconditions, pushBoulders, relational (needs the VGDL checkout), and `all` |
| `vizdoom/` | 78 | `levels/`: basic, deadly_corridor, defend_center, defend_line, health_gathering_supreme, my_way_home, predict_position, take_cover, deathmatch, freedoom1_e1m1 … freedoom1_e4m9 and freedoom2_map01 … freedoom2_map32 (every map of [Freedoom](https://freedoom.github.io/) Phase 1 and Phase 2, whose WADs ship with `vizdoom`; `VizdoomDoom*` plays the original WADs, given them); and `take_cover_defend_line`, two of them back to back |

Each backend's `common/README.md` says what its curricula play, how to set them up and what the action indices in their `keys` mean. Most are a `fixation → game → fixation` run whose game phase opens on its own instructions screen.

#### Games without a curriculum

These were tried and do not play here, so nothing in `configs/` names them:

- **MuJoCo** (ant, halfcheetah, hopper, humanoid, walker2d, swimmer, reacher, pusher, the inverted pendulums): continuous joint torques, which a keyboard cannot drive; the `mujoco` extra still installs them for the `gym` backend.
- **stable-retro homebrew** (Tobu Tobu Girl DX, Nomolos, Anguna): stable-retro has no integration for them, so they need one written before a ROM can run (see [Running stable-retro games](#running-stable-retro-games)); a game it integrates needs only its ROM and a curriculum.
- **Craftium** (Luanti voxel worlds): needs its own wheel and the Luanti engine built.
- **Tile Match**: `Discrete(84)` swaps, no keyboard play.
- **2048** (its own `reset()` crashes), **Pathery** and **Wordle** (text or placement actions, no real-time frame), **Mastermind** (needs Python ≥ 3.13).

Runtime flow: experimenter screen (**SPACE**) → "Waiting for scanner..." → scanner **trigger `=`** (anchors the session clock) → curriculum phases → done. `ESC` quits early but still saves.

**Rigs: how a run is played.** A curriculum says what is played, and plays the same on every rig. How it is played is the rig's: a **rig** is one setup -- a monitor, a scanner, a response device, a trigger line -- and a site can have several. Each has its own **rig file** in the repo, `configs/rigs/<site>/<rig>.json` (see `configs/rigs/README.md`), which every run played on that rig reads:

| key | what it says | when left out |
|---|---|---|
| `site`, `rig`, `pi`, `modality`, `monitor`, `photodiode`, `audio_path`, `trigger_hardware`, `notes` | what the rig is, for people: notes nothing plays from, filed with every run and pooled by the rig check; `site` and `rig`, if written, must be the file's folder and name | left out; `site` and `rig` are the file's folder and name |
| `"screen"` | the window: `{"size": "1024x768", "is_fullscreen": true, "monitor": 1, "has_vsync": true}`; `monitor` by index, 0 the first, and fullscreen scales `size` up to the monitor, aspect kept, black bars on the sides, without changing the monitor's mode or refresh rate (see [Timing](#timing-what-is-stamped-when)) | 1024x768 windowed on monitor 0, vsync on |
| `"has_pad"` | read a plugged-in game controller | `true` |
| `"bindings"` | which keyboard key each control of the controller presses (below) | none: a plugged-in controller is warned about and presses nothing |
| `"has_audio"` | play game audio; a block plays sound when this and its game phase's `has_audio` are both true | `true` |
| `"data_root"` | where the BIDS tree goes | `data` |
| `"triggers"` | how a run starts and what codes the recording gets, over which line (see [Triggers](#triggers-fmri-vs-megeeg)) | wait for `=`, no codes, reported as NOT SET |

`fmri-play --rig configs/rigs/<site>/<rig>.json` names the rig file, as `--curriculum` names the curriculum's, and is required. The repo ships example rigs that play out of the box (`configs/rigs/README.md` lists them): `configs/rigs/example/empty.json`, an empty `{}` and so every default (a window with sound; after SPACE the run waits for the scanner's `=`, which on a laptop you press), which QUICKSTART uses; `example/laptop.json`, the keyboard alone with no scanner (a run starts on SPACE), which the commands in this README name; one per controller at a desk, `example/xbox.json` and `example/curdes-v1.json`, bound to the curricula's keys (below); and one per scanner, `example/fmri.json` and `example/meg.json`. The console and the manifest say which rig file a run played on; when the file is missing, `fmri-play` does not start and says how to make one. Each rig gets one long rig check, which makes its file: its form asks for each field the first time (`fmri-play --subject sub-rig --ses 1 --run 1 --rig configs/rigs/<site>/<rig>.json --curriculum configs/curricula/checks/rig-check-long.json`; `python -m fmri_gym.checks rig --rig configs/rigs/<site>/<rig>.json` opens the form alone, or edits it later); commit the file it writes. In the editor, the Launch tab names the rig file and its **Edit rig...** opens the same form. The console says which rig file a run read and what it settles, and every run's manifest keeps it (`"rig"`, defaults written out).

**A response device that is not a keyboard.** Some button boxes and gamepads (a Current Designs interface with a controller on it, say) report HID *joystick* buttons rather than keystrokes: nothing a phase's `keys` names is ever pressed, the game does not move, and the run records no responses with nothing on screen to say why. A plugged-in controller is therefore read directly (`fmri_gym/pad.py`), and each of its controls presses the keyboard key the rig file binds it to (`"bindings"`): curricula stay written in keyboard keys, so the same game plays on a keyboard at the desk and on the controller in the scanner, and one map in the rig file serves every game played there. That holds because every curriculum here whose actions fit six buttons keeps to ten keys -- the arrows, `SPACE`, `X`, `C`, `Z`, `A`, `S` -- the ten a six-button controller presses with this map:

```json
"bindings": {"A": "SPACE", "X": "X", "B": "C", "Y": "Z", "LT": "A", "RT": "S",
             "UP": "UP", "DOWN": "DOWN", "LEFT": "LEFT", "RIGHT": "RIGHT"}
```

On a keyboard the same ten keys play every game: the arrows under the right hand, `Z X C` and `A S` under the left, `SPACE` under the thumb. A run on a rig with a controller and `"bindings"` does not start when a phase needs something no control presses -- a game action, a menu key, a key that dismisses a screen -- and names it; play that curriculum on a rig without a controller, or bind the key. A game with more actions than six buttons (its curriculum's README says so) is played on a keyboard.

The controls are a gamepad's: `UP` `DOWN` `LEFT` `RIGHT` (the left stick), `Y` `B` `X` `A`, `LT` `RT`, `LB` `RB`, `START` `BACK`, `DPAD_UP` `DPAD_DOWN` `DPAD_LEFT` `DPAD_RIGHT` and `RS_UP` `RS_DOWN` `RS_LEFT` `RS_RIGHT` (the right stick); a controller has the ones it has (the Current Designs pad: the left stick and `Y` `B` `X` `A` `LT` `RT`). A gamepad SDL knows, such as an Xbox controller, is read by SDL's layout of it, since its raw button numbers change with the OS and the connection; any other controller by the Current Designs pad's button numbers. The keys are key names as curricula write them (`fmri_gym/keys.py`). A control left out presses nothing. With no `"bindings"` at all, a plugged-in controller is warned about on the console and in the manifest's `pad` status, and presses nothing: a controller that types keys itself needs none. `fmri-pad` names each control as it is pressed. The rig file's `"has_pad": false` ignores the controller.

**Calibrate each stick once per rig.** A stick neither rests at 0 nor reaches 1 both ways -- one pad measured -0.03 at rest and travelled about twice as far right as left -- so a single symmetric threshold asks for twice the movement one way as the other. The calibration measures what a run needs, and every step waits for the controller rather than a clock: for each stick, hands off until it is still, roll it round its edge until every direction has reached its stop, let go until it is still again; then each of the stick's directions, asked for until it comes twice in a row, a wrong one named and asked again -- which is also the check that each control is the one it is named. Calibrating every stick the pad has asks for each of its buttons the same way. It takes about half a minute for a stick and six buttons. Each axis' rest and its two stops make a reading a fraction of the travel *available in that direction*; a stick presses a direction at half its travel and releases it under 0.3, raised for a stick that dithers at rest. A stick can be calibrated alone (a pad with two sticks, say): the other stick is copied from the calibration in use, with the time it was measured.

**One file per measurement.** Each calibration is a file of its own in `~/.config/fmri-gym/pads/` (`$XDG_CONFIG_HOME/fmri-gym/pads/`), named with the device and the time (`Controller_20261021-141500.json`), with the recording it was fitted from beside it (`.csv`); none is ever overwritten. It lives outside the checkout because it describes the hardware here, not the task. A run reads the newest file of the controller plugged in, by file time, and names it on the console and in the manifest's `pad` status; one made on another controller is never applied. One shape, every key required:

```json
{"device": "Current Designs, Inc. 932", "tuned_from": null,
 "sticks": {"left": {"made": "2026-10-21T14:15:00+00:00",
                     "x": {"rest": -0.074, "low": -0.577, "high": 1.0, "drift": 0.122},
                     "y": {"rest": 0.003, "low": -0.679, "high": 0.926, "drift": 0.002},
                     "on": 0.5, "off": 0.3, "diagonal": 1.0, "dwell_s": 0.05}}}
```

A run does not start, and says why, when the newest file of the controller plugged in is malformed (it names the file and the field), or when the rig binds a direction of a stick that has no calibration. A rig whose bindings name no stick, a controller with no stick, and a rig with no controller start without one. The rig check is the exception: it is what makes the calibration, and reads such a stick uncalibrated until it has.

**`fmri-pad` calibrates, shows and tunes it.** `fmri-pad` (`--rig configs/rigs/<site>/<rig>.json` to show the key each control is bound to there) opens a window on the controller plugged in: a tile per button, lit while it is held; the shown stick's two axes over time against its thresholds, a lane per direction, and the stick's square with the rule deciding it; and a log of what was pressed and released. Its buttons calibrate every stick and button, or one stick alone, in the same window (Cancel writes nothing); choose the stick shown; tune that stick's `on`, `off`, dwell and diagonals live; save; reload; clear the log; quit. Save writes a new file, the calibration in use with the tuning now and `"tuned_from"` naming the file it came from; the sticks' `made` stay those of the measurement, and the file it came from is left as it was. It reloads on its own, within a second, when a calibration is added or the one in use changes. Only its buttons act: the keyboard does nothing but ESC, since a program that turns a pad into keystrokes would otherwise drive it as the pad is pushed. Its window can be resized to any size from its minimum; the picture scales to fit.

**One direction at a time.** A thumb does not move one axis at a time, and a stick does not let it: on the pad this was written for, a push meant for *down* carried the horizontal axis to 0.87-0.98 of its own travel on half the trials. Thresholding the two axes separately therefore delivers diagonals nobody gave, which a game with four discrete actions -- nearly every curriculum here -- will act on. So a stick is read as one thing rather than two: how far it is from rest says *whether* a direction is held, and the angle of the push says *which*, by whichever of the four directions -- 0, 90, 180, 270 degrees -- it points nearest to. Exactly one is pressed. A game that steers can set a stick's `diagonal` below 1.0 (**Diagonals** in `fmri-pad`), which reads the axes independently again and allows two at once. The four directions are not fitted to where a cued push pointed: on the measured pad "down" came out 32 degrees round towards the left, consistently, but that skew is as much the hand that was measured as the stick, and the hand in the bore is a different one.

**A direction has to mean it.** A thumb does not leave rest along the line it is aimed at: on the way out to *down* the stick crosses the left sector, and for a frame it classifies as LEFT. Over 24 cued pushes that happened 5 times, always for one 17 ms frame, against pushes that were held for 254 ms at the shortest. So a new direction must hold for `dwell_s` (50 ms, tuned in `fmri-pad`) before it is pressed. Releases are never delayed: a key must not outlast the stick coming back. The cost is that a stick press is reported that much late, the same amount every time, which a reaction time measured off the stick has to subtract. Buttons are unaffected -- they are already clean.

**The centre is not a constant, so it is measured as one movement, not stored as one place.** Cold, this pad read -0.003 one session and -0.019 the next; after it had been deflected once it sat at -0.10 in the first and -0.165 in the second and stayed there. The gimbal does not return where it started. So the calibration measures *how far* the centre moves (`drift`) -- where the stick sits before it has been touched against where it settles after the roll -- and a run takes its centre once, the first time the stick is used and then let go still for a second, allowed to move only as far as that measurement says it can. It never re-centres while a direction is pressed, which would release the key under a subject leaning on the stick, and a stick that settles further out than the measured movement is refused with a message rather than believed: that is a stick that has drifted past its calibration, and the stops have not moved with it, so one side really has lost travel. With a 0.20 drift injected into a recorded session, a fixed centre held a key down on 43% of the frames where nobody was pushing; taking it once, 16%.

**The rig check calibrates it.** `check_controls` looks for a controller first. If one is plugged in it runs the calibration there and then, of every stick and button, on the window the check already has open, and writes a calibration of that controller -- so the measurement is part of setting the rig up rather than a separate thing to remember, and every later run on the machine with it reads what it wrote. The check then asks for its keys as usual, now through the calibration it just made; the calibration fails the check only if it is stopped before it finishes, which writes nothing. With no controller plugged in there is nothing to measure and whatever device is in use is expected to send the keys itself, which is what the key prompts test. `"has_pad": false` in the phase skips the measurement and leaves any existing calibration alone.

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
{"type": "game", "backend": "retro", "game": "SonicTheHedgehog-Genesis",
 "mode": "duration", "duration": 30.0, "fps": 60, "state_stride": 15}
```

The `retro` backend itself is verified with Airstriker (`configs/curricula/retro/levels/airstriker.json`), the one game whose ROM ships with stable-retro; any other needs both its ROM imported and an integration in stable-retro (see "Games without a curriculum").

## External checkouts (`external/`)

Three backends read their game files from a checkout of the game's own repo: `vgdl` (game, level and sprite files), `coom` (`conf.cfg`/`.wad` scenarios) and `baba_auto` (the C++ sources of `pyBaba`, plus maps and sprites). The checkout is a git submodule at `external/<backend>`, pinned to a commit, so every session plays the same tree; `git clone --recursive` (or `git submodule update --init`) fetches all three:

| backend | submodule `external/<backend>` of | commit |
|---|---|---|
| `vgdl` | [tomov/language_and_experience](https://github.com/tomov/language_and_experience) `dbp` | `d70c3248c504992068c8fb318238f1c3fbe956bf` |
| `coom` | [TTomilin/COOM](https://github.com/TTomilin/COOM) | `7929801176c6e2e036c7c1c7dd6ce9b84a9d1f3e` |
| `baba_auto` | [utilForever/baba-is-auto](https://github.com/utilForever/baba-is-auto) | `24cefb48d47ae6a6f5c0d936310d8bceb9c4279d` |

The env reads `external/<backend>` by default; a phase's `"repo"` field names another checkout for that block. To bump a pin, check the new commit out in the submodule, commit the submodule and this table together (for `baba_auto`, also `uv sync --reinstall-package baba-auto-gym`: the compiled engine is tied to the checkout it was built from).

## ARC-AGI-3 games

The `arc3` backend plays [ARC-AGI-3](https://arcprize.org/arc-agi/3) games (turn-based grid puzzles in several levels) through `arc3-gym` (`gym/arc3/`), which runs the `arc-agi` toolkit offline. It needs Python 3.12 or later, so `uv sync --extra arc3` skips it on older interpreters. The game files are downloaded once, with the network on, into `external/arc3` (gitignored): `uv run python -m arc3_gym ls20`. Playing then makes no network call and needs no API key. A game that offers a click (`ACTION6`) is refused at start-up, since the scanner has no pointer; `ls20`, `tr87`, `tu93` (ACTION1-4) and `g50t`, `re86`, `wa30` (ACTION1-5) are the keyboard-only ones at the time of writing.

## Running VGDL games

The `vgdl` backend drives the VGDL games from a gymnasium-ported fork: **[tomov/language_and_experience @ dbp](https://github.com/tomov/language_and_experience/tree/dbp)**. Because it runs under gymnasium + numpy 2, no separate env is needed — the same `fmri-gym` env works.

1. The fork is the `external/vgdl` submodule, at its pinned commit on the `dbp` branch, which has the gymnasium port (see [External checkouts](#external-checkouts-external)).

2. Install the `vgdl` extra -- **`vgdl-gym`** (`gym/vgdl/`), the standard Gymnasium env over the fork's `VGDLEnv` (a game name and level in, `reset(seed=)`, an offscreen `render()`; it puts the checkout on `sys.path`) -- and run a VGDL curriculum:

   ```bash
   uv sync --extra vgdl               # or: pip install -e gym/vgdl
   uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/laptop.json --curriculum configs/curricula/vgdl/all.json
   ```

   Game files live at `<repo>/games/<game>_v0/<game>.txt` and `<game>_lvl<level>.txt`. Available games include `aliens`, `beesAndBirds`, `avoidGeorge`, `jaws`, `missile_command`, `plaqueAttack`, `portals`, `preconditions`, `pushBoulders`, `relational`.

VGDL blocks log a symbolic per-cell object grid (`symbolic_state`) and collision `events` as analysis variables, plus a per-frame exact savestate (get/set_state) for determinism-free reconstruction.

## Running COOM games

The `coom` backend plays [TTomilin/COOM](https://github.com/TTomilin/COOM)'s own continual-RL Doom scenarios (`pitfall`, `chainsaw`, `hide_and_seek`, `health_gathering`, `arms_dealer`, `parkour`, `raise_the_roof`, `run_and_gun`, `floor_is_lava`) -- distinct from the stock ViZDoom scenarios the `vizdoom` backend already covers (DeadlyCorridor, DefendCenter, ...).

COOM's own Python package pins `gymnasium==0.28.1`, which conflicts with minihack's `gymnasium==1.2` pin in this shared env, so **the COOM package is never installed or imported**. The gym env is **`coom-gym`** (`gym/coom/`, the `coom` extra; also in `dbp`), which drives `vizdoom.DoomGame` on the scenario files of a COOM checkout at its pinned commit (the `external/coom` submodule; see [External checkouts](#external-checkouts-external)):

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/laptop.json --curriculum configs/curricula/coom/levels/pitfall.json
```

The env reads `<repo>/COOM/env/scenarios/<scenario>/conf.cfg` and `<task>.wad` (`env_kwargs.task`, default `"default"`; `run_and_gun` also ships `blue`, `red`, `hard`, ...). Without the checkout the run stops at start-up.

Every scenario always exposes exactly 4 buttons (`TURN_LEFT`, `TURN_RIGHT`, `MOVE_FORWARD`, plus one of `JUMP`/`ATTACK`/`SPEED`/`USE`), driven through COOM's own 12-action table (turn x move x execute), so a phase's `keys` are indices into it -- 8 = turn left, 4 = turn right, 2 = forward, 1 = the 4th button, sums for combos, `"": 0` for no key; each `coom__*.json` spells its map out. COOM blocks log the raw ViZDoom `game_variables` (health, ammo, position, ...) as an analysis variable; the savestate is the seed and the actions since (`ReplayState`, see "Savestates by replay" below), restored by replay. The nine COOM curricula enable audio via `env_kwargs.audio_buffer_enabled`. On Ubuntu, install OpenAL with `sudo apt install libopenal1`. Keep `fps: 35` for one Doom tic of 44.1 kHz stereo PCM per step. The rig file's `"has_audio": false` mutes playback but preserves recorded PCM; terminal frames are marked `audio_valid=false`. `audio_efx: false` disables reverb to avoid an older OpenAL crash. MIDI music requires a working engine MIDI renderer; check the startup console for errors. Queued sounds can be cut at episode end. Logged `audio_onset` estimates DAC timing, not physical speaker latency.

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
uv run fmri-play --subject sub-01 --dummy-trigger --ses 1 --run 1 --rig configs/rigs/example/laptop.json --curriculum configs/curricula/aigamestore/all.json
```

Phase fields: `game` (`"game1"`…`"game10"`, plus `/level<N>` to name the level -- `"game6/level3"`; bare `"game6"` is level 1 -- or an `http(s)://…/index.html` URL, then `game_keys` lists the keys that game listens for), `frame_skip` (default 6; `fps` must equal `60 / frame_skip`, and the block refuses to start otherwise), `headed` (show the browser window), `browser_channel` (`"chrome"` default, or `null` for the bundled Chromium), `games_dir` (override the vendored dir). A curriculum lists one game phase per level it wants played (`configs/curricula/aigamestore/game1.json` has all nine of Water Sort's). `keys` values are indices into the game's key list (`aigamestore_gym.GAME_KEYS`; game1: LEFT, RIGHT, SPACE, Z), so `{"B": 2}` puts the game's SPACE on the B key; the env relabels the on-canvas hints to match.

## Running Rush-Hour

[Rush-Hour](https://github.com/chrplr/Rush-Hour) is a sliding-block puzzle written as a psychophysics experiment in Go, with its rules in a small engine binary that both an agent and a participant play through `rushhour-gym`. The `rushhour` backend drives the package's `RushHourHuman-v0` — the experiment program's own interface as an env: car selection on four buttons, its picture, its results columns in `info` — so the adapter is a keymap plus the fields to log. One game phase is one puzzle (`"puzzle": "p07"`); the program's ready screens, blank intervals and solved feedback are `message` phases the curriculum lists around each puzzle, so every puzzle is its own block in the manifest and its own log folder. Nothing to install beyond the `rushhour` extra: on first use the package downloads the engine of its matching release into `~/.cache/rushhour-gym/` (checksum-verified; Linux x86-64, macOS arm64, Windows x86-64). On a machine without network, run a config once while online or copy that directory; `RUSHHOUR_ENV_BIN` names a binary of your own.

```bash
uv run fmri-play --subject sub-01 --dummy-trigger --ses 1 --run 1 --rig configs/rigs/example/laptop.json --curriculum configs/curricula/rushhour/levels/easy.json      # 5 min of random easy puzzles
uv run fmri-play --subject sub-01 --dummy-trigger --ses 1 --run 1 --rig configs/rigs/example/laptop.json --curriculum configs/curricula/rushhour/complete.json   # the program's session then the rest of the library: 49 puzzles, one block each
```

Controls, phase fields and the logged columns are documented in `configs/curricula/rushhour/common/README.md` and in the package's README ("A person at the board").

To develop Rush-Hour and fmri-gym together, install the package editable from a Rush-Hour checkout and point at the engine built there, which the adapter then uses. The engine's default Go build is headless (no SDL, no C toolchain), so it needs only the Go compiler:

```bash
RH=/path/to/Rush-Hour
uv pip install -e "$RH/python"
(cd "$RH" && go build -o rushhour-env ./cmd/rushhour-env)
export RUSHHOUR_ENV_BIN="$RH/rushhour-env"
```

## Running Baba Is Auto (`baba_auto`)

[utilForever/baba-is-auto](https://github.com/utilForever/baba-is-auto) is a C++ simulator of Baba Is You with the game's real ruleset and the original levels' maps (`baba_is_you`, `out_of_reach`, `off_limits`, ...), where the `baba` backend plays baba-is-ai's small generated puzzles. Its Python module exposes only the `Game`, so the Gymnasium env is **`baba-auto-gym`** (`gym/baba_auto/`, the `baba_auto` extra), one `baba_auto__<level>.json` per level the checkout ships. Installing the extra compiles the engine from the pinned checkout (see [External checkouts](#external-checkouts-external); a C++17 compiler and `python3-dev` are needed; about 30 s), and the env reads the maps and the GUI's sprites from the same checkout:

```bash
uv sync --extra dbp --extra baba_auto
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/laptop.json --curriculum configs/curricula/baba_auto/levels/baba_is_you.json
```

A phase's `game` is a map name under the checkout's `Resources/Maps` or a map file's path; `keys` index `Discrete(5)`: 0 = wait, 1 = up, 2 = down, 3 = left, 4 = right (turn-based). Each turn logs `play_state` (0 playing, 1 won, 2 lost) and the engine's `(16, H, W)` state tensor. The savestate is the seed and the moves since (`ReplayState`), restored by replay at a few ms a move.

## Running SuperTuxKart (`stk`)

`supertuxkart-gym` is the current SuperTuxKart with a gym server built in (`--gym`), plus the `stk_gym` Python client that drives it. The backend is called `stk` and the client package `stk_gym`; the backend drives `stk_gym.StkEnv` with `render_mode="rgb_array"` and `action_mode="keys"`: the game renders into a window that is created hidden, every step brings the frame back and fmri-gym shows it; the held keys go to the game's own player controller, so steering ramps and skids latch as they do for a keyboard. Participant and model are in front of the same env object, and a block replays from its launch `seed`, `episode_seeds` and `actions`. A restarted race is not bit-identical to a fresh launch (`stk_gym`'s README measures the drift), so the adapter relaunches the game for every episode (about half a second of track loading between races). The adapter is a keymap and the fields to log. (This replaced an earlier pystk2-gymnasium backend, which had no pixel obs and needed its own control mapping, so human and model play were not the same interface.)

```bash
uv sync --extra stk                 # or: pip install supertuxkart-gym
uv run fmri-play --subject sub-01 --dummy-trigger --ses 1 --run 1 --rig configs/rigs/example/laptop.json --curriculum configs/curricula/stk/levels/hacienda.json
```

There is one config per race track the asset pack ships with geometry — `stk__{hacienda,cornfield_crossing,lighthouse,scotland,snowmountain}.json`, 880–1330 m per lap, three laps, four karts. The pack's other track directories (`abyss`, `mines`, `zengarden`, … and the battle arenas and soccer fields) are metadata only: naming one kills the game process, so more tracks means a pack that carries them, not more configs. A track is fixed for the life of a process (`reset(options={"track": ...})` is refused by name), so race several by giving a curriculum one game phase per track.

No checkout and no build: the wheel is pure Python and fetches the game with a trimmed asset pack (270 MiB, five tracks) from its GitHub release the first time an env is made, into `~/.cache/supertuxkart-gym`. It says so while it downloads, and never does it twice. The pack is tied to the package version, so a wheel and its engine cannot drift apart: upgrading `supertuxkart-gym` is the whole update procedure. Linux x86_64 only for now; on anything else it says which platform it has no pack for. The engine links OpenAL and SDL2 from the system (`libopenal1 libsdl2-2.0-0`, in the install line above); without them it cannot start, and `stk_gym` reports `EngineDied: ... (exit status 127)`.

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
  config/
    curriculum.py   # Curriculum: the curriculum file, loaded once into typed phases; load, save, includes
    phases.py       # one dataclass per phase type (Fixation, Message, Survey, GamePhase, Check*), its checks
    rig.py          # the rig files (configs/rigs/<site>/<rig>.json): what it is, screen, pad, audio, data root
    run.py          # RunConfig: one run's command line and the rig it is played on
  display.py        # pygame: fixed window, aspect-fit frame, fixation, text; vsync-locked flip + call_on_flip
  logging.py        # manifest.json + one crash-safe JSONL/HDF5 log per game block
  replay.py         # replay an episode from a block's log; its frame fields as arrays
  menu.py           # the hold-a-key pause menu (reset / forfeit / resume), opt-in per game phase
  triggers.py       # run-start sync (wait/send/none) + MEG/EEG trigger codes over lsl/serial/parallel/hid
  photodiode.py     # `python -m fmri_gym.photodiode`: flash a patch to measure the flip-to-photon offset
  checks.py         # the rig check: its phases, report.html/.md, rigchecks.tsv, `pool`
  adapters/
    base.py         # EnvAdapter + FrameState (the seam)
    keymap.py       # the phase's keys -> actions, one Keymap per action space
    ale.py          # clone_state, getRAM, lossless indexed pixels
    retro.py        # em.get_state, get_ram, decoded info vars, console-button keymap
    default.py      # ANY gym env: rgb frames, seed+replay, obs-as-state
    vgdl.py         # VGDL via vgdl-gym: get_state/set_state savestate, symbolic grid + events
    crafter.py      # Crafter via crafter-gym: obs is the frame; achievements
    minihack.py     # pixel_crop as the env id defines it (minihack_levels: -Allocentric-v0 twins fixed on the map) + compass keymap; blstats/glyphs/message
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
configs/            # curricula/<backend>/, rigs/<site>/, subjects/<site>/sub-*/ses-*/ (configs/README.md)
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

`autoplay` is how an `is_turn_based` block gets through a stretch the env takes no action in. Crafter is the case it exists for: a sleeping player's action is overwritten with `sleep` until their energy fills, so the sixty-six presses a full sleep would cost under `is_turn_based` are all discarded by the engine, and how many there are is a fact about the game rather than a decision the subject is making. Returning an action says "step with this and do not wait for a press"; the loop keeps doing so, at the phase's fps and logging every frame like any other, until it returns `None` again. Nothing is skipped and nothing is hidden: the subject watches the night pass, and because waking from a blow ends the stretch, the picture stops on the frame with the zombie beside them. The default is `None`, so every other backend and every real-time phase is unaffected.

`self.env` is always a `gymnasium.Env`. A game that has no such env -- a bare engine, an old-`gym` env, a gymnasium env with habits of its own -- gets a thin one under `gym/` (see `gym/coom/`, `gym/baba/`, `gym/vgdl/`), not in its adapter; the adapter is then `_make` plus what to log.

`FrameState` carries a standard shape for **every** backend:
- `blob`: opaque bytes that `restore()` turns back into this exact state: the engine's own snapshot where it has one (ALE `clone_state`, retro `em.get_state()`, VGDL `get_state`, crafter's pickled env), else the seed and the actions since, restored by replay (`adapters/replaystate.py`, "Savestates by replay" below). `None` only for the generic `gym` backend, whose env is unknown.
- `variables`: named analysis fields surfaced uniformly (`ram`, retro's decoded `info_score`/`info_lives`, the raw `obs`, …), so the loop and downstream analysis code are identical across engines.

**Adding a new engine = writing one adapter** (~40–80 lines). Nothing else changes.

## Curriculum format

A **run** is one JSON file: a `"curriculum"` of phases, an optional `"triggers"` section (below), and comments: any key that starts with `##` (`"## why": "..."`), used sparingly -- a backend's curricula are documented in their `common/README.md`. A bare list, an unknown top-level key, an unknown phase `type` or a phase key that starts with `_` stops the run at start-up with the reason.

One config is one run. A whole scanning session is a plain shell script with one line per run, in order -- so every run is a process of its own, with a fresh interpreter, display and trigger port:

```sh
#!/bin/sh
# fmri-gym session: one line per run, in order.
set -e
SES=${1:-$(uv run fmri-ses --subject sub-01 --rig configs/rigs/example/laptop.json)}
uv run fmri-play --subject sub-01 --ses "$SES" --run 1 --rig configs/rigs/example/laptop.json --curriculum configs/pong.json
./scripts/localizer.sh "$SES"
# uv run fmri-play --subject sub-01 --ses "$SES" --run 1 --rig configs/rigs/example/laptop.json --curriculum configs/mario.json
```

**The numbers come from the script, not from the disk.** The `SES=` line picks the session once, so every run lands in it: the script's own argument if it was given one (`sh ses1.sh 003` resumes session 3), else the subject's next free session. Each run states its `--run`, which is its place among the lines that play that task -- so it is the same run number however the session went, and skipping a line renumbers nothing after it. Both flags are required of `fmri-play`: it never picks a number itself.

A run whose folder already has data is **re-acquired, never overwritten**: it writes to `..._02` (then `_03`) beside the attempt that stopped. The suffix names the folder only, so the re-acquisition replays the same episodes as the run it replaces.

`set -e` stops the script at the first run that fails or is quit with ESC (`fmri-play` then exits with status 3). A line that is not an `fmri-play` run is any command of yours, as typed; a commented line is a skipped run, which is how a stopped session is resumed. Write it by hand, or in `fmri-edit`, whose Session manager tab has two panels, each a form and the text it stands for: Session design (the list of lines -- add an existing or a new config, an external script; repeat, reorder, skip, "Start here" -- or the script itself) and Run design (the selected run's phases, or its JSON). Each panel starts with a drop-down of what `configs/` holds: the Session one opens a script, the Run one opens a config (or, in a session, adds it after the selected line); File > Open takes a `.sh` like a `.json` from anywhere. Save writes the script and the configs you edited. Run it from the repo root: `sh configs/subjects/<site>/sub-01/ses-001/<name>.sh`.

Three example sessions of about an hour each ship in `configs/subjects/boston-college/sub-01/` -- eleven runs of one DBP game apiece (5 min of play, plus the instructions screen and the start-up between runs):

```bash
sh configs/subjects/boston-college/sub-01/ses-001/dbp_mix.sh      # one run per genre: Crafter, COOM, MiniHack, Rush Hour, Baba, ViZDoom, AI GameStore...
sh configs/subjects/boston-college/sub-01/ses-002/dbp_doom.sh     # the nine COOM scenarios, then two ViZDoom ones
sh configs/subjects/boston-college/sub-01/ses-003/dbp_puzzle.sh   # nine AI GameStore puzzles, Rush Hour, Baba Is AI
sh configs/subjects/boston-college/sub-01/ses-001/dbp_mix.sh 003  # ... into session 3: how one that stopped is resumed
```

`configs/subjects/boston-college/sub-02/ses-001/dbp_multi.sh` is a session of resumed runs: eleven sequences of levels (COOM, ViZDoom, AI GameStore games 1, 4, 7 and 8, Crafter L1, Rush Hour, Baba Is Auto, MiniHack, STK), each played as two 1-min runs back to back with `--duration 60 --loop --resume`, so the second run carries on where the first stopped and a curriculum that ends starts over (see "Resuming a curriculum"). STK keeps no savestate, so its two runs play `--loop` without `--resume`.

The sub-01 sessions assume `sub-01` and a 1024x768 window, and play into the session their folder names unless given another: open one in `fmri-edit --session configs/subjects/boston-college/sub-01/ses-001/dbp_mix.sh` to change any of that, or `--dummy-trigger` a run of it at the desk.

```jsonc
{"type": "fixation", "duration": 2.0}                 // "+" for N seconds
{"type": "fixation", "duration": 1.0, "include_on_resume": true}  // also played first by a resumed run (below)
{"type": "message", "text": "Get ready", "duration": 2.0}  // text: string or list of lines; omit duration to wait for a key
                                                           // ("key": a character, or "keys_to_dismiss": key names)
                                                           // a key named in brackets, "[SPACE]", shows as
                                                           // what the participant presses on the rig: "SPACE"
                                                           // on a keyboard, "A" on a pad that binds A to SPACE;
                                                           // "[ARROWS]" shows "ARROWS", or "stick" on a pad
{"type": "survey", "n_points": 7, "questions": ["...","..."]}

{"type": "game",
 "backend": "ale",              // "ale" | "retro" | "gym"
 "game": "ALE/Pong-v5",         // env id for that backend
 "instructions": {"title": "Pong", "text": ["Move with [UP] / [DOWN].", "(press [SPACE])"],
                  "align": "center", "duration": 30.0, "keys_to_dismiss": ["SPACE"]},
                                // optional: the screen before the block, "Starting Pong" (or
                                // "Resuming Pong" when a run carries the block on) over the text,
                                // until duration runs out or one of the keys is pressed (at least
                                // one of the two). Outside the block's clock, inside the run's
 "mode": "duration",            // "duration" = replay until time up; "episode" = play N episodes
 "duration": 30.0,              // seconds the block lasts (duration mode, default 30) or lasts at
                                // most, a wall-clock cap (episode mode, default 300)
 "n_episodes": 1,               // episodes (episode mode)
 "advance_on": ["won"],         // episode mode, optional: only an episode that ends this way counts
                                // towards n_episodes; any other replays the same instance (a level
                                // is repeated until cleared). Names: won, lost, terminated,
                                // truncated, playing (the block's clock) -- see EnvAdapter.outcome
 "show_outcome_duration": 2.0,  // seconds "Final score: N" + the outcome show after each episode
 "rewind_on": ["lost"],         // optional, with rewind_frames: an episode ending this way (won,
 "rewind_frames": 1,            // lost, terminated) sends the world back rewind_frames frames and
                                // play goes on, as a new episode starting from that state (its
                                // episode_start carries it as initial_state; it does not count
                                // towards n_episodes). Session policy, not a rule of the game, so
                                // agents/agent_play.py ignores it. Needs a backend whose capture
                                // returns a savestate every frame (crafter, ale, retro, vgdl)
 "fps": 30,                     // required: steps (and frames) per second. The engine's own rate
                                // (console cores and Atari ~60, Doom 35 / frame_skip) plays the game
                                // at its real speed and fits its sound; the editor's Controls tab
                                // (Check with the engine) shows it. Any other value plays the game slower or faster: the
                                // manifest logs "speed" and the console says so when it is not 1
 "is_turn_based": false,        // step only on a key PRESS, not per frame (grid/toy_text games)
 "has_latched_keys": false,     // real-time: a fresh key PRESS beats the held-key poll, so a tap
                                // shorter than one frame is not dropped (games below ~10 fps)
 "has_live_hud": false,         // is_turn_based only: redraw the standing frame while the loop waits
                                // for the press, so the clock in the HUD keeps up with the block's.
                                // Draws pixels and nothing else -- no step, no log row, no trigger,
                                // no sound. A duration block ends on the wall clock whether anyone
                                // presses or not, so without this a subject who is thinking reads
                                // the time their last press left behind
 "has_audio": true,             // false mutes the speakers for this block; logged sound is unchanged
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

### Shared phases (the `__include` field)

The levels of one game repeat most of their game phase: the keys, the instructions, the mode, the env's settings. A phase can instead start from a shared one and write only what differs:

```jsonc
// configs/curricula/vizdoom/levels/freedoom1_e2m5.json
{"__include": "../common/freedoom1.json", "game": "VizdoomFreedoom1E2M5-S2-v0",
 "instructions": {"title": "Freedoom: Underground Hub"}}
```

`__include` names a file relative to the curriculum's own folder, which holds one phase object (`configs/curricula/vizdoom/common/freedoom1.json`: type, backend, instructions, `env_kwargs`, keys, mode, fps). The `common/README.md` beside it explains them: setup, what the keys' indices mean, the notes a level needs. The phase's own keys are laid over it: an object in both merges key by key, so the line above changes the title and keeps the instructions' text; anything else (a list, a number, a string) replaces the shared value whole. A key cannot be taken away, so a shared phase holds only what every phase that includes it keeps. A shared phase may include another, relative to its own folder. Shared phases live in folders named `common`, which hold no curricula: neither `fmri-edit`'s lists nor the tests treat them as runs. A backend's curricula of one level each sit in its `levels/` folder, beside `common/`, so they include `../common/<name>.json`; those that play several levels sit in the backend's folder and include `common/<name>.json`.

Includes are resolved when the file is read, so everything after that sees each phase in full: `fmri-play`, the editor and the manifest, which keeps the run's curriculum as it was played. `fmri-edit` keeps each phase's `__include` (a game phase shows it in its extra JSON, as a path from the working directory) and saves the phase as its include and what it adds. A phase that changes something its shared phase sets makes the editor ask before saving: the change is that curriculum's alone, and if it should hold for every level, it belongs in the file in `common/`, which every phase that includes it then shares. A phase that drops a key of its shared phase cannot be written as an include, so it is saved in full.

### Keys (the `keys` field)

Every game phase states its whole keyboard map. **There is no default**: which key does what differs between sites (a keyboard at the desk, a gamepad that types keys in the scanner), so the file says it, and a phase without `keys` is refused before the window opens. `keys` is a dict of `"<key(s)>": <action>`:

- The key is a name from `fmri_gym/keys.py` (`UP`, `DOWN`, `LEFT`, `RIGHT`, `SPACE`, `RETURN`, `LSHIFT`, `A`–`Z`, `0`–`9`, `F1`–`F12`, `KP0`–`KP9`, …); a name not in that table is refused, since it could never be pressed. Join keys with `+` for a combo (`"UP+SPACE"`).
- The action is what the env's `step` takes, as JSON, and the rule is the env's action space (`fmri_gym/adapters/keymap.py`, one class per space):
  - **`MultiBinary`** (retro, vizdoom, stk, aigamestore): the value is the **index of the button** the key holds down; every held key sets its bit, so keys combine as on a controller, and nothing held is every button up. An index outside the space is refused when the env is built, and so is a `""` entry (nothing held already means no button).
  - **`Discrete`** (ale, coom, vgdl, crafter, minihack, baba, rushhour, gym) or **`Box`** (gym): the value is the action itself, an index or a list (`"LEFT": [-1.0, 0.0]`). The most specific combo whose keys are all held wins (`"UP+SPACE"` over `"UP"`), and the **`""`** entry is what a frame with no key held sends. `""` is required unless the phase is `is_turn_based`, where nothing is sent between presses: these spaces have no action that means "do nothing" everywhere (FrozenLake's 0 is LEFT, MiniHack's is "move N"). A value outside the space is refused when the env is built.
  - Any other action space is refused when the env is built.

Each adapter's module docstring says what its indices mean, and each backend's `configs/curricula/<backend>/common/README.md` spells out the map its curricula use. For an Atari game, read its meanings:

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

Holding `key` for `hold` seconds -- even if that key does something in the game -- pauses the game and shows the options; `move` steps through them and `confirm` picks one. `reset` starts the episode over -- a new `reset()` of the env with the same seed, so the very same level instance -- `forfeit` ends the block and moves on to the next phase, `resume` continues. The menu is unavailable for the first `after` seconds of the block (0 = always), and a hold plus a two-key choice is hard to do by accident -- the point is that this is possible but never done lightly. Only `key` is required; the rest default to the values shown, and `options` may list any subset, in the order to show them. The manifest's phase entry records the settings and every pop-up (`menu.events`: when, and what was chosen); a reset or forfeited episode's `episode_outcome` is `reset` / `forfeit` (the env's own `terminated` / `truncated` flags are never touched), and a reset one does not count towards `n_episodes`. `configs/curricula/baba/levels/make_win.json` uses it.

### Resuming a curriculum (`--duration`, `--resume`)

A curriculum belongs to the subject, who works through its levels at their own pace; a run only says how much of it fits in one scanner acquisition. `--duration S` gives the run its length in seconds from the trigger: whatever phase is on screen stops 10 s before the end, mid-episode included, and the run closes on 10 s of fixation for the haemodynamic lag (a run that finishes the curriculum sooner ends on the same 10 s); `--end-fixation S` sets that fixation's length. `--resume` starts where the subject's latest run of the same config file (any session) stopped, read from that run's manifest and its last block's `block_end` line (`block_did_not_finish`, `block_time_elapsed`, `num_completed_episodes`, `extra.state`): a block it cut off goes on with the time and episodes it has left, the episode it was in restored from its savestate; otherwise the run starts after the last game block played, or at the last other phase shown, from its beginning. The phases marked `"include_on_resume": true` -- only the non-game phases the curriculum opens with -- are played first. ESC keeps the place the same way. The config may change between runs as long as the game the block plays does not: `duration`, `n_episodes`, `advance_on`, `instructions`, `keys` and the like can, and any other change to that phase stops the run with the fields that changed. With no earlier run `--resume` starts at the beginning, and on a config already played to its end it refuses. `stk` keeps no savestate, so a run that stopped one of its episodes cannot be resumed into it: `--resume` refuses before the window opens, and says to start over (a block it stopped between two races carries on with the next).

```sh
uv run fmri-play --subject sub-01 --ses "$SES" --run 1 --rig configs/rigs/example/laptop.json --duration 600 --resume --curriculum configs/curricula/rushhour/complete.json
uv run fmri-play --subject sub-01 --ses "$SES" --run 2 --rig configs/rigs/example/laptop.json --duration 600 --resume --curriculum configs/curricula/rushhour/complete.json
```

`--loop` plays the curriculum again from its first phase when it ends before `--duration`, and again, until `--duration` stops it; it needs `--duration`. Each pass plays the same phases with the same seeds; a block played again writes `block-NN_<backend>_<game>_2`, `_3`.... With `--resume`, a curriculum the last run played to its end starts again from its beginning instead of being refused. The manifest says `"loop"`.

A resumed episode starts from that savestate rather than from its seed alone, so its `episode_start` line carries it as `initial_state`, and `fmri_gym.replay.reconstruct_episode` restores it over the reset. Without `--duration` and `--resume` a run plays its config from start to end, as before.

## Triggers: fMRI vs MEG/EEG

The rig file's `"triggers"` section (see *Rigs*) says how a run starts and what the recording gets, for every curriculum played on the rig (a full example is in `configs/rigs/README.md`):

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

`backend`: `null`, `lsl`, `serial`, `parallel` or `hid` — `uv sync --extra triggers` (pylsl / pyserial / pyparallel / hidapi); `port` for serial/parallel, `lsl_stream_name` for LSL. `hid` drives the digital port of a USB DAQ wired to the trigger input, where a PC has no parallel port (a Mac): `"hid_device"` names the box (`mcc-1208fs`, a Measurement Computing USB-1208FS, the default) and `port` its digital port (`A` or `B`). It holds each code on the lines like a parallel port, and opening it sets the port to 0.

Which of these a rig needs varies: the scanner may type a key at every volume, or start its recording when the stimulus PC sends a code, or neither. So the rig file says what happens rather than naming a modality, and the editor's Triggers tab offers a template per common setup:

| setup | `sync.mode` | `backend` |
|---|---|---|
| the scanner types a key at every volume | `wait` | `null` (no trigger line) |
| the recording starts from the trigger input | `send` | `serial` / `parallel` / `hid` / `lsl` |
| the recording is started by hand, the PC gets the scanner's key | `wait` | `serial` / `parallel` / `hid` / `lsl` |
| bench test, nothing connected | `none` | `null` |

Leaving `sync.mode` or `backend` out defaults to `wait` / `null`, and says so: the experimenter screen, the console and the manifest all report it (`NOT SET in the rig file: sync.mode, backend`), as they do for `--dummy-trigger`.

What is sent: `task_start` when the clock anchors, `episode_start` at each reset, one code per frame (`"frame_every": N` to thin, `"on_frame": false` to drop), `task_stop` at the end. Codes never share bits, so two triggers on the same sample still decode: frames cycle 1–7 in the low 3 bits, `task_start`=8, `task_stop`=16, `episode_start`=32, `scanner_start`=64, and a lifecycle code is OR'd with the current frame code (all under `"codes"`; overlaps are refused). Every value sent is logged: per frame as `trigger` on the block's `frame` lines, lifecycle events with their `run_time` under `triggers` in `manifest.json`.

## Timing

Frames are shown with a vsync-locked flip and each frame's onset is logged as `flip_time`; message/fixation onsets in the manifest are flip times too. Key presses and releases are logged as they arrive (`key_time`, `key_name`, `key_down`), independent of the frame grid. The manifest records the display actually obtained (`vsync`, measured at start-up; `refresh_rate`).

- A frame is shown at the next refresh after its step, so an `fps` that divides the refresh rate (30 or 60 on a 60 Hz screen) shows every frame for the same number of refreshes; otherwise frames alternate between one and two and each onset can be up to one refresh late. `flip_time` records what happened either way. Some cores' own rate is 59.92: close enough to 60 Hz that one frame in ~800 is shown twice.
- Check that the rig locks to the refresh before a session: `python -m fmri_gym.display --fullscreen` (verdict LOCKED / NOT locked; if not, use fullscreen and disable the desktop compositor). `--no-vsync` turns the request off. Pass the rig's monitor (its `"screen"` `"monitor"`) here and to the photodiode as `--monitor`: refresh, vsync and the photon offset belong to the monitor.
- Once per rig, measure the constant flip-to-photon offset with a photodiode on the screen, then subtract it from `flip_time` and the frame triggers:

  ```bash
  python -m fmri_gym.photodiode --fullscreen --rig configs/rigs/<site>/<rig>.json     # diode into the MEG/EEG amp, codes on the rig's line
  python -m fmri_gym.photodiode --fullscreen --audio                          # diode into this PC's sound card
  python -m fmri_gym.photodiode --fullscreen --audio --audio-click --mic       # + mic on input 1: when sound is heard
  ```

  The first flashes a patch with the frame trigger on each white flip; match the triggers to the diode edges in your recording with `fmri_gym.photodiode.match_edges(trigger_times, edge_times)`. The second records the diode on the sound-card input and prints the offsets itself (`--list-audio-devices` to pick the input). `--audio-click` also plays a tone burst on each white flip through the session's audio output and reports when it reaches the DAC; with `--mic`, when a microphone on input channel 1 hears it.
- **The rig check** runs all of these as one run: `configs/curricula/checks/rig-check.json`, whose curriculum holds check phases instead of games, played by `fmri-play` with the window, trigger line and audio output a session opens. So the editor edits it like any config -- the line it tests is the rig's (its **Triggers** tab edits the rig's triggers), its **Controls** tab the buttons it asks for (Use a device layout to fill them in) -- and it can stand first in any session: add `configs/curricula/checks/rig-check.json` as its first run. A failed test does not stop the check or the session: it is listed, with why, on screen, on the console and in the report. On its own, the session `configs/rigs/rig-check.sh` plays the long check (filed under `sub-rig`, as MEG-BIDS files empty-room recordings under `sub-emptyroom`; its first line, the quick check, is skipped: un-skip it to play that one): open it in the editor, `fmri-edit --session configs/rigs/rig-check.sh`, and press Play, or run `sh configs/rigs/rig-check.sh`.

  | phase | test | fails when |
  |---|---|---|
  | `check_display` | display | flips are not locked to the refresh (missed refreshes are counted) |
  | `check_frames` | frames | a built-in test pattern, played as a game through the session's own game loop at each of `rates` fps and under each of `loads` (`cpu`: every core busy), loses a frame, shows one a refresh later than its rate allows, resets its pacing, or its frame triggers do not mark every frame in their cycle (and, given `recording_hz`, for 2 samples each) |
  | `check_triggers` | triggers | the line does not open, LSL does not read back what was sent, or the scanner's pulses do not come or come irregularly (counted when the run waits for the scanner). This PC's ports and LSL are listed first, with a warning when none is usable. Codes sent over serial/parallel/hid are printed: check them in the recording |
  | `check_controls` | controls | a key is not read back through the event queue as itself (automatic), or is not pressed when asked for on screen (the input devices plugged in are listed) |
  | `check_photodiode` | photodiode | the diode misses a flash, or sees only the flashes with a click (it hears sound). `"readout": "recording"`: matched offline to the frame triggers |
  | | audio | a tone burst (every other flash, through the session's output) never reaches the DAC, or with `"has_mic": true` a microphone at the ear on input 1 misses one |

  The rig-check configs ship with `"triggers": null`: no setup is anyone's default. The first rig check asks for one -- a preset (fMRI, MEG, EEG, behavioural) or your own -- and saves it in the config; the Triggers tab changes it later. With no screen for the dialog it stops and says so.

  Every check runs whichever fails, and each test gets its verdict. The screen says what each check is doing -- which code goes out on which line, which flash (and whether it carries a click), which rate under which load -- and then its verdict. A session's runs must share one triggers section: the editor's Check says so when they differ (each run keeps its own, so a change on the Triggers tab applies to the selected run only). `configs/curricula/checks/rig-check.json` is the **quick** check, before every session: under a minute, most of it pressing the buttons -- enough to show today's rig is the one that was measured. `configs/curricula/checks/rig-check-long.json` **measures** the rig, once per rig and after any hardware, driver or OS change (about 25 min; the frame test alone plays 60, 30, 20 and 50 fps for a minute each, idle and under CPU load): offsets to a fraction of a ms, their drift in ms/min (a sound card's clock runs apart from the PC's), missed refreshes and pulses, the TR on this PC's clock. With `sync.mode` `wait`, the check waits for the scanner like any run: start a sequence, or the trigger box's test mode.

  Each rig needs a **rig file**, `configs/rigs/<site>/<rig>.json` (see *Rigs* under [Quick start](#quick-start)): site, rig, PI's initials, modality, and what software cannot see -- monitor or projector, photodiode, sound path to the ear, trigger hardware -- with what every run on the rig uses (screen, has_pad, has_audio, data_root). When it is missing or invalid -- a machine with no rig yet, or a `--rig` it has no file for -- the check opens a form to fill it in (the `gui` extra; Save stays disabled until every field is valid) before the window; cancelled, or with no screen, it stops. `python -m fmri_gym.checks rig --rig configs/rigs/<site>/<rig>.json` reopens the form when the hardware changes.

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

  Beside the rig file, every check also files what it measured on the rig: `configs/rigs/<site>/measured/<rig>-<n>.json`, `n` one more than the last check's, when it measured any latency (a check with no photodiode or microphone files nothing). Its `latency` holds flip to light at the screen (`flip_to_photon`), flip to the sound card (`flip_to_dac`) and sound card to the ear (`dac_to_ear`: amplifier, tubes, earphones), each the check's median, spread and count, `null` where the check could not measure it or its test failed; with them every check's numbers, the verdicts, the rig file as played, the software (fmri-gym and its git commit, Python, pygame, SDL, numpy) and the machine (OS, processor, memory, displays and their refresh rates, GPUs, sound devices, input devices, what is plugged into USB). The rig file is the rig's settings, shared in the repo; the measurements are one machine's and stay on it (`.gitignore`). A run on a rig with none says so on the console; the first run of a session, run from a terminal, offers to play the quick or the long rig check first, on the same rig under `sub-rig`, and then starts.

## Output & data format

Each run writes one folder, named and numbered as BIDS does:

```
data/sub-01/ses-001/beh/sub-01_ses-001_task-pong_run-001/
data/sub-01/ses-001/beh/sub-01_ses-001_task-pong_run-002/        the same task again
data/sub-01/ses-001/beh/sub-01_ses-001_task-pong_run-002_02/     ... re-acquired
data/sub-01/ses-001/beh/sub-01_ses-001_task-crafter_run-001/
```

The task is the config's file name (letters and digits); `--subject` must be `sub-<letters/digits>`. `--ses` and `--run` are **required**: the numbers come from the session design (the script's `SES=` line and each run's place in it), never from what is on disk, so they survive a session that was interrupted, resumed or re-acquired. The rig file's `"data_root"` moves the tree (default `data`).

Data is never overwritten. A run whose folder is already there writes to `..._02`, then `_03`; the attempt that stopped stays where it is. That suffix names the folder only — the run's label, which the manifest records with the `attempt` number and which keys the seeds, stays canonical, so every attempt at a run plays the same episodes.

The names follow BIDS apart from that suffix, the contents not yet (no `_beh.tsv` / `_events.tsv`). Each folder holds:

- **`manifest.json`** — subject, curriculum, trigger epoch, per-phase onsets/offsets (+ survey responses; onsets are flip times), the `display` actually opened (size, `vsync`, `refresh_rate`, driver), the `triggers` settings + lifecycle triggers sent (+ what the config left `defaulted`), the `audio` output (device, measured device delay, chosen delay), `dummy_trigger`, the `run` it is (label and `attempt`), the `seeds` (each game phase derived or pinned), the `versions` of pygame and SDL, the run's `duration` and where it was `resumed_from` (see "Resuming a curriculum").
- **`block-NN_<backend>_<game>/`** — one folder per game block, written as the block plays (so a crash loses at most the last second), holding:
  - **`events.jsonl`** — one JSON object per line, in order; `type` says which:

    | `type` | fields |
    |--------|--------|
    | `block_start` | first line: `format`, `version`, `subject`, `block_index`, `backend`, `game`, `phase` (the config), `base_seed` |
    | `episode_start` | `episode_id`, `seed`, `initial_state` (the savestate an episode carried on from an earlier run starts from, as a frame's `state`; `null` otherwise) |
    | `frame` | `frame` (index in the block), `episode_id`, `ep_frame` (index in the episode), `action`, `reward`, `terminated`, `truncated`, `run_time` (seconds since the trigger, after the step), `flip_time` (of the **flip that showed the frame**: its onset; vsync-locked when the display reports `vsync: true`), `wall_time` (Unix), `variables` (the backend's: `ram` for ale/retro, `info_*` for retro, `obs` for gym, …), and when present `env_action` (an adapter's own reading of the keys, e.g. Rush Hour), `trigger` (the code sent on that flip), `audio_chunk` (the chunk this frame's sound was queued as, -1 none; in blocks that play sound), `state` (a savestate, zlib'd then base64, every `state_stride` frames) |
    | `input_event` | `run_time`, `name`, `down`: every key press/release during the block, stamped on arrival (~1 ms), independent of the frame grid; the pause menu's included |
    | `pacing_reset` | `flip_time`, `late`: a flip that ended a stall of more than a frame, and how many seconds late it was; the frame schedule restarted there instead of catching up with a burst of short frames (none in a clean block; why frames fall behind is open, issue #43) |
    | `episode_end` | `episode_id`, `outcome` (`won`, `lost`, `terminated`, `truncated`, `playing` for the block's clock; `quit`, `reset`, `forfeit` for the subject's) , `terminated`, `truncated`, `score`, `n_pacing_resets` |
    | `block_end` | last line: `n_episodes`, `n_frames`, `outcomes`, `n_pacing_resets`, `total_reward`, `block_time_elapsed` and `num_completed_episodes` (earlier runs' included, for a block carried on), `block_did_not_finish` (stopped by the run's clock or ESC before its own end) (the manifest entry's summary), `audio` (when the block played sound: `onsets`, `[chunk, run_time]` when each chunk reached the DAC, so a frame's sound onset is the entry for its `audio_chunk` and `onset - flip_time` the audio delay achieved; `delay_ms`, `resyncs`, `trimmed_samples`), `extra` (the adapter's block extras, e.g. an ALE palette, and `state`, the savestate the block ended in) |

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

### Savestates by replay

Most engines keep their state where Python cannot reach it (a NetHack library loaded per env, a Doom process, Rush Hour's Go process, a browser page, a C++ simulator with no serializer) and offer no snapshot, but every one of them is deterministic from `reset(seed=)` and the actions. For those, `replay_state_gym.ReplayState` (`gym/replay_state/`) wraps the env: it records the episode's seed and every action, `get_state()` returns them as bytes, and `set_state()` is a reset and a replay. The adapter over it is the same two lines as over an engine with a real snapshot, and the log's `state` field means the same thing: `restore(blob)` brings a fresh adapter to that frame. What differs is the cost of a restore (tens of µs a move for Rush Hour, a few ms for Baba Is Auto, a Doom tic for ViZDoom/COOM, tens of ms a step for a browser game) and the blob's size, which grows with the episode -- so these adapters write no savestate in the `frame` lines and one at the end of the block instead, in the `block_end` record's `extra.state` (zlib'd and base64'd like a frame's), the state a resume starts from. A crash mid-block leaves none, but the block still replays from its seeds and actions. SuperTuxKart is the one engine whose reset is not a fresh start (a restarted race drifts from a fresh one by millimetres growing to centimetres), so its adapter relaunches the game on every reset and before every restore, which makes both exact per episode.

Whether an engine really is deterministic is measured, not assumed:

```sh
uv run python -m tests.verify_replay --curriculum configs/curricula/minihack/levels/room5x5.json --max-frames 200 --n-episodes 2
PYTHONHASHSEED=7 uv run python -m tests.verify_replay --check data/sub-01/ses-001/beh/<run>   # a run that exists
```

The first form records the curriculum's game blocks with the random policy (`agents/agent_play.py`, no window) and then, in a second process under another `PYTHONHASHSEED`, rebuilds every frame from the log and compares pixels, logged variables, reward, termination and each frame's sound chunk bit for bit, then restores every savestate anchor and plays on from it to the next. The test suite runs the same check on every backend it selects (`tests/README.md`). Measured 2026-10-17 on macOS, two episodes each, all bit-exact: minihack (150 frames, 150 anchors), nethack (120, 120), baba_auto (120, 120), rushhour (160, 160), vizdoom (300, 10), coom (300, 10), aigamestore game1 levels 1-9 (120 and 12 each), crafter (232, 10). stk could not be run there (Linux x86_64 only) and is unverified. The check is what found nethack's `capture` logging each frame's variables one frame late (a view of NLE's reused buffer, pickled after the loop had moved on) and ViZDoom's first episode after `init()` sounding different from one after `new_episode()`.

## Playing a block with a model

`agents/agent_play.py` runs the same curriculum with a policy where `fmri_play.py` puts a person. It shares everything that defines the task (the config, the adapter, the episode seeds, the Logger and its log schema) and none of the session loop, which exists for a scanner: trigger wait, fps pacing, a window, a key queue.

```sh
python agents/agent_play.py --curriculum configs/curricula/crafter/levels/L4.json \
    --policy random --outdir data/model-random --n-episodes 2 --max-frames 60
export ANTHROPIC_API_KEY=...      # `--policy vlm` refuses to start without it
python agents/agent_play.py --curriculum configs/curricula/crafter/levels/L4.json \
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
- [x] **LSL / serial / parallel-port / HID triggers** and a send-mode start signal for MEG/EEG (`"triggers"` section) -- done.
- [x] **Photodiode calibration task** (`python -m fmri_gym.photodiode`) to measure the flip-to-photon offset of a rig -- done; an always-on sync square in the corner of every frame remains an option if a lab wants per-frame verification.
- [ ] **retro `.bk2` movie logging** as an alternative to per-frame states (frame-exact, tiny).
- [ ] Per-subject deterministic curriculum generation; multi-run structure with one trigger per fMRI run.
- [ ] More adapters: ViZDoom/COOM, MiniHack, crafter, MuJoCo (`qpos/qvel` as state vars) — each a small `EnvAdapter`.
- [ ] Button-box / MRI-safe response device key remapping.
- [ ] BIDS-style output layout + `events.tsv` per run.
- [ ] A replay/QC utility to render any block to video from its states.
- [ ] Crash-safe incremental logging (stream frames to disk) for long runs.
