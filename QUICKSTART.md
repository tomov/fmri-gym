# Quick start

Get fmri-gym running and play every currently supported DBP game with a one-liner. For design notes, adapters, and logging details see [README.md](README.md).

## 1. Install

One system library, then everything else, from the repo root:

```bash
sudo apt install libportaudio2                  # PortAudio; every backend needs it
curl -LsSf https://astral.sh/uv/install.sh | sh # if you don't have uv
uv sync --extra dbp --extra gui                 # every game below + the editor, into .venv/
uv run fmri-play --help                         # check: prints usage
```

That installs every game below, plus the `fmri-edit` editor, into `.venv/`; `uv run fmri-play ...` then plays one. A game is an extra and nothing is installed by default: a bare `uv sync` gets the core alone, `--extra dbp` the DBP set, `--extra baba` (or any one backend) the core plus that game, `--extra all` every backend and the editor. Each sync makes `.venv/` match exactly the extras it names, so name the whole list every time -- one left out is one uv uninstalls. `baba_auto` (hence `all`) compiles a C++ engine at install: `sudo apt install build-essential`, or `xcode-select --install` on macOS. Without [uv](https://docs.astral.sh/uv/): `pip install -e ".[dbp,gui]"` in a venv of your own, and `python fmri_play.py` in place of `fmri-play`.

COOM, Baba Is Auto and VGDL read their game files from a checkout of the game's own repo, a git submodule at `external/<backend>` pinned to a commit (README "External checkouts"). `git clone --recursive` fetches them; in a clone made without it, run `git submodule update --init` once. On an offline scanner PC, fetch them while online or copy `external/` over.

Rush Hour needs no extra step: `rushhour-gym` comes from PyPI, and on first use it downloads the matching `rushhour-env` engine from the Rush-Hour GitHub release into `~/.cache/rushhour-gym/` (checksum-verified; Linux x86-64, macOS arm64, Windows x86-64). On an offline scanner PC, run any Rush Hour config once while online, or copy that cache directory over; `RUSHHOUR_ENV_BIN` can also point at a binary you placed yourself (from a release archive, or `go build -o rushhour-env ./cmd/rushhour-env` in a Rush-Hour checkout).

Developing Rush-Hour and fmri-gym together? Install the package editable from the checkout and point at its engine, which the adapter then uses:

```bash
RH=/path/to/Rush-Hour
uv pip install -e "$RH/python"    # or pip install -e, in the same env
(cd "$RH" && go build -o rushhour-env ./cmd/rushhour-env)
export RUSHHOUR_ENV_BIN="$RH/rushhour-env"
```

(The engine's default Go build is headless — no SDL, no C toolchain — so `go build` needs nothing but the Go compiler.)

AI GameStore uses Playwright + system Chrome by default. If you don't have Chrome, install the bundled Chromium instead:

```bash
playwright install chromium   # then set "browser_channel": null in the phase if needed
```



## 2. How a session works

```bash
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/<game>.json --ses 1 --run 1
```


| Flag / key         | What it does                                                    |
| ------------------ | --------------------------------------------------------------- |
| `--subject sub-01` | Subject id used in the output folder name                       |
| `--rig scanner3T`  | Which rig plays it, when this machine has several (see below)   |
| **SPACE**          | Advance past the experimenter screen                            |
| `=`                | Scanner trigger (anchors the session clock)                     |
| **ESC**            | Quit early; data is still saved                                 |

**A rig file first.** How a run is played -- window size, fullscreen, which monitor, the controller, audio, where the data goes -- is the rig's, not the command's: one file per rig, `~/.config/fmri-gym/rigs/<name>.json`. With no rig file, `fmri-play` does not start; make one with the rig's long rig check (one per rig), which opens a form for it: `uv run fmri-play --curriculum configs/rig-check-long.json --subject sub-rig --ses 1 --run 1 --rig <name>` (or only the form: `uv run python -m fmri_gym.checks rig --rig <name>`). A machine with one rig needs no `--rig` after that (README "Rigs").


The config editor is a command of its own, `fmri-edit` (needs the `gui` extra: `uv sync --extra dbp --extra gui`). It opens on a run, on a session script (`--session ses1.sh`, one line per run) or on a new run, and its **Play** starts what it shows. To play a session without it: `sh ses1.sh`.

Each config is a short curriculum: message → fixation → game (~300 s, auto-restarts on game-over) → fixation. Output lands in `data/sub-01/ses-<ses>/beh/sub-01_ses-<ses>_task-<config>_run-<run>/`, from the `--ses` and `--run` you give it; a run that already has data is re-acquired beside it, never overwritten (see README "Output & data format").

Runtime: experimenter screen (**SPACE**) → "Waiting for scanner..." → trigger `=` → curriculum.

stable-retro games play their native audio, and ViZDoom and COOM do when their config sets `env_kwargs.audio_buffer_enabled`. Set `"audio": false` in the rig file to mute every game, or set `"audio": false` on a game phase to mute that block; logged audio is unchanged. Sound plays through the system's default output, a constant delay after its frame's flip (printed at start-up).

## 3. Run every game

All commands assume you're in the repo root with `fmri-gym` activated.

### ViZDoom

All ten stock scenarios, one line each:

```bash
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/vizdoom__basic.json --ses 1 --run 1                      # strafe and shoot one monster
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/vizdoom__deadly_corridor.json --ses 1 --run 1            # fight down a corridor to the vest
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/vizdoom__deathmatch.json --ses 1 --run 1                 # arena, full arsenal, scored by kills
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/vizdoom__defend_center.json --ses 1 --run 1              # surrounded; turn and shoot
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/vizdoom__defend_line.json --ses 1 --run 1                # they advance down a hall and respawn
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/vizdoom__health_gathering_supreme.json --ses 1 --run 1   # acid-floor maze; find medikits
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/vizdoom__my_way_home.json --ses 1 --run 1                # navigate a 9-room maze to the vest
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/vizdoom__predict_position.json --ses 1 --run 1           # lead a moving target with a rocket
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/vizdoom__take_cover.json --ses 1 --run 1                 # no weapon; dodge fireballs
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/vizdoom__take_cover_defend_line.json --ses 1 --run 1     # those last two back to back, 150 s each
```

Controls: the arrows move/turn and the buttons shoot or strafe — but each scenario only has the buttons its `.cfg` declares, so the arrows *strafe* in Basic and Take Cover (no turning), and there is no shooting in Health Gathering / My Way Home / Take Cover (no weapon). Each config's controls message lists exactly what that scenario accepts.

### COOM

COOM's own Doom scenarios (not the stock ViZDoom ones above). The env ships here as `coom-gym` (`gym/coom/`, part of the `dbp` extra) and reads the scenario files from the COOM checkout at its pinned commit (the `external/coom` submodule; README "External checkouts").

```bash
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/coom__pitfall.json --ses 1 --run 1          # cross a corridor of randomized pits
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/coom__chainsaw.json --ses 1 --run 1         # hunt maze enemies at melee range
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/coom__run_and_gun.json --ses 1 --run 1      # find and shoot maze enemies
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/coom__health_gathering.json --ses 1 --run 1 # draining floor; collect health kits
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/coom__hide_and_seek.json --ses 1 --run 1    # evade enemies, grab kits when low
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/coom__arms_dealer.json --ses 1 --run 1      # collect weapons, deliver to platforms
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/coom__floor_is_lava.json --ses 1 --run 1    # stay on the briefly-appearing platforms
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/coom__parkour.json --ses 1 --run 1          # jump the gaps and ledges
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/coom__raise_the_roof.json --ses 1 --run 1   # press wall switches before the ceiling crushes you
```

Controls: UP moves forward, LEFT/RIGHT turn, and the scenario's one extra button is A (on a keyboard, the A key). The nine configs enable native audio; OpenAL is required. MIDI background music needs a working MIDI renderer. See README for audio setup and the current episode-end playback limitation.

### Crafter

```bash
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/crafter__crafter_L4.json --ses 1 --run 1
```



### Rush Hour

```bash
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/rushhour__easy.json --ses 1 --run 1      # 5 min of random easy puzzles
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/rushhour__complete.json --ses 1 --run 1    # Rush-Hour's own session: 12 puzzles, easiest first, self-paced
```

`rushhour__complete.json` is the Rush-Hour program's own session (ready screen before each puzzle, blank interval, solved hold); see its `_session_note`.

The engine binary is fetched on first run (see §1); nothing to build.

### Baba Is AI (generated puzzles in the style of Baba Is You)

```bash
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/baba__make_win.json --ses 1 --run 1
```



### Baba Is Auto (Baba Is You's original levels on the baba-is-auto engine)

The engine is C++, compiled when the extra installs from the checkout at its pinned commit (README "External checkouts"; a C++17 compiler and `python3-dev` are needed; about 30 s):

```bash
uv sync --extra dbp --extra baba_auto
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/baba_auto__baba_is_you.json --ses 1 --run 1
```



### VGDL

Ten small arcade games from a gymnasium-ported fork, at its pinned commit (README "External checkouts"). The `vgdl` extra is separate from `dbp`:

```bash
uv sync --extra dbp --extra vgdl
uv run fmri-play --subject sub-01 --curriculum configs/demo_vgdl_all.json --ses 1 --run 1   # all ten, 15 s each
```



### AI GameStore (p5.js browser games)

```bash
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/aigamestore__game1.json --ses 1 --run 1
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/aigamestore__game2.json --ses 1 --run 1
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/aigamestore__game3.json --ses 1 --run 1
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/aigamestore__game4.json --ses 1 --run 1
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/aigamestore__game5.json --ses 1 --run 1
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/aigamestore__game6.json --ses 1 --run 1
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/aigamestore__game7.json --ses 1 --run 1
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/aigamestore__game8.json --ses 1 --run 1
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/aigamestore__game9.json --ses 1 --run 1
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/aigamestore__game10.json --ses 1 --run 1
```

Controls: arrows + A / B and other rig keys (game-dependent; each config's controls message lists them). Needs Playwright + Chrome (§1).

### MiniHack

```bash
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/minihack.json --ses 1 --run 1
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/minihack__room5x5.json --ses 1 --run 1
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/minihack__room15x15.json --ses 1 --run 1
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/minihack__mazewalk9x9.json --ses 1 --run 1
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/minihack__river.json --ses 1 --run 1
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/minihack__corridor.json --ses 1 --run 1
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/minihack__eat.json --ses 1 --run 1
```

Controls: arrow keys (N/E/S/W). Needs `setuptools<81` (already a core dependency).

### SuperTuxKart

```bash
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/stk__hacienda.json --ses 1 --run 1
```

Needs a real GL display (does **not** work under `SDL_VIDEODRIVER=dummy`); the game itself comes with `supertuxkart-gym`, which fetches it on first use (no checkout, no build). See the README section "Running SuperTuxKart". Controls: arrows steer/accelerate/brake, X fire, B skid, A nitro, Y rescue.

## Tips

- The window, the controller, the audio and the data folder are the rig's, in its rig file (README "Rigs"): `"screen": {"size": "1280x1024", "fullscreen": true, "monitor": 1, "vsync": true}`, `"pad"`, `"audio"`, `"data_root"`.
- Before a MEG/EEG session, check that flips lock to the refresh on the presentation machine: `python -m fmri_gym.display --fullscreen`.
- Once per rig, measure the flip-to-photon offset with a photodiode on the screen: `python -m fmri_gym.photodiode --fullscreen` (see README "Timing").
- Archived / unsupported configs live under `configs/dbp_games/archive/` and `configs/dbp_games/unsupported/` — see the README for the wider game list.
- Per-config `_note` / `_game` fields document setup quirks for that title.

