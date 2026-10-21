# Quick start

Get fmri-gym running and play every currently supported DBP game on a laptop -- in a window, with its sound, on the keyboard -- with a one-liner each. For design notes, adapters, and logging details see [README.md](README.md).

## 1. Install

One system library, then everything else, from the repo root:

```bash
sudo apt install libportaudio2                  # Linux only: PortAudio (macOS/Windows wheels bundle it)
brew install openal-soft                        # Apple Silicon Mac only: OpenAL Soft, for ViZDoom/COOM sound
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



## 2. Play a game on a laptop

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/minihack/levels/room5x5.json
```

`--rig configs/rigs/example/empty.json` is an empty rig file (`configs/rigs/example/empty.json`, just `{}`), so every setting is the default: a 1024x768 window with the game's sound on the laptop's speakers, the keyboard (arrows and the letter keys each game's instructions name), and the data saved under `data/`. Press **SPACE** on the first screen, then **=**: with no triggers in the rig file, a run waits for the scanner's `=` like an fMRI rig (the screen says NOT SET). **ESC** quits. Every command below plays this way: swap the curriculum for any level in `configs/curricula/<backend>/levels/<level>.json`, or any sequence of levels in `configs/curricula/<backend>/<name>.json`.


| Flag / key         | What it does                                                    |
| ------------------ | --------------------------------------------------------------- |
| `--subject sub-01` | Subject id used in the output folder name                       |
| `--rig configs/rigs/example/empty.json` | The rig file it is played on: here, all defaults |
| **SPACE**          | Advance past the experimenter screen                            |
| `=`                | Scanner trigger (anchors the session clock); on a laptop, press it yourself |
| **ESC**            | Quit early; data is still saved                                 |

**A rig file first.** How a run is played -- window size, fullscreen, which monitor, the controller, audio, where the data goes -- is the rig's, not the command's: one file per rig, in the repo, `configs/rigs/<site>/<rig>.json`. The repo ships example rigs in `configs/rigs/example/` (`configs/rigs/README.md` lists them): `empty.json`, an empty file and so every default, which the commands below name; `laptop.json`, the keyboard alone with no scanner (a run starts on SPACE, no `=`); `xbox.json` and `curdes-v1.json`, a controller bound to the curricula's keys; and `fmri.json` and `meg.json`, a scanner's triggers. For a real rig, make its file with the rig's long rig check (one per rig), which opens a form for it: `uv run fmri-play --subject sub-rig --ses 1 --run 1 --rig configs/rigs/<site>/<rig>.json --curriculum configs/curricula/checks/rig-check-long.json` (or only the form: `uv run python -m fmri_gym.checks rig --rig configs/rigs/<site>/<rig>.json`), and commit the file; `--rig` names the rig file every run plays on (README "Rigs").


The config editor is a command of its own, `fmri-edit` (needs the `gui` extra: `uv sync --extra dbp --extra gui`). It opens on a run, on a session script (`--session ses1.sh`, one line per run) or on a new run, and its **Play** starts what it shows. To play a session without it: `sh ses1.sh`.

Each config is a short curriculum: message → fixation → game (~300 s, auto-restarts on game-over) → fixation. Output lands in `data/sub-01/ses-<ses>/beh/sub-01_ses-<ses>_task-<config>_run-<run>/`, from the `--ses` and `--run` you give it; a run that already has data is re-acquired beside it, never overwritten (see README "Output & data format").

Runtime: experimenter screen (**SPACE**) → "Waiting for scanner..." until the trigger (`=`, which an fMRI trigger box types; on a laptop, press it) → curriculum. A rig whose triggers start on SPACE (`example/laptop`) skips the wait.

Sound plays through the laptop's default output, a constant delay after its frame's flip (printed at start-up). The games whose engine makes sound play it: stable-retro, ViZDoom and COOM (their curricula turn on `env_kwargs.audio_buffer_enabled`); the rest are silent games. A game phase's `"has_audio": false` mutes that block, and the rig file's `"has_audio": false` every block on that rig; logged audio is unchanged.

## 3. Run every game

All commands assume you're in the repo root with `fmri-gym` activated.

### ViZDoom

All ten stock scenarios, one line each:

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/vizdoom/levels/basic.json                      # strafe and shoot one monster
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/vizdoom/levels/deadly_corridor.json            # fight down a corridor to the vest
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/vizdoom/levels/deathmatch.json                 # arena, full arsenal, scored by kills
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/vizdoom/levels/defend_center.json              # surrounded; turn and shoot
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/vizdoom/levels/defend_line.json                # they advance down a hall and respawn
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/vizdoom/levels/health_gathering_supreme.json   # acid-floor maze; find medikits
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/vizdoom/levels/my_way_home.json                # navigate a 9-room maze to the vest
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/vizdoom/levels/predict_position.json           # lead a moving target with a rocket
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/vizdoom/levels/take_cover.json                 # no weapon; dodge fireballs
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/vizdoom/take_cover_defend_line.json     # those last two back to back, 150 s each
```

Controls: arrows move/turn, Z/X strafe, SPACE shoots — but each scenario only has the buttons its `.cfg` declares, so the arrows *strafe* in Basic and Take Cover (no turning), there is no SPACE in Health Gathering / My Way Home / Take Cover (no weapon), and Deathmatch adds N/M to switch weapons and S to run. Each config's controls message lists exactly what that scenario accepts.

### COOM

COOM's own Doom scenarios (not the stock ViZDoom ones above). The env ships here as `coom-gym` (`gym/coom/`, part of the `dbp` extra) and reads the scenario files from the COOM checkout at its pinned commit (the `external/coom` submodule; README "External checkouts").

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/coom/levels/pitfall.json          # cross a corridor of randomized pits
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/coom/levels/chainsaw.json         # hunt maze enemies at melee range
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/coom/levels/run_and_gun.json      # find and shoot maze enemies
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/coom/levels/health_gathering.json # draining floor; collect health kits
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/coom/levels/hide_and_seek.json    # evade enemies, grab kits when low
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/coom/levels/arms_dealer.json      # collect weapons, deliver to platforms
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/coom/levels/floor_is_lava.json    # stay on the briefly-appearing platforms
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/coom/levels/parkour.json          # jump the gaps and ledges
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/coom/levels/raise_the_roof.json   # press wall switches before the ceiling crushes you
```

Controls: UP moves forward, LEFT/RIGHT turn, and the scenario's one extra button is SPACE (pitfall, chainsaw, run_and_gun, parkour), LSHIFT (health_gathering, hide_and_seek, arms_dealer, floor_is_lava) or E (raise_the_roof). The nine configs enable native audio; OpenAL is required. MIDI background music needs a working MIDI renderer. See README for audio setup and the current episode-end playback limitation.

### Crafter

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/crafter/levels/L4.json
```



### Rush Hour

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/rushhour/levels/easy.json      # 5 min of random easy puzzles
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/rushhour/complete.json    # Rush-Hour's own session: 12 puzzles, easiest first, self-paced
```

`configs/curricula/rushhour/complete.json` is the Rush-Hour program's own session (ready screen before each puzzle, blank interval, solved hold); see `configs/curricula/rushhour/common/README.md`.

The engine binary is fetched on first run (see §1); nothing to build.

### Baba Is AI (generated puzzles in the style of Baba Is You)

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/baba/levels/make_win.json
```



### Baba Is Auto (Baba Is You's original levels on the baba-is-auto engine)

The engine is C++, compiled when the extra installs from the checkout at its pinned commit (README "External checkouts"; a C++17 compiler and `python3-dev` are needed; about 30 s):

```bash
uv sync --extra dbp --extra baba_auto
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/baba_auto/levels/baba_is_you.json
```



### VGDL

Ten small arcade games from a gymnasium-ported fork, at its pinned commit (README "External checkouts"). The `vgdl` extra is separate from `dbp`:

```bash
uv sync --extra dbp --extra vgdl
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/vgdl/all.json   # all ten, 15 s each
```



### AI GameStore (p5.js browser games)

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/aigamestore/game1.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/aigamestore/game2.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/aigamestore/game3.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/aigamestore/game4.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/aigamestore/game5.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/aigamestore/game6.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/aigamestore/game7.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/aigamestore/game8.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/aigamestore/game9.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/aigamestore/game10.json
```

Controls: arrows + SPACE / Z / ENTER (game-dependent). Needs Playwright + Chrome (§1).

### MiniHack

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/minihack/levels/room5x5.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/minihack/levels/room15x15.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/minihack/levels/mazewalk9x9.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/minihack/levels/river.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/minihack/levels/corridor.json
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/minihack/levels/eat.json
```

Controls: arrow keys (N/E/S/W). Needs `setuptools<81` (already a core dependency).

### SuperTuxKart

```bash
uv run fmri-play --subject sub-01 --ses 1 --run 1 --rig configs/rigs/example/empty.json --curriculum configs/curricula/stk/levels/hacienda.json
```

Needs a real GL display (does **not** work under `SDL_VIDEODRIVER=dummy`); the game itself comes with `supertuxkart-gym`, which fetches it on first use (no checkout, no build). See the README section "Running SuperTuxKart". Controls: arrows steer/accelerate/brake, SPACE fire, V skid, N nitro, BACKSPACE rescue.

## Tips

- The window, the controller, the audio and the data folder are the rig's, in its rig file (README "Rigs"): `"screen": {"size": "1280x1024", "is_fullscreen": true, "monitor": 1, "has_vsync": true}`, `"has_pad"`, `"has_audio"`, `"data_root"`.
- Before a MEG/EEG session, check that flips lock to the refresh on the presentation machine: `python -m fmri_gym.display --fullscreen`.
- Once per rig, measure the flip-to-photon offset with a photodiode on the screen: `python -m fmri_gym.photodiode --fullscreen` (see README "Timing").
- Every curriculum lives in `configs/curricula/<backend>/` (see `configs/README.md`); the README lists the games that ship none, and why.
- Each backend's `configs/curricula/<backend>/common/README.md` documents setup quirks, keys and levels; a config's own `##` comments say what is particular to it.

