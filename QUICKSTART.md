# Quick start

Get fmri-gym running and play every currently supported DBP game with a
one-liner. For design notes, adapters, and logging details see [README.md](README.md).

## 1. Install

One system library, then everything else, from the repo root:

```bash
sudo apt install libportaudio2                  # PortAudio; every backend needs it
curl -LsSf https://astral.sh/uv/install.sh | sh # if you don't have uv
uv sync --extra dbp                             # every game below, into .venv/
uv run fmri-play --help                         # check: prints usage
```

That installs every game below into `.venv/`; `uv run fmri-play ...` then
plays one. Without [uv](https://docs.astral.sh/uv/): `pip install -e ".[dbp]"`
in a venv of your own, and `python fmri_play.py` in place of `fmri-play`.

Rush Hour needs no extra step: `rushhour-gym` comes from PyPI, and on first use it
downloads the matching `rushhour-env` engine from the Rush-Hour
GitHub release into `~/.cache/rushhour-gym/` (checksum-verified; Linux x86-64,
macOS arm64, Windows x86-64). On an offline scanner PC, run any Rush Hour
config once while online, or copy that cache directory over; `RUSHHOUR_ENV_BIN`
can also point at a binary you placed yourself (from a release archive, or
`go build -o rushhour-env ./cmd/rushhour-env` in a Rush-Hour checkout).

Developing Rush-Hour and fmri-gym together? Install the package editable from
the checkout and point at its engine, which the adapter then uses:

```bash
RH=/path/to/Rush-Hour
uv pip install -e "$RH/python"    # or pip install -e, in the same env
(cd "$RH" && go build -o rushhour-env ./cmd/rushhour-env)
export RUSHHOUR_ENV_BIN="$RH/rushhour-env"
```

(The engine's default Go build is headless — no SDL, no C toolchain — so
`go build` needs nothing but the Go compiler.)

AI GameStore uses Playwright + system Chrome by default. If you don't have
Chrome, install the bundled Chromium instead:

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
| `--monitor 1`      | Which screen to open on, when there are several (0 = the first) |
| **SPACE**          | Advance past the experimenter screen                            |
| `=`                | Scanner trigger (anchors the session clock)                     |
| **ESC**            | Quit early; data is still saved                                 |

The config editor is a command of its own, `fmri-edit` (needs the `gui` extra:
`uv sync --extra dbp --extra gui`). It opens on a run, on a session script
(`--session ses1.sh`, one line per run) or on a new run, and its **Play**
starts what it shows. To play a session without it: `sh ses1.sh`.

Each config is a short curriculum: message → fixation → game (~300 s, auto-restarts
on game-over) → fixation. Output lands in `data/sub-01/ses-<ses>/beh/sub-01_ses-<ses>_task-<config>_run-<run>/`, from the `--ses` and `--run` you give it; a run that already has data is re-acquired beside it, never overwritten (see README "Output & data format").

Runtime: experimenter screen (**SPACE**) → "Waiting for scanner..." → trigger `=` → curriculum.

stable-retro games play their native audio, and ViZDoom does when its config sets
`env_kwargs.audio_buffer_enabled`. Add `--no-audio` to mute every game, or set
`"audio": false` on a game phase to mute that block; logged audio is unchanged.
Sound plays through the system's default output, a constant delay after its
frame's flip (printed at start-up).

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

Controls: arrows move/turn, Z/X strafe, SPACE shoots — but each scenario only
has the buttons its `.cfg` declares, so the arrows *strafe* in Basic and Take
Cover (no turning), there is no SPACE in Health Gathering / My Way Home / Take
Cover (no weapon), and Deathmatch adds N/M to switch weapons and S to run. Each
config's controls message lists exactly what that scenario accepts.

### COOM

COOM's own Doom scenarios (not the stock ViZDoom ones above). Set-up is a
checkout — the COOM package itself is never installed (its `gymnasium` pin
conflicts), only its scenario assets are read:

```bash
git clone https://github.com/TTomilin/COOM.git ../COOM
export COOM_REPO=../COOM          # or pass --coom-repo ../COOM per run
```

```bash
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/coom__pitfall.json          # cross a corridor of randomized pits
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/coom__chainsaw.json         # hunt maze enemies at melee range
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/coom__run_and_gun.json      # find and shoot maze enemies
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/coom__health_gathering.json # draining floor; collect health kits
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/coom__hide_and_seek.json    # evade enemies, grab kits when low
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/coom__arms_dealer.json      # collect weapons, deliver to platforms
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/coom__floor_is_lava.json    # stay on the briefly-appearing platforms
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/coom__parkour.json          # jump the gaps and ledges
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/coom__raise_the_roof.json   # press wall switches before the ceiling crushes you
```

Controls: UP moves forward, LEFT/RIGHT turn, and the scenario's one extra
button is SPACE (pitfall, chainsaw, run_and_gun, parkour), LSHIFT
(health_gathering, hide_and_seek, arms_dealer, floor_is_lava) or E
(raise_the_roof). No native audio on this backend yet.

### Crafter

```bash
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/crafter__crafter.json --ses 1 --run 1
```



### Rush Hour

```bash
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/rushhour__easy.json --ses 1 --run 1      # 5 min of random easy puzzles
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/rushhour__complete.json --ses 1 --run 1    # Rush-Hour's own session: 12 puzzles, easiest first, self-paced
```

`rushhour__complete.json` is the Rush-Hour program's own session (ready screen
before each puzzle, blank interval, solved hold); see its `_session_note`.

The engine binary is fetched on first run (see §1); nothing to build.

### Baba is AI

```bash
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/baba__make_win.json --ses 1 --run 1
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

Controls: arrows + SPACE / Z / ENTER (game-dependent). Needs Playwright + Chrome (§1).

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
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/supertuxkart__race.json --ses 1 --run 1
```

Needs a real GL display (does **not** work under `SDL_VIDEODRIVER=dummy`).
Controls: arrows steer/accelerate/brake, SPACE fire, Z drift, X nitro.

## Tips

- Useful flags: `--size 1280x1024`, `--fullscreen`, `--no-vsync`.
- Before a MEG/EEG session, check that flips lock to the refresh on the
presentation machine: `python -m fmri_gym.display --fullscreen`.
- Once per rig, measure the flip-to-photon offset with a photodiode on the
screen: `python -m fmri_gym.photodiode --fullscreen` (see README "Timing").
- Archived / unsupported configs live under `configs/dbp_games/archive/` and
`configs/dbp_games/unsupported/` — see the README for the wider game list.
- Per-config `_note` / `_game` fields document setup quirks for that title.

