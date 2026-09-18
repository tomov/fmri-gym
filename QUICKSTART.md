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
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/<game>.json
```

| Flag / key | What it does |
|---|---|
| `--subject sub-01` | Subject id used in the output folder name |
| **SPACE** | Advance past the experimenter screen |
| **`=`** | Scanner trigger (anchors the session clock) |
| **ESC** | Quit early; data is still saved |

Each config is a short curriculum: message → fixation → game (~300 s, auto-restarts
on game-over) → fixation. Output lands in `data/<subject>_<timestamp>/`.

Runtime: experimenter screen (**SPACE**) → "Waiting for scanner..." → trigger **`=`** → curriculum.

stable-retro games play their native audio, and ViZDoom does when its config sets
`env_kwargs.audio_buffer_enabled`. Add `--no-audio` to mute every game, or set
`"audio": false` on a game phase to mute that block; logged audio is unchanged.
Sound plays through the system's default output, a constant delay after its
frame's flip (printed at start-up).

## 3. Run every game

All commands assume you're in the repo root with `fmri-gym` activated.

### ViZDoom

```bash
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/vizdoom__defend_center.json
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/vizdoom__deadly_corridor.json
```

Controls: arrows move/turn, Z/X strafe, SPACE shoots.

### Crafter

```bash
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/crafter__crafter.json
```

### Rush Hour

```bash
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/rushhour__easy.json      # 5 min of random easy puzzles
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/rushhour__complete.json    # Rush-Hour's own session: 12 puzzles, easiest first, self-paced
```

`rushhour__complete.json` is the Rush-Hour program's own session (ready screen
before each puzzle, blank interval, solved hold); see its `_session_note`.

The engine binary is fetched on first run (see §1); nothing to build.

### Baba is AI

```bash
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/baba__make_win.json
```

### AI GameStore (p5.js browser games)

```bash
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/aigamestore__game1.json
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/aigamestore__game2.json
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/aigamestore__game3.json
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/aigamestore__game4.json
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/aigamestore__game5.json
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/aigamestore__game6.json
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/aigamestore__game7.json
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/aigamestore__game8.json
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/aigamestore__game9.json
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/aigamestore__game10.json
```

Controls: arrows + SPACE / Z / ENTER (game-dependent). Needs Playwright + Chrome (§1).

### MiniHack

```bash
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/minihack.json
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/minihack__room5x5.json
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/minihack__room15x15.json
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/minihack__mazewalk9x9.json
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/minihack__river.json
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/minihack__corridor.json
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/minihack__eat.json
```

Controls: arrow keys (N/E/S/W). Needs `setuptools<81` (already a core dependency).

### SuperTuxKart

```bash
uv run fmri-play --subject sub-01 --curriculum configs/dbp_games/supertuxkart__race.json
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
