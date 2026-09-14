# Quick start

Get fmri-gym running and play every currently supported DBP game with a
one-liner. For design notes, adapters, and logging details see [README.md](README.md).

Native audio plays automatically where supported (currently ViZDoom and
stable-retro). Add `--no-audio` to any launch command to mute the session, or
set `"audio": false` in one game phase to mute only that block. See
[Audio support](AUDIO.md) for the backend list and why some games are silent.

## 1. Install

```bash
conda create -n fmri-gym python=3.11
conda activate fmri-gym
pip install -r requirements.txt
```

`requirements.txt` already pulls in the common backends (crafter, minihack,
vizdoom, playwright, pystk2-gymnasium, rushhour-gym, …). One game needs an
extra step:

```bash
# Baba is AI
pip install "git+https://github.com/nacloos/baba-is-ai.git"
```

Rush Hour needs none: `rushhour-gym` comes from PyPI, and on first use it
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
pip install -e "$RH/python"
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
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/<game>.json
```

| Flag / key | What it does |
|---|---|
| `--subject sub-01` | Subject id used in the output folder name |
| `--no-audio` | Disable audio for every game block |
| **SPACE** | Advance past the experimenter screen |
| **`=`** | Scanner trigger (anchors the session clock) |
| **ESC** | Quit early; data is still saved |

Each config is a short curriculum: message → fixation → game (~300 s, auto-restarts
on game-over) → fixation. Output lands in `data/<subject>_<timestamp>/`.

Runtime: experimenter screen (**SPACE**) → "Waiting for scanner..." → trigger **`=`** → curriculum.

## 3. Run every game

All commands assume you're in the repo root with `fmri-gym` activated.

### ViZDoom

```bash
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/vizdoom__defend_center.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/vizdoom__deadly_corridor.json
```

Controls: arrows move/turn, Z/X strafe, SPACE shoots.

ViZDoom plays its native audio through the default output device. Both configs
run at 35 FPS to match Doom's native game speed (`frame_skip` defaults to 1).
Slower playback leaves gaps in the audio; with another `frame_skip`, use
`fps = 35 / frame_skip`. To disable output, use `--no-audio` or set
`"audio": false` in the game phase. Existing
`env_kwargs.audio_buffer_enabled=false` settings also remain supported.

### Crafter

```bash
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/crafter__crafter.json
```

### Rush Hour

```bash
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/rushhour__easy.json      # 5 min of random easy puzzles
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/rushhour__complete.json    # Rush-Hour's own session: 12 puzzles, easiest first, self-paced
```

`rushhour__complete.json` is the Rush-Hour program's own session (ready screen
before each puzzle, blank interval, solved hold); see its `_session_note`.

The engine binary is fetched on first run (see §1); nothing to build.

### Baba is AI

```bash
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/baba__make_win.json
```

### AI GameStore (p5.js browser games)

```bash
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/aigamestore__game1.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/aigamestore__game2.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/aigamestore__game3.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/aigamestore__game4.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/aigamestore__game5.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/aigamestore__game6.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/aigamestore__game7.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/aigamestore__game8.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/aigamestore__game9.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/aigamestore__game10.json
```

Controls: arrows + SPACE / Z / ENTER (game-dependent). Needs Playwright + Chrome (§1).

### MiniHack

```bash
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/minihack.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/minihack__room5x5.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/minihack__room15x15.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/minihack__mazewalk9x9.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/minihack__river.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/minihack__corridor.json
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/minihack__eat.json
```

Controls: arrow keys (N/E/S/W). Needs `setuptools<81` (already in `requirements.txt`).

### SuperTuxKart

```bash
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/supertuxkart__race.json
```

Needs a real GL display (does **not** work under `SDL_VIDEODRIVER=dummy`).
Controls: arrows steer/accelerate/brake, SPACE fire, Z drift, X nitro.

### stable-retro audio demo

```bash
python fmri_play.py --subject sub-test --dummy-trigger --curriculum configs/demo_retro_audio.json
```

Requires `stable-retro` and `sounddevice`; the Airstriker ROM is included.
Arrows move, Z fires. Native audio plays automatically; use `--no-audio` or
`"audio": false` to mute it. Match `fps` to
`env.unwrapped.em.get_screen_rate()` (about 59.923 for this Genesis core).
The sample rate comes directly from the emulator; changing `fps` does not
resample audio, so slower or faster playback causes gaps or accumulating delay.

## Tips

- Useful flags: `--size 1280x1024`, `--fullscreen`.
- Archived / unsupported configs live under `configs/dbp_games/archive/` and
  `configs/dbp_games/unsupported/` — see the README for the wider game list.
- Per-config `_note` / `_game` fields document setup quirks for that title.
