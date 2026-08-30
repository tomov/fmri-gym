# Quick start

Get fmri-gym running and play every currently supported DBP game with a
one-liner. For design notes, adapters, and logging details see [README.md](README.md).

## 1. Install

```bash
conda create -n fmri-gym python=3.11
conda activate fmri-gym
pip install -r requirements.txt
```

> If your default pip index is a private registry, add
> `--index-url https://pypi.org/simple`.

`requirements.txt` already pulls in the common backends (crafter, minihack,
vizdoom, playwright, pystk2-gymnasium, …). Two games need an extra step:

```bash
# Baba is AI
pip install "git+https://github.com/nacloos/baba-is-ai.git"

# Rush Hour — Python package + Go engine binary
pip install "git+https://github.com/chrplr/Rush-Hour.git#subdirectory=python"
git clone https://github.com/chrplr/Rush-Hour vendor/rush-hour-src
cd vendor/rush-hour-src && go build -o rushhour-env ./cmd/rushhour-env && cd ../..
```

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

### Crafter

```bash
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/crafter__crafter.json
```

### Rush Hour

```bash
python fmri_play.py --subject sub-01 --curriculum configs/dbp_games/rushhour__easy.json
```

Needs the Go binary from §1 (`vendor/rush-hour-src/rushhour-env`, or set `RUSHHOUR_ENV_BIN`).

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

## Tips

- Useful flags: `--size 1280x1024`, `--fullscreen`.
- Archived / unsupported configs live under `configs/dbp_games/archive/` and
  `configs/dbp_games/unsupported/` — see the README for the wider game list.
- Per-config `_note` / `_game` fields document setup quirks for that title.
