# Baba Is Auto curricula

The `baba_auto` backend's curricula: twelve Baba Is You levels on the baba-is-auto engine (language / rule-manipulation puzzles: push word blocks to rewrite the rules), one file each. Every game phase includes `common/baba_auto.json` (instructions, keys, menu, blocking mode) and states its level and the instructions' title (README, "Shared phases").

## Setup

Clone utilForever/baba-is-auto into `external/baba_auto` at the commit the README pins (README, "External checkouts"), then `uv sync --extra baba_auto`: baba-auto-gym (`gym/baba_auto/`) compiles the C++ engine from that checkout, which needs a C++17 compiler and python3-dev. A phase's `game` is a map name under the checkout's `Resources/Maps` (baba_is_you, out_of_reach, volcano, off_limits, grass_yard, pillar_yard, brick_wall, icy_waters, novice_locksmith, lock, affection, turns) or a map file path.

## Keys

`baba_auto_gym` is Discrete(5): 0=wait, 1=up, 2=down, 3=left, 4=right (`gym/baba_auto/baba_auto_gym/env.py`); the shared phase maps SPACE to wait and the arrows to the moves. The instructions' Controls text is kept in sync with `keys` by hand.

## Blocking and the menu

Blocking, as Baba Is You is: the block is the level played until won (mode `episode`, `n_episodes` 1, `advance_on` `["won"]`); a loss (YOU destroyed, or no YOU left) restarts the same level, as the game itself does, and `duration` 300 s is the cap. A puzzle can be pushed into a dead end, so holding X for 3 s pauses the game and offers Restart / Give up / Resume (UP/DOWN to choose, SPACE to select; `fmri_gym/menu.py`); Give up is the way out of a level the subject cannot solve.

## Curricula

| file | plays |
|---|---|
| `affection.json` | Affection |
| `baba_is_you.json` | Baba Is You, the game's first level |
| `brick_wall.json` | Brick Wall |
| `grass_yard.json` | Grass Yard |
| `icy_waters.json` | Icy Waters |
| `lock.json` | Lock |
| `novice_locksmith.json` | Novice Locksmith |
| `off_limits.json` | Off Limits |
| `out_of_reach.json` | Out of Reach |
| `pillar_yard.json` | Pillar Yard |
| `turns.json` | Turns |
| `volcano.json` | Volcano |
| `all.json` (in `baba_auto/`) | the twelve in a row |
