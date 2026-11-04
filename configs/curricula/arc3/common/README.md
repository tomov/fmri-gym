# ARC-AGI-3 curricula

The `arc3` backend's curricula: six [ARC-AGI-3](https://arcprize.org) games, turn-based puzzles in several levels, one file each. Every game phase includes `common/arc3.json` (the menu, the blocking mode, the turn-based pace) and states its game, keys and instructions (README, "Shared phases").

## Setup

`uv sync --extra arc3` installs arc3-gym (`gym/arc3/`, Python >= 3.12). Each game's files are fetched once, with the network on, by `uv run python -m arc3_gym <game>` (e.g. `ls20`), which puts them in `external/arc3`; playing needs no network.

## Keys

A game's `controls` index its simple actions in ascending order (`fmri_gym/adapters/arc3.py`): 0=ACTION1 (up), 1=ACTION2 (down), 2=ACTION3 (left), 3=ACTION4 (right), and, in a game that offers it, 4=ACTION5 (the game's interact action). g50t, re86 and wa30 offer ACTION1-5; ls20, tr87 and tu93 offer ACTION1-4. The instructions' Controls text is kept in sync with `controls` by hand.

## Blocking and the menu

The block is the whole game played until won (mode `episode`, `n_episodes` 1, `advance_on` `["won"]`, `duration` 600 s); a GAME_OVER restarts it from level 1. A level can be a dead end (the game can end before it is solved; in ls20, when its energy runs out), so after 15 s of play, holding X for 5 s pauses the game and offers Restart / Give up / Resume (UP/DOWN to choose, SPACE to select; `fmri_gym/menu.py`).

## Curricula

| file | plays |
|---|---|
| `g50t.json` | g50t: a grid puzzle |
| `ls20.json` | ls20: a grid puzzle |
| `re86.json` | re86: a shape-moving puzzle |
| `tr87.json` | tr87: a symbol-matching puzzle |
| `tu93.json` | tu93: a maze puzzle |
| `wa30.json` | wa30: a block-moving puzzle |
