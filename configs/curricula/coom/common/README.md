# COOM curricula

The `coom` backend's curricula: one per scenario of COOM, the Doom continual-RL suite (its own scenarios, not stock ViZDoom). Each includes `common/coom.json` (audio on, EFX off, 300 s at 35 fps) and states its scenario, instructions and keys (README, "Shared phases").

## Setup

Needs TTomilin/COOM cloned into `external/coom` at the commit the README pins (README, "External checkouts"). The coom backend reads its `conf.cfg`/`.wad` scenario assets directly; it does not install the COOM package, which pins a conflicting gymnasium version. `env_kwargs.task` picks a WAD variant: `default`, or `blue`, `red`, `green`, `hard`, `monsters`, `obstacles`, `resized`, `shadows`.

## Keys

COOM's action is Discrete(12), a table of turn x move x execute: 0=noop, 1=execute, 2=forward, 3=forward+execute, 4=turn right, 6=turn right+forward, 8=turn left, 10=turn left+forward. Every curriculum maps UP, LEFT, RIGHT and their combinations the same way, with "" (no key) as 0; the scenario decides what the execute button is, and which key carries it (table below). Keep each curriculum's Controls text in sync with its `controls` by hand.

## Audio

Requires OpenAL (Ubuntu: `sudo apt install libopenal1`). Native stereo PCM at 44100 Hz, one Doom tic per step: keep fps at 35. EFX reverb is disabled. The rig file's `"has_audio": false` mutes playback but keeps the logged PCM. MIDI music also needs a working engine MIDI renderer; check the engine's start-up log.

## Curricula

| file | scenario | execute button, key |
|---|---|---|
| `arms_dealer.json` | Arms Dealer | SPEED, SPACE |
| `chainsaw.json` | Chainsaw | ATTACK, SPACE |
| `floor_is_lava.json` | Floor is Lava | SPEED, SPACE |
| `health_gathering.json` | Health Gathering | SPEED, SPACE |
| `hide_and_seek.json` | Hide and Seek | SPEED, SPACE |
| `parkour.json` | Parkour | JUMP, SPACE |
| `pitfall.json` | Pitfall | JUMP, SPACE |
| `raise_the_roof.json` | Raise the Roof | USE, SPACE |
| `run_and_gun.json` | Run and Gun | ATTACK, SPACE |
| `all.json` (in `coom/`) | the nine in a row, as above | |
