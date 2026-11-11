# SuperTuxKart curricula

The `stk` backend's curricula: one per race track, each including `common/stk.json` (3 laps, 4 karts, difficulty 1, 640x360, `frame_skip` 2 at 60 fps, seed 1, 300 s, the instructions) and naming its `track` and title (README, "Shared phases"). Each track also comes in a `*-GMPD3-stick-nav` version, which includes `common/stk-GMPD3-stick-nav.json` instead: the same race played on a Current Designs GMPD3 (README, "Rigs"), steering and accelerating on the left stick; it plays on a rig whose `"inputs"` list the GMPD3 or an Xbox controller.

## Setup

`uv sync --extra stk` (supertuxkart-gym). The wheel fetches the game and a 270 MiB asset pack from its GitHub release on first use, for Linux x86_64 and macOS (x86_64, arm64): no checkout and no build; `STK_ENV_BIN` overrides the binary it fetched. It needs a real OpenGL display: the game renders into a hidden window and every frame is read back and shown here, so it does not work under `SDL_VIDEODRIVER=dummy` or with `--render-driver=vulkan`.

## Tracks

There is one curriculum per race track the asset pack ships with geometry. Its other track directories (abyss, mines, zengarden, ... and the battle arenas and soccer fields) are metadata only, and naming one kills the game process, so a new track needs a pack that carries it, not a new config. The track is fixed for the life of a process, so a curriculum races several by using one game phase per track.

## Keys

The env's action is MultiBinary(8) over SuperTuxKart's keys: 0=left, 1=right, 2=up (accelerate), 3=down (brake / reverse), 4=nitro, 5=skid, 6=fire, 7=rescue. Held keys combine, and the game's own player controller turns them into controls. `stk.json` maps them to the rig's ten keys (arrows, X nitro, C skid, SPACE fire, Z rescue), and `stk-GMPD3-stick-nav.json` to a controller's stick (up accelerates, down brakes), RT fire, LT skid, A nitro and Y rescue; keep each shared phase's Controls text in sync by hand.

## Timing

The game is lock-stepped: one step per frame, `frame_skip` physics ticks (120 Hz) per step, so fps must equal 120 / `frame_skip`; the adapter refuses otherwise. Measured: 2.5 ms per step at 640x360 with four karts on an Intel Arc laptop.

## Logging

Per frame: the action (the 8-vector), the reward (progress, in metres), and every `SAMPLE_FIELDS` column of stk_gym: tick/time/phase, speed, x/y/z, heading, distance_down_track, distance_to_center, on_road/on_ground/wrong_way, rank, finished_laps, powerup, nitro_energy, progress_m, and ctrl_* (the controls the kart applied). Frames are shown, not stored. A block replays from `episode_seeds` + actions with the same track, karts, seed and supertuxkart-gym version: step a fresh `stk_gym.StkEnv` through the logged actions.

## Curricula

| file | track | lap |
|---|---|---|
| `cornfield_crossing.json` | Cornfield Crossing | 1264 m |
| `hacienda.json` | Hacienda | 1166 m |
| `lighthouse.json` | Around the Lighthouse | 880 m |
| `scotland.json` | Nessie's Pond | 1329 m |
| `snowmountain.json` | Northern Resort | 1147 m |
| `<track>-GMPD3-stick-nav.json` | the same track, on a controller | |
