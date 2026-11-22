# SuperTuxKart curricula

The `stk` backend's curricula: one per race track, each including `common/stk.json` (3 laps, 4 karts, difficulty 1, 640x360, `frame_skip` 2 at 60 fps, seed 1, 300 s, the instructions) and naming its `track` and title (README, "Shared phases"). Each track also comes in a `*-GMPD3-stick-nav` version, which includes `common/stk-GMPD3-stick-nav.json` instead: the same race played on a Current Designs GMPD3 (README, "Rigs"), steering and accelerating on the left stick; it plays on a rig whose `"inputs"` list the GMPD3 or an Xbox controller. Each also comes in a `*-GMPD3-stick-analog` version (`common/stk-GMPD3-stick-analog.json`), on the same controllers: the stick steers and accelerates as far as it is pushed, not all or nothing.

## Setup

`uv sync --extra stk` installs [supertuxkart-gym](https://pypi.org/project/supertuxkart-gym/), which brings the game itself. It needs a real OpenGL display: the game renders into a hidden window and every frame is read back and shown here, so it does not work under `SDL_VIDEODRIVER=dummy` or with `--render-driver=vulkan`.

## Tracks

There is one curriculum per race track the package ships. Its other track names (abyss, mines, zengarden, ... and the battle arenas and soccer fields) have no geometry, and naming one kills the game process. The track is fixed for the life of a process, so a curriculum races several by using one game phase per track.

## Keys

The env's action is MultiBinary(8) over SuperTuxKart's keys: 0=left, 1=right, 2=up (accelerate), 3=down (brake / reverse), 4=nitro, 5=skid, 6=fire, 7=rescue. Held keys combine, and the game's own player controller turns them into controls. `stk.json` maps them to the rig's ten keys (arrows, X nitro, C skid, SPACE fire, Z rescue), and `stk-GMPD3-stick-nav.json` to a controller's stick (up accelerates, down brakes), RT fire, LT skid, A nitro and Y rescue; keep each shared phase's Controls text in sync by hand.

`stk-GMPD3-stick-analog.json` sets `"action_mode": "continuous_full"`: the action is Box(5) of the kart's controls, 0=steer (-1 left .. 1 right), 1=accelerate (0..1), 2=brake, 3=nitro, 4=skid (each on above 0.5), which the env sets directly, past the game's player controller (no steering ramp). The stick's x axis is the steering, its up side the throttle and its down side the brake (past half its travel), LT skid and A nitro. The env has no fire or rescue in this mode, so this version has neither.

## Timing

The game is lock-stepped: one step per frame, `frame_skip` physics ticks (120 Hz) per step, so fps must equal 120 / `frame_skip`; the adapter refuses otherwise. Measured: 2.5 ms per step at 640x360 with four karts on an Intel Arc laptop.

## Logging

Per frame: the action (the 8-vector; the 5 controls in `*-GMPD3-stick-analog`), the reward (progress, in metres), and every `SAMPLE_FIELDS` column of stk_gym: tick/time/phase, speed, x/y/z, heading, distance_down_track, distance_to_center, on_road/on_ground/wrong_way, rank, finished_laps, powerup, nitro_energy, progress_m, and ctrl_* (the controls the kart applied). Frames are shown, not stored. A block replays from `episode_seeds` + actions with the same track, karts, seed and supertuxkart-gym version: step a fresh `stk_gym.StkEnv` through the logged actions.

## Curricula

| file | track | lap |
|---|---|---|
| `cornfield_crossing.json` | Cornfield Crossing | 1264 m |
| `hacienda.json` | Hacienda | 1166 m |
| `lighthouse.json` | Around the Lighthouse | 880 m |
| `scotland.json` | Nessie's Pond | 1329 m |
| `snowmountain.json` | Northern Resort | 1147 m |
| `<track>-GMPD3-stick-nav.json` | the same track, on a controller | |
| `<track>-GMPD3-stick-analog.json` | the same track, on a controller's analog stick | |
