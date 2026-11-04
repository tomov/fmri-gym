# ViZDoom curricula

The `vizdoom` backend's curricula: ViZDoom's stock scenarios, one each, and every map of Freedoom Phase 1 and Phase 2. They run on COOM's engine. The Freedoom maps include `common/freedoom1.json` or `common/freedoom2.json` and state only the map and its title (README, "Shared phases"); each stock scenario has its own buttons, so it writes its phase in full.

## Setup

`uv sync --extra vizdoom`. Freedoom (https://freedoom.github.io/) is a free replacement for Doom's game data: Phase 1 (`freedoom1.wad`, E1M1–E4M9) stands in for Doom, and Phase 2 (`freedoom2.wad`, MAP01–MAP32) for Doom II. Both WADs ship inside the vizdoom package, which registers every map as `VizdoomFreedoom1E<e>M<m>-S<skill>-v0` / `VizdoomFreedoom2MAP<nn>-S<skill>-v0`, so any map runs by changing `game`; `VizdoomDoom*` plays the original WADs, which are not free. Every Freedoom curriculum here plays skill 2. A Freedoom map ends when the player reaches the exit, with reward 1; the scenario's own `episode_timeout` is 60 minutes. `freedoom1.cfg`/`freedoom2.cfg` turn the automap buffer on, which would put a second panel next to the screen in `render()` and log it every frame, so the shared phases set `env_kwargs.automap_buffer_enabled` false.

## Keys

The env's action is MultiBinary over the scenario's buttons (`env_kwargs.max_buttons_pressed` 0), so `controls` values are button indices in the scenario's `available_buttons` order, and held keys combine (turn while shooting). The controls the subject sees are the phase's `instructions`; keep their text in sync with `controls` by hand. The keys are the rig's ten (README, "Rigs") but for the Freedoom maps, whose seven button actions are more than a controller's six: they are played on a keyboard, and a rig with a controller refuses them.

| scenario | buttons, by index | reference |
|---|---|---|
| Freedoom Phase 1 and 2 (same order) | 0=ATTACK, 1=SPEED, 2=STRAFE, 3=USE, 4=MOVE_RIGHT, 5=MOVE_LEFT, 6=MOVE_BACKWARD, 7=MOVE_FORWARD, 8=TURN_RIGHT, 9=TURN_LEFT, 10–16=SELECT_WEAPON1–7, 17=SELECT_NEXT_WEAPON, 18=SELECT_PREV_WEAPON | [freedoom1.cfg](https://github.com/Farama-Foundation/ViZDoom/blob/main/scenarios/freedoom1.cfg), [freedoom2.cfg](https://github.com/Farama-Foundation/ViZDoom/blob/main/scenarios/freedoom2.cfg) |
| Basic | 0=MOVE_LEFT, 1=MOVE_RIGHT, 2=ATTACK | [basic.cfg](https://github.com/Farama-Foundation/ViZDoom/blob/main/scenarios/basic.cfg) |
| Deadly Corridor | 0=MOVE_LEFT, 1=MOVE_RIGHT, 2=ATTACK, 3=MOVE_FORWARD, 4=MOVE_BACKWARD, 5=TURN_LEFT, 6=TURN_RIGHT | [deadly_corridor.cfg](https://github.com/Farama-Foundation/ViZDoom/blob/main/scenarios/deadly_corridor.cfg) |
| Deathmatch | 0=ATTACK, 1=SPEED, 2=STRAFE, 3=MOVE_RIGHT, 4=MOVE_LEFT, 5=MOVE_BACKWARD, 6=MOVE_FORWARD, 7=TURN_RIGHT, 8=TURN_LEFT, 9–14=SELECT_WEAPON1–6, 15=SELECT_NEXT_WEAPON, 16=SELECT_PREV_WEAPON | [deathmatch.cfg](https://github.com/Farama-Foundation/ViZDoom/blob/main/scenarios/deathmatch.cfg) |
| Defend the Center, Defend the Line, Predict Position | 0=TURN_LEFT, 1=TURN_RIGHT, 2=ATTACK | [defend_the_line.cfg](https://github.com/Farama-Foundation/ViZDoom/blob/main/scenarios/defend_the_line.cfg), [predict_position.cfg](https://github.com/Farama-Foundation/ViZDoom/blob/main/scenarios/predict_position.cfg) |
| Health Gathering Supreme | 0=TURN_LEFT, 1=TURN_RIGHT, 2=MOVE_FORWARD | [health_gathering_supreme.cfg](https://github.com/Farama-Foundation/ViZDoom/blob/main/scenarios/health_gathering_supreme.cfg) |
| My Way Home | 0=TURN_LEFT, 1=TURN_RIGHT, 2=MOVE_FORWARD, 3=MOVE_LEFT, 4=MOVE_RIGHT | [my_way_home.cfg](https://github.com/Farama-Foundation/ViZDoom/blob/main/scenarios/my_way_home.cfg) |
| Take Cover | 0=MOVE_LEFT, 1=MOVE_RIGHT | [take_cover.cfg](https://github.com/Farama-Foundation/ViZDoom/blob/main/scenarios/take_cover.cfg) |

## Audio

`env_kwargs.audio_buffer_enabled` turns Doom's sound on in every curriculum here: it arrives in `obs['audio']`, is played through the session's speakers, and is logged per frame. Doom makes 1/35 s of sound per step, so `fps * env_kwargs.frame_skip` must be 35, Doom's own tic rate; the session stops the block otherwise, because sound could not follow the frames.

## Stock scenarios

| file | plays |
|---|---|
| `basic.json` | The simplest scenario: one monster on the far wall of a small room; line it up and shoot it. The episode ends on the kill (or after 300 tics). No turn buttons, only strafing. |
| `deadly_corridor.json` | Fight down a corridor to the vest at the end, with enemies attacking from side alcoves. Has move buttons: walk forward/back and strafe. |
| `deathmatch.json` | An arena with respawning monsters, weapons and ammo pickups, scored by KILLCOUNT. It also declares the mouse axes, so ViZDoom's action space is Dict{binary, continuous}; the adapter hides the axes (there is no mouse in the scanner) and keeps the keyboard buttons, so turning is TURN_LEFT/TURN_RIGHT only. |
| `defend_center.json` | Surrounded by monsters: turn and shoot. Other `Vizdoom*-v1` ids run with the same backend. |
| `defend_line.json` | Like Defend the Center, but the monsters come from one side of a hall and respawn: survive as long as possible while killing them. |
| `health_gathering_supreme.json` | An acid-floor maze: health drains every tic, so keep finding medikits. The hard variant of Health Gathering: a maze rather than an open room, with poison vials among the medikits. No weapon. |
| `my_way_home.json` | A nine-room maze with a green armour vest in one fixed room; the player starts in a random room facing a random direction. Sparse reward (the vest) minus a small per-tic cost: spatial navigation, no attack button. |
| `predict_position.json` | A rocket launcher against a monster moving across a hall. The rocket takes time to fly, so fire where the monster is going to be. One rocket per episode; the episode ends when it lands (or at the timeout). |
| `take_cover.json` | Dodge fireballs from monsters on the far wall, more of them over time, by strafing only. One point per tic survived; no weapon. |
| `take_cover_defend_line.json` | Take Cover, then Defend the Line, as two game blocks of 150 s each (so logs, seeds and keymaps stay per scenario): about 5 min, like one single-scenario curriculum. Raise both `duration`s for 300 s each. |

## Freedoom maps

| file | map | title |
|---|---|---|
| `freedoom1_e1m1.json` | E1M1 | Outer Prison |
| `freedoom1_e1m2.json` | E1M2 | Communications Center |
| `freedoom1_e1m3.json` | E1M3 | Waste Disposal |
| `freedoom1_e1m4.json` | E1M4 | Supply Depot |
| `freedoom1_e1m5.json` | E1M5 | Armory |
| `freedoom1_e1m6.json` | E1M6 | Training Facility |
| `freedoom1_e1m7.json` | E1M7 | Xenobiotic Materials Lab |
| `freedoom1_e1m8.json` | E1M8 | Outpost Quarry |
| `freedoom1_e1m9.json` | E1M9 | Nutrient Recycling |
| `freedoom1_e2m1.json` | E2M1 | Elemental Gate |
| `freedoom1_e2m2.json` | E2M2 | Shifter |
| `freedoom1_e2m3.json` | E2M3 | Reclaimed Facilities |
| `freedoom1_e2m4.json` | E2M4 | Flooded Installation |
| `freedoom1_e2m5.json` | E2M5 | Underground Hub |
| `freedoom1_e2m6.json` | E2M6 | Hidden Sector |
| `freedoom1_e2m7.json` | E2M7 | Control Complex |
| `freedoom1_e2m8.json` | E2M8 | Containment Cell |
| `freedoom1_e2m9.json` | E2M9 | Fortress 31 |
| `freedoom1_e3m1.json` | E3M1 | Land of the Lost |
| `freedoom1_e3m2.json` | E3M2 | Geothermal Tunnels |
| `freedoom1_e3m3.json` | E3M3 | Sacrificial Bastion |
| `freedoom1_e3m4.json` | E3M4 | Oblation Temple |
| `freedoom1_e3m5.json` | E3M5 | Infernal Hallows |
| `freedoom1_e3m6.json` | E3M6 | Igneous Intrusion |
| `freedoom1_e3m7.json` | E3M7 | No Regrets |
| `freedoom1_e3m8.json` | E3M8 | Ancient Lair |
| `freedoom1_e3m9.json` | E3M9 | Acquainted With Grief |
| `freedoom1_e4m1.json` | E4M1 | Maintenance Area |
| `freedoom1_e4m2.json` | E4M2 | Research Complex |
| `freedoom1_e4m3.json` | E4M3 | Central Computing |
| `freedoom1_e4m4.json` | E4M4 | Hydroponic Facility |
| `freedoom1_e4m5.json` | E4M5 | Engineering Station |
| `freedoom1_e4m6.json` | E4M6 | Command Center |
| `freedoom1_e4m7.json` | E4M7 | Waste Treatment |
| `freedoom1_e4m8.json` | E4M8 | Launch Bay |
| `freedoom1_e4m9.json` | E4M9 | Operations |
| `freedoom2_map01.json` | MAP01 | Hydroelectric Plant |
| `freedoom2_map02.json` | MAP02 | Filtration Tunnels |
| `freedoom2_map03.json` | MAP03 | Crude Processing Center |
| `freedoom2_map04.json` | MAP04 | Containment Bay |
| `freedoom2_map05.json` | MAP05 | Sludge Burrow |
| `freedoom2_map06.json` | MAP06 | Janus Terminal |
| `freedoom2_map07.json` | MAP07 | Logic Gate |
| `freedoom2_map08.json` | MAP08 | Astronomy Complex |
| `freedoom2_map09.json` | MAP09 | Datacenter |
| `freedoom2_map10.json` | MAP10 | Deadly Outlands |
| `freedoom2_map11.json` | MAP11 | Dimensional Rift Observatory |
| `freedoom2_map12.json` | MAP12 | Railroads |
| `freedoom2_map13.json` | MAP13 | Station Earth |
| `freedoom2_map14.json` | MAP14 | Nuclear Zone |
| `freedoom2_map15.json` | MAP15 | Hostile Takeover |
| `freedoom2_map16.json` | MAP16 | Urban Jungle |
| `freedoom2_map17.json` | MAP17 | City Capitol |
| `freedoom2_map18.json` | MAP18 | Aquatics Lab |
| `freedoom2_map19.json` | MAP19 | Sewage Control |
| `freedoom2_map20.json` | MAP20 | Blood Ember Fortress |
| `freedoom2_map21.json` | MAP21 | Under Realm |
| `freedoom2_map22.json` | MAP22 | Remanasu |
| `freedoom2_map23.json` | MAP23 | Underground Facility |
| `freedoom2_map24.json` | MAP24 | Abandoned Teleporter Lab |
| `freedoom2_map25.json` | MAP25 | Persistence of Memory |
| `freedoom2_map26.json` | MAP26 | Dark Depths |
| `freedoom2_map27.json` | MAP27 | Palace of Red |
| `freedoom2_map28.json` | MAP28 | Grim Redoubt |
| `freedoom2_map29.json` | MAP29 | Melting Point |
| `freedoom2_map30.json` | MAP30 | Jaws of Defeat |
| `freedoom2_map31.json` | MAP31 | Be Quiet |
| `freedoom2_map32.json` | MAP32 | Not Sure |
