# Gymnasium curricula

The `gym` backend's curricula: Gymnasium's own classic-control, Box2D and toy-text envs. Each game's phase (its id, keys and settings) is in `common/<game>.json`, shared by the game's own 300 s curriculum and by its family's sampler, which plays each game briefly in a row.

## Setup

Classic control and toy text ship with gymnasium. Box2D (`lunarlander`, `bipedalwalker`, `carracing`, `box2d`) needs `uv sync --extra box2d`.

## Keys

| game | action space | keys |
|---|---|---|
| CartPole | Discrete(2) | LEFT 0, RIGHT 1; no key 0 |
| MountainCar, Acrobot | Discrete(3) | LEFT 0, RIGHT 2; no key 1 (no push) |
| MountainCarContinuous, Pendulum | Box(1) | LEFT / RIGHT push the torque or force to its limit; no key 0 |
| LunarLander | Discrete(4) | LEFT 1, SPACE 2 (main engine), RIGHT 3; no key 0 |
| BipedalWalker | Box(4) | LEFT / RIGHT and DOWN / UP drive the first two joints to their limits |
| CarRacing | Box(3) | LEFT / RIGHT steer, UP gas, DOWN brake |
| FrozenLake, FrozenLake8x8 | Discrete(4) | arrows; `is_slippery` false, so a move goes where it is pressed |
| CliffWalking | Discrete(4) | arrows |
| Taxi | Discrete(6) | arrows, SPACE pickup, X dropoff |
| Blackjack | Discrete(2) | LEFT stick, RIGHT hit |

Pendulum, MountainCarContinuous and BipedalWalker have continuous actions: pushing joints to their limits renders and logs fine, but they are not really keyboard-playable. The toy-text games are `is_turn_based`: one step per press, rendered as RGB by gymnasium's pygame renderers.

## Curricula

| file | plays |
|---|---|
| `acrobot.json` | Acrobot-v1 |
| `bipedalwalker.json` | BipedalWalker-v3 |
| `blackjack.json` | Blackjack-v1 |
| `carracing.json` | CarRacing-v3 |
| `cartpole.json` | CartPole-v1 |
| `cliffwalking.json` | CliffWalking-v1 |
| `frozenlake.json` | FrozenLake-v1 |
| `frozenlake8x8.json` | FrozenLake8x8-v1 |
| `lunarlander.json` | LunarLander-v3 |
| `mountaincar.json` | MountainCar-v0 |
| `mountaincarcontinuous.json` | MountainCarContinuous-v0 |
| `pendulum.json` | Pendulum-v1 |
| `taxi.json` | Taxi-v3 |
| `classic_control.json` | the five classic-control games, one episode each |
| `box2d.json` | the three Box2D games, one episode each |
| `toy_text.json` | the five toy-text games, two episodes each |
| `classic_control-GMPD3-stick-nav.json` | MountainCar and Acrobot for 90 s each on a controller's left stick, from `common/<game>-GMPD3-stick-nav.json` |
| `carracing-GMPD3-stick-nav.json` | CarRacing on a controller: the left stick steers, RT is gas and LT brakes, at 25 steps per second, half its physics speed, which a slower Mac keeps up with (needs the box2d extra) |
