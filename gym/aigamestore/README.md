# AI GameStore — Game Environments

Source code for the 10 game environments featured on the AI GameStore platform, plus `aigamestore_gym`, a Gymnasium env that plays them in lock step.

Each directory (`game1` … `game10`) is a self-contained, browser-based game implemented in plain HTML and JavaScript (no build step). Open a game's `index.html` to run it.

```
game1/ … game10/
  index.html      # entry point
  *.js            # game logic (ES modules)
aigamestore_gym/
  env.py          # AIGameStoreEnv: MultiBinary keys in, canvas + getGameState() out
  lockstep.js     # init script: the page's clock and Math.random, under the env's control
```

Rendering is done on an HTML canvas in code; no external art assets are required. The games have no sound.

## The gym env

```bash
pip install -e .            # from this directory; needs a Chromium browser (system Chrome by default)
```

```python
import gymnasium as gym, aigamestore_gym
env = gym.make("AIGameStore/game6-level3-v0")   # or aigamestore_gym.AIGameStoreEnv("game6", level=3)
obs, info = env.reset(seed=1)
obs, reward, terminated, truncated, info = env.step([1, 0, 0, 0, 0])   # hold env.keys[0] == "LEFT"
```

One `step()` holds the keys the action marks and advances the game by exactly `frame_skip` of its 60 Hz frames -- no time passes between steps, so the game is the same however long the agent takes to decide, and `reset(seed=...)` plus the action sequence replays an episode. The observation is the canvas (RGB), the reward is the change in the game's `score`, and `info["state"]` is the scalar part of `getGameState()` (`gamePhase`, `score`, `currentLevel`, ...).

An episode is one level: `reset()` starts the env's `level` afresh through the game's `window.loadLevel(n)`, and the episode is over as soon as the game leaves PLAYING (a win, a loss, a level-complete screen) or its level counter moves. The end screen then stays up for `end_steps` more steps (20, i.e. 2 s at the default rate) with no key passed on, and `terminated` comes on the last of them. `GAME_LEVELS` says how many levels each game has -- game4 has none (an endless runner, the episode ends when it is lost), game6 is one long level, game7's are procedural without an upper bound. See the docstring of `env.py`.

## Changes to the games

The pages are self-contained: the libraries their HTML fetches from CDNs (p5, p5.collide2d, Matter, three.js, seedrandom, pinned by version in the URL) are answered from `vendor/`, the Google analytics tag they carry is dropped, and a request for anything else off the page's origin makes the next `step` raise, since a script from the network would draw from the seeded `Math.random` and a library served differently some day would change the game. No network is needed to play, now or later.

The games differ from the ones on the platform in three ways, all so they can be played without a keyboard in front of them: level clears and game overs advance by themselves after 3 s (no "press ENTER" / "press R" prompts); the on-canvas control hints take their key names from `?label_<KEY>=` query parameters, so a host that remaps keys (a button box, say) can show the key the player really presses; and each game's `window.loadLevel(n)` (a dev-mode hook on the platform) starts level `n` as a fresh game -- score 0, full lives, PLAYING.

game7 also takes `X` for the next weapon owned, in the order of its `1`-`5` keys, so it can be played with fewer buttons, and its `5` selects the RPG, which on the platform only a pickup equips.
