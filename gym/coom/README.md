# coom-gym

COOM's continual-RL Doom scenarios as a Gymnasium env. COOM's own package pins `gymnasium==0.28`, so this never imports it: it drives `vizdoom.DoomGame` against a [TTomilin/COOM](https://github.com/TTomilin/COOM) checkout's scenario files (`conf.cfg` and `<task>.wad`).

```bash
pip install -e ../vizdoom -e .   # from this directory; also needs the COOM checkout
```

```python
import coom_gym
env = coom_gym.COOMEnv("pitfall")                   # fmri-gym's external/coom, at the README's pinned commit
# env = coom_gym.COOMEnv("pitfall", repo="/path/to/COOM")   # another checkout
# env = gym.make("COOM-v0", scenario="pitfall")
obs, info = env.reset(seed=1)
obs, reward, terminated, truncated, info = env.step(2)   # move forward
```

The action space is COOM's 12-action table (turn x move x execute): 0 = noop, 1 = execute, 2 = forward, 3 = forward + execute, 4 = turn right, 6 = right + forward, 8 = turn left, 10 = left + forward. "Execute" is the scenario's fourth button (JUMP, ATTACK, SPEED, or USE). One step is one Doom tic. The observation is the 640x480 screen; the reward is ViZDoom's raw reward.

`audio_buffer_enabled=True` puts one 44.1 kHz stereo buffer per tic on `env.game.get_state().audio_buffer`. That needs OpenAL: `libopenal1` on Ubuntu, OpenAL Soft on macOS (`brew install openal-soft`). Without it, the env raises before the engine starts.
