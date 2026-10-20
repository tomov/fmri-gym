# SuperTuxKart determinism fixes

`determinism_fixes.patch`: 9 bugs in [chrplr/stk-code](https://github.com/chrplr/stk-code) (`gym-v0.1.3`) that
made two runs of the same seed and actions diverge -- unseeded RNGs (wind, LOD jitter, Irrlicht's particle
system) and draw order depending on pointer values instead of scene content.

```bash
git clone --branch gym-v0.1.3 https://github.com/chrplr/stk-code.git
cd stk-code
git apply /path/to/determinism_fixes.patch
cmake -B cmake_build -DCMAKE_BUILD_TYPE=Release && cmake --build cmake_build -j
```

Verified: reward, termination, truncation and logged state are bit-identical across fresh processes on every
track. Rendered pixels are not -- a GPU-driver rounding effect on alpha-tested edges, unrelated to this patch.
