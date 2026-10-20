# SuperTuxKart determinism fixes

`determinism_fixes.patch`: 9 bugs in [chrplr/stk-code](https://github.com/chrplr/stk-code) (`gym-v0.1.3`, the
engine `fmri_gym/adapters/stk.py` drives) that made two runs of the same seed and actions diverge: wind
pattern/phase seeded from `rand()`/the real device clock instead of the episode seed and simulated time, LOD
switch-distance jitter from unseeded `rand()`, three draw/upload loops iterating a pointer-keyed
`unordered_set` (order randomized by ASLR every process launch) now sorted by a stable creation-order counter,
and Irrlicht's own particle-system RNG never reseeded on reset.

Apply it to a checkout before building:

```bash
git clone --branch gym-v0.1.3 https://github.com/chrplr/stk-code.git
cd stk-code
git apply /path/to/determinism_fixes.patch
cmake -B cmake_build -DCMAKE_BUILD_TYPE=Release && cmake --build cmake_build -j
```

Verified: same seed, same actions, two fresh processes, every track, 100 steps each -- reward, termination,
truncation and all logged state variables are bit-identical. Rendered pixels are not: a GPU-driver rounding
non-determinism on alpha-tested edges (grass, foliage, cutout textures), not fixed by this patch and not
fixable from application code.
