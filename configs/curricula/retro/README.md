# stable-retro curricula

`airstriker.json` plays Airstriker, and `airstriker-GMPD3-stick-nav.json` plays it on a controller (left stick moves, RT fires), with its instructions and controls image in `common/`; Airstriker is the one game whose ROM ships with stable-retro. Any other needs its ROM imported and an integration in stable-retro (README, "Running stable-retro games").

**Keys.** This Genesis core's buttons are 0 = B, 1 = A, 2 = MODE, 3 = START, 4 = UP, 5 = DOWN, 6 = LEFT, 7 = RIGHT, 8 = C, 9 = Y, 10 = X, 11 = Z, and `controls` values are those indices (held keys combine). Airstriker moves on the D-pad and fires with B, so the map is the arrows plus X (keyboard) to B (console). The controls the subject sees are a plain message phase, kept in sync with `controls` by hand.

**Sound and fps.** The sound plays a constant delay after each frame's flip; the rig file's `"has_audio": false` mutes it. `fps` is this Genesis core's native rate: the session stops a block whose sound per step does not match 1/fps.
