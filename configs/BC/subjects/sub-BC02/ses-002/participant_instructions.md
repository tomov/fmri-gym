# Session 2: instructions for the participant (ses-002)

One section per cognitive task of `BC_sub-BC02_ses-002.sh`, in the order the script plays them. Everything is played on the 932 gamepad. The buttons:

- **Four choices** (WCST, VCST): **A, X, Y, B** pick the cards 1 to 4, left to right.
- **Two choices** (ARC, MTB): **A** is the first answer (yes, same, true, left), **B** the second (no, mirror, false, right).

## WCST (run 2)

"I want you to match a given card displayed on the bottom of the screen with one of four cards displayed on the top. Those four cards would not be changed, and only the card on the bottom would be changed for each trial. and you would press from 1 to 4 to select the card from left to right. You would see the feedback whether your answer was "right" or "wrong"."

Buttons 1 to 4 are A, X, Y, B.

*Source: `DBP_CogTasks_fMRI/WCS/README.md`, "Instruction", verbatim.*

## MTB set A (runs 4 and 8)

Each run is 18 short blocks (reading, grids, stories, movements, search, movie, towers, rotation, numbers, knots, rest), each with its own instruction screen of 5 s that names the buttons. For example:

- "Read the short story, then judge the last sentence. Press A for true, B for false."
- "Silently read each set of words. When you see the hand, press B right away."
- "Move only your RIGHT INDEX FINGER in time with the circle. Keep your head and body still."

*Source: `DBP_MTB/dbp_tasks.py`, `INSTRUCTIONS`, as shown with the gamepad (buttons 2 and 3 are A and B).*

## ARC (run 6)

"Two example pairs show an input grid of coloured cells and the output grid a hidden rule makes of it. A test input appears with a proposed output. Press accept (A) if the output follows the rule, or reject (B) if it does not."

*Source: ARC has no instruction text of its own (no README); adapted from the DBP_Welcome documentation, `docs/tools/cogtasks.md`.*

## VCST (run 9)

"I want you to match a given card displayed on the bottom of the screen with one of four cards displayed on the top. Those four cards would not be changed, and only the card on the bottom would be changed for each trial. and you would press from 1 to 4 to select the card from left to right. You would see the feedback whether your answer was "right" or "wrong"."

Buttons 1 to 4 are A, X, Y, B.

*Source: `DBP_CogTasks_fMRI/VCS/README.md`, "Instruction", verbatim.*
