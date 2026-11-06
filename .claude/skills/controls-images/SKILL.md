---
name: controls-images
description: Make, edit or check a game's controls image (a gamepad drawing with each control the game uses named by its action) that a curriculum's instructions show with an image line. Use when adding a controller curriculum, when changing a curriculum's controls, menu or Controls lines, when a curriculum's instructions name an image under configs/images/controls/, or when asked whether the images are up to date.
---

# Controls images

A screen's text shows an image where a line is only `![alt](path)`, the path from `configs/` (`fmri_gym/display.py`, `IMAGE_LINE`); a missing or unreadable image stops the run at load. The controls images are one PNG per controller curriculum's shared phase, `configs/images/controls/<backend>/<common file's name>.png`, each shown above that phase's Controls lines, which stay as text. They are drawn, never edited by hand: `configs/images/controls/labels.json` names the underlay (the Xbox 360 drawing in `configs/images/controllers/`) and, per image, each control (as a curriculum writes it, the stick as `LEFT_STICK`) with the lines beside it; `tools/controls_images.py` draws them all. A control with no label is greyed out, as are the drawing's controls no controller rig has (bumpers, D-pad, right stick, Back, Start, Guide), and the names printed on the drawing are painted over; the areas are constants at the top of the tool, in the underlay's pixels.

## Make or change one

1. Edit its entry in `labels.json`: one key per control the phase binds in `controls`, plus the menu's `control` (labelled "hold: pause menu"); the text is the action the adapter's docstring gives that index, worded as the Controls lines word it.
2. Redraw: `uv run python tools/controls_images.py configs/images/controls/labels.json`. A control the drawing does not have stops it, with the list of the ones it has.
3. For a new image, add the line to the instructions' `text` in the `common/` file, above the Controls lines with an empty line after it: `"![controls](images/controls/<backend>/<name>.png)"`.
4. Look at the image, and at the screen it is on, at the rigs' 1024x768: the text and the image must fit, the heading and the "Press X to start" line included. To see one game's screen:
   ```bash
   SDL_VIDEODRIVER=dummy uv run python -c "
   import pygame, sys
   from fmri_gym.config.curriculum import Curriculum
   from fmri_gym.display import Display
   game = next(p for p in Curriculum.load(sys.argv[1]).phases if getattr(p, 'instructions', None))
   d = Display(size=(1024, 768), vsync=False); m = game.instructions.message(False)
   d.draw_text(m.text, align=m.align); pygame.image.save(d.canvas, '/tmp/screen.png')
   " configs/curricula/aigamestore/game1-GMPD3-ABXY-nav.json
   ```
5. Commit `labels.json` and the PNGs together.

## Check they are up to date

Whenever a branch touches a curriculum, or a `common/` file one includes, whose instructions show a controls image (`rg -l images/controls configs/curricula`), and above all when it changes `controls`, `menu` or the Controls lines:

- **Labels match the bindings.** For each such phase, the controls in its image's entry are exactly the ones it binds (any `CONTROLLER_AXIS_LEFTX_*` or `CONTROLLER_AXIS_LEFTY_*` counts as `LEFT_STICK`) plus the menu's `control`, and each label says the action that control's index is bound to: a remapped button with its old label, a binding with no label, or a label with no binding is a mismatch. Labels and the Controls lines say the same thing.
- **Images match the labels.** Redraw, then `git status --short configs/images/controls`: a PNG that changed was stale; commit it. Nothing changed means the images are what `labels.json` says.
- **Every image line has an entry** in `labels.json`, and every entry is shown by some curriculum.

No test checks the first and last points, so the agent does, and says in the PR which images it checked.
