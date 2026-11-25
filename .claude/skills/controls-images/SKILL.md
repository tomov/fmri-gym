---
name: controls-images
description: Make, edit or check a game's controls image (a gamepad drawing with each control the game uses named by its action) that a curriculum's instructions show with an image line. Use when adding a controller curriculum, when changing a curriculum's controls or menu, when a curriculum's instructions name an image under configs/images/controls/, or when asked whether the images are up to date.
---

# Controls images

A screen's text shows an image where a line is only `![alt](path)`, the path from `configs/` (`humans/outputs/display.py`, `IMAGE_LINE`); a missing or unreadable image stops the run at load. The controls images are one PNG per controller curriculum's shared phase, `configs/images/controls/<backend>/<common file's name>.png`, each shown in that phase's instructions in place of Controls lines: the image is the only list of the controls the participant reads. They are drawn, never edited by hand: `configs/images/controls/labels.json` names the underlay (the Xbox 360 drawing in `configs/images/controllers/`) and, per image, each control (as a curriculum writes it, the stick as `LEFT_STICK`) with the lines beside it; `tools/controls_images.py` draws them all, keeping a blank line between two controls' labels so each reads as one. A control with no label is greyed out, as are the drawing's controls no controller rig has (bumpers, D-pad, right stick, Back, Start, Guide), and the names printed on the drawing are painted over; the areas are constants at the top of the tool, in the underlay's pixels.

## Make or change one

1. Edit its entry in `labels.json`: one key per control the phase binds in `controls`, plus the menu's `control` and the hint's, each labelled with how long to hold it when its `hold` is above zero ("hold 3 s: menu", "hold: show controls") and, when its `after` is above zero, how long into the block it opens, centered under that line by making the two a list of their own (`["hold 3 s: menu", "(after 3 min)"]`); the text is the action the adapter's docstring gives that index, in a few words a participant reads at a glance.
2. Redraw: `uv run python tools/controls_images.py configs/images/controls/labels.json`. A control the drawing does not have stops it, with the list of the ones it has.
3. For a new image, put the line in the instructions' `text` in the `common/` file where the Controls lines were, removing them, with an empty line before the "Press X to start" line: `"![controls](images/controls/<backend>/<name>.png)"`. A curriculum that overrides the instructions' `text` does the same. A phase with a `hint` shows the same line at the top of the hint's `text`.
4. Look at the image, and at the screen it is on, at the rigs' 1024x768: the text and the image must fit, the heading and the "Press X to start" line included. To see one game's screen:
   ```bash
   SDL_VIDEODRIVER=dummy uv run python -c "
   import pygame, sys
   from humans.config.curriculum import Curriculum
   from humans.outputs.display import Display
   game = next(p for p in Curriculum.load(sys.argv[1]).phases if getattr(p, 'instructions', None))
   d = Display(size=(1024, 768), vsync=False); m = game.instructions.message(False)
   d.draw_text(m.text, align=m.align); pygame.image.save(d.canvas, '/tmp/screen.png')
   " configs/curricula/aigamestore/game1-GMPD3-ABXY-nav.json
   ```
5. Commit `labels.json` and the PNGs together.

## Check they are up to date

Whenever a branch touches a curriculum, or a `common/` file one includes, whose instructions show a controls image (`rg -l images/controls configs/curricula`), and above all when it changes `controls` or `menu`:

- **Labels match the bindings.** For each such phase, the controls in its image's entry are exactly the ones it binds (`CONTROLLER_AXIS_LEFTX`, `CONTROLLER_AXIS_LEFTY` and any of their sides count as `LEFT_STICK`) plus the menu's and the hint's `control`, and each label says the action that control's index is bound to, and the menu's and the hint's labels say their `hold` and `after`: a remapped button with its old label, a binding with no label, a label with no binding, or a changed `hold` or `after` is a mismatch. Instructions that show the image have no Controls lines.
- **Images match the labels.** Redraw, then `git status --short configs/images/controls`: a PNG that changed was stale; commit it. Nothing changed means the images are what `labels.json` says.
- **Every image line has an entry** in `labels.json`, and every entry is shown by some curriculum; a hint's `text` shows the same image as its phase's instructions.

No test checks the first and last points, so the agent does, and says in the PR which images it checked.
