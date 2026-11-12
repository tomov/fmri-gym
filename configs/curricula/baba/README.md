# Baba Is AI curricula

`make_win.json` plays Baba Is AI's `env/make_win`, a generated puzzle in the style of Baba Is You: push word blocks to rewrite the rules. The env's other ids (`env/goto_win`, `env/you_win` and their distractor variants) play with the same phase and another `game`.

**Setup.** `uv sync --extra baba` installs `baba-gym` (`gym/baba/`), which pulls baba-is-ai from GitHub.

**Keys.** `BabaIsYouEnv.Actions` ([baba/grid.py](https://github.com/nacloos/baba-is-ai/blob/main/baba/grid.py)): 0 = idle, 1 = up, 2 = right, 3 = down, 4 = left. The instructions' Controls text is kept in sync with `controls` by hand.

**Blocking, as Baba Is You is.** The block is the level played until won (`mode` `episode`, `n_episodes` 1, `advance_on` `["won"]`); a loss (YOU destroyed, or no YOU left) restarts the same level, as the game itself does, and `duration` 300 s is the cap.

**Menu.** A puzzle can be pushed into a dead end, so after 15 s of play holding X for 5 s pauses the game and offers Restart / Give up / Resume (UP/DOWN to choose, SPACE to select; `humans/popup.py`). Give up is the way out of a level the subject cannot solve.
