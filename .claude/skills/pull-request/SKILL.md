---
name: pull-request
description: Take an fmri-gym change from a branch to a merged pull request on tomov/fmri-gym: the worktree, the local test run, the lint, what the PR description must say (core changes, backends not played locally), draft versus ready for review, and watching CI. Use when asked to open, update or finish a PR, or to push a branch.
---

# From a branch to a merged PR

## Branch

Work on a branch off `main`, in a git worktree when the main checkout is in use (`git worktree add ../fmri-gym-worktrees/<branch> -b <branch> main`). A fresh worktree's `.venv` is empty until `uv sync --extra <the backends you touch>`; `uv run --no-sync` afterwards keeps uv from re-syncing on every command. Submodules come with `git submodule update --init`.

## Before pushing

```bash
uv run pytest --changed
uv run ruff check fmri_gym fmri_play.py agents tests gym/*/tests
PYTHONPATH="$(printf '%s:' gym/*/)" uv run lint-imports   # who imports whom: pyproject.toml's contracts
```

All clean, or the PR says which backend you could not play here and why (the summary's skip line is the sentence to quote). The `testing` skill explains the options and how to read a failure. Commit messages carry the reasoning behind a change; the code, the docstrings and the configs describe the present only (AGENTS.md, rule 3).

Then read every curriculum the branch touches, including each `common/` file the touched ones `__include`, and check by eye that what the subject is told matches what the phase binds: the Controls lines of the instructions' `text` name the same controls as the game phase's `controls` map, each with the action the adapter's docstring gives that index (a line saying "Y shoot" needs `"CONTROLLER_BUTTON_Y"` bound to the shoot action, and a bound control with no line, or a line with no binding, is a mismatch); the "Press X to start" line names a control in `controls_to_dismiss`; a menu's hint names its `control`, `move` and `confirm`; a survey's text names its `controls`. No test checks this, so the agent does, and says in the PR which files it read. When one of those curricula shows a controls image (`rg -l images/controls configs/curricula`), check the image against the bindings too and redraw it (the `controls-images` skill, "Check they are up to date"), and say in the PR which images were checked and which were redrawn.

## The title

The title names what the branch makes true, in the present tense: "A muted rig logs the game's sound", "ALE: a resumed episode opens where the stopped one ended". It does not name the old behaviour ("Fix missing audio"), the symptom ("audio.h5 missing"), or how the bug was found.

## The description

A reviewer has to be able to follow the change without reading the diff first, and to reproduce both sides of it on their own machine. Write these six sections, in this order, each under its own heading. Keep the before and the after in their own sections: a paragraph that mixes "used to" and "now" is the thing to avoid.

1. **Before.** What the code did, observably: what a run wrote, showed or played. Use concrete facts (a file that is missing, a frame that is wrong, a number), not adjectives.
2. **Why that was a problem.** What it costs a session or an analysis: data lost, a subject shown the wrong thing, a log that cannot be trusted. Name the rule it broke if there is one (AGENTS.md rule 4, say).
3. **Now.** What the code does after this branch, observably, in the same terms as *Before*, so the two can be read side by side. Then the files changed and how. List core changes (any `fmri_gym/*.py`, any file under `config/`, `inputs/`, `outputs/`, `sync/` or `save/`, `adapters/base.py`, `adapters/keymap.py`), and say whether each is additive and default-off and names no backend (AGENTS.md, rule 2).
4. **Why that solves it.** The causal link from the change to the problem going away, in a few sentences, including what it deliberately leaves alone.
5. **How it was verified.** What you ran and what came out: the tests added, the backends the suite played and skipped (with the reason), the runs made by hand, with their numbers.
6. **See it yourself.** Steps a person can paste, for the before and for the after, each producing the output that shows the difference:
   - which ref to check out, for each side (`git switch --detach origin/main`, `git switch --detach origin/<branch>`), or a worktree for each;
   - what to sync on each side, by extra (`uv sync --extra vizdoom`), and the submodules or game files it needs;
   - the exact commands, with every path spelled out and nothing left as `<config>`;
   - what to look for in the output on each side: the line, the file, the number that differs.

   Run these steps yourself, on both refs, before posting them. A step that does not work as written is worse than none. If they need something the reviewer may lack (a sound card, a GL display, a ROM), say so and give the closest substitute.

Also in the description: the backends the suite could not play here and why (quote the summary's skip line), and for a new backend, its config, how it installs, and that `tests/backends.py` has its row.

## Draft, ready, CI

Open as a draft while the work is in flux; CI does not run on drafts, and GitHub runs no `pull_request` workflow on a PR that conflicts with `main` either, silently: a PR with no checks at all has usually fallen behind `main`, so merge it in first. Marking it ready for review runs the suite (`.github/workflows/tests.yml`: a core job, then one job per backend a Linux runner can play, every config, no skipping), and every push after that runs it again. Watch with `gh pr checks <number> --watch`; a failed backend job keeps its run folders as an artifact. Make the checks green before asking for review, or say in the PR which job fails and why that is expected.
