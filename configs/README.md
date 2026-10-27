# configs/

Three kinds of file, kept apart because each is shared differently: a curriculum is the same for every subject of every team, a rig is the same for every run on one setup, and a session is one subject's one visit. Curricula are shared by everyone; each team keeps its rigs and its sessions in a folder of its own, named after its PI's initials; the examples show both.

```text
configs/
  curricula/<backend>/levels/<name>.json            what one run plays: one level
  curricula/<backend>/<name>.json                   what one run plays: several levels in a row
  curricula/<backend>/common/<name>.json            a game phase several of them include
  curricula/<backend>/common/README.md              what they play, and how
  examples/rigs/<rig>.json                          example rigs, one per kind of setup (examples/rigs/README.md)
  examples/sessions/<name>.sh                       example sessions, subject and rig given when started
  <team>/rigs/<rig>.json                            how one of the team's setups plays every run
  <team>/subjects/sub-<label>/ses-<NNN>/<name>.sh   which runs one subject plays in one session
```

A team's folder is made the first time one of its rigs is made with `rig-config` (New rig... asks for the team and the rig's name) or one of its sessions is saved with `session-editor`. A team with several places names each rig after its place: `MT/rigs/mgh-prisma.json`, `MT/rigs/mit-desk.json`.

## curricula/

One folder per backend (the `backend` its game phases name), plus `checks/` (the rig check, whose phases are checks rather than games), and `demos/` (runs that span backends). A file is either one level, in the backend's `levels/` folder -- `minihack/levels/room5x5.json`, `vizdoom/levels/freedoom1_e1m1.json` -- or a sequence of levels, in the backend's folder itself -- `rushhour/complete.json`, `vgdl/all.json`. `aigamestore/` has no `levels/`: its ten games keep one layout, one file each in the folder, the one-level games too. The name says the level, not the backend: the folder already does. What a backend's curricula share -- a game's keys, instructions and settings -- is written once, in `<backend>/common/<name>.json`, and each phase includes it (`"__include": "common/<name>.json"`, or `"../common/<name>.json"` from `levels/`) and states only what differs (README, "Shared phases"); `common/` holds no curricula. `<backend>/common/README.md` is the backend's documentation: what each curriculum plays, setup, what the keys' indices mean, and what a level needs said. Comments in a curriculum (`"## ..."` keys) are few, for what is particular to that one file.

A curriculum's path after `curricula/`, in camelCase (BIDS labels are letters and digits only), is its BIDS task label: `curricula/minihack/levels/room5x5.json` is `task-minihackLevelsRoom5x5`, `curricula/vgdl/all.json` is `task-vgdlAll`. Moving or renaming a file renames the task, and the task keys the run's seeds, so a level that has been recorded keeps its file name.

Every file here is played by the test suite (`tests/test_configs.py` validates each one; `pytest --thorough` plays every game phase of each), so a curriculum that does not play does not stay. The games tried and left out are listed in the top-level README, "Games without a curriculum".

## Rigs: `<team>/rigs/`

One file per rig, `<team>/rigs/<rig>.json`: the screen, how the participant responds, the sound, the data folder, the triggers, and the equipment its rig check measures with (README, "Rigs"). Make and change them with `uv run rig-config`, and commit them. Every run names its rig file (`fmri-play --rig configs/<team>/rigs/<rig>.json`); the data records the team as the rig's site, and the file's name as the rig. Beside the rig files, `measured/<rig>-<n>.json` holds what each rig check measured on this computer; git ignores them. The example rigs are described in `examples/rigs/README.md`.

## Sessions: `<team>/subjects/`

One shell script per session, `<team>/subjects/sub-<label>/ses-<NNN>/<name>.sh`, as BIDS lays out the data it records: one `fmri-play` line per run, in order (README, "Curriculum format"). A script names its subject and rig at the top (`SUBJECT=`, `RIG=`), or leaves either to whoever starts it, and plays into the session its folder names (`SES=${1:-001}`); its own argument overrides that, which is how a session that stopped is resumed into a new one. `session-editor --session <script>` opens one; run it from the repo root with `sh <script>`. `examples/sessions/pilot.sh` is an hour-long example: five games, ten minutes each (`SUBJECT=sub-01 RIG=configs/examples/rigs/laptop.json sh configs/examples/sessions/pilot.sh`).
