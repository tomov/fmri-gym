#!/bin/sh
# fmri-gym session: one line per run, in order.
set -e
FMRI_GYM_PATH=/Users/psychology/GitHub/DBP/dev/fmri-gym-worktrees/BC02_ses-002_hy
DBP_MTB_PATH=/Users/psychology/GitHub/DBP/dev/DBP_MTB-worktrees/BC02_ses-002
DBP_FMRI_PATH=/Users/psychology/GitHub/DBP/dev/DBP_CogTasks_fMRI-worktrees/BC02_ses-002
SESSION_DIR=/Users/psychology/GitHub/DBP/dev/scheduler-worktrees/BC02_ses-002/data/sub-mock/ses-02
SUBJECT=sub-mock
RIG=configs/BC/rigs/lapmac_keyboard.json
SES=${1:-002}
#RUN
uv run --directory "$FMRI_GYM_PATH" --no-sync fmri-play --curriculum configs/curricula/stk/levels/lighthouse.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG" --duration 600 --loop --resume
mkdir -p "$SESSION_DIR/blocks/02_wcst" && uv run --directory "$DBP_FMRI_PATH" --no-sync python run.py WCST --run 01 --subject sub-mock --session ses-02 --input keyboard --rig "$FMRI_GYM_PATH/$RIG" --controls controls/wcst.json --data-dir "$FMRI_GYM_PATH/data/sub-mock/ses-002/beh"  # scheduler: {"task": "wcst", "duration_s": 450.0, "parameters": {"run": "01", "input": "keyboard", "rig": "session rig", "controls": "wcst", "data_dir": "/Users/psychology/GitHub/DBP/dev/fmri-gym-worktrees/BC02_ses-002_hy/data/sub-mock/ses-002/beh"}}
uv run --directory "$FMRI_GYM_PATH" --no-sync fmri-play --curriculum configs/curricula/vizdoom/levels/deathmatch.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG" --duration 600 --loop --resume
mkdir -p "$SESSION_DIR/blocks/04_mtb_set_a" && uv run --directory "$DBP_MTB_PATH" --no-sync python run.py sub-mock set_a 1 --run 01 --rig "$FMRI_GYM_PATH/$RIG" --controls controls/mtb.json  # scheduler: {"task": "mtb_set_a", "duration_s": 600.0, "parameters": {"order": "1", "run": "01", "no_ttl": false, "rig": "session rig", "controls": "mtb"}}
uv run --directory "$FMRI_GYM_PATH" --no-sync fmri-play --curriculum configs/curricula/stk/levels/hacienda.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG" --duration 600 --loop --resume
mkdir -p "$SESSION_DIR/blocks/06_arc" && uv run --directory "$DBP_FMRI_PATH" --no-sync python run.py ARC --run 01 --subject sub-mock --session ses-02 --input keyboard --rig "$FMRI_GYM_PATH/$RIG" --controls controls/arc.json --data-dir "$FMRI_GYM_PATH/data/sub-mock/ses-002/beh"  # scheduler: {"task": "arc", "duration_s": 450.0, "parameters": {"run": "01", "input": "keyboard", "rig": "session rig", "controls": "arc", "data_dir": "/Users/psychology/GitHub/DBP/dev/fmri-gym-worktrees/BC02_ses-002_hy/data/sub-mock/ses-002/beh"}}
uv run --directory "$FMRI_GYM_PATH" --no-sync fmri-play --curriculum configs/curricula/vizdoom/levels/freedoom2_map01.json --subject "$SUBJECT" --ses "$SES" --run 1 --rig "$RIG" --duration 600 --loop --resume
mkdir -p "$SESSION_DIR/blocks/08_mtb_set_a" && uv run --directory "$DBP_MTB_PATH" --no-sync python run.py sub-mock set_a 1 --run 02 --rig "$FMRI_GYM_PATH/$RIG" --controls controls/mtb.json  # scheduler: {"task": "mtb_set_a", "duration_s": 600.0, "parameters": {"order": "1", "run": "02", "no_ttl": false, "rig": "session rig", "controls": "mtb"}}
mkdir -p "$SESSION_DIR/blocks/09_vcst" && uv run --directory "$DBP_FMRI_PATH" --no-sync python run.py VCST --run 01 --subject sub-mock --session ses-02 --input keyboard --rig "$FMRI_GYM_PATH/$RIG" --controls controls/vcst.json --data-dir "$FMRI_GYM_PATH/data/sub-mock/ses-002/beh"  # scheduler: {"task": "vcst", "duration_s": 450.0, "parameters": {"run": "01", "input": "keyboard", "rig": "session rig", "controls": "vcst", "data_dir": "/Users/psychology/GitHub/DBP/dev/fmri-gym-worktrees/BC02_ses-002_hy/data/sub-mock/ses-002/beh"}}
