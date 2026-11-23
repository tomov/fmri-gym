#!/bin/sh
# DBP session: one line per run, in order. Run from the fmri-gym folder.
# BEHAVIOURAL ONLY: training for ses-003's tasks outside the scanner, on an Xbox controller.
# Not a scan session; 900s are practice. Each run starts on SPACE (no scanner); stop and
# answer the participant's questions between runs.
# CogTasks use the practice stimulus set 00, for 120 s (+10 s feedback); MTB runs are 74 s. ~10 min in all.

set -e  # Exit on error

# Session parameters
SUBJECT=sub-BC01
RIG=$PWD/configs/BC/rigs/lapmac_xbox.json
SES=901
COG_DURATION=120
COG=$HOME/GitHub/DBP/dev/DBP_CogTasks_fMRI   # branch run-duration (upstream main + --duration, gamepad fixes)
MTB=$HOME/GitHub/DBP/dev/DBP_MTB             # branch rig-structured-fields + the dmgrd_mrot batteries
BEH=$PWD/data/$SUBJECT/ses-$(printf %03d "$SES")/beh   # CogTasks and MTB data, in the games' folder

# The session's recording (session-logger): started first, stopped when the script ends.
RECORDING=$(uv run session-logger start --subject "$SUBJECT" --ses "$SES" --rig "$RIG")
trap 'uv run session-logger stop "$RECORDING"' EXIT
trap 'exit 130' INT TERM HUP

# Runs, one per line, each after a mark in the session's recording

uv run session-logger mark "$RECORDING" "1/5 WCST practice (CogTasks run 00)"
(cd "$COG" && uv run python run.py WCST --subject $SUBJECT --session $SES --run 00 --duration $COG_DURATION --input gamepad --data-dir "$BEH" --rig "$RIG" --controls controls/wcst-GMPD3-ABXY-nav.json)

uv run session-logger mark "$RECORDING" "2/5 VCST practice (CogTasks run 00)"
(cd "$COG" && uv run python run.py VCST --subject $SUBJECT --session $SES --run 00 --duration $COG_DURATION --input gamepad --data-dir "$BEH" --rig "$RIG" --controls controls/vcst-GMPD3-ABXY-nav.json)

uv run session-logger mark "$RECORDING" "3/5 ARC practice (CogTasks run 00)"
(cd "$COG" && uv run python run.py ARC --subject $SUBJECT --session $SES --run 00 --duration $COG_DURATION --input gamepad --data-dir "$BEH" --rig "$RIG" --controls controls/arc-GMPD3-ABXY-nav.json)

uv run session-logger mark "$RECORDING" "4/5 spatial working memory practice (MTB run 1)"
(cd "$MTB" && uv run python run.py $SUBJECT dmgrd_mrot_practice --run 1 --no-ttl --data-dir "$BEH" --rig "$RIG" --controls controls/mtb-GMPD3-ABXY-nav.json)

uv run session-logger mark "$RECORDING" "5/5 mental rotation practice (MTB run 2)"
(cd "$MTB" && uv run python run.py $SUBJECT dmgrd_mrot_practice --run 2 --no-ttl --data-dir "$BEH" --rig "$RIG" --controls controls/mtb-GMPD3-ABXY-nav.json)
