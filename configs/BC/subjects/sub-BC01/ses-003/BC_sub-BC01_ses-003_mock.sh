#!/bin/sh
# DBP session: one line per run, in order. Run from the fmri-gym folder.
# MOCK of BC_sub-BC01_ses-003.sh to rehearse on the laptop alone: the same runs, in the same
# order, but on the laptop rig (lapmac_xbox.json: one window, the Xbox controller) and under
# sub-mock, so --resume never moves where sub-BC01's real session picks up. No scanner: games
# start on RETURN, CogTasks on SPACE, and MTB waits for its trigger key: press 5.
# Games (fmri-gym, BC's curricula with LT and RT swapped) interleaved with DBP_CogTasks_fMRI and
# DBP_MTB runs, all on the 932 gamepad. Games 600 s, CogTasks 420 s (+10 s feedback),
# MTB 574 s (+ scoreboard until SPACE): ~81 min of runs.

set -e  # Exit on error

# Session parameters
SUBJECT=sub-mock
RIG=$PWD/configs/BC/rigs/lapmac_xbox.json
SES=3
RUN_DURATION=600
COG=$HOME/GitHub/DBP/dev/DBP_CogTasks_fMRI   # branch run-duration (upstream main + --duration, gamepad fixes)
MTB=$HOME/GitHub/DBP/dev/DBP_MTB             # branch rig-structured-fields + the dmgrd_mrot batteries
BEH=$PWD/data/$SUBJECT/ses-$(printf %03d "$SES")/beh   # CogTasks and MTB data, in the games' folder

# The session's recording (session-logger): started first, stopped when the script ends.
RECORDING=$(uv run session-logger start --subject "$SUBJECT" --ses "$SES" --rig "$RIG")
trap 'uv run session-logger stop "$RECORDING"' EXIT
trap 'exit 130' INT TERM HUP

# Runs, one per line, each after a mark in the session's recording

uv run session-logger mark "$RECORDING" "1/9 STK lighthouse (fmri-play run 1)"
uv run fmri-play --subject $SUBJECT --ses $SES --run 1 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/BC/curricula/stk/levels/lighthouse-GMPD3-stick-nav.json

uv run session-logger mark "$RECORDING" "2/9 WCST (CogTasks run 01)"
(cd "$COG" && uv run python run.py WCST --subject $SUBJECT --session 003 --run 01 --input gamepad --data-dir "$BEH" --rig "$RIG" --controls controls/wcst-GMPD3-ABXY-nav.json)

uv run session-logger mark "$RECORDING" "3/9 ViZDoom Deathmatch (fmri-play run 2)"
uv run fmri-play --subject $SUBJECT --ses $SES --run 2 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/BC/curricula/vizdoom/levels/deathmatch-GMPD3-stick-nav.json

uv run session-logger mark "$RECORDING" "4/9 spatial working memory (MTB dmgrd_mrot run 1)"
(cd "$MTB" && uv run python run.py $SUBJECT dmgrd_mrot --run 1 --data-dir "$BEH" --rig "$RIG" --controls controls/mtb-GMPD3-ABXY-nav.json)

uv run session-logger mark "$RECORDING" "5/9 STK hacienda (fmri-play run 3)"
uv run fmri-play --subject $SUBJECT --ses $SES --run 3 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/BC/curricula/stk/levels/hacienda-GMPD3-stick-nav.json

uv run session-logger mark "$RECORDING" "6/9 ARC (CogTasks run 01)"
(cd "$COG" && uv run python run.py ARC --subject $SUBJECT --session 003 --run 01 --input gamepad --data-dir "$BEH" --rig "$RIG" --controls controls/arc-GMPD3-ABXY-nav.json)

uv run session-logger mark "$RECORDING" "7/9 ViZDoom Freedoom 2 MAP01 (fmri-play run 4)"
uv run fmri-play --subject $SUBJECT --ses $SES --run 4 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/BC/curricula/vizdoom/levels/freedoom2_map01-GMPD3-stick-nav.json

uv run session-logger mark "$RECORDING" "8/9 mental rotation (MTB dmgrd_mrot run 2)"
(cd "$MTB" && uv run python run.py $SUBJECT dmgrd_mrot --run 2 --data-dir "$BEH" --rig "$RIG" --controls controls/mtb-GMPD3-ABXY-nav.json)

uv run session-logger mark "$RECORDING" "9/9 VCST (CogTasks run 01)"
(cd "$COG" && uv run python run.py VCST --subject $SUBJECT --session 003 --run 01 --input gamepad --data-dir "$BEH" --rig "$RIG" --controls controls/vcst-GMPD3-ABXY-nav.json)
