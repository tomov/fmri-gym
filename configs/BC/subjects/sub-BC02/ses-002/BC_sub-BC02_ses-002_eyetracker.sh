#!/bin/sh
# DBP session: one line per run, in order. Run from the fmri-gym folder.
# BC_sub-BC02_ses-002.sh with the EyeLink: a calibration before the first run, then every run
# recorded by eyelink-track (one EDF per run, in beh/, t=0 = the scanner's first 5).
# Games (fmri-gym, BC's curricula with LT and RT swapped) interleaved with DBP_CogTasks_fMRI and
# DBP_MTB runs, all on the 932 gamepad (932 in GMPD3 mode the whole session: the scanner's
# pulses arrive as 5). Games 600 s, CogTasks 420 s (+10 s feedback), MTB set A 574 s
# (+ scoreboard until SPACE): ~81 min of runs.

set -e  # Exit on error

# Session parameters
SUBJECT=sub-BC02
RIG=$PWD/configs/BC/rigs/3T_MIT_GMPD3.json
SES=2
RUN_DURATION=600
COG=$HOME/GitHub/DBP/dev/DBP_CogTasks_fMRI-worktrees/BC02_ses-002   # main + PRs #4 (rig_handler.py) and #5 (gamepad on macOS)
MTB=$HOME/GitHub/DBP/dev/DBP_MTB-worktrees/BC02_ses-002             # main + PRs #4 (rig_handler.py, --data-dir) and #5 (gamepad on macOS)
MTB_ORDER=1   # set A block order: one per subject, in rotation at the site (DBP_MTB README)
BEH=$PWD/data/$SUBJECT/ses-$(printf %03d "$SES")/beh   # CogTasks, MTB and eyetracker data, in the games' folder

# The eyetracker: eyelink-track, Shrecki's photodiode_cli (eyetrack_record) + our commit reading the
# trigger key on macOS (branch scanner-keys, not pushed yet). Its rig: start key 5 from the 932, the
# projector's size and distance (as in 3T_MIT_GMPD3.json).
EYE=$HOME/GitHub/DBP/dev/eyelink-track-worktrees/scanner-keys_hy/.venv/bin
EYE_RIG=$HOME/GitHub/DBP/dev/eyelink-rigs/BC_3T.json
EYE_SES=$(printf %03d "$SES")
EDF=$BEH/${SUBJECT}_ses-$EYE_SES   # the EDFs: ${EDF}_task-<task>_run-<n>_eyetrack.edf
REC=   # the eye recorder of the current run, while it records

# edf_run TASK: the run number of TASK's next EDF: 1, or the next free one for a redo (an EDF is
# never overwritten)
edf_run() {
    n=1
    while [ -e "${EDF}_task-$1_run-${n}_eyetrack.edf" ]; do n=$((n + 1)); done
    echo "$n"
}

# ask WHAT: after a failure of the eyetracker, retry (r), go on without it (c) or quit (q)
ask() {
    while :; do
        printf '%s r: retry, c: go on without the eyetracker, q: quit. ' "$1"
        read -r answer || exit 1
        case $answer in r) return 0 ;; c) return 1 ;; q) exit 1 ;; esac
    done
}

# The calibration, on the projector. ESCAPE before 5 skips it (the EyeLink keeps its last one).
calibrate() {
    echo "Eyetracker calibration: 5 opens the setup; C calibrates (ENTER accepts the first dot), V validates, O leaves; then ENTER here. ESCAPE before 5 skips it."
    until "$EYE/eyetrack_calibration" --rig "$EYE_RIG" --screen 1 --no-lpt --subject "$SUBJECT" \
            --session "$EYE_SES" --task calib --run "$(edf_run calib)" --directory "$BEH"; do
        ask "The calibration failed." || return 0
    done
}

# start_recorder TASK: starts eyetrack_record (REC) for TASK's next EDF and waits until it records and
# waits for the trigger. Its output: the .log beside the EDF.
start_recorder() {
    mkdir -p "$BEH"
    run=$(edf_run "$1")
    log=${EDF}_task-$1_run-${run}_eyetrack.log
    lines=0
    [ ! -e "$log" ] || lines=$(wc -l < "$log")
    "$EYE/eyetrack_record" --run-simultaneous --rig "$EYE_RIG" --screen 1 --subject "$SUBJECT" \
        --session "$EYE_SES" --task "$1" --run "$run" --directory "$BEH" >> "$log" 2>&1 &
    REC=$!
    until tail -n "+$((lines + 1))" "$log" | grep -q "Waiting for the trigger"; do
        if ! kill -0 "$REC" 2>/dev/null; then
            wait "$REC" || true
            REC=
            tail -n 3 "$log"
            return 1
        fi
        sleep 0.2
    done
    echo "Eyetracker: recording $(basename "$log" .log).edf, waiting for the trigger"
}

# stop_recorder: stops the eye recorder, if one records: it saves the EDF, and says whether the
# trigger reached it
stop_recorder() {
    [ -n "$REC" ] || return 0
    kill "$REC" 2>/dev/null || true
    wait "$REC" || echo "WARNING: the eye recorder ended with an error: see $log"
    REC=
    tail -n "+$((lines + 1))" "$log" | grep -E "^(Saved|WARNING)" || true
}

# eyes TASK COMMAND...: one run, recorded by the eyetracker: the recorder starts first and writes the
# scanner's first 5 to the EDF (t=0); it is stopped when the run ends.
eyes() {
    task=$1
    shift
    until start_recorder "$task"; do
        ask "The eye recorder did not start (see $log)." || break
    done
    status=0
    "$@" || status=$?
    stop_recorder
    return $status
}

# The session's recording (session-logger): started first, stopped when the script ends, after the
# eye recorder if the script ends during a run.
RECORDING=$(uv run session-logger start --subject "$SUBJECT" --ses "$SES" --rig "$RIG" --session-pid $$)
trap 'stop_recorder; uv run session-logger stop "$RECORDING"' EXIT
trap 'exit 130' INT TERM HUP

uv run session-logger mark "$RECORDING" "0/9 Eyetracker calibration"
calibrate

# Runs, one per line, each after a mark in the session's recording, each recorded by the eyetracker

uv run session-logger mark "$RECORDING" "1/9 STK lighthouse (fmri-play run 1)"
eyes stklighthouse uv run fmri-play --subject $SUBJECT --ses $SES --run 1 --rig "$RIG" --duration $RUN_DURATION --loop --curriculum configs/BC/curricula/stk/levels/lighthouse-GMPD3-stick-nav-analog.json

uv run session-logger mark "$RECORDING" "2/9 WCST (CogTasks run 01)"
eyes wcst uv run --directory "$COG" python run.py WCST --subject $SUBJECT --session 002 --run 01 --input xbox --data-dir "$BEH" --rig "$RIG" --controls controls/wcst-GMPD3.json

uv run session-logger mark "$RECORDING" "3/9 ViZDoom all levels (fmri-play run 2)"
eyes vizdoomall uv run fmri-play --subject $SUBJECT --ses $SES --run 2 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/BC/curricula/vizdoom/all-GMPD3-stick-nav.json

uv run session-logger mark "$RECORDING" "4/9 MTB set A run 1 (order $MTB_ORDER)"
eyes mtb uv run --directory "$MTB" python run.py $SUBJECT set_a $MTB_ORDER --run 1 --data-dir "$BEH" --rig "$RIG" --controls controls/mtb-GMPD3.json

uv run session-logger mark "$RECORDING" "5/9 STK hacienda (fmri-play run 3)"
eyes stkhacienda uv run fmri-play --subject $SUBJECT --ses $SES --run 3 --rig "$RIG" --duration $RUN_DURATION --loop --curriculum configs/BC/curricula/stk/levels/hacienda-GMPD3-stick-nav-analog.json

uv run session-logger mark "$RECORDING" "6/9 ARC (CogTasks run 01)"
eyes arc uv run --directory "$COG" python run.py ARC --subject $SUBJECT --session 002 --run 01 --input xbox --data-dir "$BEH" --rig "$RIG" --controls controls/arc-GMPD3.json

uv run session-logger mark "$RECORDING" "7/9 ViZDoom all levels (fmri-play run 4)"
eyes vizdoomall uv run fmri-play --subject $SUBJECT --ses $SES --run 4 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/BC/curricula/vizdoom/all-GMPD3-stick-nav.json

uv run session-logger mark "$RECORDING" "8/9 MTB set A run 2 (order $MTB_ORDER)"
eyes mtb uv run --directory "$MTB" python run.py $SUBJECT set_a $MTB_ORDER --run 2 --data-dir "$BEH" --rig "$RIG" --controls controls/mtb-GMPD3.json

uv run session-logger mark "$RECORDING" "9/9 VCST (CogTasks run 01)"
eyes vcst uv run --directory "$COG" python run.py VCST --subject $SUBJECT --session 002 --run 01 --input xbox --data-dir "$BEH" --rig "$RIG" --controls controls/vcst-GMPD3.json
