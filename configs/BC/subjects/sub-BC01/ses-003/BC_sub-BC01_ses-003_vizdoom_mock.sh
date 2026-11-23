#!/bin/sh
# fmri-gym session: one line per run, in order.
# MOCK on the laptop alone (lapmac_xbox.json: one window, the Xbox controller; data under
# sub-mock): one run per kind of BC ViZDoom level, to try each -- the 9 scenarios, then one map
# of each Freedoom campaign (every other map plays as these two). Each run plays its level from
# the start (no --resume) and ends with it: after its episodes or its time, whichever is first.
# RETURN starts a run; CTRL+RETURN ends it and goes on to the next; CTRL+ESCAPE stops the session.

set -e  # Exit on error

# Session parameters
SUBJECT=sub-mock
RIG=configs/BC/rigs/lapmac_xbox.json
SES=3
RUN_DURATION=600

# The session's recording (session-logger): started first, stopped when the script ends.
RECORDING=$(uv run session-logger start --subject "$SUBJECT" --ses "$SES" --rig "$RIG" --session-pid $$)
trap 'uv run session-logger stop "$RECORDING"' EXIT
trap 'exit 130' INT TERM HUP

# Runs, one per line, each after a mark in the session's recording

uv run session-logger mark "$RECORDING" "1/11 ViZDoom Basic"
uv run fmri-play --subject $SUBJECT --ses $SES --run 1 --rig "$RIG" --duration $RUN_DURATION --curriculum configs/BC/curricula/vizdoom/levels/basic-GMPD3-stick-nav.json

uv run session-logger mark "$RECORDING" "2/11 ViZDoom Defend the Center"
uv run fmri-play --subject $SUBJECT --ses $SES --run 2 --rig "$RIG" --duration $RUN_DURATION --curriculum configs/BC/curricula/vizdoom/levels/defend_center-GMPD3-stick-nav.json

uv run session-logger mark "$RECORDING" "3/11 ViZDoom Defend the Line"
uv run fmri-play --subject $SUBJECT --ses $SES --run 3 --rig "$RIG" --duration $RUN_DURATION --curriculum configs/BC/curricula/vizdoom/levels/defend_line-GMPD3-stick-nav.json

uv run session-logger mark "$RECORDING" "4/11 ViZDoom Predict Position"
uv run fmri-play --subject $SUBJECT --ses $SES --run 4 --rig "$RIG" --duration $RUN_DURATION --curriculum configs/BC/curricula/vizdoom/levels/predict_position-GMPD3-stick-nav.json

uv run session-logger mark "$RECORDING" "5/11 ViZDoom Take Cover"
uv run fmri-play --subject $SUBJECT --ses $SES --run 5 --rig "$RIG" --duration $RUN_DURATION --curriculum configs/BC/curricula/vizdoom/levels/take_cover-GMPD3-stick-nav.json

uv run session-logger mark "$RECORDING" "6/11 ViZDoom Health Gathering Supreme"
uv run fmri-play --subject $SUBJECT --ses $SES --run 6 --rig "$RIG" --duration $RUN_DURATION --curriculum configs/BC/curricula/vizdoom/levels/health_gathering_supreme-GMPD3-stick-nav.json

uv run session-logger mark "$RECORDING" "7/11 ViZDoom My Way Home"
uv run fmri-play --subject $SUBJECT --ses $SES --run 7 --rig "$RIG" --duration $RUN_DURATION --curriculum configs/BC/curricula/vizdoom/levels/my_way_home-GMPD3-stick-nav.json

uv run session-logger mark "$RECORDING" "8/11 ViZDoom Deadly Corridor"
uv run fmri-play --subject $SUBJECT --ses $SES --run 8 --rig "$RIG" --duration $RUN_DURATION --curriculum configs/BC/curricula/vizdoom/levels/deadly_corridor-GMPD3-stick-nav.json

uv run session-logger mark "$RECORDING" "9/11 ViZDoom Deathmatch"
uv run fmri-play --subject $SUBJECT --ses $SES --run 9 --rig "$RIG" --duration $RUN_DURATION --curriculum configs/BC/curricula/vizdoom/levels/deathmatch-GMPD3-stick-nav.json

uv run session-logger mark "$RECORDING" "10/11 ViZDoom Freedoom 1 E1M1, Outer Prison"
uv run fmri-play --subject $SUBJECT --ses $SES --run 10 --rig "$RIG" --duration $RUN_DURATION --curriculum configs/BC/curricula/vizdoom/levels/freedoom1_e1m1-GMPD3-stick-nav.json

uv run session-logger mark "$RECORDING" "11/11 ViZDoom Freedoom 2 MAP01, Hydroelectric Plant"
uv run fmri-play --subject $SUBJECT --ses $SES --run 11 --rig "$RIG" --duration $RUN_DURATION --curriculum configs/BC/curricula/vizdoom/levels/freedoom2_map01-GMPD3-stick-nav.json
