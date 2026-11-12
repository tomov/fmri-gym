#!/bin/sh
# fmri-gym session: one line per run, in order.
# BEHAVIOURAL ONLY: controller practice outside the scanner, on an Xbox controller. Not a scan session; 900s are practice.

set -e  # Exit on error

# Session parameters
SUBJECT=sub-01
RIG=configs/MT/rigs/xbox.json
SES=900
RUN_DURATION=180

# Runs, one per line

uv run fmri-play --subject $SUBJECT --ses $SES --run 1 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/curricula/gym/classic_control-GMPD3-stick-nav.json

uv run fmri-play --subject $SUBJECT --ses $SES --run 2 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/curricula/gym/carracing-GMPD3-stick-nav.json

uv run fmri-play --subject $SUBJECT --ses $SES --run 3 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/curricula/ale/mspacman-GMPD3-stick-nav.json

uv run fmri-play --subject $SUBJECT --ses $SES --run 4 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/curricula/ale/spaceinvaders-GMPD3-stick-nav.json

uv run fmri-play --subject $SUBJECT --ses $SES --run 5 --rig "$RIG" --duration $RUN_DURATION --loop --resume --curriculum configs/curricula/retro/airstriker-GMPD3-stick-nav.json

uv run fmri-play --subject $SUBJECT --ses $SES --run 6 --rig "$RIG" --duration 600 --loop --resume --curriculum configs/curricula/minihack/practice-GMPD3-stick-nav.json
