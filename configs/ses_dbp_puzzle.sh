#!/bin/sh
# fmri-gym session: one line per run, in order.
set -e
SES=${1:-$(uv run fmri-ses --subject sub-01)}
uv run fmri-play --curriculum configs/dbp_games/aigamestore__game1.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/aigamestore__game2.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/aigamestore__game3.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/aigamestore__game4.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/aigamestore__game5.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/aigamestore__game6.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/aigamestore__game7.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/aigamestore__game8.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/aigamestore__game9.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/rushhour__easy.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/baba__make_win.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
