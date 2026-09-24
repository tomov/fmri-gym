#!/bin/sh
# fmri-gym session: one line per run, in order.
set -e
SES=${1:-$(uv run fmri-ses --subject sub-01)}
uv run fmri-play --curriculum configs/dbp_games/crafter__crafter.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/coom__pitfall.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/minihack__room5x5.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/rushhour__easy.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/baba__make_win.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/vizdoom__defend_center.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/aigamestore__game1.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/coom__run_and_gun.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/minihack__mazewalk9x9.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/aigamestore__game2.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
uv run fmri-play --curriculum configs/dbp_games/coom__chainsaw.json --subject sub-01 --ses "$SES" --run 1 --size 1024x768
