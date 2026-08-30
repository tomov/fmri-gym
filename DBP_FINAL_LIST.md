# DBP final game list — integration status

The games settled on for the DBP study, and how each runs in **fmri-gym**.
All 9 (excluding the `*` no-gymnasium ones) are verified working end-to-end —
each renders a real frame and runs through the framework. (Zork and Stepmania
are skipped: no Gymnasium interface.)

## Results

| Category | Game | Backend / how | Config | Status |
|---|---|---|---|---|
| Action/shooter | **COOM** | `vizdoom` (COOM's exact Doom engine) | `dbp_games/vizdoom__defend_center.json` | ✅ |
| Action/shooter, Puzzle | **AI GameStore** | `aigamestore` (p5.js via headless browser) | `dbp_games/aigamestore__game1.json` … | ✅ |
| Building/open-world | **Crafter** | `crafter` | `dbp_games/crafter__crafter.json` | ✅ |
| Building/open-world | **Craftium** | `gym` + `import_module` (Luanti voxel) | `dbp_games/craftium__choptree.json` | ✅ |
| Puzzle | **Rush Hour** | `rushhour` (Go engine + colored board) | `dbp_games/rushhour__easy.json` | ✅ |
| Language | **Baba Is You** | `baba` (baba-is-ai) | `dbp_games/baba__make_win.json` | ✅ |
| Adventure | **MiniHack** | `minihack` | `dbp_games/minihack__room5x5.json` | ✅ |
| Sports/racing | **SuperTuxKart** | `supertuxkart` (pystk2, 3D) | `dbp_games/supertuxkart__race.json` | ✅ |
| Social | **Overcooked** | `overcooked` (overcooked_ai) | `dbp_games/overcooked__cramped_room.json` | ✅ |
| Interactive fiction | Zork* | — (no Gymnasium) | — | ⏭️ skipped |
| Motor/music | Stepmania* | — (no Gymnasium) | — | ⏭️ skipped |

Each was verified to render a real frame (PNG spot-checks for the SuperTuxKart
3D view, the Rush Hour board, and Craftium) and to run end-to-end.

## The hard calls / honest notes

1. **COOM → ViZDoom.** Real COOM pins `gymnasium==0.28`, which breaks
   MiniHack/NetHack/retro (need 1.2). ViZDoom is the *identical Doom engine*
   COOM is built on, works with our gymnasium, and provides the action-shooter
   scenarios — so we use it. If COOM's specific continual-learning scenario
   WADs are ever needed, that's a separate `gymnasium==0.28` env.

2. **SuperTuxKart needs a real GL display.** pystk2-gymnasium is state-only
   (no pixels), so we drive `pystk2` directly for the 3D render — but Irrlicht
   needs a real GL context (works on `DISPLAY=:1`, **not** under headless
   `SDL_VIDEODRIVER=dummy`). Fine for the fMRI presentation machine.

3. **Rush Hour needs a Go build.** Installed Go (conda-forge), built its
   `rushhour-env` binary; the adapter auto-finds `vendor/rush-hour-src/rushhour-env`
   (or `RUSHHOUR_ENV_BIN`). The Go source is gitignored (build-from-source dep;
   the config `_note` has the clone+build commands).

4. **Craftium** pins `gymnasium 0.29` but runs fine on 1.2, so it stays in the
   one shared env via the `gym` backend + `import_module=craftium` (no new
   adapter). Installed from the prebuilt wheel on the mikelma/craftium releases.

5. **`vizdoom` swapped `pygame` → `pygame-ce`** (a drop-in replacement) —
   verified the display and all other backends still work.

6. **Slow-start backends.** aigamestore (browser launch) and SuperTuxKart
   (engine init) take several seconds to start; short `timeout`-killed test runs
   can miss the block save even though they work.

## Dependency reality

Everything above coexists in **one** `fmri-gym` conda env (gymnasium 1.2,
numpy 1.26, setuptools<81; `vizdoom` brings pygame-ce). The only true conflicts
were COOM (gymnasium 0.28) and, nominally, craftium (0.29 pin — but runs on
1.2). Non-PyPI installs: `baba` (GitHub), `rushhour` (GitHub + Go build),
`craftium` (release wheel). See `requirements.txt` for the full list and the
per-config `_note` fields for game-specific setup.
