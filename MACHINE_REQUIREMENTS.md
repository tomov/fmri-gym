# Machine requirements

These are deployment recommendations for running one fmri-gym game at a time on a
dedicated experiment computer. The minimum column is a starting point for testing a
candidate machine; it has not been certified by benchmarking low-end hardware.

The scope is the 22 game/scenario entries in the
[local test log](docs/local-testing/2026-09-14.md), targeting native Ubuntu 22.04 LTS
(64-bit), a 1024 x 768 game window, and the existing curriculum settings. Other
backends, higher resolutions, and higher update rates need separate assessment.

## Minimum and recommended configuration

| Component | Minimum deployment recommendation | Recommended configuration |
|---|---|---|
| CPU | Dual-core x86-64 processor with SSE3 | Quad-core x86-64 processor with SSE3 and comparable or better single-core performance |
| System memory | **4 GB RAM** | **8 GB RAM** |
| Graphics | Integrated or discrete graphics providing **OpenGL 3.3 or later and working WebGL**; reference models below | The same API requirements, with a stable driver; **a discrete GPU is not required** |
| Available graphics memory | **512 MB** | **1 GB** |
| Storage | Space for the selected environments, their assets, and experiment logs | SSD, with capacity planned for retained experiment data |
| Operating system | Native Ubuntu 22.04 LTS, 64-bit, with a working graphical desktop | The same OS and dependency versions validated on the experiment computer |

Choose one compatible graphics device. The following models are the examples listed
in the visualization requirements for
[PySuperTuxKart2 0.7.4](https://pypi.org/project/PySuperTuxKart2/0.7.4/#description):

| Vendor | Upstream minimum reference model | Type |
|---|---|---|
| AMD | Radeon HD 6870 | Discrete graphics |
| Intel | HD Graphics 4000 | Integrated graphics |
| NVIDIA | GeForce GTX 470 | Discrete graphics |

These examples are not equal-performance tiers or the absolute lowest models that
could work. A newer model still needs a suitable driver. Check that the actual Linux
driver exposes OpenGL 3.3 or later and that Chrome can create a WebGL context.
The CPU does not need integrated graphics if another compatible graphics device is
available.

Integrated graphics may use shared system memory. The graphics-memory figures do
not require dedicated VRAM or a fixed BIOS reservation; shared graphics allocations
also consume part of system RAM. The recommended 8 GB RAM and 1 GB graphics memory
provide operating headroom, rather than describing measured game usage.

## Which games need 3D graphics?

| Games in the assessed set | Application rendering requirement |
|---|---|
| AI GameStore game3: Floor Sweep / KIA | WebGL for the Three.js scene; Canvas 2D for the HUD |
| SuperTuxKart | OpenGL 3.3 or later |
| AI GameStore games 1, 2, and 4-10 | Canvas 2D; no application-level WebGL requirement |
| ViZDoom: Defend the Center and Deadly Corridor | Software rendering |
| Crafter, Baba make_win, and Rush Hour easy | Software-generated RGB frames |
| MiniHack: Room 5x5, Room 15x15, MazeWalk 9x9, River, Corridor, and Eat | Tile-based RGB frames |

All games need a working display. Canvas 2D and desktop composition can still use
GPU acceleration; the table describes the game rendering paths, not a test with all
GPU acceleration disabled. The 20 entries without a 3D API requirement do not inherit
SuperTuxKart's graphics requirement when deployed on their own.

The graphics paths were checked against the adapters and vendored games.
[ViZDoom documents its software renderer](https://vizdoom.farama.org/1.3.0.dev1/faq/).
The browser games use
[p5.js 1.4.0's default 2D renderer](https://github.com/processing/p5.js/blob/v1.4.0/src/core/rendering.js),
except game3, whose pinned
[Three.js r160 renderer](https://github.com/mrdoob/three.js/blob/r160/src/renderers/WebGLRenderer.js)
tries WebGL 2 and then WebGL 1. CPU-based OpenGL/WebGL fallback was not validated.

## Basis and performance target

The graphics models, OpenGL requirement, and 512 MB graphics-memory starting point
come from PySuperTuxKart2's upstream requirements, not a low-end fmri-gym benchmark.
[Chrome's Linux requirements](https://support.google.com/chrome/answer/16737616?hl=en)
include a 64-bit system and SSE3. The 4 GB system-memory starting point also considers
the [Ubuntu community desktop guidance](https://help.ubuntu.com/community/Installation/SystemRequirements);
it is specific to the deployment scope above, not every Ubuntu release.

The local measurements target the project's configured rates: 10 updates/s for AI
GameStore, approximately 35 updates/s for ViZDoom, and 30 updates/s for SuperTuxKart
at SD quality with a 600 x 400 internal image and three karts. Turn-based games advance
on input. These targets do not promise 60 FPS or measure physical display refresh.

The tested workstation had substantially more resources than either configuration.
Observed game process memory did not justify requiring 8 GB for the game alone, but
process RSS does not include the complete desktop, OS, or shared graphics allocation.
Confirm responsiveness and memory pressure on the intended experiment computer.
The browser results for games 3-10 also depend on a local state-transfer fix recorded
in the [test log](docs/local-testing/2026-09-14.md#tested-software-and-local-changes).
