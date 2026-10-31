"""AI GameStore games as a Gymnasium environment.

AI GameStore (https://aigamestore.org) is a benchmark of ten LLM-generated
browser games -- plain HTML + JavaScript, nine on p5.js and one on three.js --
built to compare vision-language models with human players. The paper's own
harness drives the page in real time with Playwright: it pauses the game while
the model thinks, then holds keys for 0.2 s segments on the wall clock. That
has no ``step()`` and no seed, so it cannot be trained against or replayed.

This env keeps the browser (the games are the browser) but takes over its
clock. ``lockstep.js`` runs in the page before the game and replaces
``requestAnimationFrame``, ``setTimeout``, ``performance.now``, ``Date`` and
``Math.random``; one :meth:`step` then dispatches key events, advances the game
by exactly ``frame_skip`` of its 60 Hz frames and returns the canvas pixels and
the game's own ``getGameState()``. Time does not pass between steps, so a model
that deliberates for a minute and a human at the keyboard meet the same game,
and ``reset(seed=...)`` plus the action sequence replays an episode exactly.

Observation: the game canvas as an RGB ``Box`` (each game has its own size,
e.g. 600x400). Action: ``MultiBinary(len(keys))`` -- which of the game's keys are
held this step; see :data:`GAME_KEYS`. Reward: the change in the game's
``score``. ``info["state"]`` is the game's full state, exactly as
``getGameState()`` returns it (``gamePhase``, ``score``, ``currentLevel``, and
every other field the game exposes, however nested).

An episode is one level: ``reset`` starts ``level`` afresh (score 0, full
lives) through the game's ``window.loadLevel(n)`` hook, and the episode is over
as soon as the game leaves PLAYING (a win, a loss or a level-complete screen)
or its level counter moves -- the games differ in which of these they do when a
level is cleared, and this rule covers all of them. The game is then stepped
``end_steps`` more times with no key held, so the win / lose / level-complete
screen stays up long enough to be read (2 s at the default rate) instead of
flashing for one frame; ``terminated`` is reported on the last of them. A game
without levels (game4, an endless runner) has episodes that end when it is lost.

Not supported: sound (the games have none) and savestates (replay from seed).
Needs a Chromium-based browser (the system Chrome by default; pass
``browser_channel=None`` for Playwright's bundled Chromium after
``playwright install chromium``).

The page is self-contained. The games' HTML loads p5, three.js, Matter and
seedrandom from CDNs and an analytics tag from Google; the env answers the
library URLs with the copies under ``vendor/`` (:data:`VENDORED`), drops the
analytics tag, and refuses any other request that leaves the page's origin --
a script that arrived from the network would draw from the seeded
``Math.random`` and schedule timers of its own, and a library the CDN serves
differently some day would change the game. Either breaks replay without a
sign, so the step after such a request raises instead.
"""

from __future__ import annotations

import base64
import contextlib
import functools
import http.server
import io
import re
import socketserver
import threading
import urllib.parse
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

_HERE = Path(__file__).resolve().parent
_GAMES_DIR = _HERE.parent          # game1/ .. game10/ sit beside this package
_LOCKSTEP_JS = _HERE / "lockstep.js"
_VENDOR_DIR = _GAMES_DIR / "vendor"

#: The libraries the games' pages fetch, and the file under ``vendor/`` served
#: for each (pinned by version in the URL; the game modules import three.js by
#: this URL too).
VENDORED: dict[str, str] = {
    "https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.4.0/p5.js": "p5-1.4.0.js",
    "https://unpkg.com/p5.collide2d@0.7.3/p5.collide2d.js": "p5.collide2d-0.7.3.js",
    "https://cdnjs.cloudflare.com/ajax/libs/matter-js/0.19.0/matter.min.js": "matter-0.19.0.min.js",
    "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js": "three-0.160.0.module.js",
    "https://cdn.jsdelivr.net/npm/seedrandom@3.0.5/seedrandom.min.js": "seedrandom-3.0.5.min.js",
}
#: The analytics tag every page carries; dropped, never fetched.
_ANALYTICS = "https://www.googletagmanager.com/"

#: The keys each vendored game listens for during play (its ``input.js``);
#: this is the env's action vocabulary. Enter / Escape / R start, pause and
#: restart and are the env's business, not the agent's.
GAME_KEYS: dict[str, list[str]] = {
    "game1": ["LEFT", "RIGHT", "SPACE", "Z"],
    "game2": ["LEFT", "RIGHT", "UP", "DOWN", "SPACE", "Z"],
    "game3": ["W", "A", "S", "D", "LEFT", "RIGHT", "UP", "DOWN", "SPACE", "Z", "LSHIFT"],
    "game4": ["SPACE", "UP"],
    "game5": ["LEFT", "RIGHT", "UP"],
    "game6": ["LEFT", "RIGHT", "UP", "SPACE", "Z"],
    "game7": ["LEFT", "RIGHT", "UP", "DOWN", "Z", "SPACE", "LSHIFT", "1", "2", "3", "4", "5", "X"],
    "game8": ["LEFT", "RIGHT", "SPACE", "Z"],
    "game9": ["LEFT", "RIGHT", "UP", "DOWN", "Z", "SPACE", "LSHIFT"],
    "game10": ["LEFT", "RIGHT", "UP", "DOWN", "LSHIFT", "SPACE", "Z"],
}

#: How many levels each vendored game has that can be played on their own:
#: ``0`` for no levels at all (episodes end when the game is lost), ``None``
#: for procedurally generated ones without an upper bound. game6 is one long
#: level with checkpoints; game9's 8th level is a gauntlet of the bosses beaten
#: so far, which is empty (an instant win) when started fresh, so it is left out.
GAME_LEVELS: dict[str, int | None] = {
    "game1": 9, "game2": 9, "game3": 7, "game4": 0, "game5": 6,
    "game6": 1, "game7": None, "game8": 9, "game9": 7, "game10": 6,
}

# Headless Chrome falls back to SwiftShader (software GL) unless told to use
# the GPU; the three.js game then takes ~130 ms a frame instead of ~2 ms.
_BROWSER_ARGS = ["--use-gl=angle", "--use-angle=gl", "--ignore-gpu-blocklist"]
_BOOT_FRAMES = 600                 # 10 s of game time for p5 to set up


class AIGameStoreEnv(gym.Env):
    """One AI GameStore game in a lock-stepped browser page.

    :param game: ``"game1"`` .. ``"game10"`` (served from ``games_dir``), or the
        ``http(s)://`` URL of a game's ``index.html``.
    :param level: the level each episode plays, from 1; see :data:`GAME_LEVELS`.
        Defaults to 1 for a game with levels, and must be ``None`` for one
        without (game4 starts from its START screen instead).
    :param keys: the game's keys, in action-vector order. Defaults to
        :data:`GAME_KEYS`; required for a game given by URL.
    :param frame_skip: game frames (at 60 Hz) per :meth:`step`.
    :param end_steps: steps the end-of-level screen stays up before the episode
        is reported ``terminated`` (20 = 2 s at the default 10 steps/s).
    :param headless: run the browser without a window.
    :param browser_channel: Playwright browser channel (``"chrome"``), or
        ``None`` for the bundled Chromium.
    :param key_labels: ``{game key: label}`` for the on-canvas control hints, for
        a host whose player presses something other than the named key.
    :param games_dir: directory holding ``game1/`` .. ``game10/``.
    :param start_key: tapped on reset to leave the START screen.
    :param render_mode: only ``"rgb_array"``.
    """

    metadata: ClassVar[dict] = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        game: str = "game1",
        level: int | None = None,
        *,
        keys: list[str] | None = None,
        frame_skip: int = 6,
        end_steps: int = 20,
        headless: bool = True,
        browser_channel: str | None = "chrome",
        key_labels: dict[str, str] | None = None,
        games_dir: str | Path | None = None,
        start_key: str = "RETURN",
        render_mode: str | None = "rgb_array",
    ) -> None:
        from playwright.sync_api import sync_playwright

        if keys is None and game not in GAME_KEYS:
            raise ValueError(f"{game!r} is not one of {sorted(GAME_KEYS)}; pass keys=[...] for it")
        self.game = game
        self.level = _check_level(game, level)
        self.keys = list(GAME_KEYS[game] if keys is None else keys)
        self.frame_skip = int(frame_skip)
        self.end_steps = int(end_steps)
        self.start_key = start_key
        self.render_mode = render_mode
        self.metadata = {**self.metadata, "render_fps": 60 / self.frame_skip}

        self._server, self._url = None, game
        if not game.startswith(("http://", "https://")):
            self._server, base = _serve(Path(games_dir or _GAMES_DIR))
            self._url = f"{base}/{game}/index.html"
        labels = {f"label_{k}": v for k, v in (key_labels or {}).items()}
        self._url += ("&" if "?" in self._url else "?") + urllib.parse.urlencode(labels)

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=headless, args=_BROWSER_ARGS,
            **({"channel": browser_channel} if browser_channel else {}))
        self._page = self._browser.new_page(viewport={"width": 900, "height": 700})
        self._page.add_init_script(path=str(_LOCKSTEP_JS))
        self._origin = "{0.scheme}://{0.netloc}/".format(urllib.parse.urlsplit(self._url))
        self._left_the_page: list[str] = []
        # Routes match newest first: the page's own files and rng.js go through,
        # everything else is answered here.
        self._page.route("**/*", self._route)
        self._page.route("**/rng.js", self._patch_rng)

        self._frame: np.ndarray | None = None
        self._score = 0.0
        self._ended = 0                # steps taken since the game left the level
        self._load(seed=0)
        unknown = set(self.keys) - set(self._page.evaluate("() => window.__aigs.keys"))
        if unknown:
            self.close()
            raise ValueError(f"keys {sorted(unknown)} are not in lockstep.js's key table")
        frame, _ = self._step([], 0)
        self.observation_space = spaces.Box(0, 255, frame.shape, np.uint8)
        self.action_space = spaces.MultiBinary(len(self.keys))

    # ------------------------------------------------------------------ gym
    def reset(self, *, seed: int | None = None, options: dict | None = None
              ) -> tuple[np.ndarray, dict]:
        """Reload the game with ``seed`` and start :attr:`level` afresh.

        :param seed: seeds the page's ``Math.random``; ``None`` draws one.
        :param options: unused.
        :return: ``(frame, {"state": ...})`` -- the first PLAYING frame.
        :raises RuntimeError: if the game did not come up PLAYING the level.
        """
        super().reset(seed=seed)
        self._load(seed=int(self.np_random.integers(1, 2**31)) if seed is None else seed)
        if self.level is None:
            frame, state = self._step([self.start_key], 1)
        else:
            self._page.evaluate("n => window.loadLevel(n)", self.level)
            frame, state = self._step([], 1)
        if self._over(state):
            self.close()
            raise RuntimeError(f"{self.game}: asked for level {self.level} but the game is in "
                               f"{state.get('gamePhase')!r} at level {_level(state)!r}")

        self._page.evaluate("() => window.__aigs.reseedRandom()")
        self._score = float(state.get("score", 0.0))
        self._ended = 0
        return frame, {"state": state}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Hold the keys ``action`` marks and advance ``frame_skip`` frames.

        Once the game has left PLAYING or changed level the keys are no longer
        passed on (a press could skip the end screen or start the next level),
        and ``terminated`` comes ``end_steps`` steps later.

        :param action: 0/1 per entry of :attr:`keys`.
        :return: ``(frame, score delta, terminated, False, {"state": ...})``.
        """
        names = [] if self._ended else [k for k, on in zip(self.keys, action) if on]
        frame, state = self._step(names, self.frame_skip)
        score = float(state.get("score", self._score))
        reward, self._score = score - self._score, score
        if self._ended or self._over(state):
            self._ended += 1
        return frame, reward, self._ended > self.end_steps, False, {"state": state}

    def render(self) -> np.ndarray:
        """Return the canvas as it stood after the last step, RGB ``(H, W, 3)``."""
        return self._frame

    def close(self) -> None:
        """Tear down the browser, Playwright, and the file server.

        Each is attempted even if an earlier one raises, so a wedged browser
        cannot leak the server thread and its port.
        """
        closers = [self._browser.close, self._playwright.stop]
        if self._server:
            closers.append(self._server.shutdown)
        for shutdown in closers:
            with contextlib.suppress(Exception):
                shutdown()

    # -------------------------------------------------------------- helpers
    def _over(self, state: dict) -> bool:
        """Whether ``state`` is outside the episode: not PLAYING, or on another level.

        A game that keeps no level counter (game6) is judged on its phase alone.
        """
        level = _level(state)
        return state.get("gamePhase") != "PLAYING" or (level is not None and level != self.level)

    def _load(self, seed: int) -> None:
        """Navigate to the game with ``seed`` and tick it up to its START screen."""
        self._seed = seed          # read by _patch_rng, which the coming navigation triggers
        self._page.goto(f"{self._url}&seed={seed}")
        self._page.evaluate("n => window.__aigs.boot(n)", _BOOT_FRAMES)

    def _route(self, route: Any) -> None:
        """Answer a request the page makes: its own files, a vendored library, or nothing."""
        url = route.request.url
        if url.startswith(self._origin):
            route.continue_()
        elif url in VENDORED:
            route.fulfill(path=str(_VENDOR_DIR / VENDORED[url]))
        else:
            if not url.startswith(_ANALYTICS):
                self._left_the_page.append(url)
            route.abort()

    def _patch_rng(self, route: Any) -> None:
        """Force game7's own vendored RNG module (``rng.js``) onto this episode's seed.
        """
        response = route.fetch()
        body = re.sub(r"export function setSeed\(newSeed\) \{.*?\}",
                       f"export function setSeed(newSeed) {{ seed = {self._seed}; "
                       f"state = {self._seed}; window.__aigs.reseedRandom(); }}",
                       response.text(), count=1, flags=re.DOTALL)
        route.fulfill(response=response, body=body)

    def _step(self, names: list[str], frames: int) -> tuple[np.ndarray, dict]:
        """Hold exactly ``names``, tick ``frames``, and read canvas + state.

        :raises RuntimeError: the page asked for a file off its origin that is
            not in :data:`VENDORED` (vendor it, or the game is not replayable).
        """
        if self._left_the_page:
            raise RuntimeError(f"{self.game} requested {self._left_the_page} from outside the "
                               "page; add the file to vendor/ and VENDORED, or the game "
                               "depends on the network and cannot be replayed")
        out = self._page.evaluate("([n, f]) => window.__aigs.step(n, f)", [names, frames])
        self._frame = _decode_png(out["png"])
        return self._frame, out["state"]


def _check_level(game: str, level: int | None) -> int | None:
    """Return the level to play, or raise if ``level`` does not fit ``game``.

    A game given by URL is not in :data:`GAME_LEVELS`; any level goes.
    """
    if game not in GAME_LEVELS:
        return level
    n = GAME_LEVELS[game]
    if n == 0:
        if level is not None:
            raise ValueError(f"{game} has no levels; leave level=None")
        return None
    level = 1 if level is None else int(level)
    if level < 1 or (n is not None and level > n):
        raise ValueError(f"{game} has levels 1..{n if n else 'inf'}, not {level}")
    return level


def _level(state: dict) -> int | None:
    """The level counter of a game state (``currentLevel`` or, in some games, ``level``)."""
    level = state.get("currentLevel", state.get("level"))
    return None if level is None else int(level)


def _decode_png(data_url: str) -> np.ndarray:
    """RGB array from a ``data:image/png;base64,...`` URL."""
    from PIL import Image

    png = base64.b64decode(data_url.split(",", 1)[1])
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """``SimpleHTTPRequestHandler`` without the per-request stderr line."""

    def log_message(self, *args: Any) -> None:
        """Drop the request log."""


def _serve(directory: Path) -> tuple[socketserver.TCPServer, str]:
    """Serve ``directory`` over HTTP on a free local port, on a daemon thread.

    The games are ES modules, which browsers refuse to load over ``file://``.

    :return: ``(server, base_url)``; the caller shuts the server down.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"games directory {directory} does not exist")
    handler = functools.partial(_QuietHandler, directory=str(directory))
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"
