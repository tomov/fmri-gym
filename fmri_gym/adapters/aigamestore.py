"""AI GameStore adapter -- p5.js browser games (https://aigamestore.org).

AI GameStore is a benchmark of LLM-generated browser games (plain HTML + JS +
p5.js) built to compare frontier VLMs against human players. There is no
Gymnasium env behind it: the *page* is the environment and the *keyboard* is the
action space, for models as much as for people. The official VLM harness
(github.com/lance-ying/aigamestore_harness) drives the same page with Playwright,
sends the model canvas screenshots plus the page's own controls text, and has it
press ARROWS / SPACE / ENTER / R. This adapter is the human-side twin of that,
so a subject in the scanner sees the pixels a model would see.

Per step it holds exactly the keys the keymap names, screenshots the <canvas>
for the display, and reads ``window.getGameState()`` -- every game exposes one --
for the score (reward is its delta) and the ``gamePhase`` (START / PLAYING /
LEVEL_COMPLETE / GAME_OVER_WIN / ...), which gives termination and the variables
to log. Game over does not end the episode: the episode lasts until the game
goes back to its START screen, which the next ``reset`` then leaves. The vendored
games never do that on their own -- level clears and game overs move on after 3 s
straight into PLAYING, with no key -- so in practice a block is one episode and a
reload never throws away the subject's progress.

Keys are the one thing NOT passed through. A game hard-codes its keyCodes in
p5's ``keyPressed``, but a subject in the scanner holds a button box, so combo
VALUES here are the keys the *game* wants and combo keys are whatever the
subject actually presses: ``"keys": {"B1": "LEFT", "B3": "SPACE"}``. The games
also PAINT their control hints onto the canvas ("PRESS SPACE FOR NEXT LEVEL").
The vendored games relabel those from ``label_<KEY>`` query parameters, which
``_make`` fills from ``keys`` (if two keys stand in for one game key, the last
one names it); for a game loaded from a URL, the message phase should say which
button stands in for which key.

Not supported: seeding (the games draw from ``Math.random`` with no seed hook,
so ``reset`` reloads the page but ``episode_seeds`` does not determine the
episode) and savestates (``blob`` is always ``None``).

Requires ``pip install playwright pillow`` and a browser -- the system Chrome by
default; set ``"browser_channel": null`` for Playwright's bundled Chromium.

Phase fields: ``game`` ("game1".."game10", a dir under vendor/aigamestore/, or
an http(s):// URL to an index.html), ``games_dir``, ``headed``,
``browser_channel``, ``start_key`` (tapped on reset to leave START, default
"RETURN"), ``load_ms`` (settle time after each reload, default 2000).
"""

from __future__ import annotations

import contextlib
import functools
import http.server
import io
import os
import socketserver
import threading
import urllib.parse
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .keyspec import PassthroughKeySpec
from .base import EnvAdapter, FrameState

if TYPE_CHECKING:
    from playwright.sync_api import Browser, Page, Playwright

# Keys the vendored games listen for, as Playwright key names. Combo values are
# looked up here, so this doubles as the vocabulary a curriculum keymap may
# point a physical key at.
_GAME_KEYS: dict[str, str] = {
    "LEFT": "ArrowLeft", "RIGHT": "ArrowRight", "UP": "ArrowUp", "DOWN": "ArrowDown",
    "W": "w", "A": "a", "S": "s", "D": "d",
    "SPACE": " ", "Z": "z", "X": "x", "RETURN": "Enter", "LSHIFT": "Shift",
    "R": "r", "ESCAPE": "Escape",
    "1": "1", "2": "2", "3": "3", "4": "4", "5": "5",
}

_VENDOR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "vendor", "aigamestore")


@dataclass
class _Browser:
    """The Playwright objects and local server backing one game block."""

    playwright: Playwright
    browser: Browser
    page: Page
    server: socketserver.TCPServer
    held: set[str] = field(default_factory=set)
    score: float = 0.0
    phase: str = ""


class AIGameStoreAdapter(EnvAdapter):
    name: str = "aigamestore"

    def _make(self, spec: dict) -> _Browser:
        """Serve the games directory and open the game in a Playwright browser.

        :param spec: game-phase config dict from the curriculum.
        :return: the browser session, stored as ``self.env``.
        """
        from playwright.sync_api import sync_playwright

        server, base_url = _serve(spec.get("games_dir", _VENDOR))
        game = spec["game"]
        url = game if game.startswith("http") else f"{base_url}/{game}/index.html"
        labels = {f"label_{want}": pressed for pressed, want in spec.get("keys", {}).items()
                  if pressed != want}
        if labels:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(labels)

        playwright = sync_playwright().start()
        channel = spec.get("browser_channel", "chrome")
        browser = playwright.chromium.launch(
            headless=not spec.get("headed", False),
            **({"channel": channel} if channel else {}))
        page = browser.new_page(viewport={"width": 900, "height": 620})
        page.goto(url)
        return _Browser(playwright, browser, page, server)

    def _keyspec(self) -> PassthroughKeySpec:
        """Return the identity keymap over every key the games understand.

        :return: a :class:`PassthroughKeySpec` whose values are game keys, so a
            curriculum keymap can bind any of them to any physical key.
        """
        return PassthroughKeySpec(combos={frozenset([k]): k for k in _GAME_KEYS},
                                  noop="")

    def reset(self, seed: int | None) -> tuple[Any, dict]:
        """Reload the page for a fresh episode and leave the START screen.

        :param seed: ignored -- the games seed themselves from ``Math.random``.
        :return: ``(None, info)`` with the game's state dict in ``info``.
        """
        page = self.env.page
        page.reload()
        page.wait_for_selector("canvas")
        page.wait_for_timeout(self.spec.get("load_ms", 2000))
        page.click("canvas")            # focus, so key events reach the game
        start_key = self.spec.get("start_key", "RETURN")
        _key(page, start_key, down=True)
        _key(page, start_key, down=False)

        state = _state(page)
        self.env.held = set()
        self.env.score = _score(state, 0.0)
        self.env.phase = _phase(state, "START")
        return None, {"state": state}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        """Hold exactly the keys ``action`` names, then read the game's state.

        :param action: "+"-joined game keys from the keymap ("" = nothing held).
        :return: ``(None, score delta, episode over, False, {"state": ...})``;
            see :func:`_episode_over`.
        """
        page = self.env.page
        want = set(action.split("+")) if action else set()
        # Edge-triggered, so the page sees real keydown/keyup pairs.
        for name in self.env.held - want:
            _key(page, name, down=False)
        for name in want - self.env.held:
            _key(page, name, down=True)
        self.env.held = want
        page.wait_for_timeout(1)        # yield to the browser's own frame loop

        state = _state(page)
        score = _score(state, self.env.score)
        reward, self.env.score = score - self.env.score, score
        phase = _phase(state, self.env.phase)
        done = _episode_over(self.env.phase, phase)
        self.env.phase = phase
        return None, reward, done, False, {"state": state}

    def render(self) -> np.ndarray:
        """Return the game canvas as an RGB frame.

        These are the same pixels the VLM harness screenshots and sends to the
        model, so subject and model see the one stimulus.

        :return: RGB frame ``(H, W, 3)`` uint8.
        """
        from PIL import Image

        canvas = self.env.page.query_selector("canvas")
        png = canvas.screenshot() if canvas else self.env.page.screenshot()
        return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> FrameState:
        """Log the scalar fields of the game's own state dict.

        Scalars only (score, level, gamePhase, lives, ...): the boards and
        entity lists some games also expose have no fixed shape to log.

        :param obs: unused -- the canvas is the observation.
        :param info: info dict from the latest :meth:`step` / :meth:`reset`.
        :param want_blob: unused -- a browser game has no savestate.
        :return: a :class:`FrameState` with one ``state_*`` variable per scalar.
        """
        state = (info or {}).get("state") or {}
        return FrameState(blob=None, variables={
            f"state_{name}": value for name, value in state.items()
            if isinstance(value, (int, float, bool, str))})

    def close(self) -> None:
        """Tear down browser, Playwright, and the local server.

        Each is attempted even if an earlier one raises, so a wedged browser
        cannot leak the server thread and its port for the rest of the session.
        """
        for shutdown in (self.env.browser.close, self.env.playwright.stop,
                         self.env.server.shutdown):
            with contextlib.suppress(Exception):
                shutdown()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """``SimpleHTTPRequestHandler`` without the per-request stderr logging."""

    def log_message(self, *args: Any) -> None:
        """Drop the request log line; nothing should print in the frame loop."""


def _serve(directory: str) -> tuple[socketserver.TCPServer, str]:
    """Serve ``directory`` over HTTP on a free local port, on a daemon thread.

    The games are ES modules, which browsers refuse to load over ``file://``.

    :param directory: the games directory to serve.
    :return: ``(server, base_url)``; the caller shuts the server down.
    """
    handler = functools.partial(_QuietHandler, directory=directory)
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _key(page: Page, name: str, down: bool) -> None:
    """Press or release one game key in the page.

    :param page: the page running the game.
    :param name: a game key name; one missing from :data:`_GAME_KEYS` is
        ignored, so a stray curriculum entry costs that key, not the block.
    :param down: ``True`` to press, ``False`` to release.
    """
    playwright_key = _GAME_KEYS.get(name)
    if playwright_key:
        (page.keyboard.down if down else page.keyboard.up)(playwright_key)


def _state(page: Page) -> dict | None:
    """Return the game's own ``window.getGameState()`` dict, or ``None``.

    Evaluation errors are swallowed: the page is briefly unreachable while it
    reloads, and one unreadable frame should not end the block.

    :param page: the page running the game.
    :return: the state dict, or ``None`` if it could not be read.
    """
    try:
        state = page.evaluate("() => window.getGameState ? window.getGameState() : null")
    except Exception:
        return None
    return state if isinstance(state, dict) else None


def _phase(state: dict | None, default: str) -> str:
    """Return the ``gamePhase`` in ``state``, or ``default`` if it could not be read.

    Keeping the last phase through an unreadable frame stops a reload blip from
    reading as a phase change.
    """
    return str(state.get("gamePhase", "")) if state else default


def _episode_over(before: str, after: str) -> bool:
    """Whether the step from phase ``before`` to ``after`` ends the episode.

    :param before: the phase at the previous step (or reset).
    :param after: the phase at this step.
    :return: ``True`` when the game returns to START, or ends.
    """
    return (after == "START" and before != "START") or after == "ENDED"


def _score(state: dict | None, default: float) -> float:
    """Return the score in ``state``, or ``default`` if it has no usable one.

    Falling back to the last score (rather than 0) keeps an unreadable frame
    reading as "no change" instead of a large negative reward.

    :param state: a state dict from :func:`_state`, or ``None``.
    :param default: the score to keep when this frame has no usable one.
    :return: the current score.
    """
    try:
        return float((state or {}).get("score"))
    except (TypeError, ValueError):
        return default
