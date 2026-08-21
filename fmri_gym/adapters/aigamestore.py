"""AI GameStore adapter -- p5.js / browser games (https://aigamestore.org).

AI GameStore is a benchmark of LLM-generated **browser games** (plain HTML +
JavaScript + p5.js). There is no standard p5.js<->Gymnasium bridge, so this
adapter builds one with a headless (or headed) browser via Playwright:

  * a tiny local HTTP server serves the vendored games (ES modules need http://,
    not file://);
  * per step, currently-held keyboard keys are pressed/released in the page;
  * the game's <canvas> is screenshotted -> the RGB frame for display;
  * each game exposes `window.getGameState()` returning a state object with a
    `score` and a `gamePhase` (START / PLAYING / GAMEOVER / ...), which we map to
    reward (score delta) and done. State is logged as an analysis variable.

This mirrors AI GameStore's own harness (pause -> read state -> act -> resume),
adapted to the fMRI real-time loop. Games are keyboard-driven, so -- like the
retro backend -- the "action" is the *set* of held keys, not a single index.

Requires: `pip install playwright` and a browser. Uses the system Chrome by
default (channel="chrome"); set phase "browser_channel": null to use Playwright's
bundled Chromium (`playwright install chromium`).

Phase fields:
    game        : "game1" ... "game10" (a vendored dir under vendor/aigamestore/)
                  or an absolute http(s):// URL to an index.html
    games_dir   : override the vendored games directory (default: repo vendor/)
    headed      : true to show the browser window (default headless)
    browser_channel : "chrome" (default) | "msedge" | null (bundled chromium)
    start_key   : key pressed once on reset to leave the START screen (default "Enter")
"""

from __future__ import annotations

import os
import threading
import functools
import http.server
import socketserver

import numpy as np

from .base import EnvAdapter, FrameState, KeySpec

# Physical keys we forward to the browser, and their Playwright key names.
_KEY_TO_PLAYWRIGHT = {
    "LEFT": "ArrowLeft", "RIGHT": "ArrowRight", "UP": "ArrowUp", "DOWN": "ArrowDown",
    "SPACE": " ", "Z": "z", "X": "x", "RETURN": "Enter", "LSHIFT": "Shift",
    "R": "r", "ESCAPE": "Escape",
}
_VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "vendor", "aigamestore")


class _Session:
    """Holds the Playwright browser/page + local server for one game block."""

    def __init__(self, playwright, browser, page, server, held):
        self.pw = playwright
        self.browser = browser
        self.page = page
        self.server = server
        self.held = held           # currently-pressed keys (set of key names)
        self.prev_score = 0.0


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


class AIGameStoreAdapter(EnvAdapter):
    name = "aigamestore"

    def make(self, spec):
        from playwright.sync_api import sync_playwright

        games_dir = spec.get("games_dir", _VENDOR)
        game = spec["game"]
        # Serve the games dir over HTTP (ES modules are blocked over file://).
        handler = functools.partial(_QuietHandler, directory=games_dir)
        server = socketserver.TCPServer(("127.0.0.1", 0), handler)
        server.allow_reuse_address = True
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()

        if game.startswith("http://") or game.startswith("https://"):
            url = game
        else:
            url = f"http://127.0.0.1:{port}/{game}/index.html"

        pw = sync_playwright().start()
        channel = spec.get("browser_channel", "chrome")
        launch = {"headless": not spec.get("headed", False)}
        if channel:
            launch["channel"] = channel
        browser = pw.chromium.launch(**launch)
        page = browser.new_page(viewport={"width": 900, "height": 620})
        page.goto(url)
        page.wait_for_timeout(spec.get("load_ms", 2200))
        # Focus the canvas so keyboard events reach the game.
        cv = page.query_selector("canvas")
        if cv:
            try:
                cv.click(timeout=1000)
            except Exception:
                pass
        self._start_key = spec.get("start_key", "Enter")
        return _Session(pw, browser, page, server, set())

    def keymap(self, env) -> KeySpec:
        combos = {frozenset([k]): k for k in _KEY_TO_PLAYWRIGHT}
        ks = KeySpec(combos=combos, noop=frozenset(),
                     help="keys are forwarded to the browser (hold several at once); "
                          "exact actions vary per game",
                     controls=[("ARROWS", "move / navigate"),
                               ("SPACE", "primary action / select"),
                               ("Z", "secondary action"),
                               ("X / SHIFT", "extra actions"),
                               ("ENTER", "start / confirm"),
                               ("R", "restart")])
        # Override resolve to return the FULL set of held keys as a stable
        # "+"-joined string (keyboard games use simultaneous keys; a string logs
        # cleanly to npz, unlike a frozenset). "" == no keys held.
        known = set(_KEY_TO_PLAYWRIGHT)
        ks.resolve = lambda held: "+".join(sorted(held & known))
        return ks

    def reset(self, env, seed, spec):
        page = env.page
        # Try to (re)start a fresh episode: reload keeps things deterministic-ish.
        page.reload()
        page.wait_for_timeout(spec.get("load_ms", 2000))
        cv = page.query_selector("canvas")
        if cv:
            try:
                cv.click(timeout=1000)
            except Exception:
                pass
        # Leave the START screen.
        page.keyboard.press(self._start_key)
        page.wait_for_timeout(200)
        env.held = set()
        env.prev_score = _score(page)
        return None, {}

    def step(self, env, action):
        page = env.page
        # action is a "+"-joined key string (from resolve) or any iterable.
        if isinstance(action, str):
            want = set(action.split("+")) if action else set()
        else:
            want = set(action or ())
        # Release keys no longer held; press newly held keys (edge-triggered so
        # the browser sees keydown/keyup like a real keyboard).
        for k in env.held - want:
            _key(page, k, down=False)
        for k in want - env.held:
            _key(page, k, down=True)
        env.held = want
        # Let the game advance for roughly one frame.
        page.wait_for_timeout(1)
        st = _state(page)
        score = float(st.get("score", 0.0) or 0.0) if isinstance(st, dict) else 0.0
        reward = score - env.prev_score
        env.prev_score = score
        phase = (st.get("gamePhase") or st.get("phase") or "") if isinstance(st, dict) else ""
        done = str(phase).upper() in ("GAMEOVER", "GAME_OVER", "WIN", "WON", "COMPLETE")
        env._last_state = st
        return None, reward, done, False, {"state": st}

    def render(self, env):
        cv = env.page.query_selector("canvas")
        png = cv.screenshot() if cv else env.page.screenshot()
        return _png_to_rgb(png)

    def capture(self, env, obs, info, want_blob=True) -> FrameState:
        st = (info or {}).get("state")
        variables = {}
        if isinstance(st, dict):
            # Log compact scalar fields as analysis regressors (skip big nested
            # structures like full board/tube arrays).
            for k, v in st.items():
                if isinstance(v, (int, float, bool, str)):
                    variables[f"state_{k}"] = v
        return FrameState(blob=None, variables=variables)

    def close(self, env) -> None:
        try:
            env.browser.close()
        finally:
            try:
                env.pw.stop()
            finally:
                env.server.shutdown()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _key(page, name, down):
    pw_key = _KEY_TO_PLAYWRIGHT.get(name)
    if not pw_key:
        return
    (page.keyboard.down if down else page.keyboard.up)(pw_key)


def _state(page):
    try:
        return page.evaluate("window.getGameState ? window.getGameState() : null")
    except Exception:
        return None


def _score(page):
    st = _state(page)
    return float(st.get("score", 0.0) or 0.0) if isinstance(st, dict) else 0.0


def _png_to_rgb(png_bytes):
    import io
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        return np.asarray(img)
    except Exception:
        # Fallback via pygame if Pillow is unavailable.
        import pygame
        surf = pygame.image.load(io.BytesIO(png_bytes))
        return pygame.surfarray.array3d(surf).transpose(1, 0, 2)
