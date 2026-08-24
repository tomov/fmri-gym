"""SuperTuxKart adapter (pystk2) -- the DBP "sports/racing" pick.

pystk2-gymnasium exposes only *state features* (no pixel render), so we drive
the underlying pystk2 engine directly: it renders the 3D scene to an image
(`race.render_data[0].image`) which we display, and we translate held keys into
a pystk2.Action (steer / accelerate / brake / drift / fire / nitro).

IMPORTANT: SuperTuxKart's renderer (Irrlicht) needs a real GL context -- it does
NOT work under SDL_VIDEODRIVER=dummy. Run it on a real display (or Xvfb with
GLX). The fMRI presentation machine has a display, so this is fine there;
headless CI without GL cannot render it.

Controls: LEFT/RIGHT steer, UP accelerate, DOWN brake, SPACE fire item,
Z drift, X nitro. Reward = distance progress per step; kart finish -> done.
"""

from __future__ import annotations

import numpy as np

from .base import EnvAdapter, FrameState, KeySpec

_STARTED = {"init": False}


class SuperTuxKartAdapter(EnvAdapter):
    name = "supertuxkart"

    def make(self, spec):
        import pystk2
        self._pystk2 = pystk2
        w = int(spec.get("width", 600))
        h = int(spec.get("height", 400))
        # pystk2 must be init'd once per process (re-init crashes the engine).
        if not _STARTED["init"]:
            gc = pystk2.GraphicsConfig.sd()
            gc.screen_width, gc.screen_height = w, h
            pystk2.init(gc)
            _STARTED["init"] = True
        cfg = pystk2.RaceConfig(num_kart=int(spec.get("num_kart", 3)),
                                laps=int(spec.get("laps", 3)))
        if spec.get("track"):
            cfg.track = spec["track"]
        cfg.players[0].controller = pystk2.PlayerConfig.Controller.PLAYER_CONTROL
        race = pystk2.Race(cfg)
        race.start()
        race.step()  # first frame
        self._race = race
        self._ws = pystk2.WorldState()
        self._prev_dist = 0.0
        return race

    def keymap(self, env) -> KeySpec:
        # Actions are assembled from the held-key set in step(); the combos here
        # just declare which keys are meaningful (resolve returns the held set).
        keys = ["LEFT", "RIGHT", "UP", "DOWN", "SPACE", "Z", "X"]
        combos = {frozenset([k]): k for k in keys}
        ks = KeySpec(combos=combos, noop=frozenset(),
                     help="hold several keys at once",
                     controls=[("LEFT/RIGHT", "steer"), ("UP", "accelerate"),
                               ("DOWN", "brake"), ("SPACE", "fire item"),
                               ("Z", "drift"), ("X", "nitro")])
        ks.resolve = lambda held: "+".join(sorted(held & set(keys)))
        return ks

    def reset(self, env, seed, spec):
        # pystk2.Race has no reset(); restart the race for a fresh episode.
        try:
            env.restart()
        except Exception:
            pass
        env.step()
        self._prev_dist = 0.0
        return None, {}

    def step(self, env, action):
        pystk2 = self._pystk2
        held = set(action.split("+")) if isinstance(action, str) and action else \
            (set(action) if action else set())
        a = pystk2.Action()
        a.steer = (1.0 if "RIGHT" in held else 0.0) - (1.0 if "LEFT" in held else 0.0)
        a.acceleration = 1.0 if "UP" in held else 0.0
        a.brake = "DOWN" in held
        a.fire = "SPACE" in held
        a.drift = "Z" in held
        a.nitro = "X" in held
        env.step(a)
        self._ws.update()
        kart = self._ws.karts[0] if self._ws.karts else None
        dist = float(getattr(kart, "overall_distance", 0.0)) if kart else 0.0
        reward = dist - self._prev_dist
        self._prev_dist = dist
        finished = bool(getattr(kart, "finished_laps", 0) >= self._race.config.laps) if kart else False
        return None, reward, finished, False, {"distance": dist}

    def render(self, env):
        return np.asarray(env.render_data[0].image)

    def capture(self, env, obs, info, want_blob=True) -> FrameState:
        return FrameState(blob=None, variables={"distance": (info or {}).get("distance", 0.0)})

    def close(self, env) -> None:
        try:
            env.stop()
        except Exception:
            pass
