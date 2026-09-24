"""ViZDoom adapter -- Doom action-shooter scenarios (the COOM engine).

COOM (the DBP "action/shooter" pick) is a continual-RL suite built on ViZDoom,
but COOM itself pins gymnasium 0.28 which conflicts with the other backends
(minihack/nle need 1.2). ViZDoom -- the same Doom engine COOM uses -- ships
Gymnasium environments that work with our gymnasium and cover the same
action-shooter category (DefendCenter, DeadlyCorridor, HealthGathering,
TakeCover, MyWayHome, Deathmatch, and full Doom E1M1..). So this backend uses
ViZDoom directly.

The env's observation is a dict {"screen": (H,W,3) uint8, "gamevariables": ...};
`env.render()` (rgb_array) returns the screen for display, and we log
gamevariables (health, ammo, ...) as an analysis variable.

Actions are the scenario's small Discrete(n) button set, and keymaps (both the
default one below and curriculum `keys` overrides) are ALWAYS Discrete action
indices. Setting `env_kwargs.max_buttons_pressed` to 0 switches the env to a
MultiBinary action space so several buttons can be pressed at once; the keymap
is unchanged, we just OR the buttons of every held key (e.g. forward + turn).

Sound is opt-in per curriculum: `env_kwargs.audio_buffer_enabled` puts one tic
of stereo PCM in `obs["audio"]` (so a model sees the same observation a subject
hears), which `sound()` hands to the session's speakers and `capture()` logs.
Doom produces 1/35 s of sound per step whatever the frame rate, so audio only
runs in real time when `fps * env_kwargs.frame_skip == 35`; below that it plays
with gaps, above it lags further behind every frame.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import gymnasium as gym

from .keyspec import KeySpec, MultiKeySpec, SingleKeySpec
from .base import EnvAdapter, FrameState, Sound

# Physical key -> preferred Doom button (first available for the scenario wins).
# The gymnasium wrapper's Discrete action i presses the buttons set in
# env.unwrapped.button_map[i]; index 0 is the no-op (all buttons up).
# See e.g. https://github.com/Farama-Foundation/ViZDoom/blob/main/scenarios/deathmatch.cfg
_DEFAULT_KEY_TO_BUTTON_MAP: dict[str, list[str]] = {
    "UP": ["MOVE_FORWARD"],
    "DOWN": ["MOVE_BACKWARD"],
    "LEFT": ["TURN_LEFT"],
    "RIGHT": ["TURN_RIGHT"],
    "Z": ["MOVE_LEFT"],
    "X": ["MOVE_RIGHT"],
    "SPACE": ["ATTACK"],
    "ENTER": ["USE"],
    "N": ["SELECT_PREV_WEAPON"],
    "M": ["SELECT_NEXT_WEAPON"],
}


def _get_button_map(env: gym.Env) -> list[list[int]]:
    """Return ``Discrete action index -> per-button 0/1 row`` for this scenario.

    ViZDoom only builds ``env.unwrapped.button_map`` for a Discrete action
    space; under MultiBinary (``max_buttons_pressed=0``) we rebuild the same
    single-button table, so Discrete indices mean the same thing in both modes.

    :param env: a ViZDoom Gymnasium environment.
    :return: one 0/1 row per Discrete action index.
    """
    button_map = getattr(env.unwrapped, "button_map", None)
    if button_map is not None:
        return [[int(v) for v in row] for row in np.asarray(button_map)]
    n = len(env.unwrapped.game.get_available_buttons())
    return [list(row) for row in itertools.product((0, 1), repeat=n)
            if sum(row) <= 1]


def _get_button_to_action_map(env: gym.Env) -> dict[str, int]:
    """Map each available Doom button name to its Discrete action index.

    Only single-button rows of the button map are included; the first index
    for each button wins.

    :param env: a ViZDoom Gymnasium environment.
    :return: ``{BUTTON_NAME: discrete_action_index}``.
    """
    u = env.unwrapped
    names = [str(b).split(".")[-1] for b in u.game.get_available_buttons()]
    out: dict[str, int] = {}
    for i, row in enumerate(_get_button_map(env)):
        on = [names[j] for j, v in enumerate(row) if v]
        if len(on) == 1 and on[0] not in out:
            out[on[0]] = i
    return out


def _get_default_key_to_action_map(env: gym.Env) -> KeySpec:
    """Build the default keyboard->action :class:`KeySpec` for this scenario.

    For each physical key in :data:`_DEFAULT_KEY_TO_BUTTON_MAP`, picks the
    first preferred Doom button that exists in the env's button map.

    :param env: a ViZDoom Gymnasium environment.
    :return: a :class:`KeySpec` with single-key combos and ``noop=0``, both
        given as Discrete action indices. MultiBinary envs get a
        :class:`MultiKeySpec` that ORs the buttons of every held key.
    """
    btn_idx = _get_button_to_action_map(env)
    combos: dict[frozenset[str], int] = {}
    for key, prefs in _DEFAULT_KEY_TO_BUTTON_MAP.items():
        for b in prefs:
            if b in btn_idx:
                combos[frozenset([key])] = btn_idx[b]
                break
    if isinstance(env.action_space, gym.spaces.MultiBinary):
        return MultiKeySpec(combos=combos, noop=0,
                            button_map=_get_button_map(env))
    return SingleKeySpec(combos=combos, noop=0)


class VizDoomAdapter(EnvAdapter):
    name: str = "vizdoom"

    def _make(self, spec: dict) -> gym.Env:
        """Create a ViZDoom Gymnasium environment for one game block.

        :param spec: game-phase config dict from the curriculum.
        :return: a ViZDoom Gymnasium environment.
        """
        from vizdoom import gymnasium_wrapper  # noqa: F401  (registers Vizdoom*-v1)
        env = gym.make(spec["game"], render_mode="rgb_array",
                        **spec.get("env_kwargs", {}))
        if env.unwrapped.game.is_audio_buffer_enabled():
            # ViZDoom 1.3.0 ships an assert-enabled OpenAL Soft whose EFX
            # (reverb) filter setup aborts with "gain > 0.00001f" the moment the
            # audio buffer is on, segfaulting the process inside game.init().
            # Doom's reverb only colours the buffer, so turn EFX off; the audio
            # buffer itself still carries the real sound.
            env.unwrapped.game.add_game_args("+snd_efx 0")
        return env

    def _keyspec(self) -> KeySpec:
        return _get_default_key_to_action_map(self.env)

    def render(self) -> np.ndarray:
        return np.asarray(self.env.render())

    def sound(self) -> Sound | None:
        """Return the frame's Doom audio, or ``None`` if there is none to play.

        ``audio_buffer`` is ``None`` unless the curriculum turned the buffer on,
        and ``state`` itself is ``None`` on a terminal frame -- the episode is
        over, so there is nothing left to hear.

        :return: one tic of stereo PCM at the scenario's rate, or ``None``.
        """
        state = self.env.unwrapped.state
        if state is None or state.audio_buffer is None:
            return None
        return Sound(state.audio_buffer,
                     self.env.unwrapped.game.get_audio_sampling_rate())

    def capture(self, obs: Any, info: dict, want_blob: bool = True) -> FrameState:
        variables = {}
        if isinstance(obs, dict) and "gamevariables" in obs:
            variables["gamevariables"] = np.asarray(obs["gamevariables"])
        if isinstance(obs, dict) and "audio" in obs:
            # What the subject heard this frame. Taken from obs, not sound(),
            # because obs has a (zeroed) audio buffer on the terminal frame too,
            # keeping this series the same length as actions and rewards.
            variables["audio"] = np.asarray(obs["audio"])
        return FrameState(blob=None, variables=variables)

    def block_extra(self) -> dict | None:
        """Block-level arrays merged into the npz (the audio sample rate).

        :return: ``{"audio_sampling_rate": Hz}`` when sound is on, else
            ``None``. Without it the logged ``audio`` is not playable back.
        """
        game = self.env.unwrapped.game
        if not game.is_audio_buffer_enabled():
            return None
        return {"audio_sampling_rate": game.get_audio_sampling_rate()}

    def rich_state(self, obs: Any, info: dict) -> dict | None:
        """Optional, opt-in counterpart to :meth:`capture` -- see
        :mod:`fmri_gym.recording`'s ``save_rich_state`` phase field. Thin
        wrapper so :func:`Session` (which calls ``adapter.rich_state(obs,
        info)`` by name) finds this hook; the actual gathering lives in
        :meth:`get_rich_state`, kept as its own method (same signature,
        every adapter's ``get_rich_state`` matches it) so it can also be
        called directly -- e.g. from a notebook -- without going through
        the ``rich_state`` hook-lookup name. ViZDoom itself never reads
        ``obs``/``info`` here: its own state lives on the env
        (``self.env.unwrapped.state``).
        """
        return self.get_rich_state(obs, info)

    def get_rich_state(self, obs: Any, info: dict) -> dict | None:
        """Every piece of scene state ViZDoom's backend can report right
        now -- not just the handful of ``gamevariables`` a scenario's own
        ``available_game_variables`` config happens to declare (Defend the
        Center, e.g., only declares ``AMMO2``/``HEALTH``; that config
        controls what ``obs["gamevariables"]`` contains, not what the engine
        actually tracks).

        - ``game_variables``: **every** ``vzd.GameVariable`` the installed
          ViZDoom build knows about (position/angle/pitch/roll/velocity,
          per-weapon ammo, kill/hit/damage/item/secret counts, dead/
          on-ground/attack-ready flags, camera state, ...), queried directly
          via ``game.get_game_variable(var)`` rather than limited to
          ``game.get_available_game_variables()`` -- confirmed safe to call
          for any scenario: undeclared/inapplicable variables just read back
          ``0.0``, they don't raise (see ``analysis/`` for the probe that
          checked this against Defend the Center specifically).
        - ``episode``: state ``GameVariable`` doesn't cover -- engine tic
          (``state.tic``) vs. this adapter's own frame ordinal
          (``state.number``), wall-clock ``episode_time``, the reward
          accounting ViZDoom itself keeps (``total_reward``/``last_reward``/
          ``living_reward``), ``is_player_dead``, the last low-level action
          applied, and the map name.
        - ``objects``: every actor currently in the level (monsters, items,
          decorations, the player itself) with its name, position, and
          orientation -- ViZDoom's ground-truth entity list, independent of
          what's actually on screen. Empty unless the scenario's
          ``env_kwargs`` set ``objects_info_enabled: true``
          (``game.set_objects_info_enabled``, off by default in ViZDoom).
        - ``labels``: the subset of those objects actually visible on screen
          this frame, each with its on-screen bounding box (``x``, ``y``,
          ``width``, ``height``) and category (e.g. ``"Monster"``) in
          addition to name/position -- the closest thing here to MiniHack's
          glyph classification or ``AIGameStoreAdapter``'s tube layout.
          Empty unless ``env_kwargs`` set ``labels_buffer_enabled: true``.
        - ``sectors``: floor/ceiling height per level sector. Static for the
          whole episode (wall geometry, ``sector.lines``, is not included --
          this is the coarse per-sector heights only), so logging it every
          frame is redundant; a short ``rich_state_stride`` (or reading it
          from just one frame) is enough. Empty unless ``env_kwargs`` set
          ``sectors_info_enabled: true``.

        :return: the dict described above, or ``None`` on a terminal frame
            (``state`` is ``None`` once the episode ends -- same case
            :meth:`sound` already guards against).
        """
        state = self.env.unwrapped.state
        if state is None:
            return None
        game = self.env.unwrapped.game

        import vizdoom as vzd  # lazy: only needed to enumerate GameVariable

        game_variables = {}
        for name in dir(vzd.GameVariable):
            if name.startswith("_") or name in ("name", "value"):
                continue
            game_variables[name] = float(game.get_game_variable(getattr(vzd.GameVariable, name)))

        objects = [{
            "id": o.id, "name": o.name,
            "position": [o.position_x, o.position_y, o.position_z],
            "angle": o.angle, "pitch": o.pitch, "roll": o.roll,
            "velocity": [o.velocity_x, o.velocity_y, o.velocity_z],
        } for o in (state.objects or [])]

        labels = [{
            "object_id": l.object_id, "object_name": l.object_name,
            "category": l.object_category, "value": l.value,
            "bbox": [l.x, l.y, l.width, l.height],
            "position": [l.object_position_x, l.object_position_y, l.object_position_z],
        } for l in (state.labels or [])]

        sectors = [{"floor_height": s.floor_height, "ceiling_height": s.ceiling_height}
                   for s in (state.sectors or [])]

        episode = {
            "tic": state.tic, "frame_number": state.number,
            "episode_time": game.get_episode_time(),
            "total_reward": game.get_total_reward(),
            "last_reward": game.get_last_reward(),
            "living_reward": game.get_living_reward(),
            "is_player_dead": game.is_player_dead(),
            "last_action": [float(a) for a in game.get_last_action()],
            "doom_map": game.get_doom_map(),
        }

        return {"game_variables": game_variables, "episode": episode,
                "objects": objects, "labels": labels, "sectors": sectors}
