# Audio support

Native game audio plays automatically when the adapter supports it. No
`"audio": true` setting is required. ViZDoom and stable-retro currently provide
audio to the shared player; other backends keep their existing behavior.

## Turn audio off

Mute every game block with the CLI flag:

```bash
python fmri_play.py --subject sub-test --dummy-trigger --no-audio --curriculum configs/dbp_games/vizdoom__defend_center.json
```

To mute just one block, add `"audio": false` to that game phase in the curriculum.
For example:

```json
{
  "type": "game",
  "backend": "vizdoom",
  "game": "VizdoomDefendCenter-v1",
  "mode": "duration",
  "duration": 30,
  "fps": 35,
  "audio": false
}
```

Omitting `audio`, or setting it to `true`, allows supported audio. `--no-audio`
overrides every game phase, including an explicit `"audio": true`.
ViZDoom's existing `env_kwargs.audio_buffer_enabled=false` setting also disables
audio; neither `"audio": true` nor an omitted setting overrides that engine mute.
Conversely, `"audio": false` wins over `env_kwargs.audio_buffer_enabled=true`.

These controls govern fmri-gym's shared audio output. They do not create an audio
interface for an unsupported environment or control an arbitrary third-party
environment's independent sound player.

## Why some games are silent

The relevant source of sound is the environment package used in the experiment.
A commercial game or standalone application may have music and sound effects
that its Gym implementation does not include or expose.

The table covers all registered adapter families. Findings refer to the reviewed
versions and source snapshots below, as of September 14, 2026; they are not a
guarantee about future releases or every third-party game.

| Backend | Current audio behavior | Reason / evidence |
|---|---|---|
| `vizdoom` | Plays automatically | The engine exposes native PCM through its audio buffer. See the [adapter](fmri_gym/adapters/vizdoom.py). |
| `retro` | Plays automatically | stable-retro 1.0.1 exposes PCM and its native rate through [emulator bindings](https://github.com/Farama-Foundation/stable-retro/blob/v1.0.1/src/retro.cpp). |
| `crafter` | Silent in the reviewed environment | Crafter 1.8.3 has no audio generation or playback interface in the reviewed [environment](https://github.com/danijar/crafter/blob/e04542a2159f1aad3d4c5ad52e8185717380ee3a/crafter/env.py) and [GUI](https://github.com/danijar/crafter/blob/e04542a2159f1aad3d4c5ad52e8185717380ee3a/crafter/run_gui.py). |
| `baba` | Silent in the reviewed environment | baba-is-ai 0.0.1 is a benchmark implementation, with no audio in the reviewed [game code](https://github.com/nacloos/baba-is-ai/blob/40ae1b29dbc67f2ecea06cc161ed5ca51719e5e3/baba/grid.py) or [GUI](https://github.com/nacloos/baba-is-ai/blob/40ae1b29dbc67f2ecea06cc161ed5ca51719e5e3/baba/play.py). Commercial Baba Is You audio is not part of this package. |
| `minihack`, `nethack` | No exposed PCM | Both use NLE. Its [observations](https://github.com/NetHack-LE/nle/blob/2319f2989f0035685017e9ea13c83b2546fe477c/nle/nethack/nethack.py) do not include audio, and its [terminal callback](https://github.com/NetHack-LE/nle/blob/2319f2989f0035685017e9ea13c83b2546fe477c/src/nle.c#L91) ignores bell events. |
| `aigamestore` (`p5`) | Silent for vendored game1-game10 | The checked-in [game sources and assets](vendor/aigamestore/) contain no audio assets or playback calls. This finding does not cover arbitrary browser games. |
| `supertuxkart` (`stk`) | Requires upstream support | PySTK2's Python engine [build excludes sound support](https://github.com/bpiwowar/pystk2/blob/dd70f6823f248ae1df2ce513839a9b2c8c940c39/CMakeLists.txt#L444). Enabling sound in the standalone SuperTuxKart application does not enable it here. |
| `rushhour` | Requires upstream support | The standalone Go application has a [success sound](https://github.com/chrplr/Rush-Hour/blob/16c9322c3a6fea50d66860b875b559035fed92da/main.go#L287), but the reviewed rushhour-gym 0.6.0 [Human environment](https://github.com/chrplr/Rush-Hour/blob/16c9322c3a6fea50d66860b875b559035fed92da/python/src/rushhour_gym/human.py) and JSON engine path do not expose audio. |
| `ale` | Audio exists; playback integration deferred | ALE 0.12.1 exposes raw audio, but its buffer timing needs further work before continuous native playback can be supported. See below. |
| `gym` | No shared PCM integration | The [generic adapter](fmri_gym/adapters/default.py) accepts many environments; audio capability must be checked per environment. This is not a claim that all Gymnasium games are silent. |
| `vgdl` | No shared PCM integration | The current [VGDL adapter](fmri_gym/adapters/vgdl.py) supplies offscreen frames and state. Audio support has not been established for every game in the external checkout. |
| `overcooked` | No shared PCM integration | The [adapter](fmri_gym/adapters/overcooked.py) uses overcooked-ai's environment and StateVisualizer. Sound from a commercial Overcooked game does not establish an audio interface in this benchmark. |

ALE's raw buffer contains [512 samples per frame](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/common/SoundRaw.hxx),
while its [default generation rate is 31,400 Hz](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/v0.12.1/src/ale/emucore/Settings.cxx).
At 60 frames per second, this produces only 30,720 samples per second. The inferred
mismatch means that simply forwarding those buffers would leave gaps. This is a
timing limitation of the reviewed raw path, not an absence of Atari sound.

Adapters do not invent beeps, music, or reward sounds to fill these gaps. Sound
can provide task information, so new sound behavior belongs in the environment
used by both humans and model evaluation. Once an environment exposes suitable
native audio, its adapter can use the same shared player.

## Native format and timing

The player preserves the engine's sample rate, PCM dtype, and channel count.
fmri-gym does not resample audio or choose a common rate for all games. The
operating system or audio device may still perform its own conversion.

Game speed must match the rate at which the engine generates audio:

- **ViZDoom:** use `fps * env_kwargs.frame_skip == 35`; `frame_skip` defaults to 1.
  The supplied ViZDoom configs use 35 FPS.
- **stable-retro:** match `fps` to `env.unwrapped.em.get_screen_rate()`. The
  included [Airstriker demo](configs/demo_retro_audio.json) uses approximately
  59.923 FPS and native 44,100 Hz stereo audio. Other cores may use other rates.

Running too slowly leaves gaps; running too quickly can accumulate queued audio
and delay. The player opens when an episode first supplies samples and closes
when the episode ends, including early exit. It does not carry queued sounds
into later episodes or fixation periods.

Playback support does not establish precise audiovisual synchronization at the
scanner. Measure timing with the actual display, audio device, and scanner setup
before relying on sound onset times in an experiment.

## If expected audio is missing

First check the backend table and mute settings. For a supported backend, ensure
that `sounddevice` is installed (it is included in `requirements.txt`) and that
the system's default output device is available and unmuted. List the devices
visible to the audio library with:

```bash
python -m sounddevice
```

The output device must accept the native stream format. Device or dependency
errors are reported rather than silently converting the stream. Use `--no-audio`
to run without opening an audio output device.
