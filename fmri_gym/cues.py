"""Synthesized feedback cues -- the sounds an adapter makes for what happened.

A cue is not game audio. Doom hands us a PCM buffer per tic and the adapter
just forwards it (``vizdoom.sound``); a cue is generated here because the
engine has none: crafter ships 56 assets and every one is a PNG. What the
subject hears is therefore a design decision of the paradigm, and these three
waveforms are it.

Three events, because a subject in the bore needs to know three different
things and cannot be told any of them in words:

- ``score`` -- an achievement just unlocked, the only "+1" crafter scores.
- ``hit`` -- the last press landed on a creature. Chopping a zombie takes five
  presses bare-handed and two with a stone sword, and until the kill nothing on
  screen says a blow connected, so without this cue the subject cannot tell
  "wrong tile" from "not dead yet".
- ``blocked`` -- the press was legal input but the engine refused it (no
  pickaxe for that rock, nothing craftable here). Otherwise a refusal and a
  dropped button press look identical.

They are separated by TIMBRE, not pitch: ``hit`` is a noise burst, ``score`` a
harmonic chime, ``blocked`` a low sine. Audio output queues rather than mixes
(:class:`fmri_gym.audio.Audio`), so the adapter plays at most one per frame,
picking by priority.

Queueing makes cue length a timing constraint, not just a stylistic one: a cue
longer than a frame delays the next one by the difference. All three fit inside
the scanner block's 400 ms turn -- ``hit`` 60 ms, ``blocked`` 100 ms, ``score``
exactly 400 -- so no cue can ever push the next one into a later frame than the
press that earned it, and a cue always names the press the subject just made.

``score`` is the one that had to be cut to get there. It is a port of the rig's
``celebrate_wav``, whose third note rings for 0.45 s (670 ms in all), and at
that length an unlock followed immediately by another cue delayed it ~270 ms,
with three unlocks on consecutive frames drifting ~800 ms. Shortened to 0.18 s
on 2026-09-20: the two frontends now differ in the tail of the reward note,
which is the cost, and the attack that identifies it is untouched.

Everything is synthesized from these numbers rather than shipped as a wav, so
the exact stimulus a session presented is recoverable from the commit hash.
``hit`` uses a fixed RNG seed for the same reason: same bytes every run.
"""

from __future__ import annotations

import numpy as np

from .adapters.base import Sound

#: Cue playback rate. 44.1 kHz because that is what the rig's own celebration
#: jingle was authored at and what any scanner-side audio chain expects.
SAMPLE_RATE = 44100

#: Peak amplitude as a fraction of full scale. The jingle was raised to this
#: after a pilot found the earlier cue inaudible over scanner-adjacent
#: playback; the other two are matched to it so relative loudness is a
#: property of the waveform, not of three different normalisations.
_PEAK = 0.92


def _pcm(wave: np.ndarray, peak: float = _PEAK) -> np.ndarray:
    """Normalise a float waveform to int16 mono PCM shaped ``(n, 1)``.

    :param wave: float samples, any scale.
    :param peak: target peak as a fraction of full scale.
    :return: ``(n_samples, 1)`` int16, the shape :class:`~fmri_gym.audio.Audio`
        opens the output stream from.
    """
    wave = np.asarray(wave, np.float64)
    wave = wave / max(np.abs(wave).max(), 1e-12) * peak
    return (wave * 32767).astype("<i2")[:, None]


def score_cue(rate: int = SAMPLE_RATE) -> Sound:
    """The "+1" chime: a three-note ascending arpeggio, exactly 0.40 s.

    B5, E6, B6, each a sine plus a quieter octave harmonic under a 5 ms attack
    and an exponential decay. Ported from the rig's ``celebrate_wav``, with the
    third note cut from 0.45 s to 0.18 so the whole cue fits one 400 ms turn
    (see the module docstring). The two short notes that make it recognisable
    are the rig's own.

    :param rate: sample rate in Hz.
    :return: the cue as a :class:`~fmri_gym.adapters.base.Sound`.
    """
    parts = []
    for freq, dur in ((987.77, 0.11), (1318.51, 0.11), (1975.53, 0.18)):
        t = np.arange(int(rate * dur)) / rate
        tone = np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(4 * np.pi * freq * t)
        envelope = np.minimum(t / 0.005, 1.0) * np.exp(-t / (0.7 * dur))
        parts.append(tone * envelope)
    return Sound(_pcm(np.concatenate(parts)), rate)


def hit_cue(rate: int = SAMPLE_RATE) -> Sound:
    """The "that landed" thud: a 60 ms noise burst with a dropping body.

    Deliberately not a note. It fires up to five times on one zombie and would
    have to be agreed with the ``score`` chime it can immediately precede, so
    it is separated by texture: a smoothed noise burst (the impact) over a
    220 Hz sine sliding down an octave (the body), under a 1 ms attack and a
    fast decay. Short enough that five in a row at 2.5 Hz stay distinct.

    :param rate: sample rate in Hz.
    :return: the cue as a :class:`~fmri_gym.adapters.base.Sound`.
    """
    dur = 0.06
    t = np.arange(int(rate * dur)) / rate
    # Seeded, so the waveform is a constant of the paradigm rather than a
    # fresh draw per session.
    noise = np.random.default_rng(20260916).standard_normal(t.size)
    # Boxcar-smooth the noise: drops the hiss that makes a raw burst read as
    # static rather than as an impact.
    noise = np.convolve(noise, np.ones(8) / 8, mode="same")
    body = np.sin(2 * np.pi * 220 * t * np.exp(-t / dur))
    envelope = np.minimum(t / 0.001, 1.0) * np.exp(-t / (0.25 * dur))
    return Sound(_pcm((0.7 * noise + 0.6 * body) * envelope), rate)


def blocked_cue(rate: int = SAMPLE_RATE) -> Sound:
    """The "nothing happened" blip: 0.1 s of low sine, quieter than the rest.

    Low and dull on purpose. It reports a non-event, it can repeat while a
    subject probes the tech tree, and it must never be mistaken for the reward
    chime an octave and a half above it.

    :param rate: sample rate in Hz.
    :return: the cue as a :class:`~fmri_gym.adapters.base.Sound`.
    """
    dur = 0.1
    t = np.arange(int(rate * dur)) / rate
    tone = np.sin(2 * np.pi * 160 * t) + 0.25 * np.sin(2 * np.pi * 80 * t)
    envelope = np.minimum(t / 0.004, 1.0) * np.minimum(1.0, (dur - t) / 0.02)
    return Sound(_pcm(tone * envelope, peak=0.55), rate)
