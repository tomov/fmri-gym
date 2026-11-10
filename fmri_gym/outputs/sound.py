"""Sound: one chunk of PCM to play, with the rate it has to be played at."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Sound:
    """One chunk of PCM to play right now.

    The samples, plus the one thing an array cannot carry: the rate they have to
    be played at, which differs from one source to the next.

    :ivar pcm: samples shaped ``(n_samples, n_channels)``; the dtype is the
        sample format (``int16`` for most sources).
    :ivar sample_rate: samples per second at which ``pcm`` must be played.
    """

    pcm: np.ndarray
    sample_rate: float
