"""A scan's clock: seconds since the scanner trigger, on ``perf_counter``, and the wall clock."""

from __future__ import annotations

import time


class Clock:
    """Anchored at the scanner trigger; gives run + wall-clock time."""

    def __init__(self) -> None:
        """Create an untriggered clock (``t0_*`` are ``None`` until :meth:`trigger`)."""
        self.t0_perf = None
        self.t0_epoch = None

    def trigger(self) -> None:
        """Anchor the clock at the current time (call on scanner trigger)."""
        self.t0_perf = time.perf_counter()
        self.t0_epoch = time.time()

    def run_time(self) -> float:
        """Seconds since the scanner trigger (``perf_counter`` based).

        :return: seconds since this run's trigger.
        """
        return time.perf_counter() - self.t0_perf

    def from_perf(self, t_perf: float) -> float:
        """Convert a ``perf_counter`` stamp (e.g. a flip time) to run time.

        :param t_perf: a ``time.perf_counter()`` value.
        :return: seconds since the scanner trigger.
        """
        return t_perf - self.t0_perf

    def wall_time(self) -> float:
        """Current wall-clock epoch time.

        :return: ``time.time()`` seconds since the Unix epoch.
        """
        return time.time()
