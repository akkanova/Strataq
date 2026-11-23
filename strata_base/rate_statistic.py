"""A rate (an amount per second, such as bytes moved) measured over a series of intervals."""

from __future__ import annotations


class RateStatistic:
    """Follows a rate across intervals: its exponential moving average, the lowest and highest rate of one interval,
    and the total amount and time, whose ratio is the overall average.  The caller measures the intervals and passes
    their lengths to `add`, so the statistic needs no clock: time that is not added (a pause) does not count."""

    def __init__(self, decay_per_second: float):
        """`decay_per_second`: the weight the moving average gives what it was one second earlier (0.5 halves it
        each second; nearer 1 is smoother)."""
        if not 0 <= decay_per_second <= 1:
            raise ValueError("decay_per_second must be between 0 and 1")
        self.decay_per_second = decay_per_second
        self.total = 0.0        # the amount over all intervals
        self.seconds = 0.0      # their length
        self.moving: float | None = None   # the moving average; None before the first interval
        self.low: float | None = None
        self.high: float | None = None

    def add(self, amount: float, seconds: float) -> None:
        """Counts `amount` over an interval of `seconds`; an interval of no length is ignored."""
        if seconds <= 0:
            return
        rate, keep = amount / seconds, self.decay_per_second ** seconds
        self.moving = rate if self.moving is None else keep * self.moving + (1 - keep) * rate
        self.low = rate if self.low is None else min(self.low, rate)
        self.high = rate if self.high is None else max(self.high, rate)
        self.total += amount
        self.seconds += seconds

    @property
    def average(self) -> float | None:
        """Gets the rate over all intervals together, or None before the first."""
        return self.total / self.seconds if self.seconds else None

    def to_dict(self) -> dict:
        """Gets the readings as a JSON-ready {"moving", "average", "low", "high"}, each None before the first
        interval."""
        return {"moving": self.moving, "average": self.average, "low": self.low, "high": self.high}
