"""A probe: a standing demonstration that a graph guarantee holds under strain.

A probe is not a unit test and not a benchmark. It is a small, honest
report: it walks a graph built to stress one guarantee, takes real
readings from the engine's own code, and states in one line whether the
guarantee held. The readings are numbers a skeptic can check, not
adjectives. A probe that stops holding is a guarantee that broke, and the
test suite runs every probe so that break fails the build.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Probe:
    prober: str
    guarantee: str
    holds: bool
    readings: dict[str, object] = field(default_factory=dict)

    def line(self) -> str:
        status = "holds" if self.holds else "BROKEN"
        return f"{self.prober}: {status} -- {self.guarantee}"

    def detail(self) -> str:
        pairs = ", ".join(f"{k}={v}" for k, v in self.readings.items())
        return f"{self.line()}\n    readings: {pairs}" if pairs else self.line()
