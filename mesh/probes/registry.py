"""The probe registry: every guarantee this engine claims, in one place.

Each probe function returns a Probe with real readings. The registry
collects them so the CLI can report them and the test suite can assert
that none is broken. Adding a probe is adding its function to PROBES; the
frame test counts dynamically so a new probe never breaks the frame.
"""

from __future__ import annotations

from collections.abc import Callable

from mesh.probes.probe import Probe

# probe functions are registered here as they are written
_REGISTERED: list[Callable[[], Probe]] = []


def register(fn: Callable[[], Probe]) -> Callable[[], Probe]:
    _REGISTERED.append(fn)
    return fn


def all_probes() -> list[Probe]:
    return [fn() for fn in _REGISTERED]


def broken() -> list[Probe]:
    return [p for p in all_probes() if not p.holds]


def report() -> str:
    probes = all_probes()
    lines = [p.line() for p in probes]
    lines.append(f"\n{len(probes)} probe(s), {len(broken())} broken")
    return "\n".join(lines)
