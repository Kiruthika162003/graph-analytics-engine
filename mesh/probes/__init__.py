"""The probes organ: standing demonstrations that the engine's guarantees hold.

Importing this package registers every probe by importing the modules that
define them. The registry exposes all_probes(), broken(), and report().
"""

from __future__ import annotations

# importing probe modules registers their probes as they are added
from mesh.probes import (  # noqa: F401
    agreements,
    census,
    chance,
    classes,
    counting,
    demonstrations,
    hardness,
    history,
    invariants,
    readings,
    rendering,
    reports,
    spectral,
    streams,
    structure,
    tooling,
    walks,
)
from mesh.probes.probe import Probe
from mesh.probes.registry import all_probes, broken, register, report

__all__ = ["Probe", "all_probes", "broken", "register", "report"]
