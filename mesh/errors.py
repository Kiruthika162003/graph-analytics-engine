"""The error family: every refusal in this engine descends from one name."""

from __future__ import annotations


class MeshError(Exception):
    """Base for everything this graph engine refuses to do."""


class Invalid(MeshError):
    """The request contradicts itself or the graph's configuration."""


class Missing(MeshError):
    """The node or edge addressed does not exist."""


class Unreachable(MeshError):
    """No path exists between the endpoints asked about."""


class Cyclic(MeshError):
    """A cycle was found where an acyclic graph was required."""


class Negative(MeshError):
    """A negative cycle was found where the algorithm forbids one."""
