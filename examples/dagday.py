"""A DAG day: one build graph ordered, counted, timed, pruned, and dominated.

Run with: python -m examples.dagday
"""

from __future__ import annotations

from mesh.dagpaths import DagPaths
from mesh.dominators import Dominators
from mesh.graph import Graph
from mesh.toposort import TopologicalSort
from mesh.toposortcount import TopologicalCount
from mesh.transitiveclosure import TransitiveClosure


def build_pipeline() -> Graph:
    g = Graph(directed=True)
    steps = [
        ("fetch", "parse", 2), ("parse", "typecheck", 5), ("parse", "lint", 1),
        ("typecheck", "compile", 8), ("lint", "compile", 0), ("compile", "test", 6),
        ("compile", "package", 3), ("test", "release", 1), ("package", "release", 1),
        ("fetch", "compile", 0),
    ]
    for a, b, minutes in steps:
        g.add_node(a)
        g.add_node(b)
        g.add_edge(a, b, minutes)
    return g


def main() -> int:
    g = build_pipeline()
    print(f"steps: {g.node_count()}, dependencies: {g.edge_count()}")

    order = TopologicalSort(g).order()
    print(f"one valid order: {order}")
    print(f"valid orders in all: {TopologicalCount(g).count}")

    paths = DagPaths(g, "fetch")
    print(f"critical path: {paths.critical_path()}")
    print(f"  minutes if everything else waits on it: {paths.longest_to('release'):.0f}")

    closure = TransitiveClosure(g)
    print(f"dependencies implied by others: {closure.removed_count()} of {g.edge_count()}")

    dom = Dominators(g, "fetch")
    print(f"steps every path to release must pass: {dom.dominators_of('release')}")
    print(dom.note())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
