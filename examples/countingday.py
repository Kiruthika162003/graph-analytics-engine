"""A counting day: colorings, spanning trees, perfect matchings, and triads, all exact.

Run with: python -m examples.countingday
"""

from __future__ import annotations

from mesh.chromaticpolynomial import ChromaticPolynomial
from mesh.factories import complete, cycle, path, wheel
from mesh.graph import Graph
from mesh.graphproduct import GraphProduct
from mesh.matchingcount import PerfectMatchingCount
from mesh.triadcensus import TriadCensus
from mesh.tuttecount import TutteCount


def build_office_arrows() -> Graph:
    g = Graph(directed=True)
    reports = [
        ("ada", "ben"), ("ada", "cal"), ("ben", "cal"), ("cal", "ben"),
        ("dee", "ada"), ("dee", "ben"), ("eve", "dee"), ("cal", "eve"),
    ]
    for a, b in reports:
        g.add_node(a)
        g.add_node(b)
        g.add_edge(a, b)
    return g


def main() -> int:
    for name, g in (("pentagon", cycle(5)), ("wheel", wheel(5)), ("K4", complete(4))):
        cp = ChromaticPolynomial(g)
        print(f"{name:<9} {cp.note()}")
        print(f"{'':<9} with 3 colors: {cp.evaluate(3)}, with 4 colors: {cp.evaluate(4)}")

    for name, g in (("pentagon", cycle(5)), ("wheel", wheel(5)), ("K5", complete(5))):
        tc = TutteCount(g)
        agree = "agrees" if tc.agrees_with_kirchhoff() else "DISAGREES"
        print(f"{name:<9} {tc.note()}; {agree} with Kirchhoff")

    for rungs in (3, 4, 5, 6):
        ladder = GraphProduct(path(2), path(rungs)).cartesian()
        print(f"ladder of {rungs}: {PerfectMatchingCount(ladder).count} perfect matching(s)")

    census = TriadCensus(build_office_arrows())
    print(f"office triads: {census.note()}")
    print(f"transitive share of three-arrow triples: {census.transitivity_share():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
