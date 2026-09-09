"""A tool day: two rosters merged, renamed, cached, and checked against a contract.

Run with: python -m examples.toolday
"""

from __future__ import annotations

from mesh.graphbuilder import GraphBuilder
from mesh.graphcache import ReadingCache, fingerprint
from mesh.graphmerge import GraphMerge
from mesh.graphsummary import GraphSummary
from mesh.graphvalidate import Contract
from mesh.relabel import Relabel


def main() -> int:
    spring = GraphBuilder().text("Ada Ben 3\nBen Cal 1\nCal Dee 2\nDee Ada 1\n").build()
    autumn = GraphBuilder().text("ada ben 1\nben eve 2\neve dee 1\ndee ada 4\n").build()
    folded, _inverse = Relabel(spring).by_function(str.lower)
    print(f"spring folded to lower case: {sorted(folded.nodes())}")

    gm = GraphMerge(folded, autumn, rule="sum")
    print(f"merge: {gm.note()}")
    union = gm.union()
    print(f"ada-ben over both seasons weighs {union.weight('ada', 'ben'):g}")
    changed = [(u, v) for u, v, _w in gm.symmetric_difference().edges()]
    print(f"changed in either season: {changed}")

    cache = ReadingCache()
    cache.register("summary", lambda g: GraphSummary(g).lines()[0])
    for _ in range(3):
        print(cache.get("summary", union))
    print(f"cache: {cache.note()}; fingerprint {fingerprint(union)}")

    contract = Contract().connected().no_isolated().weights_between(0, 5)
    contract.requires(["ada", "fox"])
    print(f"contract on the union: {contract.note(union)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
