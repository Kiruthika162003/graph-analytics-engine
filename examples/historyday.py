"""A history day: a network edited step by step, replayed, read as a sequence, asked what if.

Run with: python -m examples.historyday
"""

from __future__ import annotations

from mesh.eventlog import EventLog
from mesh.graphsequence import GraphSequence
from mesh.scenario import Scenario


def main() -> int:
    log = EventLog()
    for n in ("ada", "ben", "cal", "dee", "eve"):
        log.add_node(n)
    log.add_edge("ada", "ben")
    log.add_edge("ben", "cal")
    week_one = log.replay()
    log.add_edge("cal", "dee")
    log.add_edge("dee", "eve")
    week_two = log.replay()
    log.remove_edge("ben", "cal")
    log.add_edge("ada", "eve")
    week_three = log.replay()
    print(f"log: {log.note()}")
    print(f"week one again: {log.replay(upto=7).edge_count()} edge(s)")

    seq = GraphSequence([week_one, week_two, week_three])
    print(f"sequence: {seq.note()}")
    for step, report in enumerate(seq.reports(), start=1):
        print(f"  week {step} to {step + 1}: {report[0]}; {report[-1]}")

    sc = Scenario(week_three)
    print(f"if dee left: {sc.note(sc.remove_node('dee'))}")
    print(f"if ben met eve: {sc.note(sc.add_edge('ben', 'eve'))}")
    worst = sc.sweep_nodes(by="largest")[0]
    print(f"most damaging departure: {worst[0]} ({worst[1]:+g} to the largest piece)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
