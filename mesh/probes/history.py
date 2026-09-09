"""History probes: logs, sequences, and scenarios held to what a record must respect.

Replaying every event gives the live graph, a text round trip gives an
equal log, a constant sequence persists fully with its core intact, and
a scenario never touches the graph it was asked about.
"""

from __future__ import annotations

from mesh.eventlog import EventLog
from mesh.factories import cycle, star
from mesh.graphcache import fingerprint
from mesh.graphio import same_graph
from mesh.graphsequence import GraphSequence
from mesh.probes.probe import Probe
from mesh.probes.registry import register
from mesh.scenario import Scenario


def _session() -> EventLog:
    log = EventLog()
    for n in ("a", "b", "c", "d"):
        log.add_node(n)
    log.add_edge("a", "b")
    log.add_edge("b", "c")
    log.add_edge("c", "d")
    log.remove_edge("b", "c")
    log.add_edge("a", "d", 2.0)
    log.remove_node("c")
    return log


@register
def replaying_every_event_gives_the_live_graph() -> Probe:
    # the log's reconstruction and its live graph must share a fingerprint
    log = _session()
    replayed = log.replay()
    return Probe(
        prober="eventlog",
        guarantee="replaying every recorded event rebuilds exactly the live graph",
        holds=fingerprint(replayed) == fingerprint(log.graph) and log.contiguous(),
        readings={"events": len(log.events), "edges": log.graph.edge_count()},
    )


@register
def log_text_round_trip_is_equal() -> Probe:
    # dump then load must give the same events and the same graph
    log = _session()
    back = EventLog.load(log.dump())
    return Probe(
        prober="eventlog",
        guarantee="a log written to text and read back has the same events and graph",
        holds=back.events == log.events and same_graph(back.graph, log.graph),
        readings={"lines": len(log.dump().splitlines())},
    )


@register
def constant_sequence_persists_fully() -> Probe:
    # three identical snapshots have persistence one, no churn, and a full core
    seq = GraphSequence([cycle(6), cycle(6), cycle(6)])
    return Probe(
        prober="graphsequence",
        guarantee="a constant sequence has persistence one, no churn, and a core equal to it",
        holds=seq.persistence() == [1.0, 1.0]
        and seq.churn() == [0, 0]
        and same_graph(seq.core(), cycle(6))
        and seq.core_inside_every_snapshot(),
        readings={"persistence": seq.persistence(), "core_edges": seq.core().edge_count()},
    )


@register
def scenarios_never_touch_the_original() -> Probe:
    # every kind of change runs on a copy, so the readings of the original hold
    sc = Scenario(star(5))
    sc.remove_node("0")
    sc.remove_edge("0", "1")
    sc.add_edge("1", "2")
    hub_first = sc.sweep_nodes(by="largest")[0][0] == "0"
    return Probe(
        prober="scenario",
        guarantee="what-if changes leave the original graph's readings untouched",
        holds=sc.untouched() and hub_first,
        readings={"pieces": sc.before["pieces"], "largest": sc.before["largest"]},
    )
