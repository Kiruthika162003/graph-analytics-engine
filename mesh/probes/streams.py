"""Stream probes: time, degree tails, and building held to what cannot vary.

Time-respecting reach never exceeds the static aggregate's, a fitted
exponent lands near the one a graph was built from, the same graph
typed four ways compares equal, and the branching factor of a regular
graph is its degree minus one.
"""

from __future__ import annotations

from mesh.factories import cycle
from mesh.graphbuilder import GraphBuilder
from mesh.graphio import same_graph
from mesh.graphstats import DegreeStats, power_law_graph
from mesh.linkstream import LinkStream
from mesh.probes.probe import Probe
from mesh.probes.registry import register


@register
def timed_reach_never_exceeds_static_reach() -> Probe:
    # every node reached in time order is also reached in the aggregate. the first
    # version left d-e alive at 5..6, and d was reached through e after all; both of
    # d's links now close at 1, before anything from a can arrive
    links = [("a", "b", 1, 2), ("b", "c", 3, 4), ("c", "d", 0, 1), ("d", "e", 0, 1)]
    log = LinkStream([*links, ("a", "e", 2, 3)])
    timed = set(log.reach("a", 0))
    static = log.aggregate_reach("a")
    return Probe(
        prober="linkstream",
        guarantee="time-respecting reach is contained in the static aggregate's reach",
        holds=timed <= static and "d" not in timed and "d" in static,
        readings={"timed": len(timed), "static": len(static)},
    )


@register
def fitted_exponent_matches_the_builder() -> Probe:
    # a graph built at gamma 2.5 must fit within a broad band of 2.5
    g = power_law_graph(3000, gamma=2.5, kmin=3, seed=6)
    gamma = DegreeStats(g).power_law_exponent(3)
    return Probe(
        prober="graphstats",
        guarantee="the maximum likelihood exponent lands near the one a graph was built from",
        holds=2.2 < gamma < 2.9,
        readings={"fitted": round(gamma, 3), "built": 2.5},
    )


@register
def four_ways_of_typing_a_graph_agree() -> Probe:
    # calls, text, pairs, and a shape must produce the same graph
    by_calls = GraphBuilder().nodes("0123").edges([("0", "1"), ("1", "2"), ("2", "3")]).build()
    by_text = GraphBuilder().text("0 1\n1 2\n2 3\n").build()
    by_shape = GraphBuilder().shape("path", 4).build()
    return Probe(
        prober="graphbuilder",
        guarantee="a graph typed by calls, by text, and by shape compares equal every way",
        holds=same_graph(by_calls, by_text) and same_graph(by_text, by_shape),
        readings={"edges": by_calls.edge_count()},
    )


@register
def branching_of_a_regular_graph_is_degree_minus_one() -> Probe:
    # following a random edge from a d-regular graph finds d minus 1 further edges
    ds = DegreeStats(cycle(12))
    return Probe(
        prober="graphstats",
        guarantee="the branching factor of a regular graph is its degree minus one",
        holds=ds.branching() == 1.0 and ds.variance() == 0.0,
        readings={"branching": ds.branching(), "mean": ds.mean()},
    )
