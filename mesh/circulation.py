"""Circulations with lower bounds: is there a flow that meets every minimum on every arc.

A plain maximum flow answers how much can move. A circulation with
lower bounds answers a different question: given a floor and a
ceiling on every arc, is there any flow at all that conserves at every
node and stays inside the bounds, with no source or sink, so that
whatever enters a node leaves it. Crew rosters, pipelines with minimum
throughput, and round-robin schedules with required meetings all
reduce to it. The reduction is Hoffman's: subtract the floor from each
arc's capacity, and charge the floor as a demand, the arc's head owing
that much and its tail being owed it. Add a super source with an arc
to every node that is owed and a super sink from every node that owes,
each with capacity equal to the imbalance, and run a maximum flow. A
feasible circulation exists exactly when the flow saturates every
source arc, and the circulation is that flow plus the floors. The
engine runs the reduction on Edmonds-Karp, reports feasibility with the
shortfall when there is none, rebuilds the circulation, and verifies
it: every arc within its bounds and every node conserving to within
rounding. It also answers the bounded flow question, a source and sink
with lower bounds, by adding an unbounded return arc from sink to
source. Antiparallel arcs are refused, because the flow modules report
net flow across such a pair and the reconstruction would blur two
arcs into one.
"""

from __future__ import annotations

from mesh.edmondskarp import EdmondsKarp
from mesh.errors import Invalid
from mesh.graph import Graph

SOURCE = "__source__"
SINK = "__sink__"


class Circulation:
    def __init__(self, graph: Graph, lower: dict[tuple[str, str], float]) -> None:
        if not graph.directed:
            raise Invalid("a circulation lives on a directed graph")
        for u, v, _w in graph.edges():
            if graph.has_edge(v, u):
                raise Invalid(f"arcs {u}->{v} and {v}->{u} are antiparallel; split one")
        self.graph = graph
        self.lower = dict(lower)
        for (u, v), floor in self.lower.items():
            if not graph.has_edge(u, v):
                raise Invalid(f"no arc {u}->{v} to bound")
            if floor < 0 or floor > graph.weight(u, v):
                raise Invalid(f"floor {floor} on {u}->{v} is outside 0..{graph.weight(u, v)}")
        self.demand_total = 0.0
        self.flow = self._solve()
        self.feasible = abs(self.flow.value - self.demand_total) < 1e-9
        self.circulation = self._rebuild() if self.feasible else {}

    def _solve(self) -> EdmondsKarp:
        aux = Graph(directed=True)
        for n in self.graph.nodes():
            aux.add_node(n)
        aux.add_node(SOURCE)
        aux.add_node(SINK)
        balance = dict.fromkeys(self.graph.nodes(), 0.0)
        for u, v, cap in self.graph.edges():
            floor = self.lower.get((u, v), 0.0)
            if cap - floor > 0:
                aux.add_edge(u, v, cap - floor)
            balance[v] += floor
            balance[u] -= floor
        for n, b in balance.items():
            if b > 0:
                aux.add_edge(SOURCE, n, b)
                self.demand_total += b
            elif b < 0:
                aux.add_edge(n, SINK, -b)
        self.aux = aux
        return EdmondsKarp(aux, SOURCE, SINK)

    def _rebuild(self) -> dict[tuple[str, str], float]:
        out: dict[tuple[str, str], float] = {}
        for u, v, _cap in self.graph.edges():
            floor = self.lower.get((u, v), 0.0)
            extra = self.flow.flow_on(u, v) if self.aux.has_edge(u, v) else 0.0
            out[(u, v)] = floor + max(extra, 0.0)
        return out

    def shortfall(self) -> float:
        return self.demand_total - self.flow.value

    def within_bounds(self) -> bool:
        return all(
            self.lower.get((u, v), 0.0) - 1e-9 <= f <= self.graph.weight(u, v) + 1e-9
            for (u, v), f in self.circulation.items()
        )

    def conserves(self) -> bool:
        net = dict.fromkeys(self.graph.nodes(), 0.0)
        for (u, v), f in self.circulation.items():
            net[u] -= f
            net[v] += f
        return all(abs(x) < 1e-9 for x in net.values())

    def note(self) -> str:
        if not self.feasible:
            return f"infeasible: the floors demand {self.shortfall():g} more than can circulate"
        moved = sum(self.circulation.values())
        return f"feasible circulation over {len(self.circulation)} arc(s) moving {moved:g}"


class BoundedFlow:
    def __init__(
        self,
        graph: Graph,
        source: str,
        sink: str,
        lower: dict[tuple[str, str], float],
    ) -> None:
        if graph.has_edge(sink, source) or graph.has_edge(source, sink):
            raise Invalid("source and sink must not be joined directly; a return arc is added")
        widened = Graph(directed=True)
        for n in graph.nodes():
            widened.add_node(n)
        for u, v, w in graph.edges():
            widened.add_edge(u, v, w)
        total = sum(w for _u, _v, w in graph.edges()) + 1.0
        widened.add_edge(sink, source, total)
        self.circulation = Circulation(widened, lower)
        self.feasible = self.circulation.feasible
        found = self.circulation.circulation
        self.value = found.get((sink, source), 0.0) if self.feasible else 0.0

    def note(self) -> str:
        if not self.feasible:
            return "no flow meets every floor"
        return f"a flow of {self.value:g} meets every floor"
