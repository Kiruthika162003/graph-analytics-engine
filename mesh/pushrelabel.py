"""Push-relabel: flood the source's neighbors, then push excess downhill by height.

Augmenting-path algorithms build a flow one full source-to-sink path at a
time. Push-relabel works locally instead. It begins by saturating every
edge out of the source, dumping flow onto the source's neighbors, which
leaves those nodes holding excess, more flow in than out, a state called a
preflow that would be illegal in a finished flow but is the whole idea
here. Every node carries a height, the source fixed at the node count and
everything else starting at zero, and the rule is that excess may only be
pushed downhill, from a node to a neighbor exactly one unit lower along
an edge with residual capacity. When a node holds excess but has no
downhill neighbor with capacity, it is relabeled: its height rises to one
more than its lowest neighbor with residual capacity, so a push becomes
possible. Excess drains toward the sink because the sink sits at height
zero, and excess that cannot reach the sink climbs above the source's
height and drains back to the source. When no node other than source and
sink holds excess, the preflow is a flow, and it is maximum because the
height labels make a valid distance labeling that forbids any augmenting
path. The generic version runs in nodes squared times edges; choosing the
highest-excess node first improves it, and the FIFO discipline used here,
a queue of active nodes, is the simple version with the same bound. Each
node also keeps a current-edge pointer so relabel scans do not restart
from the beginning every time. The solver runs to completion, returns the
flow value and per-edge flow, refuses a negative capacity and a source
equal to the sink, and reports the push and relabel counts, because a
relabel count far above the node count is a network where excess kept
getting stranded and climbing, the shape where a highest-label heuristic
would have paid off.
"""

from __future__ import annotations

from collections import deque

from mesh.errors import Invalid, Missing
from mesh.graph import Graph


class PushRelabel:
    def __init__(self, graph: Graph, source: str, sink: str) -> None:
        if not graph.directed:
            raise Invalid("a flow network is a directed graph")
        if source == sink:
            raise Invalid("the source and sink must be different nodes")
        for endpoint in (source, sink):
            if not graph.has_node(endpoint):
                raise Missing(f"'{endpoint}' is not in the graph")
        self.graph = graph
        self.source = source
        self.sink = sink
        self.residual: dict[str, dict[str, float]] = {n: {} for n in graph.nodes()}
        for u, v, cap in graph.edges():
            if cap < 0:
                raise Invalid(f"edge {u}->{v} has negative capacity {cap}")
            self.residual[u][v] = self.residual[u].get(v, 0.0) + cap
            self.residual[v].setdefault(u, 0.0)
        self.height: dict[str, int] = dict.fromkeys(graph.nodes(), 0)
        self.excess: dict[str, float] = dict.fromkeys(graph.nodes(), 0.0)
        self.pushes = 0
        self.relabels = 0
        self.value = self._run()

    def _push(self, u: str, v: str) -> None:
        amount = min(self.excess[u], self.residual[u][v])
        self.residual[u][v] -= amount
        self.residual[v][u] += amount
        self.excess[u] -= amount
        self.excess[v] += amount
        self.pushes += 1

    def _relabel(self, u: str) -> None:
        # rise to one above the lowest neighbor that still has capacity
        lowest = min(
            self.height[v] for v, cap in self.residual[u].items() if cap > 0
        )
        self.height[u] = lowest + 1
        self.relabels += 1

    def _run(self) -> float:
        # the initial preflow saturates every edge leaving the source
        self.height[self.source] = self.graph.node_count()
        active: deque[str] = deque()
        for v, cap in list(self.residual[self.source].items()):
            if cap > 0:
                self.excess[self.source] += cap
                self._push(self.source, v)
                if v != self.sink:
                    active.append(v)
        pointer = dict.fromkeys(self.residual, 0)
        order = {u: list(nbrs) for u, nbrs in self.residual.items()}
        while active:
            u = active.popleft()
            while self.excess[u] > 0:
                if pointer[u] >= len(order[u]):
                    self._relabel(u)
                    pointer[u] = 0
                    continue
                v = order[u][pointer[u]]
                downhill = self.height[u] == self.height[v] + 1
                if self.residual[u][v] > 0 and downhill:
                    was_inactive = self.excess[v] == 0
                    self._push(u, v)
                    if was_inactive and v not in (self.source, self.sink):
                        active.append(v)
                else:
                    pointer[u] += 1
        return self.excess[self.sink]

    def flow_on(self, u: str, v: str) -> float:
        # the residual merges u->v with the reverse of v->u, so this is the NET
        # flow across the pair, negative when v->u carries more; a first guess
        # that it was the per-direction flow was refuted by antiparallel edges
        if not self.graph.has_edge(u, v):
            raise Missing(f"no edge from '{u}' to '{v}'")
        return self.graph.weight(u, v) - self.residual[u][v]

    def note(self) -> str:
        return (
            f"maximum flow {self.value} after {self.pushes} push(es) and "
            f"{self.relabels} relabel(s) over {self.graph.node_count()} node(s); "
            "relabels far above the node count is excess that kept getting stranded"
        )
