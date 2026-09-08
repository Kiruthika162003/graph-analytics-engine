"""PageRank: importance as the stationary share of a random surfer's attention.

PageRank ranks nodes of a directed graph by imagining a surfer who follows
links at random. At each step the surfer either follows one of the current
node's outgoing edges chosen uniformly, with probability given by a damping
factor, or teleports to a uniformly random node with the remaining
probability. The rank of a node is the long-run fraction of time the surfer
spends there. A node is important if important nodes point to it, a
recursive definition that the random walk resolves: each node passes its
rank out evenly along its edges every step, and a node's new rank is the
damped sum of what flows in plus its share of the teleport. Iterating that
update from a uniform start converges to the stationary distribution, the
power iteration, because damping makes the transition matrix irreducible
and aperiodic so a unique fixed point exists and the iteration reaches it
geometrically. Two details keep the numbers honest. A dangling node, one
with no outgoing edges, would swallow rank forever, so its rank is instead
spread evenly over all nodes each step, as if the surfer teleports from a
dead end. And the damping factor, conventionally 0.85, is what stops rank
from pooling in closed loops that never link out; with damping at one the
iteration can fail to converge and rank sinks into any trap. The ranker
iterates until the total change between rounds falls under a tolerance,
returns the rank per node summing to one, and refuses a damping factor
outside the open unit interval. It reports the iterations taken and the
top node, because a slow convergence is a graph with a near-trap structure
where the damping is doing real work to keep the walk mixing.
"""

from __future__ import annotations

from mesh.errors import Invalid
from mesh.graph import Graph


class PageRank:
    def __init__(
        self,
        graph: Graph,
        damping: float = 0.85,
        tolerance: float = 1e-9,
        max_iterations: int = 500,
    ) -> None:
        if not 0.0 < damping < 1.0:
            raise Invalid("damping must lie strictly between 0 and 1")
        if graph.node_count() == 0:
            raise Invalid("an empty graph has nothing to rank")
        self.graph = graph
        self.damping = damping
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.iterations = 0
        self.rank: dict[str, float] = self._run()

    def _run(self) -> dict[str, float]:
        nodes = self.graph.nodes()
        n = len(nodes)
        rank = dict.fromkeys(nodes, 1.0 / n)
        out_degree = {u: len(self.graph.neighbors(u)) for u in nodes}
        for _ in range(self.max_iterations):
            self.iterations += 1
            # rank stuck on dangling nodes is spread evenly, a teleport from a dead end
            dangling = sum(rank[u] for u in nodes if out_degree[u] == 0)
            base = (1.0 - self.damping) / n + self.damping * dangling / n
            new = dict.fromkeys(nodes, base)
            for u in nodes:
                if out_degree[u] == 0:
                    continue
                share = self.damping * rank[u] / out_degree[u]
                for v in self.graph.neighbors(u):
                    new[v] += share
            change = sum(abs(new[v] - rank[v]) for v in nodes)
            rank = new
            if change < self.tolerance:
                break
        return rank

    def top(self, k: int = 3) -> list[tuple[str, float]]:
        ranked = sorted(self.rank.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]

    def total(self) -> float:
        return sum(self.rank.values())

    def note(self) -> str:
        node, score = self.top(1)[0]
        return (
            f"converged in {self.iterations} iteration(s), top '{node}' at "
            f"{score:.3f}; slow convergence is a near-trap structure the damping "
            "is working to keep mixing"
        )
