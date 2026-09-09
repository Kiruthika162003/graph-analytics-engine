"""Independence polynomial: how many independent sets of each size a graph has.

The independence polynomial lists, as coefficients, the number of
independent sets of every size: the constant term is one for the
empty set, the linear term is the node count, the quadratic term is
the number of non-edges, and the degree is the independence number.
It is computed by a recursion on a chosen node v: every independent
set either leaves v out, which is an independent set of G minus v, or
takes v in, which is an independent set of G minus v and its
neighbors, counted one size larger. Picking the highest-degree node
each time shrinks the graph fastest. The polynomial is worth having
beside the clique-finding modules because evaluating it at one counts
every independent set at once, and its degree is the independence
number found by a route that never enumerates the sets themselves,
which the module checks against the complement's clique search. A
matching polynomial is the independence polynomial of the line graph,
which is how this module also counts matchings of every size when
asked, through the line graph module. Known forms pin the arithmetic:
an edgeless graph on n nodes gives (1 plus x) to the n, a complete
graph gives 1 plus n x, and a path on n nodes has coefficients that
follow Fibonacci when summed. Above twenty nodes the recursion is
refused, and a directed graph is refused.
"""

from __future__ import annotations

from functools import cache

from mesh.complement import Complement
from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.linegraph import LineGraph

Poly = tuple[int, ...]


def _add(a: Poly, b: Poly) -> Poly:
    size = max(len(a), len(b))
    out = [0] * size
    for i, c in enumerate(a):
        out[i] += c
    for i, c in enumerate(b):
        out[i] += c
    return tuple(out)


def _shift(a: Poly) -> Poly:
    return (0, *a)


class IndependencePolynomial:
    def __init__(self, graph: Graph) -> None:
        if graph.directed:
            raise Invalid("independent sets ignore direction; pass an undirected graph")
        if graph.node_count() > 20:
            raise Invalid("the recursion is exponential; keep it to twenty nodes")
        self.graph = graph
        self.calls = 0
        self.coefficients = self._solve(frozenset(graph.nodes()))

    def _solve(self, nodes: frozenset[str]) -> Poly:
        @cache
        def rec(remaining: frozenset[str]) -> Poly:
            self.calls += 1
            if not remaining:
                return (1,)
            # the busiest remaining node shrinks the graph fastest when taken
            def busy(n: str) -> tuple[int, str]:
                return sum(1 for m in self.graph.neighbors(n) if m in remaining), n

            v = max(remaining, key=busy)
            without = remaining - {v}
            closed = without - set(self.graph.neighbors(v))
            return _add(rec(without), _shift(rec(closed)))

        return rec(nodes)

    def evaluate(self, x: int) -> int:
        return sum(c * x**i for i, c in enumerate(self.coefficients))

    def independence_number(self) -> int:
        return len(self.coefficients) - 1

    def total_sets(self) -> int:
        return self.evaluate(1)

    def matches_complement_clique(self) -> bool:
        return self.independence_number() == Complement(self.graph).independence_number()

    def matching_polynomial(self) -> Poly:
        # matchings are independent sets of edges, so run the recursion on the line graph
        line = LineGraph(self.graph).line
        return IndependencePolynomial(line).coefficients

    def note(self) -> str:
        return (
            f"I(x) coefficients {list(self.coefficients)}: {self.total_sets()} independent "
            f"set(s), independence number {self.independence_number()}"
        )
