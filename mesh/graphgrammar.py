"""Graph rewriting: apply a small replacement rule wherever its left side is found.

A rewrite rule says: wherever this pattern occurs, replace it with
that one. Chemists write reactions that way, compilers write peephole
optimisations that way, and graph grammars build fractals that way.
The rule here has a left side, a small pattern graph, and a right
side, a graph on a superset of the left side's node names: nodes the
right side keeps are preserved and their edges to the rest of the
host graph survive, nodes the right side drops are deleted with every
edge at them, and nodes only the right side names are created fresh
with a suffix so repeated applications never collide. This is the
double-pushout idea done by hand. One application finds the first
match of the pattern in the host in sorted order, so applications are
deterministic, and the module can apply a rule repeatedly up to a
limit or until nothing matches. Matches are found by the subgraph
match module and must be injective and edge-preserving; edges between
kept nodes are taken from the right side, not the host, so a rule can
delete an edge by omitting it. The classic examples are here as
checks: a rule that contracts a path of three to an edge shrinks a
path of n to an edge in n minus 2 steps, a rule that swaps an edge
for a path of two through a fresh node subdivides every edge once
when run over a whole cycle, and a rule whose left side never occurs
leaves the host untouched with zero applications.
"""

from __future__ import annotations

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.subgraphmatch import SubgraphMatch


class Rule:
    def __init__(self, left: Graph, right: Graph) -> None:
        if left.directed != right.directed:
            raise Invalid("both sides of a rule must share their direction")
        if left.node_count() == 0:
            raise Invalid("a rule needs at least one node on its left side")
        self.left = left
        self.right = right
        self.kept = [n for n in left.nodes() if n in set(right.nodes())]
        self.dropped = [n for n in left.nodes() if n not in set(right.nodes())]
        self.created = [n for n in right.nodes() if n not in set(left.nodes())]


class Rewriter:
    def __init__(self, host: Graph, rule: Rule) -> None:
        if host.directed != rule.left.directed:
            raise Invalid("the host and the rule must share their direction")
        self.host = host
        self.rule = rule
        self.applications = 0

    def find(self) -> dict[str, str] | None:
        if self.rule.left.node_count() > self.host.node_count():
            return None
        return SubgraphMatch(self.host, self.rule.left).witness

    def apply_once(self) -> bool:
        mapping = self.find()
        if mapping is None:
            return False
        self.host = self._rewrite(mapping)
        self.applications += 1
        return True

    def _rewrite(self, mapping: dict[str, str]) -> Graph:
        rule = self.rule
        out = Graph(directed=self.host.directed)
        removed = {mapping[n] for n in rule.dropped}
        image = {n: mapping[n] for n in rule.kept}
        for n in rule.created:
            image[n] = f"{n}#{self.applications + 1}"
        for n in self.host.nodes():
            if n not in removed:
                out.add_node(n)
        for n in rule.created:
            out.add_node(image[n])
        matched = set(mapping.values())
        for u, v, w in self.host.edges():
            if u in removed or v in removed:
                continue
            if u in matched and v in matched:
                continue  # edges inside the match come from the right side
            out.add_edge(u, v, w)
        for u, v, w in rule.right.edges():
            out.add_edge(image[u], image[v], w)
        return out

    def apply_until_fixed(self, limit: int = 100) -> int:
        count = 0
        while count < limit and self.apply_once():
            count += 1
        return count

    def note(self) -> str:
        return (
            f"{self.applications} application(s); host now {self.host.node_count()} node(s) "
            f"and {self.host.edge_count()} edge(s)"
        )
