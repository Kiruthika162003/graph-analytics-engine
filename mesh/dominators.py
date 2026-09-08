"""Dominators: which nodes every path from the entry must pass through.

In a control-flow graph with a single entry, a node d dominates a node n
if every path from the entry to n passes through d. Every node dominates
itself, the entry dominates everything, and the immediate dominator of n
is the closest strict dominator, the last unavoidable node before n. The
immediate dominators form a tree rooted at the entry, the dominator tree,
and it is the structure that compilers lean on for placing phi functions
in SSA form, for hoisting loop-invariant code, and for knowing which
checks are guaranteed to have already run when a block executes. The
brute-force definition, remove d and see whether n becomes unreachable,
costs a search per pair. The iterative algorithm of Cooper, Harvey, and
Kennedy computes all immediate dominators by a fixed-point sweep. Number
the nodes in reverse postorder of a depth-first search from the entry,
so that a node's predecessors mostly come earlier. Set the entry's
immediate dominator to itself and every other node's to undefined. Then
sweep the nodes in reverse postorder, setting each node's immediate
dominator to the intersection of its processed predecessors, where the
intersection of two nodes is found by walking each up the current
dominator tree toward the root until they meet, comparing by reverse
postorder number. Repeat the sweep until nothing changes. The
intersection walk is the whole algorithm and it converges in a few
passes on real control-flow graphs, matching the classic Lengauer-Tarjan
in practice with far less machinery. The engine returns the immediate
dominator of every reachable node, tests whether one node dominates
another, lists a node's dominators, and reports the tree's depth, because
a deep dominator tree is a long chain of unavoidable checkpoints and a
shallow, wide one is code where most blocks are reachable by many routes.
"""

from __future__ import annotations

from mesh.errors import Invalid, Missing
from mesh.graph import Graph


class Dominators:
    def __init__(self, graph: Graph, entry: str) -> None:
        if not graph.directed:
            raise Invalid("dominance is defined on a directed control-flow graph")
        if not graph.has_node(entry):
            raise Missing(f"entry '{entry}' is not in the graph")
        self.graph = graph
        self.entry = entry
        self.order = self._reverse_postorder()
        self._rank = {n: i for i, n in enumerate(self.order)}
        self.idom: dict[str, str] = {}
        self.passes = 0
        self._solve()

    def _reverse_postorder(self) -> list[str]:
        seen = {self.entry}
        post: list[str] = []
        stack: list[tuple[str, list[str]]] = [
            (self.entry, sorted(self.graph.neighbors(self.entry)))
        ]
        while stack:
            node, pending = stack[-1]
            if pending:
                nbr = pending.pop()
                if nbr not in seen:
                    seen.add(nbr)
                    stack.append((nbr, sorted(self.graph.neighbors(nbr))))
            else:
                post.append(node)
                stack.pop()
        post.reverse()
        return post

    def _intersect(self, a: str, b: str) -> str:
        # walk both up the current tree, lower rank first, until they meet
        while a != b:
            while self._rank[a] > self._rank[b]:
                a = self.idom[a]
            while self._rank[b] > self._rank[a]:
                b = self.idom[b]
        return a

    def _solve(self) -> None:
        preds: dict[str, list[str]] = {n: [] for n in self.order}
        reachable = set(self.order)
        for u in self.order:
            for v in self.graph.neighbors(u):
                if v in reachable:
                    preds[v].append(u)
        self.idom = {self.entry: self.entry}
        changed = True
        while changed:
            self.passes += 1
            changed = False
            for node in self.order[1:]:
                processed = [p for p in preds[node] if p in self.idom]
                new = processed[0]
                for p in processed[1:]:
                    new = self._intersect(p, new)
                if self.idom.get(node) != new:
                    self.idom[node] = new
                    changed = True

    def dominators_of(self, node: str) -> list[str]:
        if node not in self.idom:
            raise Missing(f"'{node}' is not reachable from the entry")
        chain = [node]
        while chain[-1] != self.entry:
            chain.append(self.idom[chain[-1]])
        return chain

    def dominates(self, d: str, n: str) -> bool:
        return d in self.dominators_of(n)

    def tree_depth(self) -> int:
        return max(len(self.dominators_of(n)) - 1 for n in self.idom)

    def note(self) -> str:
        return (
            f"dominator tree of depth {self.tree_depth()} over {len(self.idom)} "
            f"reachable node(s) in {self.passes} pass(es); deep is a long chain of "
            "unavoidable checkpoints, shallow and wide is many routes to most blocks"
        )
