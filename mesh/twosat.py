"""2-SAT by strongly connected components: clauses become arrows, a contradiction a cycle.

A 2-SAT instance is a conjunction of clauses with two literals each,
and it is satisfiable exactly when no variable ends up in the same
strongly connected component as its negation. The reduction is the
implication graph: a clause (a or b) means not-a implies b and not-b
implies a, so each clause becomes two arrows on the nodes for every
literal and its negation. If x and not-x reach each other, then
choosing either forces the other and the instance has no solution. If
they never do, a solution comes from the condensation: process the
components in reverse topological order and set a literal true when
its component comes before its negation's, which Aspvall, Plass, and
Tarjan showed is always consistent. The engine builds the implication
graph, runs an iterative Tarjan to find the components and their
order, decides satisfiability, names the first variable that
contradicts itself, produces an assignment when one exists, and
checks that assignment against every clause, which is the test that
matters because a wrong component order produces a confident wrong
answer. A clause naming a variable and its own negation is a
tautology and is kept, since it constrains nothing; a clause with the
same literal twice is a unit clause and works as written. Variables
are strings, and a literal is a pair of the variable and a truth
value.
"""

from __future__ import annotations

from mesh.errors import Invalid

Literal = tuple[str, bool]
Clause = tuple[Literal, Literal]


class TwoSat:
    def __init__(self, clauses: list[Clause]) -> None:
        self.clauses = list(clauses)
        self.variables = sorted({var for clause in clauses for var, _val in clause})
        self.arrows: dict[Literal, list[Literal]] = {}
        for var in self.variables:
            self.arrows[(var, True)] = []
            self.arrows[(var, False)] = []
        for a, b in clauses:
            self.arrows[self._neg(a)].append(b)
            self.arrows[self._neg(b)].append(a)
        self.component: dict[Literal, int] = {}
        self._tarjan()
        clashing = (
            v for v in self.variables if self.component[(v, True)] == self.component[(v, False)]
        )
        self.contradiction = next(clashing, None)
        self.satisfiable = self.contradiction is None

    @staticmethod
    def _neg(lit: Literal) -> Literal:
        return (lit[0], not lit[1])

    def _tarjan(self) -> None:
        # iterative Tarjan; components are numbered in the order they complete, which is
        # reverse topological order of the condensation
        index: dict[Literal, int] = {}
        low: dict[Literal, int] = {}
        on_stack: set[Literal] = set()
        stack: list[Literal] = []
        counter = 0
        comp = 0
        for root in self.arrows:
            if root in index:
                continue
            work = [(root, iter(self.arrows[root]))]
            index[root] = low[root] = counter
            counter += 1
            stack.append(root)
            on_stack.add(root)
            while work:
                node, children = work[-1]
                advanced = False
                for child in children:
                    if child not in index:
                        index[child] = low[child] = counter
                        counter += 1
                        stack.append(child)
                        on_stack.add(child)
                        work.append((child, iter(self.arrows[child])))
                        advanced = True
                        break
                    if child in on_stack:
                        low[node] = min(low[node], index[child])
                if advanced:
                    continue
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
                if low[node] == index[node]:
                    while True:
                        top = stack.pop()
                        on_stack.discard(top)
                        self.component[top] = comp
                        if top == node:
                            break
                    comp += 1

    def assignment(self) -> dict[str, bool]:
        if not self.satisfiable:
            raise Invalid(f"unsatisfiable: '{self.contradiction}' implies its own negation")
        # a lower component number completes earlier, so it sits later in topological order
        return {
            v: self.component[(v, True)] < self.component[(v, False)] for v in self.variables
        }

    def satisfies(self, values: dict[str, bool]) -> bool:
        return all(values[a[0]] == a[1] or values[b[0]] == b[1] for a, b in self.clauses)

    def note(self) -> str:
        if not self.satisfiable:
            return f"unsatisfiable: '{self.contradiction}' and its negation share a component"
        chosen = ", ".join(f"{v}={'T' if val else 'F'}" for v, val in self.assignment().items())
        return f"satisfiable over {len(self.variables)} variable(s): {chosen}"
