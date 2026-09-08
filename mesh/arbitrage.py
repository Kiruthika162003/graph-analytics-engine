"""Arbitrage: a loop of exchange rates multiplying past one, found as a negative cycle.

Currencies are nodes and exchange rates are edges: an edge from dollars
to euros carrying the rate means one dollar buys that many euros. A
sequence of trades around a cycle multiplies the rates, and if the
product exceeds one the trader ends with more than they started, an
arbitrage. Products are awkward for shortest-path machinery, which
adds, so take logarithms: the log of a product is the sum of the logs,
and a product above one is a sum above zero. Negating turns that into a
sum below zero, so an arbitrage cycle is exactly a negative cycle in the
graph whose edge weights are minus the log of each rate, and Bellman-
Ford's detection of a negative cycle is a detection of free money. The
detector this engine already has refuses when a negative cycle exists,
naming one node on it; a trader needs the whole loop. The engine runs
the relaxation passes, takes a node still improving on the extra pass,
walks its parent pointers back n times to be sure of landing inside the
cycle, then follows the pointers around until the start repeats. The
product of the original rates around that loop is reported directly,
so the reader sees the profit rather than a negated logarithm. Rates
must be positive, since a zero or negative rate has no logarithm and no
meaning as an exchange, and the engine refuses them. It reports the
cycle, its multiplier, and how far above one the multiplier sits,
because a loop at one point zero zero one is a rounding artifact or a
spread the market will eat, while a loop at one point one is a mispriced
market.
"""

from __future__ import annotations

import math

from mesh.errors import Invalid, Missing


class Arbitrage:
    def __init__(self) -> None:
        self.rates: dict[tuple[str, str], float] = {}
        self.currencies: set[str] = set()

    def add_rate(self, source: str, target: str, rate: float) -> None:
        if rate <= 0:
            raise Invalid(f"rate {source}->{target} of {rate} has no logarithm")
        self.rates[(source, target)] = rate
        self.currencies.update((source, target))

    def _edges(self) -> list[tuple[str, str, float]]:
        return [(u, v, -math.log(r)) for (u, v), r in self.rates.items()]

    def find_cycle(self) -> list[str] | None:
        if not self.currencies:
            raise Missing("no rates have been added")
        nodes = sorted(self.currencies)
        n = len(nodes)
        # start everything at zero so a cycle anywhere is reachable
        dist = dict.fromkeys(nodes, 0.0)
        parent: dict[str, str | None] = dict.fromkeys(nodes, None)
        edges = self._edges()
        improved: str | None = None
        for _ in range(n):
            improved = None
            for u, v, w in edges:
                if dist[u] + w < dist[v] - 1e-12:
                    dist[v] = dist[u] + w
                    parent[v] = u
                    improved = v
        if improved is None:
            return None
        # walk back n times to be sure of standing on the cycle, then loop it
        node = improved
        for _ in range(n):
            node = parent[node]  # type: ignore[assignment]
        cycle = [node]
        cur = parent[node]
        while cur != node:
            cycle.append(cur)  # type: ignore[arg-type]
            cur = parent[cur]  # type: ignore[index]
        cycle.reverse()
        return cycle

    def multiplier(self, cycle: list[str]) -> float:
        product = 1.0
        for i in range(len(cycle)):
            product *= self.rates[(cycle[i], cycle[(i + 1) % len(cycle)])]
        return product

    def note(self) -> str:
        cycle = self.find_cycle()
        if cycle is None:
            return "no arbitrage: every cycle of rates multiplies to at most one"
        gain = self.multiplier(cycle)
        verdict = "a mispriced market" if gain > 1.01 else "a spread the market will eat"
        return (
            f"arbitrage {' -> '.join([*cycle, cycle[0]])} multiplies to {gain:.4f}, "
            f"{(gain - 1) * 100:.2f}% above one: {verdict}"
        )
