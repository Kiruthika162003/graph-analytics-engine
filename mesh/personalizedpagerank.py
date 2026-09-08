"""Personalized PageRank: importance as seen from one node's neighborhood.

Plain PageRank teleports to a uniformly random node, so it measures
importance from everyone's point of view at once. Personalized PageRank
changes one thing: the teleport lands on a chosen seed set instead of
anywhere. The random surfer follows links as before, but whenever the
damping sends it home it returns to one of the seeds, so the stationary
distribution concentrates around the seeds and fades with distance from
them along the graph's own paths. The result is a proximity score, how
close every node is to the seed set in the sense of random-walk
reachability, which is what a recommender uses when it asks which items
are near the ones this user already likes, and what a search engine uses
to bias rankings toward a topic. The scores are a probability
distribution summing to one, the seeds themselves score highest, and a
node reachable from the seeds only through long chains scores nearly
zero. Dangling nodes are handled as in plain PageRank, except that their
mass returns to the seeds rather than spreading uniformly, which keeps
the personalization honest: mass never leaks away from the seed
neighborhood through a dead end. The measure iterates until the change
falls under a tolerance, returns the scores, the top nodes excluding the
seeds themselves, which is the recommendation, and refuses an empty seed
set or a seed outside the graph. It reports how much of the total mass
sits within one hop of the seeds, because a share near one says the
seeds' neighborhood is a closed pocket the walk rarely leaves, while a
low share says the seeds sit in a well-mixed region and their
personalization reaches far.
"""

from __future__ import annotations

from mesh.errors import Invalid, Missing
from mesh.graph import Graph


class PersonalizedPageRank:
    def __init__(
        self,
        graph: Graph,
        seeds: list[str],
        damping: float = 0.85,
        tolerance: float = 1e-10,
        max_iterations: int = 1000,
    ) -> None:
        if not seeds:
            raise Invalid("personalization needs at least one seed")
        for s in seeds:
            if not graph.has_node(s):
                raise Missing(f"seed '{s}' is not in the graph")
        if not 0.0 < damping < 1.0:
            raise Invalid("damping must lie strictly between 0 and 1")
        self.graph = graph
        self.seeds = sorted(set(seeds))
        self.damping = damping
        self.iterations = 0
        self.rank = self._run(tolerance, max_iterations)

    def _run(self, tolerance: float, max_iterations: int) -> dict[str, float]:
        nodes = self.graph.nodes()
        seed_share = 1.0 / len(self.seeds)
        home = {n: (seed_share if n in self.seeds else 0.0) for n in nodes}
        rank = dict(home)
        out_degree = {u: len(self.graph.neighbors(u)) for u in nodes}
        for _ in range(max_iterations):
            self.iterations += 1
            dangling = sum(rank[u] for u in nodes if out_degree[u] == 0)
            # teleport and dangling mass both return to the seeds, not everywhere
            new = {n: (1.0 - self.damping + self.damping * dangling) * home[n] for n in nodes}
            for u in nodes:
                if out_degree[u] == 0:
                    continue
                share = self.damping * rank[u] / out_degree[u]
                for v in self.graph.neighbors(u):
                    new[v] += share
            change = sum(abs(new[n] - rank[n]) for n in nodes)
            rank = new
            if change < tolerance:
                break
        return rank

    def recommendations(self, k: int = 3) -> list[tuple[str, float]]:
        others = [(n, s) for n, s in self.rank.items() if n not in self.seeds]
        return sorted(others, key=lambda kv: (-kv[1], kv[0]))[:k]

    def total(self) -> float:
        return sum(self.rank.values())

    def neighborhood_share(self) -> float:
        near = set(self.seeds)
        for s in self.seeds:
            near.update(self.graph.neighbors(s))
        return sum(self.rank[n] for n in near)

    def note(self) -> str:
        recs = self.recommendations(1)
        top = recs[0][0] if recs else "nothing outside the seeds"
        return (
            f"{self.neighborhood_share() * 100:.0f}% of the mass within one hop of "
            f"{self.seeds}, top recommendation '{top}' after {self.iterations} "
            "iteration(s); near 100 is a closed pocket, low is a well-mixed region"
        )
