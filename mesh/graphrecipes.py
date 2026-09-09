"""Recipes: named multi-step analyses that run several modules in a fixed order and explain.

A reader who wants to know whether a network has communities does not
want to pick a modularity method; they want the question answered
with the steps shown. A recipe is a named question with a fixed
sequence of module calls and a written verdict at the end. Four live
here. The community recipe reads transitivity against its null model,
runs spectral clustering at the k the Laplacian gap suggests, and
reports the groups with the crossing edge count. The bottleneck
recipe runs the Gomory-Hu tree to find the pair with the smallest cut
and names the edges of that cut. The robustness recipe removes nodes
in order of degree and reports how many removals it takes to halve
the largest piece, beside the same count for random removals, which
is the Albert, Jeong, and Barabasi comparison. The influence recipe
ranks nodes by three centralities and reports which node tops the
most rankings. Each recipe returns its verdict as a list of lines, so
the command line can print it, and each is tested on a graph whose
answer is known: three cliques on bridges have three communities and
a bottleneck of one edge, a star halves after removing its hub, and a
hub tops every ranking. The recipes refuse a directed graph where
the modules under them do, and say which module refused.
"""

from __future__ import annotations

import random
from collections import Counter, deque

from mesh.errors import Invalid, MeshError
from mesh.gomoryhu import GomoryHu
from mesh.graph import Graph
from mesh.harmonic import Harmonic
from mesh.laplacianspectrum import LaplacianSpectrum
from mesh.nullmodel import NullModel
from mesh.spectralclustering import SpectralClustering
from mesh.stresscentrality import StressCentrality


def _largest_piece(graph: Graph, removed: set[str]) -> int:
    seen: set[str] = set(removed)
    best = 0
    for start in graph.nodes():
        if start in seen:
            continue
        size = 1
        seen.add(start)
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for other in graph.neighbors(node):
                if other not in seen:
                    seen.add(other)
                    size += 1
                    queue.append(other)
        best = max(best, size)
    return best


class Recipes:
    def __init__(self, graph: Graph, seed: int = 0) -> None:
        self.graph = graph
        self.seed = seed

    def _undirected(self, recipe: str) -> None:
        if self.graph.directed:
            raise Invalid(f"the {recipe} recipe runs on an undirected graph")

    def communities(self) -> list[str]:
        self._undirected("community")
        if self.graph.node_count() < 3:
            return ["too few nodes to look for communities"]
        null = NullModel(self.graph, samples=10, seed=self.seed).transitivity()
        spectrum = LaplacianSpectrum(self.graph).values
        gaps = [(spectrum[i + 1] - spectrum[i], i + 1) for i in range(1, len(spectrum) - 1)]
        k = max(gaps, key=lambda g: (g[0], -g[1]))[1] if gaps else 1
        k = max(1, min(k, self.graph.node_count()))
        sc = SpectralClustering(self.graph, k)
        lines = [
            f"transitivity {null['observed']:.3f} against a null of {null['mean']:.3f} "
            f"(z {null['z']:.1f})",
            f"the Laplacian gap suggests {k} group(s): {sc.note()}",
        ]
        for group in sc.groups:
            lines.append("  " + ", ".join(group))
        few_crossing = sc.crossing_edges() < self.graph.edge_count() / 4
        clear = null["z"] > 2 and few_crossing
        verdict = "clear communities" if clear else "no strong community structure"
        lines.append(f"verdict: {verdict}")
        return lines

    def bottleneck(self) -> list[str]:
        self._undirected("bottleneck")
        if self.graph.node_count() < 2:
            return ["too few nodes for a bottleneck"]
        gh = GomoryHu(self.graph)
        pairs = [
            (gh.min_cut(a, b), a, b) for i, a in enumerate(gh.nodes) for b in gh.nodes[i + 1 :]
        ]
        value, a, b = min(pairs, key=lambda t: (t[0], t[1], t[2]))
        lines = [f"the thinnest cut separates {a} from {b} at {value:g}"]
        if value == 0:
            lines.append("verdict: the graph is already in pieces")
        else:
            lines.append(f"verdict: {value:g} unit(s) of capacity is all that joins them")
        return lines

    def robustness(self) -> list[str]:
        self._undirected("robustness")
        n = self.graph.node_count()
        if n < 2:
            return ["too few nodes to attack"]
        target = _largest_piece(self.graph, set()) / 2
        by_degree = sorted(self.graph.nodes(), key=lambda x: (-self.graph.degree(x), x))
        rng = random.Random(self.seed)
        at_random = list(self.graph.nodes())
        rng.shuffle(at_random)
        counts = {}
        for label, order in (("targeted", by_degree), ("random", at_random)):
            removed: set[str] = set()
            steps = 0
            for node in order:
                if _largest_piece(self.graph, removed) <= target:
                    break
                removed.add(node)
                steps += 1
            counts[label] = steps
        lines = [
            f"halving the largest piece takes {counts['targeted']} targeted removal(s) "
            f"and {counts['random']} random one(s)"
        ]
        if counts["targeted"] < counts["random"]:
            lines.append("verdict: fragile to a targeted attack, sturdier against random loss")
        else:
            lines.append("verdict: no easier to break by targeting hubs than at random")
        return lines

    def influence(self) -> list[str]:
        if self.graph.node_count() == 0:
            return ["no nodes to rank"]
        stress = StressCentrality(self.graph)
        rankings = {
            "degree": sorted(self.graph.nodes(), key=lambda x: (-self.graph.degree(x), x)),
            "betweenness": stress.ranking("betweenness"),
            "harmonic": Harmonic(self.graph).ranking(),
        }
        tops = Counter(order[0] for order in rankings.values())
        leader, wins = tops.most_common(1)[0]
        lines = [f"{label}: {', '.join(order[:3])}" for label, order in rankings.items()]
        lines.append(f"verdict: {leader} tops {wins} of 3 rankings")
        return lines

    def run(self, name: str) -> list[str]:
        table = {
            "communities": self.communities,
            "bottleneck": self.bottleneck,
            "robustness": self.robustness,
            "influence": self.influence,
        }
        if name not in table:
            return [f"no recipe called '{name}'; try {', '.join(table)}"]
        try:
            return table[name]()
        except MeshError as exc:
            return [f"the {name} recipe stopped: {exc}"]
