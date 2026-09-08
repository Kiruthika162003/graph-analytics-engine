"""A generators day: three random models, and the fingerprints that tell them apart.

Run with: python -m examples.generatorsday
"""

from __future__ import annotations

from mesh.connectedcomponents import ConnectedComponents
from mesh.degreedistribution import DegreeDistribution
from mesh.diameter import Diameter
from mesh.generators import barabasi_albert, erdos_renyi, fingerprint, watts_strogatz
from mesh.graph import Graph


def largest_component(g: Graph) -> Graph:
    biggest = ConnectedComponents(g).components()[0]
    sub = Graph()
    for n in biggest:
        sub.add_node(n)
    for u, v, w in g.edges():
        if u in biggest and v in biggest:
            sub.add_edge(u, v, w)
    return sub


def main() -> int:
    n = 150
    models = {
        "erdos-renyi": erdos_renyi(n, 4 / (n - 1), seed=1),
        "barabasi-albert": barabasi_albert(n, 2, seed=1),
        "watts-strogatz": watts_strogatz(n, 4, 0.1, seed=1),
    }
    for name, g in models.items():
        f = fingerprint(g)
        dd = DegreeDistribution(g)
        core = largest_component(g)
        span = Diameter(core).diameter()
        print(f"{name:<16} hubs x{f['max_over_mean']:.1f}  clustering {f['clustering']:.3f}")
        print(f"{'':<16} variance/mean {dd.variance_ratio():.2f}  diameter {span}")
    print("hubs mark preferential attachment, clustering marks the small world")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
