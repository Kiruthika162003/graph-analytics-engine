"""SIR epidemics on a graph: who catches it, how many in the end, and the spectral threshold.

The susceptible-infected-recovered model runs in rounds: every
infected node passes the infection to each susceptible neighbor with
probability beta, then recovers with probability gamma and is immune
after. The outcome that matters is the final size, the fraction of
nodes that were ever infected, averaged over seeded trials, and the
question that matters is whether an outbreak takes off at all. On a
graph the threshold is spectral: an outbreak is expected to die out
when beta over gamma is below one over the largest adjacency
eigenvalue and to spread when it is above, which Wang and colleagues
showed and Chakrabarti made standard, and which puts the spectral
radius from the energy module to work. The engine runs trials with a
seeded generator from a chosen patient zero or a random one, records
the final size and the peak number infected in one round, sweeps
beta to find where the mean final size first exceeds a tenth of the
graph, and compares that with the spectral estimate gamma over the
spectral radius. The identities the tests hold are exact: with beta
zero only the seed is ever infected, with beta one and gamma one on a
connected graph the whole graph is infected in one round per hop from
the seed, and the peak never exceeds the final size. A directed graph
is followed along its arcs, since infection is directional, and the
spectral estimate is then reported as absent because the adjacency
matrix is not symmetric.
"""

from __future__ import annotations

import random

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.graphenergy import GraphEnergy


class Epidemic:
    def __init__(self, graph: Graph, beta: float, gamma: float, seed: int = 0) -> None:
        if not 0.0 <= beta <= 1.0 or not 0.0 < gamma <= 1.0:
            raise Invalid("beta lies in 0..1 and gamma in (0, 1]")
        self.graph = graph
        self.beta = beta
        self.gamma = gamma
        self.rng = random.Random(seed)
        self.nodes = graph.nodes()

    def run(self, patient_zero: str | None = None) -> tuple[int, int, int]:
        # returns (ever infected, peak infected in one round, rounds)
        if not self.nodes:
            return 0, 0, 0
        start = patient_zero if patient_zero is not None else self.rng.choice(self.nodes)
        if start not in self.nodes:
            raise Invalid(f"'{start}' is not a node of the graph")
        infected = {start}
        recovered: set[str] = set()
        peak = 1
        rounds = 0
        while infected:
            rounds += 1
            fresh: set[str] = set()
            for node in sorted(infected):
                for other in self.graph.neighbors(node):
                    untouched = other not in infected and other not in recovered
                    if untouched and other not in fresh and self.rng.random() < self.beta:
                        fresh.add(other)
            healed = {n for n in infected if self.rng.random() < self.gamma}
            recovered |= healed
            infected = (infected - healed) | fresh
            peak = max(peak, len(infected))
        return len(recovered), peak, rounds

    def final_size(self, trials: int = 20, patient_zero: str | None = None) -> float:
        if trials < 1:
            raise Invalid("at least one trial is needed")
        if not self.nodes:
            return 0.0
        total = sum(self.run(patient_zero)[0] for _ in range(trials))
        return total / (trials * len(self.nodes))

    def spectral_threshold(self) -> float | None:
        if self.graph.directed:
            return None
        radius = GraphEnergy(self.graph).spectral_radius()
        return self.gamma / radius if radius > 0 else None

    def observed_threshold(self, steps: int = 10, trials: int = 20) -> float:
        for k in range(steps + 1):
            beta = k / steps
            trial = Epidemic(self.graph, beta, self.gamma, seed=self.rng.randrange(1 << 30))
            if trial.final_size(trials) > 0.1:
                return beta
        return 1.0

    def note(self, trials: int = 20) -> str:
        size = self.final_size(trials)
        spectral = self.spectral_threshold()
        shown = f"{spectral:.3f}" if spectral is not None else "absent on a directed graph"
        return (
            f"beta {self.beta} gamma {self.gamma}: mean final size {size:.2f} of the graph "
            f"over {trials} trial(s); spectral threshold for beta {shown}"
        )
