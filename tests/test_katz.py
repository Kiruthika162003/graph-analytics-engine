from __future__ import annotations

from itertools import combinations

import pytest

from mesh.errors import Invalid
from mesh.graph import Graph
from mesh.katz import Katz


def _hub_and_tail() -> Graph:
    g = Graph()
    for n in "abcd":
        g.add_node(n)
    for x, y in combinations("abc", 2):
        g.add_edge(x, y)
    g.add_edge("a", "d")
    return g


class TestScores:
    def test_the_best_connected_node_ranks_first(self):
        assert Katz(_hub_and_tail(), alpha=0.2).top(1)[0][0] == "a"

    def test_the_result_solves_the_katz_equation(self):
        g = _hub_and_tail()
        k = Katz(g, alpha=0.2)
        # the unnormalized solution satisfies x = beta + alpha A x; after
        # scaling to unit length that becomes (I - alpha A) s = constant, so
        # the residual s[n] - alpha * sum of neighbor scores is the same for
        # every node. A first version divided s by beta + alpha A s instead,
        # which is not constant once s is normalized, and failed.
        residual = [
            k.score[n] - 0.2 * sum(k.score[m] for m in g.neighbors(n)) for n in g.nodes()
        ]
        assert max(residual) - min(residual) < 1e-6
        assert residual[0] > 0

    def test_a_small_alpha_tracks_degree(self):
        g = _hub_and_tail()
        k = Katz(g, alpha=0.01)
        by_degree = sorted(g.nodes(), key=lambda n: (-g.degree(n), n))
        by_katz = [n for n, _s in k.top(4)]
        assert by_katz == by_degree

    def test_the_vector_has_unit_length(self):
        k = Katz(_hub_and_tail(), alpha=0.2)
        assert sum(x * x for x in k.score.values()) == pytest.approx(1.0)

    def test_direction_is_respected_on_a_digraph(self):
        g = Graph(directed=True)
        for n in "abc":
            g.add_node(n)
        g.add_edge("a", "c")
        g.add_edge("b", "c")
        k = Katz(g, alpha=0.3)
        assert k.top(1)[0][0] == "c"


class TestAlphaLimit:
    def test_alpha_at_the_limit_is_refused(self):
        g = _hub_and_tail()
        radius = Katz(g, alpha=0.01).spectral_radius
        with pytest.raises(Invalid) as caught:
            Katz(g, alpha=1.0 / radius)
        assert "diverge" in str(caught.value)

    def test_the_spectral_radius_of_a_complete_graph_is_k_minus_one(self):
        g = Graph()
        for n in "abcde":
            g.add_node(n)
        for x, y in combinations("abcde", 2):
            g.add_edge(x, y)
        assert Katz(g, alpha=0.1).spectral_radius == pytest.approx(4.0)

    def test_alpha_share_is_alpha_times_radius(self):
        g = _hub_and_tail()
        k = Katz(g, alpha=0.2)
        assert k.alpha_share() == pytest.approx(0.2 * k.spectral_radius)


class TestRefusals:
    def test_a_non_positive_alpha_is_refused(self):
        with pytest.raises(Invalid):
            Katz(_hub_and_tail(), alpha=0.0)

    def test_an_empty_graph_is_refused(self):
        with pytest.raises(Invalid):
            Katz(Graph())


class TestReport:
    def test_the_note_states_the_alpha_share_and_top(self):
        note = Katz(_hub_and_tail(), alpha=0.2).note()
        assert "% of its limit" in note
        assert "top 'a'" in note
