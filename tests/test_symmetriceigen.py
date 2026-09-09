from __future__ import annotations

import random
from math import cos, pi, sqrt

import pytest

from mesh.errors import Invalid
from mesh.factories import complete, cycle, path, star
from mesh.graph import Graph
from mesh.symmetriceigen import SymmetricEigen


def _adjacency(g: Graph) -> list[list[float]]:
    nodes = g.nodes()
    return [[1.0 if g.has_edge(a, b) else 0.0 for b in nodes] for a in nodes]


class TestSmallMatrices:
    def test_a_diagonal_matrix_returns_its_diagonal_sorted(self):
        se = SymmetricEigen([[3.0, 0.0], [0.0, 1.0]])
        assert se.values == [1.0, 3.0]
        assert se.rotations == 0

    def test_a_two_by_two_with_known_roots(self):
        se = SymmetricEigen([[2.0, 1.0], [1.0, 2.0]])
        assert se.values == pytest.approx([1.0, 3.0])

    def test_identities_and_residual_hold_on_random_symmetric_matrices(self):
        rng = random.Random(677)
        for _ in range(10):
            n = 6
            m = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(i, n):
                    m[i][j] = m[j][i] = rng.uniform(-3, 3)
            se = SymmetricEigen(m)
            assert se.trace_identity_holds()
            assert se.norm_identity_holds()
            assert se.residual() < 1e-8


class TestGraphSpectra:
    def test_a_complete_graph_has_n_minus_one_and_minus_one(self):
        se = SymmetricEigen(_adjacency(complete(5)))
        assert se.values == pytest.approx([-1, -1, -1, -1, 4])

    def test_a_cycle_has_two_cosine_eigenvalues(self):
        n = 6
        se = SymmetricEigen(_adjacency(cycle(n)))
        expected = sorted(2 * cos(2 * pi * k / n) for k in range(n))
        assert se.values == pytest.approx(expected, abs=1e-9)

    def test_a_star_has_plus_and_minus_root_of_the_leaves(self):
        se = SymmetricEigen(_adjacency(star(4)))
        assert se.values[0] == pytest.approx(-2.0)
        assert se.values[-1] == pytest.approx(2.0)
        assert se.values[1:-1] == pytest.approx([0, 0, 0])

    def test_squares_sum_to_twice_the_edges_on_a_path(self):
        g = path(7)
        se = SymmetricEigen(_adjacency(g))
        assert sum(v * v for v in se.values) == pytest.approx(2 * g.edge_count())
        assert se.values[-1] == pytest.approx(2 * cos(pi / 8))
        assert sqrt(2) < se.values[-1] < 2


class TestRefusal:
    def test_a_ragged_or_asymmetric_matrix_is_refused_by_entry(self):
        with pytest.raises(Invalid, match="row 1 has 1 entries in a matrix of size 2"):
            SymmetricEigen([[1.0, 2.0], [2.0]])
        with pytest.raises(Invalid, match=r"entry \(0,1\)"):
            SymmetricEigen([[1.0, 2.0], [3.0, 1.0]])


class TestReport:
    def test_the_note_prints_sorted_values_and_the_residual(self):
        note = SymmetricEigen([[2.0, 1.0], [1.0, 2.0]]).note()
        assert note.startswith("eigenvalues [1.0000, 3.0000] after 1 rotation(s)")
        assert "residual" in note
