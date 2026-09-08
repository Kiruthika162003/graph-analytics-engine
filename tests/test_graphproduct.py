from __future__ import annotations

from collections import deque

import pytest

from mesh.errors import Invalid
from mesh.factories import complete, cycle, path
from mesh.graph import Graph
from mesh.graphproduct import GraphProduct


def _distances(g: Graph, start: str) -> dict[str, int]:
    dist = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for other in g.neighbors(node):
            if other not in dist:
                dist[other] = dist[node] + 1
                queue.append(other)
    return dist


def _edge() -> Graph:
    g = Graph()
    g.add_node("0")
    g.add_node("1")
    g.add_edge("0", "1")
    return g


class TestCartesian:
    def test_a_path_times_a_path_is_a_grid(self):
        grid = GraphProduct(path(3), path(4)).cartesian()
        assert grid.node_count() == 12
        assert grid.edge_count() == 3 * 3 + 4 * 2
        corners = [n for n in grid.nodes() if grid.degree(n) == 2]
        assert len(corners) == 4

    def test_edge_times_edge_times_edge_is_a_cube(self):
        square = GraphProduct(_edge(), _edge()).cartesian()
        cube = GraphProduct(square, _edge()).cartesian()
        assert cube.node_count() == 8
        assert cube.edge_count() == 12
        assert all(cube.degree(n) == 3 for n in cube.nodes())

    def test_distances_add_across_the_coordinates(self):
        a, b = path(4), cycle(5)
        prod = GraphProduct(a, b).cartesian()
        da = _distances(a, "0")
        db = _distances(b, "0")
        dp = _distances(prod, "0,0")
        for x in a.nodes():
            for y in b.nodes():
                assert dp[f"{x},{y}"] == da[x] + db[y]

    def test_degrees_add(self):
        prod = GraphProduct(complete(3), path(3)).cartesian()
        assert prod.degree("0,1") == 2 + 2
        assert prod.degree("0,0") == 2 + 1


class TestTensor:
    def test_degrees_multiply_and_edges_double_the_product(self):
        a, b = complete(3), path(3)
        prod = GraphProduct(a, b).tensor()
        assert prod.edge_count() == 2 * a.edge_count() * b.edge_count()
        assert prod.degree("0,1") == 2 * 2
        assert prod.degree("0,0") == 2 * 1

    def test_an_edge_times_an_edge_is_two_separate_edges(self):
        prod = GraphProduct(_edge(), _edge()).tensor()
        assert prod.edge_count() == 2
        assert prod.has_edge("0,0", "1,1")
        assert prod.has_edge("0,1", "1,0")
        assert not prod.has_edge("0,0", "0,1")


class TestStrong:
    def test_the_strong_product_is_the_union_of_the_other_two(self):
        gp = GraphProduct(path(3), path(3))
        strong = gp.strong()
        assert strong.edge_count() == gp.cartesian().edge_count() + gp.tensor().edge_count()
        # the centre of a three by three board has eight king moves
        assert strong.degree("1,1") == 8

    def test_predicted_counts_match_built_counts_on_every_product(self):
        gp = GraphProduct(cycle(4), path(3))
        for kind, build in (
            ("cartesian", gp.cartesian),
            ("tensor", gp.tensor),
            ("strong", gp.strong),
        ):
            assert build().edge_count() == gp.predicted_edges(kind)


class TestRefusal:
    def test_directed_factors_and_unknown_kinds_are_refused(self):
        with pytest.raises(Invalid):
            GraphProduct(Graph(directed=True), path(2))
        with pytest.raises(Invalid):
            GraphProduct(path(2), path(2)).predicted_edges("lexicographic")


class TestReport:
    def test_the_note_shows_built_beside_predicted(self):
        note = GraphProduct(_edge(), _edge()).note()
        assert "2x2 nodes" in note
        assert "cartesian 4 edge(s), predicted 4" in note
        assert "tensor 2 edge(s), predicted 2" in note
        assert "strong 6 edge(s), predicted 6" in note
