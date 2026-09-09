from __future__ import annotations

from mesh.factories import path
from mesh.graph import Graph
from mesh.graphquery import GraphQuery


def _office() -> Graph:
    g = Graph()
    for n in ("ada", "ben", "cal", "dee", "eve"):
        g.add_node(n)
    for a, b in [("ada", "ben"), ("ben", "cal"), ("cal", "dee")]:
        g.add_edge(a, b)
    return g


class TestVerbs:
    def test_degree_adjacent_and_neighbors(self):
        q = GraphQuery(_office())
        assert q.ask(["degree", "ben"]) == "2"
        assert q.ask(["adjacent", "ada", "ben"]) == "yes"
        assert q.ask(["adjacent", "ada", "cal"]) == "no"
        assert q.ask(["neighbors", "ben"]) == "ada, cal"
        assert q.ask(["neighbors", "eve"]) == "none"

    def test_path_and_distance(self):
        q = GraphQuery(_office())
        assert q.ask(["path", "ada", "dee"]) == "ada > ben > cal > dee"
        assert q.ask(["distance", "ada", "dee"]) == "3"
        assert q.ask(["path", "ada", "eve"]) == "no path from ada to eve"
        assert q.ask(["distance", "ada", "eve"]) == "unreachable"

    def test_counts_component_and_summary(self):
        q = GraphQuery(_office())
        assert q.ask(["nodes"]) == "5"
        assert q.ask(["edges"]) == "3"
        assert q.ask(["component", "cal"]) == "ada, ben, cal, dee"
        assert q.ask(["summary"]).startswith("undirected graph with 5 node(s)")


class TestMistakes:
    def test_unknown_verbs_wrong_counts_and_missing_nodes_are_named(self):
        q = GraphQuery(path(3))
        assert q.ask(["fly"]) == "unknown verb 'fly'; try help"
        assert q.ask(["degree"]) == "'degree' takes 1 argument(s), got 0"
        assert q.ask(["adjacent", "0"]) == "'adjacent' takes 2 argument(s), got 1"
        assert q.ask(["degree", "zz"]) == "no node called 'zz'"

    def test_an_empty_query_and_help_list_the_verbs(self):
        q = GraphQuery(path(2))
        listing = q.ask([])
        assert listing.startswith("verbs: degree/1, adjacent/2, path/2")
        assert q.ask(["help"]) == listing
        assert "summary/0" in listing
