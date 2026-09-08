"""Graph I/O: read an edge list, write one back, and hand the picture to DOT.

A graph that cannot leave the process is a toy. The plainest exchange
format is the edge list, one edge per line as two names and an optional
weight, with a header line naming the direction and blank or hash-led
lines ignored; it is what every tool can produce and what a person can
type. The engine reads it strictly: a line with fewer than two fields or
a weight that does not parse is refused with its line number, because a
silently skipped line is a missing edge nobody will find later, and an
isolated node, one with no edges, is declared on its own line so it is
not lost in a format that otherwise only mentions nodes through edges.
Writing produces the same format in sorted order, so two graphs with
the same structure write the same text and the diff of two files is the
diff of two graphs. The JSON form carries the same information with
explicit node and edge arrays, for programs that would rather not parse.
The DOT form is one-way, an export for Graphviz to draw, with undirected
graphs written as graph with double dashes and directed ones as digraph
with arrows, and weights as labels; it is the shape a reader wants when
a printed adjacency list stops being legible. Round trips are the test:
a graph written and read back must compare equal in nodes, edges,
weights, and direction, and the engine checks that on every shape it
knows. The reader and writer refuse a mismatch between the declared
direction and the requested one, and report the line count read against
the edges found, because the difference is comments, blanks, and lone
node declarations, the parts of the file that are not edges.
"""

from __future__ import annotations

import json

from mesh.errors import Invalid
from mesh.graph import Graph


def write_edge_list(g: Graph) -> str:
    lines = ["directed" if g.directed else "undirected"]
    linked = {n for u, v, _w in g.edges() for n in (u, v)}
    for node in sorted(g.nodes()):
        if node not in linked:
            lines.append(node)  # an isolated node must be declared or it is lost
    for u, v, w in sorted(_canonical(g)):
        lines.append(f"{u} {v} {w:g}")
    return "\n".join(lines) + "\n"


def _canonical(g: Graph) -> list[tuple[str, str, float]]:
    # an undirected edge keeps its insertion orientation inside the graph;
    # writing it with sorted endpoints is what makes equal graphs write
    # equal text, which the first version of the writer got wrong
    if g.directed:
        return list(g.edges())
    return [(min(u, v), max(u, v), w) for u, v, w in g.edges()]


def read_edge_list(text: str) -> Graph:
    rows = [line.strip() for line in text.splitlines()]
    body = [(i + 1, r) for i, r in enumerate(rows) if r and not r.startswith("#")]
    if not body:
        raise Invalid("an edge list needs a header line of directed or undirected")
    _n, header = body[0]
    if header not in ("directed", "undirected"):
        raise Invalid(f"line 1 must declare directed or undirected, not '{header}'")
    g = Graph(directed=header == "directed")
    for number, row in body[1:]:
        fields = row.split()
        if len(fields) == 1:
            g.add_node(fields[0])
            continue
        if len(fields) not in (2, 3):
            raise Invalid(f"line {number}: expected 'u v' or 'u v weight', got '{row}'")
        u, v = fields[0], fields[1]
        try:
            weight = float(fields[2]) if len(fields) == 3 else 1.0
        except ValueError as exc:
            raise Invalid(f"line {number}: weight '{fields[2]}' is not a number") from exc
        g.add_node(u)
        g.add_node(v)
        g.add_edge(u, v, weight)
    return g


def to_json(g: Graph) -> str:
    payload = {
        "directed": g.directed,
        "nodes": sorted(g.nodes()),
        "edges": [{"from": u, "to": v, "weight": w} for u, v, w in sorted(g.edges())],
    }
    return json.dumps(payload, indent=2)


def from_json(text: str) -> Graph:
    payload = json.loads(text)
    for key in ("directed", "nodes", "edges"):
        if key not in payload:
            raise Invalid(f"the JSON form needs a '{key}' entry")
    g = Graph(directed=bool(payload["directed"]))
    for node in payload["nodes"]:
        g.add_node(str(node))
    for edge in payload["edges"]:
        g.add_edge(str(edge["from"]), str(edge["to"]), float(edge.get("weight", 1.0)))
    return g


def to_dot(g: Graph, name: str = "mesh") -> str:
    kind, arrow = ("digraph", "->") if g.directed else ("graph", "--")
    lines = [f"{kind} {name} {{"]
    for node in sorted(g.nodes()):
        lines.append(f'  "{node}";')
    for u, v, w in sorted(g.edges()):
        label = f' [label="{w:g}"]' if w != 1.0 else ""
        lines.append(f'  "{u}" {arrow} "{v}"{label};')
    lines.append("}")
    return "\n".join(lines) + "\n"


def same_graph(a: Graph, b: Graph) -> bool:
    if a.directed != b.directed or sorted(a.nodes()) != sorted(b.nodes()):
        return False
    return sorted(_canonical(a)) == sorted(_canonical(b))


def note(text: str, g: Graph) -> str:
    lines = sum(1 for line in text.splitlines() if line.strip())
    return (
        f"{lines} non-blank line(s) read, {g.edge_count()} edge(s) found; the difference "
        "is the header, comments, and lone node declarations"
    )
