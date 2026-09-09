"""A final day: one graph walked through the engine from first look to picture to what if.

Run with: python -m examples.finalday
"""

from __future__ import annotations

from mesh.benchmark import Benchmark
from mesh.factories import cycle
from mesh.graph import Graph
from mesh.graphbuilder import GraphBuilder
from mesh.graphquery import GraphQuery
from mesh.graphrecipes import Recipes
from mesh.graphsummary import GraphSummary
from mesh.graphtext import TextRender
from mesh.layout import Layout
from mesh.quadcensus import QuadCensus
from mesh.scenario import Scenario
from mesh.svgrender import SvgRenderer

ROSTER = """
ada ben
ada cal
ben cal
cal dee
dee eve
dee fox
eve fox
fox gus
gus hal
hal ivy
ivy gus
"""


def main() -> int:
    g = GraphBuilder().text(ROSTER).build()
    print("== first look")
    print(GraphSummary(g).note())
    print("== a question")
    print("path ada to ivy: " + GraphQuery(g).ask(["path", "ada", "ivy"]))
    print("== shapes of four")
    print(QuadCensus(g).note())
    print("== recipe")
    for line in Recipes(g, seed=1).run("bottleneck"):
        print("  " + line)
    print("== picture")
    pos = Layout(g).spring(rounds=100, seed=2)
    print(SvgRenderer(g, pos).note())
    print(TextRender(g).histogram(width=10))
    print("== what if")
    sc = Scenario(g)
    worst = sc.sweep_nodes(by="pieces")[0]
    print(f"the departure that splits most: {worst[0]} ({worst[1]:+g} piece(s))")
    print("== growth")
    def count(h: Graph) -> int:
        return QuadCensus(h).total()

    bench = Benchmark(cycle, [8, 32]).add("quads", count, count)
    bench.run()
    print(bench.note())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
