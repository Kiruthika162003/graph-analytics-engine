"""Full report: every family of reading the engine has, on one graph, as one document.

The summary is a first look and the recipes answer single questions;
a full report is the whole file on a graph, the document a reader
prints and takes to a meeting. It gathers, in a fixed order, the
summary, the degree statistics, the centralization readings, the
class recognisers, the census of three-node and four-node shapes,
the spectral readings, the community recipe, the bottleneck, the
robustness comparison, the influence ranking, the self-check tally,
and a text histogram, each under a heading, and it never stops on a
reading that refuses: a directed graph, an empty graph, or a graph
too large for an exact count gets a line saying which reading was
skipped and why. The report is a list of lines, so the command line
prints it and the tests read it, and it ends with a line counting the
sections written and skipped. The tests hold that a small undirected
graph fills every section, that a directed graph skips the ones it
must and says so, that the empty graph produces a report of skips
rather than an error, and that the section count on the last line
matches the headings above it. Sections are written by small
functions so a caller can ask for one by name.
"""

from __future__ import annotations

from collections.abc import Callable

from mesh.centralization import Centralization
from mesh.errors import MeshError
from mesh.graph import Graph
from mesh.graphcheck import SelfCheck
from mesh.graphenergy import GraphEnergy
from mesh.graphrecipes import Recipes
from mesh.graphstats import DegreeStats
from mesh.graphsummary import GraphSummary
from mesh.graphtext import TextRender
from mesh.laplacianspectrum import LaplacianSpectrum
from mesh.quadcensus import QuadCensus
from mesh.triadcensus import TriadCensus

Section = Callable[[Graph], list[str]]


def _summary(g: Graph) -> list[str]:
    return GraphSummary(g).lines()


def _degrees(g: Graph) -> list[str]:
    return [DegreeStats(g).note()]


def _centralization(g: Graph) -> list[str]:
    return [Centralization(g).note()]


def _classes(g: Graph) -> list[str]:
    found = GraphSummary(g).classes()
    return ["classes: " + (", ".join(found) if found else "none recognised")]


def _census(g: Graph) -> list[str]:
    return [f"triads: {TriadCensus(g).note()}", f"quads: {QuadCensus(g).note()}"]


def _spectrum(g: Graph) -> list[str]:
    return [GraphEnergy(g).note(), LaplacianSpectrum(g).note()]


def _recipe(name: str) -> Section:
    return lambda g: Recipes(g, seed=1).run(name)


def _selfcheck(g: Graph) -> list[str]:
    return [SelfCheck(g).note()]


def _histogram(g: Graph) -> list[str]:
    return TextRender(g).histogram(width=12).splitlines()


SECTIONS: dict[str, Section] = {
    "summary": _summary,
    "degrees": _degrees,
    "centralization": _centralization,
    "classes": _classes,
    "census": _census,
    "spectrum": _spectrum,
    "communities": _recipe("communities"),
    "bottleneck": _recipe("bottleneck"),
    "robustness": _recipe("robustness"),
    "influence": _recipe("influence"),
    "selfcheck": _selfcheck,
    "histogram": _histogram,
}


class FullReport:
    def __init__(self, graph: Graph, names: list[str] | None = None) -> None:
        self.graph = graph
        self.names = list(SECTIONS) if names is None else list(names)
        unknown = [n for n in self.names if n not in SECTIONS]
        if unknown:
            raise MeshError(f"no section called '{unknown[0]}'")
        self.written = 0
        self.skipped = 0
        self.lines = self._build()

    def _build(self) -> list[str]:
        out: list[str] = []
        for name in self.names:
            out.append(f"== {name}")
            try:
                body = SECTIONS[name](self.graph)
            except MeshError as exc:
                out.append(f"  skipped: {exc}")
                self.skipped += 1
                continue
            out.extend("  " + line for line in body)
            self.written += 1
        out.append(f"{self.written} section(s) written, {self.skipped} skipped")
        return out

    def section(self, name: str) -> list[str]:
        if name not in SECTIONS:
            raise MeshError(f"no section called '{name}'")
        return SECTIONS[name](self.graph)

    def note(self) -> str:
        return self.lines[-1]

    def text(self) -> str:
        return "\n".join(self.lines)
