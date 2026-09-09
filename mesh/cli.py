"""The mesh command line: report the probes, check them, summarize them, and ask a graph.

    python -m mesh.cli probes                 print every probe with its readings
    python -m mesh.cli check                  exit non-zero if any probe is broken
    python -m mesh.cli summary                print the count and the broken count
    python -m mesh.cli describe FILE          print the first-look summary of an edge list
    python -m mesh.cli ask FILE VERB [ARG..]  answer one query verb on an edge list file

An edge list file starts with a line saying directed or undirected and
then holds one edge per line as two names and an optional weight; a
lone name declares an isolated node.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load(path: str):
    from mesh.errors import MeshError
    from mesh.graphio import read_edge_list

    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"cannot read '{path}': {exc.strerror or exc}")
        return None
    try:
        return read_edge_list(text)
    except MeshError as exc:
        print(f"'{path}' is not an edge list: {exc}")
        return None


def main(argv: list[str] | None = None) -> int:
    from mesh.graphquery import GraphQuery
    from mesh.graphsummary import GraphSummary
    from mesh.probes import all_probes, broken, report

    parser = argparse.ArgumentParser(prog="mesh")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("probes", help="print every probe with its readings")
    sub.add_parser("check", help="exit non-zero if any probe is broken")
    sub.add_parser("summary", help="print the count and the broken count")
    describe = sub.add_parser("describe", help="print the first-look summary of an edge list")
    describe.add_argument("file")
    ask = sub.add_parser("ask", help="answer one query verb on an edge list file")
    ask.add_argument("file")
    ask.add_argument("words", nargs="*")
    args = parser.parse_args(argv)

    if args.command == "probes":
        for probe in all_probes():
            print(probe.detail())
        return 0
    if args.command == "check":
        bad = broken()
        if bad:
            for probe in bad:
                print(probe.line())
            print(f"{len(bad)} probe(s) broken")
            return 1
        print("all probes hold")
        return 0
    if args.command == "summary":
        print(report())
        return 0
    if args.command == "describe":
        graph = _load(args.file)
        if graph is None:
            return 1
        print(GraphSummary(graph).note())
        return 0
    if args.command == "ask":
        graph = _load(args.file)
        if graph is None:
            return 1
        print(GraphQuery(graph).ask(args.words))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
