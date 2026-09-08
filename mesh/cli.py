"""The mesh command line: report the probes, check them, summarize them.

    python -m mesh.cli probes     print every probe with its readings
    python -m mesh.cli check      exit non-zero if any probe is broken
    python -m mesh.cli summary    print the count and the broken count
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    from mesh.probes import all_probes, broken, report

    parser = argparse.ArgumentParser(prog="mesh")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("probes", help="print every probe with its readings")
    sub.add_parser("check", help="exit non-zero if any probe is broken")
    sub.add_parser("summary", help="print the count and the broken count")
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
    return 2


if __name__ == "__main__":
    sys.exit(main())
