"""Strict line counter: count real code lines, excluding docstrings and blanks.

Strict lines exclude comments, string tokens (so docstrings do not count),
and the structural whitespace tokens. What is left is the lines that carry
actual code. Run with: python scripts/strictcount.py
"""

from __future__ import annotations

import glob
import tokenize

_SKIP = {
    tokenize.COMMENT,
    tokenize.STRING,
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.ENCODING,
    tokenize.ENDMARKER,
}


def count_file(path: str) -> int:
    lines: set[int] = set()
    with open(path, "rb") as fh:
        for tok in tokenize.tokenize(fh.readline):
            if tok.type not in _SKIP and tok.start[0]:
                lines.add(tok.start[0])
    return len(lines)


def main() -> int:
    total = 0
    patterns = ["mesh/**/*.py", "tests/**/*.py", "examples/**/*.py", "scripts/**/*.py"]
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            total += count_file(path)
    print(f"strict {total} ({total / 30000 * 100:.1f}%) remaining {30000 - total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
