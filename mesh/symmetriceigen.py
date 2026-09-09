"""Symmetric eigenvalues by Jacobi rotations, the tool the spectral readings were missing.

Adjacency and Laplacian matrices of undirected graphs are symmetric,
and a symmetric matrix has real eigenvalues that an orthogonal
rotation can expose one pair at a time. The Jacobi method picks the
largest off-diagonal entry, rotates the two rows and columns it lives
in by the angle that zeroes it, and repeats; each rotation lowers the
off-diagonal mass and the diagonal converges to the eigenvalues. It
is slower than Householder tridiagonalisation but it is short, exact
to floating precision, and never needs a library, which is what this
engine runs on. The module takes any symmetric matrix as a list of
rows, refuses a non-square or asymmetric one with the offending entry
named, and returns the eigenvalues sorted ascending and the
eigenvectors as columns of the accumulated rotation. Three identities
check the arithmetic: the eigenvalues sum to the trace, their squares
sum to the squared Frobenius norm, and the product of the matrix with
each eigenvector equals the eigenvalue times the vector to within a
tolerance the module reports. On graphs the identities are readable
facts, since the adjacency trace is zero and the squared norm is
twice the edge count, so the spectrum's squares sum to twice the
number of edges, which is a check any caller can run without
understanding rotations.
"""

from __future__ import annotations

from math import atan2, cos, hypot, sin

from mesh.errors import Invalid

Matrix = list[list[float]]


class SymmetricEigen:
    def __init__(self, matrix: Matrix, tolerance: float = 1e-12, sweeps: int = 100) -> None:
        self.n = len(matrix)
        for i, row in enumerate(matrix):
            if len(row) != self.n:
                raise Invalid(f"row {i} has {len(row)} entries in a matrix of size {self.n}")
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if abs(matrix[i][j] - matrix[j][i]) > 1e-12:
                    raise Invalid(f"entry ({i},{j}) differs from ({j},{i}); not symmetric")
        self.original = [list(row) for row in matrix]
        self.tolerance = tolerance
        self.rotations = 0
        a = [list(row) for row in matrix]
        v = [[1.0 if i == j else 0.0 for j in range(self.n)] for i in range(self.n)]
        for _ in range(sweeps):
            p, q, size = self._largest_off_diagonal(a)
            if size < tolerance:
                break
            self._rotate(a, v, p, q)
        order = sorted(range(self.n), key=lambda i: a[i][i])
        self.values = [a[i][i] for i in order]
        self.vectors = [[v[r][i] for i in order] for r in range(self.n)]

    @staticmethod
    def _largest_off_diagonal(a: Matrix) -> tuple[int, int, float]:
        best = (0, 1, 0.0)
        n = len(a)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > best[2]:
                    best = (i, j, abs(a[i][j]))
        return best

    def _rotate(self, a: Matrix, v: Matrix, p: int, q: int) -> None:
        # the angle that zeroes a[p][q] after rotating rows and columns p and q
        theta = 0.5 * atan2(2 * a[p][q], a[q][q] - a[p][p])
        c, s = cos(theta), sin(theta)
        n = self.n
        for k in range(n):
            akp, akq = a[k][p], a[k][q]
            a[k][p] = c * akp - s * akq
            a[k][q] = s * akp + c * akq
        for k in range(n):
            apk, aqk = a[p][k], a[q][k]
            a[p][k] = c * apk - s * aqk
            a[q][k] = s * apk + c * aqk
        for k in range(n):
            vkp, vkq = v[k][p], v[k][q]
            v[k][p] = c * vkp - s * vkq
            v[k][q] = s * vkp + c * vkq
        self.rotations += 1

    def trace_identity_holds(self) -> bool:
        trace = sum(self.original[i][i] for i in range(self.n))
        return abs(sum(self.values) - trace) < 1e-8

    def norm_identity_holds(self) -> bool:
        squared = sum(x * x for row in self.original for x in row)
        return abs(sum(v * v for v in self.values) - squared) < 1e-8

    def residual(self) -> float:
        worst = 0.0
        for k in range(self.n):
            vec = [self.vectors[r][k] for r in range(self.n)]
            image = [
                sum(self.original[r][c] * vec[c] for c in range(self.n)) for r in range(self.n)
            ]
            diff = [image[r] - self.values[k] * vec[r] for r in range(self.n)]
            worst = max(worst, hypot(*diff) if diff else 0.0)
        return worst

    def note(self) -> str:
        shown = ", ".join(f"{x:.4f}" for x in self.values)
        return (
            f"eigenvalues [{shown}] after {self.rotations} rotation(s), "
            f"residual {self.residual():.2e}"
        )
