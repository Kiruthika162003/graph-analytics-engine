"""Hungarian assignment: pair every worker with a job at minimum total cost.

The assignment problem gives a square cost matrix, workers by jobs, and
asks for a perfect matching, every worker to exactly one job and every job
to exactly one worker, whose total cost is least. Trying every permutation
is factorial. The Hungarian algorithm, in its potential form, solves it in
cubic time by maintaining a price on each worker and each job, the
potentials, that together never exceed any edge's cost, and growing the
matching one worker at a time along a path of tight edges, edges whose cost
equals exactly the sum of its two potentials. For each new worker it runs
a search that behaves like Dijkstra over the reduced costs, cost minus the
two potentials, finding the cheapest way to reach a free job through
alternating chains of tight and matched edges; when the search would get
stuck, it raises the potentials along the visited chain by the smallest
slack, which makes at least one more edge tight without breaking any
constraint, and continues. When a free job is reached the alternating path
is flipped and the worker is assigned. Optimality follows from the
potentials: the matching's cost equals the sum of all potentials, and no
matching can cost less than that sum since every edge is at least its two
potentials combined, so a perfect matching on tight edges is provably
minimal. This is the same duality that makes max-flow equal min-cut, in
assignment form. The solver takes the cost matrix, refuses a non-square
one, returns the assignment and its total, and reports the total against
the sum of row minima, a lower bound any assignment must meet, because a
total sitting exactly on that bound means every worker got their own
cheapest job and the problem was easy, while a gap measures how much the
one-to-one constraint cost.
"""

from __future__ import annotations

import math

from mesh.errors import Invalid


class Hungarian:
    def __init__(self, cost: list[list[float]]) -> None:
        n = len(cost)
        if n == 0 or any(len(row) != n for row in cost):
            raise Invalid("the cost matrix must be square and non-empty")
        self.cost = [list(row) for row in cost]
        self.n = n
        self.assignment: list[int] = []
        self.total = self._solve()

    def _solve(self) -> float:
        n = self.n
        # 1-indexed potentials and matchings, the standard compact formulation
        u = [0.0] * (n + 1)  # worker potentials
        v = [0.0] * (n + 1)  # job potentials
        job_of_worker = [0] * (n + 1)
        worker_of_job = [0] * (n + 1)  # 0 means the job is free
        for worker in range(1, n + 1):
            worker_of_job[0] = worker
            current_job = 0
            min_slack = [math.inf] * (n + 1)
            came_from = [0] * (n + 1)
            visited = [False] * (n + 1)
            while True:
                visited[current_job] = True
                w = worker_of_job[current_job]
                delta = math.inf
                next_job = 0
                for job in range(1, n + 1):
                    if visited[job]:
                        continue
                    reduced = self.cost[w - 1][job - 1] - u[w] - v[job]
                    if reduced < min_slack[job]:
                        min_slack[job] = reduced
                        came_from[job] = current_job
                    if min_slack[job] < delta:
                        delta = min_slack[job]
                        next_job = job
                # raise potentials along the visited chain by the smallest slack
                for job in range(n + 1):
                    if visited[job]:
                        u[worker_of_job[job]] += delta
                        v[job] -= delta
                    else:
                        min_slack[job] -= delta
                current_job = next_job
                if worker_of_job[current_job] == 0:
                    break  # a free job was reached
            while current_job:
                # flip the alternating path back to the start
                prev_job = came_from[current_job]
                worker_of_job[current_job] = worker_of_job[prev_job]
                current_job = prev_job
        for job in range(1, n + 1):
            job_of_worker[worker_of_job[job]] = job
        self.assignment = [job_of_worker[w] - 1 for w in range(1, n + 1)]
        return sum(self.cost[w][self.assignment[w]] for w in range(n))

    def row_minima_bound(self) -> float:
        return sum(min(row) for row in self.cost)

    def note(self) -> str:
        bound = self.row_minima_bound()
        return (
            f"optimal total {self.total} against a row-minima bound of {bound}; "
            "the gap is what the one-to-one constraint cost over everyone taking "
            "their own cheapest job"
        )
