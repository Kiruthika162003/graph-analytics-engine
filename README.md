# graph-analytics-engine

A graph analytics engine in pure Python, package `mesh`, with no dependencies
beyond the standard library. It holds the classical algorithms on adjacency
lists, the readings a network analyst asks for, the graph classes whose hard
problems turn easy, the spectral tools that need an eigenvalue solver, the
stochastic models that need a seed, and the tooling that turns a graph into a
report, a picture, or an answer at the terminal. Every module carries a
docstring that says what it computes and why the identity it is checked
against is the right one, and every module has a test file that holds it to
that identity.

## Layout

```
mesh/          198 modules: the engine itself, plus the probes organ and the CLI
mesh/probes/   71 standing probes in 20 files, each a guarantee with readings
tests/         199 test files, 2341 tests
examples/      28 runnable walkthroughs, one per theme
scripts/       strictcount.py, the odometer the project was built against
```

The measured size is 30,000 strict lines: lines that carry code once
docstrings, comments, and blanks are tokenized away. `python
scripts/strictcount.py` prints the number.

## What is in it

**Core and traversal.** `Graph` with directed and undirected modes and
weights; breadth-first and depth-first search, 0-1 BFS, bidirectional
search, iterative deepening.

**Shortest paths.** Dijkstra, Bellman-Ford, SPFA, Floyd-Warshall, Johnson,
A*, DAG paths, Yen's k shortest, widest paths, arbitrage as a negative
cycle, temporal paths, and a path explanation with per-edge slack.

**Connectivity and cuts.** Union-find with rollback, components, Tarjan and
Kosaraju, condensation, articulation points, bridges, biconnected and
2-edge-connected components, Menger, vertex separators, minimum cuts,
Stoer-Wagner, Gomory-Hu trees, Cheeger constants.

**Trees and flows.** Kruskal, Prim, Boruvka, reverse-delete, second-best
trees, Kirchhoff and deletion-contraction counts, arborescences, Steiner
trees; Edmonds-Karp, Dinic, push-relabel, min-cost flow, flow
decomposition, circulations with lower bounds.

**Matching and covering.** Hopcroft-Karp, Hungarian, blossom, stable
matching, vertex cover, edge cover, path cover, dominating sets, feedback
vertex sets, perfect matching counts against Ryser's permanent.

**Centrality and community.** Degree, closeness, harmonic, betweenness,
stress, load, eigenvector, Katz, PageRank, personalized PageRank, HITS,
vitality, centralization; label propagation, Louvain, Girvan-Newman,
spectral clustering, clique percolation, k-core, k-truss, core-periphery,
rich club, null models, partition comparison.

**Graph classes.** Chordal, split, interval, cograph, threshold,
permutation, tournament, bipartite, planarity obstructions, tree
isomorphism by canonical names, Weisfeiler-Lehman hashing, exact
isomorphism, subgraph matching, maximum common subgraph, edit distance,
walk kernels.

**Counting and spectra.** Chromatic and independence polynomials, triad and
quad censuses, graphlet orbits, walk counts, Jacobi eigenvalues, graph
energy, the Laplacian spectrum, commute times, entropy.

**Hard problems with bounds.** Exact chromatic number, treewidth and
pathwidth, bandwidth, burning number, travelling salesman by Held-Karp with
the double-tree bound, Kernighan-Lin bisection, maximum cut, 2-SAT.

**Randomness with seeds.** Percolation, SIR epidemics, random walks,
sampling by node, edge, snowball, and walk, degree-preserving shuffles.

**Tooling.** Summaries, queries, recipes, full reports, self-checks,
contracts, change reports, sequences of snapshots, event logs, scenarios,
merging, relabeling, builders, caches, layouts, SVG and text rendering, a
benchmark frame, and graph I/O in edge list, JSON, and DOT.

## Running it

```bash
python -m pytest tests/ -q
```

```bash
python -m ruff check mesh/ tests/ examples/ scripts/
```

```bash
python -m mesh.cli summary
```

`python -m mesh.cli probes` prints every probe with its readings, `check`
exits non-zero if any is broken, `describe FILE` prints a first look at an
edge list file, and `ask FILE VERB ARGS` answers a query verb. Every example
runs as `python -m examples.<name>`, for instance `python -m
examples.finalday`.

## The probes

A probe builds a graph made to stress one guarantee, runs the modules that
must agree, and reports `holds` or `BROKEN` with the numbers it saw: the
max-flow equals the min-cut over every cut, Kruskal and Prim agree, Tarjan
and Kosaraju partition alike, Konig's theorem holds on a random bipartite
graph, deletion-contraction agrees with Kirchhoff, the Gomory-Hu tree
matches a direct flow on every pair, the Cheeger constant sits in its
spectral bracket, and so on through 71 of them. The probes are the
engine's own standing evidence, separate from the tests, and the command
line exists to run them.

## Honest measurement

When a guess in a test or a docstring was refuted by the measured result,
the wrong guess stays recorded beside the measured truth, in the test's
comment and in the commit message. The history holds many of these: a
wheel is not chordal because the hub is not on its rim; a path of four is a
split graph; two K4s on a bridge have 256 spanning trees, not 32; Gusfield's
later-only reparenting gives an equivalent flow tree rather than a cut
tree; a cycle node's vitality is below its own distance sum, not above;
the natural split of two cliques and the split that halves them share no
mutual information at all. A reader can grep the tests for "the guess was"
to find them.

## Conventions

Line length 96, ruff with the E, F, W, I, N, UP, B, A, C4, RET, SIM, ARG,
PL, and RUF families, sparse comments, no stubs, and no em dashes anywhere.
Every refusal names the node, edge, line, or value at fault. Modules that
are exponential say so and state their size limit.

Written by Kiruthika Subramani in collaboration with Claude, Anthropic's AI assistant.
