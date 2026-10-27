# ConTree-ACP

ConTree-ACP adds the ancestor-correlation penalty (ACP) of ACP-Gini to ConTree, the optimal classification tree
solver for continuous features of Brița, van der Linden and Demirović (AAAI 2025). It still returns the optimal tree,
now for the penalized objective.

The solver code in `code/` is ConTree at upstream commit
[`61ebd49`](https://github.com/ConSol-Lab/contree/tree/61ebd49f3b2aa607bba84c2bd3e218b9a793a40c) with the changes
listed below; `datasets/` holds the 16 datasets of the ConTree paper as distributed in that repository. ConTree is MIT
licensed (`LICENSE`).

## Objective

ConTree minimizes misclassifications by the recursion

```
C(D, 0) = L(D)
C(D, d) = min{ L(D), min_{f,t} [ L(D) - g(f, t) ] },    g(f, t) = L(D) - [C(D_L, d-1) + C(D_R, d-1)]
```

where `L(D)` is the misclassification count of a leaf. ConTree-ACP carries the set `A` of features already used on the
path and keeps only the share `P(f, A)` of the misclassifications a split removes:

```
C(D, d, A) = min{ L(D), min_{f,t} [ L(D) - g(f, t) * P(f, A) ] },   children solved with A + {f}
P(f, A)    = prod_{p in A, p != f} (1 - alpha * |r(X_f, X_p)|)
```

`|r|` is the absolute Pearson correlation on the training data. `alpha = 0` gives ConTree exactly.

## Changes to ConTree

- `Configuration` carries `acp_alpha`, the correlation matrix and the path's ancestor features (`penalty_factor`,
  `child_penalty_factors`, `add_ancestor`).
- General solver: P is constant over the thresholds of a feature, so the threshold search and all of ConTree's pruning
  run unchanged on the raw cost `left + right`; the bounds are translated into those units per feature. Both subtrees get
  the split feature as an extra ancestor.
- Depth-two solver: the depth-one splits below the root feature use `P(g, A + {f})`; the skipping rule uses the
  equivalent integer score to beat for each feature.
- Cache entries are keyed by the ancestor set as well (an entry for another ancestor set is not reused).
- Objectives are `double` (non-integer with ACP); a tree node stores its factor and offset so the objective can be
  rechecked.
- `main.cpp`: new parameter `-acp-alpha`, the correlation matrix computed from the training file, and the output lines
  `ACP alpha` and `Within time limit`.

The Python binding (`code/binding.cpp`) is not updated; the experiments call the command-line program.

## Build (Windows with MinGW, or Linux)

```bash
cmake -S contree-acp/code -B contree-acp/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build contree-acp/build
contree-acp/build/ConTree.exe -file contree-acp/datasets/rice.txt -max-depth 3 -acp-alpha 0.6
```

## Checks

- `alpha = 0` returns the same tree as upstream ConTree on all 16 datasets at depth 3.
- `python contree-acp/tests/brute_force_check.py` compares the solver with an exhaustive dynamic program on small
  subsamples (depth 2 and 3, alpha 0 to 1); all 32 cases reach the optimum.

## Experiments

```bash
python contree-acp/experiments/run_experiments.py --workers 14                    # depth 3, results/runs.jsonl
python contree-acp/experiments/run_experiments.py --depths 1,2 --workers 3 \
    --results contree-acp/results/runs_shallow.jsonl                              # depths 1 and 2
python contree-acp/experiments/summarize.py                                       # tables, Wilcoxon tests, alpha curve
```

Both runners are resumable. The protocol follows ConTree (maximum depth 3, depth tuned on five inner 80/20 splits),
but uses 25 outer splits (stratified five-fold cross-validation repeated five times) to allow a paired test on each
dataset. In every outer split, five inner 80/20 splits of the training part choose ConTree's depth in
{1, 2, 3} (highest mean inner accuracy). ConTree-ACP uses the same depth and chooses alpha from {0, 0.2, ..., 1} on the
same inner splits: keep the alphas whose mean inner accuracy is at least ConTree's mean minus one standard error
(the one-standard-error rule of Breiman et al.), then take the lowest mean importance-weighted redundancy R_w. `summarize.py` also reports the comparison at fixed
depth 3.
