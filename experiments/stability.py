"""Stability of the selected features across the outer training sets (ConTree vs ConTree-ACP).

For every dataset, the trees fitted on the outer training parts are compared pairwise:
- Jaccard similarity of the used feature sets,
- agreement of the root feature,
- Spearman correlation of the normalized impurity-decrease importances (unused features count as zero).
Both the main pipeline (ConTree's tuned depth, ACP's selected alpha) and fixed alphas at depth 3 are reported.

    python contree-acp/experiments/stability.py
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiments as rx  # noqa: E402
import summarize as sm  # noqa: E402


def importance(tree, X, y):
    k = int(y.max()) + 1
    values = np.zeros(X.shape[1])
    for kind, feature, decrease in rx.walk(tree, X, y, k):
        if kind == "split":
            values[feature] += decrease
    return values / values.sum() if values.sum() > 0 else values


def pairwise(trees):
    """Mean pairwise Jaccard, root agreement and importance rank correlation over a list of (tree, importance)."""
    jaccard, root, rank = [], [], []
    for (tree_a, imp_a), (tree_b, imp_b) in itertools.combinations(trees, 2):
        used_a, used_b = set(np.flatnonzero(imp_a > 0)), set(np.flatnonzero(imp_b > 0))
        jaccard.append(len(used_a & used_b) / len(used_a | used_b) if used_a | used_b else 1.0)
        root.append(float(isinstance(tree_a, list) and isinstance(tree_b, list) and tree_a[0] == tree_b[0]))
        if imp_a.std() > 0 and imp_b.std() > 0:
            rank.append(spearmanr(imp_a, imp_b).statistic)
    return np.mean(jaccard), np.mean(root), (np.mean(rank) if rank else np.nan)


def main():
    runs = sm.load_runs(sm.DEFAULT_RESULTS)
    pairs = sm.paired_table(runs, tune_depth=True)
    outer_runs = runs[runs.inner < 0]
    cache = rx.DataCache()
    rows = []
    for dataset, group in pairs.groupby("dataset"):
        X, y, splits, _ = cache.get(dataset)
        variants = {"contree": [], "acp": []}
        for _, pair in group.iterrows():
            train = splits[int(pair.outer)][0]
            for name, alpha in [("contree", 0.0), ("acp", pair.alpha)]:
                run = outer_runs[(outer_runs.dataset == dataset) & (outer_runs.outer == pair.outer)
                                 & (outer_runs.depth == pair.depth) & (outer_runs.alpha == alpha)].iloc[0]
                tree = json.loads(run.tree)
                variants[name].append((tree, importance(tree, X[train], y[train])))
        row = {"dataset": dataset, "splits": len(group)}
        for name, trees in variants.items():
            row[f"jaccard_{name}"], row[f"root_{name}"], row[f"rank_{name}"] = pairwise(trees)
        # fixed alphas at depth 3, every finished outer split
        for alpha in [0.0, 0.2, 0.6, 1.0]:
            at = outer_runs[(outer_runs.dataset == dataset) & (outer_runs.depth == 3) & (outer_runs.alpha == alpha)]
            trees = [(json.loads(r.tree), importance(json.loads(r.tree), X[splits[int(r.outer)][0]], y[splits[int(r.outer)][0]]))
                     for r in at.itertuples()]
            row[f"jaccard_d3_a{alpha}"] = pairwise(trees)[0] if len(trees) > 1 else np.nan
        rows.append(row)
    table = pd.DataFrame(rows).set_index("dataset")
    out = rx.ROOT / "results"
    table.to_csv(out / "stability.csv")
    pd.set_option("display.width", 220)
    print(table.round(3).to_string())
    print()
    for metric in ["jaccard", "root", "rank"]:
        diff = (table[f"{metric}_acp"] - table[f"{metric}_contree"]).dropna()
        nonzero = diff[diff.abs() > 1e-12]
        p = wilcoxon(nonzero).pvalue if len(nonzero) else float("nan")
        print(f"{metric:8s} ACP - ConTree: mean {diff.mean():+.3f}, higher on {(diff > 1e-12).sum()}, "
              f"lower on {(diff < -1e-12).sum()} datasets, Wilcoxon p = {p:.3g}")


if __name__ == "__main__":
    main()
