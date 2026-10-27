"""Tables and tests for ConTree vs ConTree-ACP from the run files.

Main comparison (as in the ConTree paper): in every outer split, ConTree's depth (1, 2 or 3) is the one with the
highest mean inner accuracy at alpha = 0 (ties to the smaller depth). ConTree-ACP uses the same depth and the alpha
chosen from the inner runs at that depth (see run_experiments.py). A secondary comparison fixes the depth at 3.
Per-dataset values are means over the outer splits; the Wilcoxon signed-rank test pairs the 16 dataset means, as in
the ConTree paper.

    python contree-acp/experiments/summarize.py [--results runs.jsonl runs_shallow.jsonl] [--out results/]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = [ROOT / "results" / "runs.jsonl", ROOT / "results" / "runs_shallow.jsonl"]
METRICS = ["eval_accuracy", "r_w", "path_r", "n_features_used", "n_leaves", "train_errors", "solver_seconds"]
N_INNER = 5


def load_runs(paths):
    records = [json.loads(line) for path in paths if path.exists() for line in path.read_text().splitlines() if line.strip()]
    runs = pd.DataFrame(records)
    if "depth" not in runs:
        runs["depth"] = 3
    runs["depth"] = runs.depth.fillna(3).astype(int)  # the first depth-3 runs were recorded without a depth field
    return runs


def select_alpha(inner):
    """One-standard-error rule (Breiman et al. 1984; Hastie et al. 2009, Sec. 7.10): alphas whose mean inner accuracy is
    within one standard error of ConTree's mean inner accuracy; among them the lowest mean R_w (computed on the inner
    training data only), ties to the smaller alpha."""
    by_alpha = inner.groupby("alpha").agg(acc=("eval_accuracy", "mean"), r_w=("r_w_train_corr", "mean"))
    base = inner[inner.alpha == 0.0].eval_accuracy
    standard_error = base.std(ddof=1) / np.sqrt(len(base))
    eligible = by_alpha[by_alpha.acc >= base.mean() - standard_error - 1e-12]
    lowest = eligible[eligible.r_w <= eligible.r_w.min() + 1e-12]
    return float(lowest.index.min())


def select_depth(inner):
    """ConTree's depth: the highest mean inner accuracy at alpha = 0, ties to the smaller depth."""
    accuracy = inner[inner.alpha == 0.0].groupby("depth").eval_accuracy.mean()
    return int(accuracy[accuracy >= accuracy.max() - 1e-12].index.min())


def complete(inner, depth, alphas):
    at_depth = inner[inner.depth == depth]
    return all((at_depth.alpha == alpha).sum() >= N_INNER for alpha in alphas)


def paired_table(runs, tune_depth):
    alphas = sorted(runs[runs.depth > 1].alpha.unique())
    rows = []
    for (dataset, outer), group in runs.groupby(["dataset", "outer"]):
        inner, outer_runs = group[group.inner >= 0], group[group.inner < 0]
        if tune_depth:
            if not (complete(inner, 1, [0.0]) and complete(inner, 2, alphas) and complete(inner, 3, alphas)):
                continue  # this outer split is not finished yet
            depth = select_depth(inner)
        else:
            depth = 3
            if not complete(inner, 3, alphas):
                continue
        # at depth 1 the penalty has no effect (no ancestors), so ConTree-ACP is ConTree
        alpha = 0.0 if depth == 1 else select_alpha(inner[inner.depth == depth])
        at_depth = outer_runs[outer_runs.depth == depth].set_index("alpha")
        if 0.0 not in at_depth.index or alpha not in at_depth.index:
            continue
        base, acp = at_depth.loc[0.0], at_depth.loc[alpha]
        row = {"dataset": dataset, "outer": outer, "depth": depth, "alpha": alpha,
               "timeouts": int((~group[group.depth == depth].within_time_limit).sum())}
        for metric in METRICS:
            row[f"{metric}_contree"] = base[metric]
            row[f"{metric}_acp"] = acp[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(pairs, out, suffix):
    pairs.to_csv(out / f"paired_splits{suffix}.csv", index=False)
    per_dataset = pairs.groupby("dataset").agg(
        splits=("outer", "size"), depth_mean=("depth", "mean"), alpha_median=("alpha", "median"),
        acc_contree=("eval_accuracy_contree", "mean"), acc_acp=("eval_accuracy_acp", "mean"),
        rw_contree=("r_w_contree", "mean"), rw_acp=("r_w_acp", "mean"),
        pathr_contree=("path_r_contree", "mean"), pathr_acp=("path_r_acp", "mean"),
        feat_contree=("n_features_used_contree", "mean"), feat_acp=("n_features_used_acp", "mean"),
        sec_contree=("solver_seconds_contree", "mean"), sec_acp=("solver_seconds_acp", "mean"),
        timeouts=("timeouts", "sum"))
    per_dataset["acc_diff_pts"] = 100 * (per_dataset.acc_acp - per_dataset.acc_contree)
    per_dataset["rw_change_pct"] = 100 * (per_dataset.rw_acp - per_dataset.rw_contree) / per_dataset.rw_contree.replace(0, np.nan)
    per_dataset["rw_lower_splits"] = pairs.assign(lower=pairs.r_w_acp < pairs.r_w_contree - 1e-12).groupby("dataset").lower.sum()
    per_dataset.to_csv(out / f"per_dataset{suffix}.csv")

    pd.set_option("display.width", 220)
    print(per_dataset[["splits", "depth_mean", "alpha_median", "acc_contree", "acc_acp", "acc_diff_pts", "rw_contree",
                       "rw_acp", "rw_change_pct", "rw_lower_splits", "feat_contree", "feat_acp", "timeouts"]].round(4).to_string())

    tests = {}
    for label, a, b in [("R_w", "rw_acp", "rw_contree"), ("path redundancy", "pathr_acp", "pathr_contree"),
                        ("test accuracy", "acc_acp", "acc_contree"), ("features used", "feat_acp", "feat_contree")]:
        diff = per_dataset[a] - per_dataset[b]
        nonzero = diff[diff.abs() > 1e-12]
        p = wilcoxon(nonzero).pvalue if len(nonzero) >= 1 else float("nan")
        tests[label] = {"mean_diff": float(diff.mean()), "median_diff": float(diff.median()),
                        "lower": int((diff < -1e-12).sum()), "higher": int((diff > 1e-12).sum()), "wilcoxon_p": float(p)}
    print()
    for label, t in tests.items():
        print(f"{label:16s} ACP - ConTree: mean {t['mean_diff']:+.4f}, median {t['median_diff']:+.4f}, "
              f"lower on {t['lower']}, higher on {t['higher']} datasets, Wilcoxon p = {t['wilcoxon_p']:.4g}")
    (out / f"tests{suffix}.json").write_text(json.dumps(tests, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, nargs="+", default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    runs = load_runs(args.results)
    args.out.mkdir(parents=True, exist_ok=True)
    for title, tune_depth, suffix in [("Main: ConTree depth tuned in {1, 2, 3}, ACP at the same depth", True, ""),
                                      ("Secondary: depth fixed at 3", False, "_depth3")]:
        pairs = paired_table(runs, tune_depth)
        print(f"\n=== {title} ({len(pairs)} outer splits) ===")
        if len(pairs):
            summarize(pairs, args.out, suffix)

    # alpha trade-off on the test parts at depth 3 (every alpha fitted on the outer training parts)
    outer_runs = runs[(runs.inner < 0) & (runs.depth == 3)]
    curve = outer_runs.groupby(["dataset", "alpha"])[["eval_accuracy", "r_w", "n_features_used"]].mean().reset_index()
    curve.to_csv(args.out / "alpha_curve.csv", index=False)


if __name__ == "__main__":
    main()
