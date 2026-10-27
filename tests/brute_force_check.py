"""Check that ConTree-ACP returns the optimal penalized objective.

An exhaustive dynamic program (no pruning) computes the optimum of the ACP objective on small subsamples of the
ConTree datasets; the C++ solver must reach the same value for every alpha.

    python contree-acp/tests/brute_force_check.py
"""
import functools
import itertools
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build" / ("ConTree.exe" if sys.platform == "win32" else "ConTree")
# ConTree treats sorted values closer than this to their predecessor as equal (EPSILON in configuration.h)
VALUE_TOLERANCE = float(np.float32(1e-7))


def split_positions(sorted_values):
    """Positions i such that a threshold between sorted_values[i] and sorted_values[i + 1] is a ConTree split."""
    return np.flatnonzero(np.diff(sorted_values.astype(np.float64)) >= VALUE_TOLERANCE)


def exact_objective(X, y, depth, alpha):
    """Optimal ACP objective by exhaustive recursion over all axis-aligned trees of the given depth."""
    n, d = X.shape
    k = int(y.max()) + 1
    abs_r = np.abs(np.nan_to_num(np.corrcoef(X.astype(np.float64), rowvar=False)))

    def leaf(rows):
        return len(rows) - np.bincount(y[list(rows)], minlength=k).max()

    def factor(f, ancestors):
        return float(np.prod([1 - alpha * abs_r[f, p] for p in ancestors if p != f]))

    def best_stump(rows, ancestors, leaf_errors):
        """Remaining depth one, vectorised: every threshold of every feature at once."""
        rows_array = np.array(rows)
        best = float(leaf_errors)
        for f in range(d):
            p = factor(f, ancestors)
            if p <= 1e-12:
                continue
            order = np.argsort(X[rows_array, f], kind="stable")
            values = X[rows_array, f][order]
            onehot = np.eye(k, dtype=int)[y[rows_array][order]]
            left = np.cumsum(onehot, axis=0)[:-1]
            right = onehot.sum(axis=0) - left
            valid = split_positions(values)
            if len(valid) == 0:
                continue
            sizes = np.arange(1, len(rows))
            raw = (sizes - left.max(axis=1)) + ((len(rows) - sizes) - right.max(axis=1))
            best = min(best, p * raw[valid].min() + (1 - p) * leaf_errors)
        return best

    @functools.lru_cache(maxsize=None)
    def solve(rows, remaining, ancestors):
        best = leaf(rows)
        if remaining == 0 or best == 0:
            return float(best)
        if remaining == 1:
            return best_stump(rows, ancestors, best)
        rows_array = np.array(rows)
        for f in range(d):
            values = X[rows_array, f]
            p = factor(f, ancestors)
            if p <= 1e-12:
                continue
            child_ancestors = tuple(sorted(set(ancestors) | {f}))
            sorted_values = np.sort(values)
            for threshold in sorted_values[split_positions(sorted_values)]:
                left = tuple(rows_array[values <= threshold])
                right = tuple(rows_array[values > threshold])
                raw = solve(left, remaining - 1, child_ancestors) + solve(right, remaining - 1, child_ancestors)
                best = min(best, p * raw + (1 - p) * leaf(rows))
        return float(best)

    return solve(tuple(range(n)), depth, ())


def contree_objective(X, y, depth, alpha):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.txt"
        np.savetxt(path, np.column_stack([y, X]), fmt=["%d"] + ["%.9g"] * X.shape[1])
        out = subprocess.run([str(EXE), "-file", str(path), "-max-depth", str(depth), "-acp-alpha", str(alpha)],
                             capture_output=True, text=True, check=True).stdout
    return float(next(line for line in out.splitlines() if line.startswith("Objective:")).split(":")[1])


def main():
    rng = np.random.default_rng(0)
    # (dataset, sample size, depth, number of features); few features keep the exhaustive depth-three search small
    cases = [("raisin", 60, 2, 6), ("rice", 60, 2, 6), ("bean", 60, 2, 6), ("segment", 40, 2, 6),
             ("magic", 40, 3, 4), ("eeg", 40, 3, 4), ("bean", 40, 3, 4), ("fault", 40, 3, 4)]
    failures = 0
    for (name, n, depth, n_features), alpha in itertools.product(cases, [0.0, 0.3, 0.7, 1.0]):
        data = np.loadtxt(ROOT / "datasets" / f"{name}.txt")
        rows = rng.choice(len(data), n, replace=False)
        y = data[rows, 0].astype(int)
        y = np.unique(y, return_inverse=True)[1]  # contiguous labels, as ConTree expects
        X = data[rows, 1:].astype(np.float32)[:, :n_features]
        exact = exact_objective(X, y, depth, alpha)
        solver = contree_objective(X, y, depth, alpha)
        ok = abs(exact - solver) <= 1e-4 * (1 + exact)
        failures += not ok
        print(f"{name:9s} n={n} depth={depth} alpha={alpha:.1f}  exact={exact:10.4f}  contree={solver:10.4f}  {'OK' if ok else 'MISMATCH'}")
    print("all optimal" if failures == 0 else f"{failures} mismatches")
    return failures


if __name__ == "__main__":
    sys.exit(main())
