"""ConTree vs ConTree-ACP on the 16 ConTree datasets (maximum depth 3).

Protocol: the protocol of ConTree (maximum depth 3, depth tuned on five inner 80/20 splits), but with 25 outer splits
(stratified five-fold cross-validation repeated five times) to allow a paired test on each dataset. In every outer split, five stratified inner 80/20 splits of the training part choose ConTree's depth
(1, 2 or 3, highest mean inner accuracy, as in the ConTree paper). ConTree-ACP uses that same depth and chooses alpha:
keep the alphas whose mean inner accuracy is at least the ConTree (alpha = 0) mean minus one standard error (the
one-standard-error rule of Breiman et al.), and take the one with the lowest mean importance-weighted redundancy R_w (ties to the smaller alpha). Every depth and alpha
is also fitted on the outer training part, so the alpha trade-off can be shown on the test parts. At depth 1 the
penalty has no effect (the root has no ancestors), so only alpha = 0 is run there.

Each solver run is appended to the results file as soon as it finishes; rerunning skips finished runs.

    python contree-acp/experiments/run_experiments.py [--workers 14] [--depths 3] [--datasets rice,raisin] [--repeats 5]
    python contree-acp/experiments/run_experiments.py --depths 1,2 --workers 3 --results contree-acp/results/runs_shallow.jsonl
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "build" / ("ConTree.exe" if sys.platform == "win32" else "ConTree")
DATA_DIR = ROOT / "datasets"
RESULTS = ROOT / "results" / "runs.jsonl"

DATASETS = ["avila", "bank", "bean", "bidding", "eeg", "fault", "htru", "magic",
            "occupancy", "page", "raisin", "rice", "room", "segment", "skin", "wilt"]
ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
MAX_DEPTH = 3
N_SPLITS, N_REPEATS, SEED = 5, 5, 42
TIME_LIMIT = 3600  # seconds per solver run; runs that hit it are flagged in the results


def load(name):
    data = np.loadtxt(DATA_DIR / f"{name}.txt")
    return data[:, 1:].astype(np.float32), data[:, 0].astype(int)


def outer_splits(y):
    return list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(np.zeros(len(y)), y))


def inner_splits(y_train, outer_index):
    return list(StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=1000 + outer_index).split(np.zeros(len(y_train)), y_train))


# ---------------------------------------------------------------- trees and metrics

def predict(tree, X):
    if not isinstance(tree, list):
        return np.full(len(X), tree)
    feature, threshold, left, right = tree
    go_left = X[:, feature] <= np.float32(threshold)
    out = np.empty(len(X), dtype=int)
    out[go_left] = predict(left, X[go_left])
    out[~go_left] = predict(right, X[~go_left])
    return out


def gini(y, k):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=k) / len(y)
    return 1.0 - float(p @ p)


def walk(tree, X, y, k, path=()):
    """Yields ('split', feature, impurity decrease) for internal nodes and ('leaf', path, n) for leaves."""
    if not isinstance(tree, list):
        yield "leaf", path, len(y)
        return
    feature, threshold, left, right = tree
    go_left = X[:, feature] <= np.float32(threshold)
    decrease = len(y) * gini(y, k) - go_left.sum() * gini(y[go_left], k) - (~go_left).sum() * gini(y[~go_left], k)
    yield "split", feature, decrease
    yield from walk(left, X[go_left], y[go_left], k, path + (feature,))
    yield from walk(right, X[~go_left], y[~go_left], k, path + (feature,))


def redundancy_metrics(tree, X, y, abs_r):
    """Importance-weighted redundancy R_w, path redundancy, and the features used (same definitions as the ACP-Gini paper)."""
    k = int(y.max()) + 1
    importance = np.zeros(X.shape[1])
    leaves = []
    for kind, a, b in walk(tree, X, y, k):
        if kind == "split":
            importance[a] += b
        else:
            leaves.append((a, b))
    used = sorted({f for path, _ in leaves for f in path})
    total = importance.sum()
    importance = importance / total if total > 0 else importance
    pairs = [(i, j) for ii, i in enumerate(used) for j in used[ii + 1:]]
    weight = sum(importance[i] * importance[j] for i, j in pairs)
    r_w = sum(importance[i] * importance[j] * abs_r[i, j] for i, j in pairs) / weight if weight > 0 else 0.0
    path_r, path_high, n_total = 0.0, 0.0, sum(n for _, n in leaves)
    for path, n in leaves:
        distinct = sorted(set(path))
        values = [abs_r[i, j] for ii, i in enumerate(distinct) for j in distinct[ii + 1:]]
        path_r += n * (np.mean(values) if values else 0.0)
        path_high += n * any(v > 0.7 for v in values)
    return {"r_w": float(r_w), "path_r": float(path_r / n_total) if n_total else 0.0,
            "path_high_share": float(path_high / n_total) if n_total else 0.0,
            "n_features_used": len(used), "features_used": used, "n_leaves": len(leaves)}


def abs_correlation(X):
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.abs(np.corrcoef(X.astype(np.float64), rowvar=False))
    r = np.nan_to_num(r, nan=0.0)
    np.fill_diagonal(r, 1.0)
    return r


# ---------------------------------------------------------------- solver runs

def run_solver(X, y, alpha, depth):
    labels, y_contiguous = np.unique(y, return_inverse=True)  # ConTree expects labels 0..k-1
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "train.txt"
        np.savetxt(path, np.column_stack([y_contiguous, X]), fmt=["%d"] + ["%.9g"] * X.shape[1])
        start = time.perf_counter()
        out = subprocess.run([str(EXE), "-file", str(path), "-max-depth", str(depth), "-acp-alpha", repr(alpha),
                              "-time", str(TIME_LIMIT)], capture_output=True, text=True, check=True).stdout
        wall = time.perf_counter() - start
    fields = dict(line.split(": ", 1) for line in out.splitlines() if ": " in line)
    tree = json.loads(fields["Optimal tree"])

    def relabel(node):
        return [node[0], node[1], relabel(node[2]), relabel(node[3])] if isinstance(node, list) else int(labels[node])

    return {"tree": relabel(tree), "objective": float(fields["Objective"]),
            "train_errors": int(fields["Misclassification score"]), "within_time_limit": fields["Within time limit"] == "1",
            "solver_seconds": float(fields["Average time taken to get the decision tree"].split()[0]), "wall_seconds": wall}


def task_key(task):
    depth = task.get("depth", MAX_DEPTH)  # the first depth-3 runs were recorded without a depth field
    key = f"{task['dataset']}|{task['outer']}|{task['inner']}|{task['alpha']}"
    return key if depth == MAX_DEPTH else f"{key}|d{depth}"


def make_tasks(datasets, repeats, depths):
    tasks = []
    for repeat in range(repeats):  # repeat by repeat, so early results cover every dataset
        for name in datasets:
            for fold in range(N_SPLITS):
                outer = repeat * N_SPLITS + fold
                for depth in depths:
                    for alpha in ([0.0] if depth == 1 else ALPHAS):
                        for inner in [-1] + list(range(N_SPLITS)):
                            tasks.append({"dataset": name, "outer": outer, "inner": inner, "alpha": alpha, "depth": depth})
    return tasks


class DataCache:
    """Loads every dataset once and keeps its splits and full-data correlation."""

    def __init__(self):
        self._lock, self._data = threading.Lock(), {}

    def get(self, name):
        with self._lock:
            if name not in self._data:
                X, y = load(name)
                self._data[name] = (X, y, outer_splits(y), abs_correlation(X))
            return self._data[name]


def run_task(task, cache):
    X, y, splits, abs_r_full = cache.get(task["dataset"])
    train, test = splits[task["outer"]]
    if task["inner"] >= 0:
        inner_train, inner_val = inner_splits(y[train], task["outer"])[task["inner"]]
        train, test = train[inner_train], train[inner_val]
    result = run_solver(X[train], y[train], task["alpha"], task["depth"])
    result["train_accuracy"] = float(np.mean(predict(result["tree"], X[train]) == y[train]))
    result["eval_accuracy"] = float(np.mean(predict(result["tree"], X[test]) == y[test]))
    metrics_train = redundancy_metrics(result["tree"], X[train], y[train], abs_correlation(X[train]))
    metrics_full = redundancy_metrics(result["tree"], X[train], y[train], abs_r_full)
    result.update({k: v for k, v in metrics_train.items() if k not in ("r_w", "path_r", "path_high_share")})
    result.update({"r_w_train_corr": metrics_train["r_w"], "r_w": metrics_full["r_w"],
                   "path_r": metrics_full["path_r"], "path_high_share": metrics_full["path_high_share"],
                   "n_train": int(len(train)), "n_eval": int(len(test))})
    result["tree"] = json.dumps(result["tree"])
    return {**task, **result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--depths", default=str(MAX_DEPTH))
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()

    results = args.results
    results.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if results.exists():
        done = {task_key(json.loads(line)) for line in results.read_text().splitlines() if line.strip()}
    depths = [int(d) for d in args.depths.split(",")]
    tasks = [t for t in make_tasks(args.datasets.split(","), args.repeats, depths) if task_key(t) not in done]
    print(f"{len(tasks)} runs to do ({len(done)} already done), {args.workers} workers", flush=True)

    cache, write_lock, start = DataCache(), threading.Lock(), time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, open(results, "a") as out:
        futures = {pool.submit(run_task, t, cache): t for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            try:
                record = future.result()
            except Exception as error:  # keep going; the run is retried on the next invocation
                print(f"FAILED {task_key(task)}: {error!r}", flush=True)
                continue
            with write_lock:
                out.write(json.dumps(record) + "\n")
                out.flush()
            if i % 50 == 0 or i == len(tasks):
                elapsed = time.time() - start
                print(f"{i}/{len(tasks)} runs, {elapsed / 60:.1f} min elapsed, ~{elapsed / i * (len(tasks) - i) / 60:.0f} min left", flush=True)


if __name__ == "__main__":
    main()
