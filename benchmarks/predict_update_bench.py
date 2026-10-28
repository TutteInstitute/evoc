"""Benchmarks for EVoC's search, predict and update.

Fits on the 60,000 training rows of MNIST / Fashion-MNIST and predicts or adds
the 10,000 test rows, and prints markdown tables:

* search: recall@10 against brute force, batch throughput and single-query
  latency of ``evoc.search.SearchIndex`` per dtype (and of pynndescent's cosine
  search on float data, if pynndescent is installed);
* predict: per layer, ARI between predicted test labels and the test rows'
  labels from fits on all 70,000 rows, next to the same ARI between two such
  full fits with different seeds; ARI with the true class labels; noise rates;
  timings;
* update: k-NN recall of the updated graph against a from-scratch graph, cost
  per stage for 100 / 1,000 / 10,000 new rows, label stability on the existing
  rows (against a refit), agreement with refits, and embedding drift;
* moving window: fit on rows 0-60k of train+test, then twice expire the oldest
  5k rows and add the next 5k, comparing each step with refitting the window.

Layers of different fits are paired by closest number of clusters.

Usage::

    python benchmarks/predict_update_bench.py \\
        [--datasets mnist fashion mnist-uint8 mnist-int8] [--seeds 3] \\
        [--data-dir ~/.cache/evoc_bench] [--skip search predict update window]
"""
import argparse
import gzip
import os
import pickle
import time
import urllib.request

import numba
import numpy as np
from sklearn.metrics import adjusted_rand_score

from evoc import EVoC
from evoc.knn_graph import knn_graph
from evoc.numba_kdtree import build_kdtree, parallel_tree_query
from evoc.search import SearchIndex
from evoc.search.kernels import prepare_data, query_distances

SOURCES = {
    "mnist": "https://raw.githubusercontent.com/fgnt/mnist/master/",
    "fashion": "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
}
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}
DTYPES = {
    "float": lambda x: x.astype(np.float32),
    "uint8": lambda x: np.packbits(x > 127, axis=1),
    "int8": lambda x: (x // 2).astype(np.int8),
}


def load(name, data_dir):
    source = "fashion" if name.startswith("fashion") else "mnist"
    dtype = name.split("-")[1] if "-" in name else "float"
    directory = os.path.join(os.path.expanduser(data_dir), source)
    os.makedirs(directory, exist_ok=True)
    arrays = {}
    for key, fname in FILES.items():
        path = os.path.join(directory, fname)
        if not os.path.exists(path):
            print(f"Downloading {SOURCES[source] + fname}", flush=True)
            urllib.request.urlretrieve(SOURCES[source] + fname, path)
        offset = 16 if key.endswith("_x") else 8
        array = np.frombuffer(gzip.open(path).read(), np.uint8, offset=offset)
        arrays[key] = array.reshape(-1, 784) if key.endswith("_x") else array
    convert = DTYPES[dtype]
    return (
        convert(arrays["train_x"]),
        arrays["train_y"],
        convert(arrays["test_x"]),
        arrays["test_y"],
    )


def mean_std(values, fmt="{:.3f}"):
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return fmt.format(values[0])
    return (fmt + " ± " + fmt).format(values.mean(), values.std())


def table(title, header, rows):
    print(f"\n#### {title}\n")
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join("---" for _ in header) + "|")
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")
    print(flush=True)


def n_clusters(labels):
    return len(np.unique(labels[labels >= 0]))


def closest_layer(target, layers, rows=slice(None)):
    """The layer with the closest number of clusters to ``target`` (log scale),
    ties broken by ARI on ``rows``."""
    gaps = np.array(
        [abs(np.log1p(n_clusters(l)) - np.log1p(n_clusters(target))) for l in layers]
    )
    candidates = np.flatnonzero(np.isclose(gaps, gaps.min()))
    return max(candidates, key=lambda c: adjusted_rand_score(target, layers[c][rows]))


def brute_force_kth(train, queries, k):
    data, inv = prepare_data(train)
    qdata, qinv = prepare_data(queries)
    kth = np.empty(queries.shape[0], dtype=np.float32)
    for start in range(0, queries.shape[0], 500):
        stop = min(start + 500, queries.shape[0])
        inds = np.tile(np.arange(data.shape[0], dtype=np.int32), (stop - start, 1))
        dists = query_distances(
            qdata[start:stop],
            qinv[start:stop] if qinv.shape[0] else qinv,
            data,
            inv,
            inds,
        )
        kth[start:stop] = np.partition(dists, k - 1, axis=1)[:, k - 1]
    return kth


def tie_aware_recall(found_dists, kth):
    """Fraction of returned neighbours at least as close as the true k-th."""
    return float(np.mean(found_dists <= kth[:, None] + 1e-6))


# ---------------------------------------------------------------------- search
def timed_queries(fn, n_queries):
    fn()
    best = np.inf
    for _ in range(3):
        t = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - t)
    return result, n_queries / best


def single_query_latency(fn, repeats=1000):
    fn()
    t = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t) / repeats * 1e6


def bench_search(name, X, Q, n_eval=1000):
    rows = []
    nn_inds, nn_dists = knn_graph(X, n_neighbors=15, random_state=0)
    t = time.perf_counter()
    index = SearchIndex(X, nn_inds, nn_dists, random_state=0)
    build = time.perf_counter() - t
    kth = brute_force_kth(X, Q[:n_eval], 10)
    for parallel in (False, True):
        for epsilon in (0.0, 0.1, 0.2):
            (inds, dists), qps = timed_queries(
                lambda: index.query(Q, 10, epsilon=epsilon, parallel=parallel),
                Q.shape[0],
            )
            rows.append(
                [
                    name,
                    "evoc" + (" (parallel)" if parallel else ""),
                    epsilon,
                    f"{tie_aware_recall(dists[:n_eval], kth):.4f}",
                    f"{qps:,.0f}",
                    "",
                ]
            )
    rows[1][5] = f"{single_query_latency(lambda: index.query(Q[:1], 10, parallel=False)):.1f}"
    if X.dtype == np.float32:
        try:
            import pynndescent
        except ImportError:
            pynndescent = None
        if pynndescent is not None:
            nnd = pynndescent.NNDescent(
                X, metric="cosine", n_neighbors=15, random_state=0
            )
            nnd.prepare()
            data, inv = prepare_data(X)
            qdata, qinv = prepare_data(Q[:n_eval])
            for epsilon in (0.0, 0.1, 0.2):
                (inds, _), qps = timed_queries(
                    lambda: nnd.query(Q, 10, epsilon=epsilon), Q.shape[0]
                )
                # Compare in EVoC's units via the returned indices
                d = query_distances(
                    qdata, qinv, data, inv, inds[:n_eval].astype(np.int32)
                )
                rows.append(
                    [
                        name,
                        "pynndescent (cosine)",
                        epsilon,
                        f"{tie_aware_recall(d, kth):.4f}",
                        f"{qps:,.0f}",
                        "",
                    ]
                )
            rows[-2][5] = f"{single_query_latency(lambda: nnd.query(Q[:1], 10)):.1f}"
    rows.append([name, "index build (s)", "", "", "", f"{build:.2f}"])
    return rows


# --------------------------------------------------------------------- predict
def bench_predict(name, X, y, Q, yq, seeds):
    n = X.shape[0]
    full = np.vstack([X, Q])
    model = EVoC(random_state=0, prediction_data=True).fit(X)
    t = time.perf_counter()
    model.prepare_prediction()
    prepare = time.perf_counter() - t
    refits = [EVoC(random_state=100 + s).fit(full) for s in range(max(seeds, 2))]
    timings, predictions = {}, {}
    for method in ("vote", "embedding"):
        t = time.perf_counter()
        predictions[method] = model.predict_layers(Q, method=method)[0]
        timings[method] = time.perf_counter() - t

    rows = []
    for layer, train_labels in enumerate(model.cluster_layers_):
        row = [name, layer, n_clusters(train_labels), f"{np.mean(train_labels < 0):.3f}"]
        # Refit layers paired with this layer by cluster count
        paired = [r.cluster_layers_[closest_layer(train_labels, [l[:n] for l in r.cluster_layers_])] for r in refits]
        baseline = [
            adjusted_rand_score(
                paired[i][n:],
                refits[j].cluster_layers_[
                    closest_layer(paired[i], refits[j].cluster_layers_)
                ][n:],
            )
            for i in range(len(refits))
            for j in range(len(refits))
            if i != j
        ]
        for method in ("vote", "embedding"):
            labels = predictions[method][layer]
            row += [
                mean_std([adjusted_rand_score(p[n:], labels) for p in paired]),
                f"{adjusted_rand_score(yq, labels):.3f}",
                f"{np.mean(labels < 0):.3f}",
            ]
        row += [
            mean_std(baseline),
            "/".join(str(n_clusters(p)) for p in paired),
            f"{adjusted_rand_score(y, train_labels):.3f}",
        ]
        rows.append(row)
    latency = single_query_latency(lambda: model.predict(Q[:1]), 200) / 1000
    info = [
        name,
        f"{prepare:.2f}",
        f"{timings['vote']:.2f}",
        f"{timings['embedding']:.2f}",
        f"{latency:.2f}",
        model._labels_layer(),
    ]
    return rows, info


# ---------------------------------------------------------------------- update
def bench_update(name, X, y, Q, yq, seeds):
    import evoc.clustering as clustering
    import evoc.update as update_module

    n = X.shape[0]
    full = np.vstack([X, Q])
    y_full = np.concatenate([y, yq])
    model = EVoC(random_state=0, prediction_data=True).fit(X)
    model.prepare_prediction()
    blob = pickle.dumps(model)
    old_labels = model.labels_.copy()
    old_embedding = model.prediction_data_.embedding.copy()

    timings = {}

    def timed(stage, fn):
        def wrapper(*args, **kwargs):
            t = time.perf_counter()
            result = fn(*args, **kwargs)
            timings[stage] = timings.get(stage, 0.0) + time.perf_counter() - t
            return result

        return wrapper

    patched = [
        (update_module, "update_knn_graph", "k-NN"),
        (clustering, "neighbor_graph_matrix", "fuzzy graph"),
        (clustering, "node_embedding", "embedding"),
        (clustering, "refine_positions", "embedding"),
        (clustering, "cluster_embedding", "recluster"),
    ]
    originals = [getattr(module, attr) for module, attr, _ in patched]
    for (module, attr, stage), original in zip(patched, originals):
        setattr(module, attr, timed(stage, original))
    cost_rows = []
    try:
        # Warm up (loads cached compiled functions for this dtype)
        warm = pickle.loads(blob)
        warm.update(Q[:200], refresh_index=False)
        for n_new in (100, 1000, Q.shape[0]):
            m = pickle.loads(blob)
            m.prepare_prediction()
            timings.clear()
            t = time.perf_counter()
            m.update(Q[:n_new], refresh_index=False)
            total = time.perf_counter() - t
            t = time.perf_counter()
            m.prepare_prediction()
            refresh = time.perf_counter() - t
            stages = ("k-NN", "fuzzy graph", "embedding", "recluster")
            cost_rows.append(
                [name, n_new]
                + [f"{timings.get(s, 0.0):.2f}" for s in stages]
                + [f"{refresh:.2f}", f"{total:.2f}"]
            )
    finally:
        for (module, attr, _), original in zip(patched, originals):
            setattr(module, attr, original)

    t = time.perf_counter()
    refits = [EVoC(random_state=100 + s).fit(full) for s in range(seeds)]
    refit_time = (time.perf_counter() - t) / seeds

    # k-NN recall of the updated graph against a from-scratch graph
    m = pickle.loads(blob)
    m.update(Q, refresh_index=False)
    rng = np.random.RandomState(0)
    old_rows = rng.choice(n, 1000, replace=False)
    new_rows = n + rng.choice(Q.shape[0], 1000, replace=False)
    kth_old = brute_force_kth(full, full[old_rows], 15)
    kth_new = brute_force_kth(full, full[new_rows], 15)
    _, scratch_dists = knn_graph(full, n_neighbors=15, random_state=0)
    knn_row = [
        name,
        f"{tie_aware_recall(m.nn_dists_[old_rows], kth_old):.4f}",
        f"{tie_aware_recall(scratch_dists[old_rows], kth_old):.4f}",
        f"{tie_aware_recall(m.nn_dists_[new_rows], kth_new):.4f}",
        f"{tie_aware_recall(scratch_dists[new_rows], kth_new):.4f}",
    ]

    dists, _ = parallel_tree_query(
        build_kdtree(old_embedding), old_embedding, np.int64(2), False
    )
    nn_dist = float(np.median(dists[:, 1]))

    def refit_layer(r, labels, rows):
        return r.cluster_layers_[closest_layer(labels, [l[rows] for l in r.cluster_layers_])]

    stability_rows = []
    for optimize, kwargs in (
        ("new", {}),
        ("all", {}),
        ("all", {"initial_alpha": 0.05}),
        ("all", {"n_epochs": 5}),
    ):
        stab, agree, truth, drift = [], [], [], []
        for seed in range(seeds):
            m = pickle.loads(blob)
            m.update(Q, optimize=optimize, random_state=seed, refresh_index=False, **kwargs)
            stab.append(adjusted_rand_score(old_labels, m.labels_[:n]))
            agree.append(
                np.mean([adjusted_rand_score(refit_layer(r, m.labels_, slice(None)), m.labels_) for r in refits])
            )
            truth.append(adjusted_rand_score(y_full, m.labels_))
            moved = np.linalg.norm(m.prediction_data_.embedding[:n] - old_embedding, axis=1)
            drift.append(np.median(moved) / nn_dist)
        label = optimize + (
            " (" + ", ".join(f"{k}={v}" for k, v in kwargs.items()) + ")" if kwargs else ""
        )
        stability_rows.append(
            [name, label, mean_std(stab), mean_std(agree), mean_std(truth), mean_std(drift, "{:.2f}")]
        )
    refit_stability = [
        adjusted_rand_score(old_labels, refit_layer(r, old_labels, slice(0, n))[:n])
        for r in refits
    ]
    refit_agreement = [
        adjusted_rand_score(
            refits[i].labels_,
            refit_layer(refits[j], refits[i].labels_, slice(None)),
        )
        for i in range(seeds)
        for j in range(seeds)
        if i != j
    ]
    stability_rows.append(
        [
            name,
            "refit on all rows",
            mean_std(refit_stability),
            mean_std(refit_agreement) if refit_agreement else "",
            mean_std([adjusted_rand_score(y_full, r.labels_) for r in refits]),
            "",
        ]
    )
    return cost_rows, knn_row, stability_rows, [name, f"{refit_time:.2f}"]


# ----------------------------------------------------------------- moving window
def bench_window(name, X, y, Q, yq, window=60_000, step=5_000, n_steps=2):
    full = np.vstack([X, Q])
    y_full = np.concatenate([y, yq])
    model = EVoC(random_state=0, prediction_data=True).fit(full[:window])
    model.prepare_prediction()
    warm = pickle.loads(pickle.dumps(model))
    warm.update(full[window : window + 200], X_expired=np.arange(200), refresh_index=False)
    previous_refit = EVoC(random_state=100).fit(full[:window])
    rows = []
    for i in range(n_steps):
        start, stop = (i + 1) * step, window + (i + 1) * step
        before = dict(zip(model.sample_ids_, model.labels_))
        t = time.perf_counter()
        model.update(full[stop - step : stop], X_expired=np.arange(step))
        elapsed = time.perf_counter() - t
        t = time.perf_counter()
        refit = EVoC(random_state=101 + i).fit(full[start:stop])
        refit_time = time.perf_counter() - t

        shared = model.sample_ids_[: window - step]
        old = np.array([before[s] for s in shared])
        refit_old = previous_refit.labels_[step:]
        rng = np.random.RandomState(i)
        sample = rng.choice(window, 1000, replace=False)
        kth = brute_force_kth(full[start:stop], full[start:stop][sample], 15)
        _, scratch_dists = knn_graph(full[start:stop], n_neighbors=15, random_state=0)
        rows.append(
            [
                name,
                i + 1,
                f"{elapsed:.2f}",
                f"{refit_time:.2f}",
                f"{tie_aware_recall(model.nn_dists_[sample], kth):.4f}",
                f"{tie_aware_recall(scratch_dists[sample], kth):.4f}",
                f"{adjusted_rand_score(old, model.labels_[: window - step]):.3f}",
                f"{adjusted_rand_score(refit_old, refit.labels_[: window - step]):.3f}",
                f"{adjusted_rand_score(y_full[start:stop], model.labels_):.3f}",
                f"{adjusted_rand_score(y_full[start:stop], refit.labels_):.3f}",
            ]
        )
        previous_refit = refit
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--datasets", nargs="+", default=["mnist", "fashion", "mnist-uint8", "mnist-int8"]
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--data-dir", default="~/.cache/evoc_bench")
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["search", "predict", "update", "window"],
    )
    args = parser.parse_args()

    print(f"numba threads: {numba.get_num_threads()}, seeds: {args.seeds}", flush=True)
    search_rows, predict_rows, predict_info = [], [], []
    cost_rows, knn_rows, stability_rows, refit_times = [], [], [], []
    window_rows = []
    for name in args.datasets:
        X, y, Q, yq = load(name, args.data_dir)
        print(f"{name}: train {X.shape} {X.dtype}, test {Q.shape}", flush=True)
        if "search" not in args.skip:
            search_rows += bench_search(name, X, Q)
        if "predict" not in args.skip:
            rows, info = bench_predict(name, X, y, Q, yq, args.seeds)
            predict_rows += rows
            predict_info.append(info)
        if "update" not in args.skip:
            cost, knn_row, stability, refit_time = bench_update(name, X, y, Q, yq, args.seeds)
            cost_rows += cost
            knn_rows.append(knn_row)
            stability_rows += stability
            refit_times.append(refit_time)
        if "window" not in args.skip:
            window_rows += bench_window(name, X, y, Q, yq)

    if search_rows:
        table(
            "Search: recall@10 (vs brute force, tie-aware), queries/s for 10k queries, single-query latency",
            ["data", "method", "epsilon", "recall@10", "queries/s", "single query (µs) / build (s)"],
            search_rows,
        )
    if predict_rows:
        table(
            "Predict: per layer; 'vs refit' is ARI with the test rows' labels from fits on all 70k rows",
            [
                "data", "layer", "clusters", "train noise",
                "vote: vs refit", "vote: vs truth", "vote: noise",
                "embedding: vs refit", "embedding: vs truth", "embedding: noise",
                "refit vs refit (test rows)", "refit layer clusters", "train vs truth",
            ],
            predict_rows,
        )
        table(
            "Predict: timings",
            ["data", "prepare_prediction (s)", "10k vote (s)", "10k embedding (s)", "single predict (ms)", "labels_ layer"],
            predict_info,
        )
    if cost_rows:
        table(
            "Update: cost per stage (s)",
            ["data", "new rows", "k-NN", "fuzzy graph", "embedding", "recluster", "index refresh", "total (excl. refresh)"],
            cost_rows,
        )
        table("Refit on all 70k rows (s)", ["data", "fit"], refit_times)
        table(
            "Update: k-NN recall@15 (tie-aware) after adding the 10k test rows",
            ["data", "old rows: updated", "old rows: scratch", "new rows: updated", "new rows: scratch"],
            knn_rows,
        )
        table(
            "Update (10k rows): labels_ stability on existing rows, agreement with refits, ARI with true labels, median drift of existing rows (in median embedding NN distances)",
            ["data", "optimize", "stability", "agreement with refit", "vs truth", "drift"],
            stability_rows,
        )

    if window_rows:
        table(
            "Moving window (60k rows; each step expires the oldest 5k and adds 5k): update vs refitting the window",
            [
                "data", "step", "update (s)", "refit (s)",
                "k-NN recall: updated", "k-NN recall: scratch",
                "stability: update", "stability: refit",
                "vs truth: update", "vs truth: refit",
            ],
            window_rows,
        )

if __name__ == "__main__":
    main()
