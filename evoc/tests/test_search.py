import pickle

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from evoc.knn_graph import knn_graph
from evoc.search import SearchIndex, prepare_search
from evoc.search.diversify import diversify
from evoc.search.kernels import (
    internal_distances,
    internal_to_output,
    output_to_internal,
    pair_distances,
    prepare_data,
    query_distances,
)

N_TRAIN = 3000
N_TEST = 500
K = 15


def _blobs(seed=0):
    X, _ = make_blobs(
        n_samples=N_TRAIN + N_TEST,
        n_features=48,
        centers=12,
        cluster_std=3.0,
        random_state=seed,
    )
    return X


def _as_dtype(X, dtype):
    if dtype == "float":
        return X.astype(np.float32)
    if dtype == "uint8":
        # Sign of a fixed random projection: 256 bits per point
        projection = np.random.RandomState(0).normal(size=(X.shape[1], 256))
        return np.packbits(X @ projection > 0, axis=1)
    scaled = X / np.abs(X).max() * 127
    return np.clip(np.round(scaled), -127, 127).astype(np.int8)


@pytest.fixture(scope="module", params=["float", "uint8", "int8"])
def fitted(request):
    X = _as_dtype(_blobs(), request.param)
    train, test = X[:N_TRAIN], X[N_TRAIN:]
    nn_inds, nn_dists = knn_graph(train, n_neighbors=K, random_state=42)
    index = SearchIndex(train, nn_inds, nn_dists, random_state=42)
    return request.param, train, test, nn_inds, nn_dists, index


def _brute_force(train, queries, k):
    data, inv = prepare_data(train)
    qdata, qinv = prepare_data(queries)
    all_inds = np.tile(np.arange(train.shape[0], dtype=np.int32), (queries.shape[0], 1))
    dists = query_distances(qdata, qinv, data, inv, all_inds)
    order = np.argsort(dists, axis=1, kind="stable")[:, :k]
    return order, np.take_along_axis(dists, order, axis=1)


def test_kernels_match_knn_graph(fitted):
    _, train, _, nn_inds, nn_dists, _ = fitted
    data, inv = prepare_data(train)
    rows = np.repeat(np.arange(train.shape[0]), K - 1).astype(np.int32)
    cols = nn_inds[:, 1:].ravel().astype(np.int32)
    dists = pair_distances(data, inv, rows, cols).reshape(-1, K - 1)
    np.testing.assert_allclose(dists, nn_dists[:, 1:], rtol=1e-5, atol=1e-6)


def test_unit_conversions_round_trip(fitted):
    dtype, train, _, nn_inds, nn_dists, _ = fitted
    data, inv = prepare_data(train)
    rows = np.repeat(np.arange(200), K - 1).astype(np.int32)
    cols = nn_inds[:200, 1:].ravel().astype(np.int32)
    internal = internal_distances(data, inv, rows, cols)
    assert np.all(internal <= 0)
    np.testing.assert_allclose(
        internal_to_output(internal), nn_dists[:200, 1:].ravel(), rtol=1e-5, atol=1e-5
    )
    input_dtype = {"float": np.float32, "uint8": np.uint8, "int8": np.int8}[dtype]
    np.testing.assert_allclose(
        output_to_internal(nn_dists[:200, 1:].ravel(), input_dtype),
        internal,
        rtol=1e-5,
        atol=1e-6,
    )


def test_diversify_keeps_nearest(fitted):
    _, train, _, nn_inds, nn_dists, _ = fitted
    data, inv = prepare_data(train)
    rows = np.ascontiguousarray(nn_inds[:, 1:], dtype=np.int32)
    dists = np.ascontiguousarray(nn_dists[:, 1:], dtype=np.float32)
    new_rows, new_dists = diversify(rows.copy(), dists.copy(), data, inv, 0, 1.0)
    np.testing.assert_array_equal(new_rows[:, 0], rows[:, 0])
    kept = new_rows >= 0
    assert np.all(kept.sum(axis=1) >= 1)
    assert kept.sum() < 0.9 * kept.size
    # Retained neighbours keep their order and come from the original row
    for i in range(0, train.shape[0], 97):
        row = new_rows[i][kept[i]]
        positions = [list(rows[i]).index(r) for r in row]
        assert positions == sorted(positions)


def test_tree_leaves_contain_training_points(fitted):
    """Descending the tree with a training point reaches the leaf holding it,
    unless the point lies on a splitting hyperplane (ties are resolved by rank
    when building and at random when searching; common for bit data)."""
    dtype, train, _, _, _, index = fitted
    tree = index._tree
    data = index._data
    if dtype == "int8":
        tree_data = data.astype(np.float32) * index._inv_norms[:, None]
    else:
        tree_data = data
    n_bytes = tree_data.shape[1]
    checked = 0
    misses = 0
    for i in range(0, train.shape[0], 7):
        node = 0
        tied = False
        while tree.children[node, 0] > 0:
            plane = tree.hyperplanes[node]
            if dtype == "uint8":
                bits = np.unpackbits(tree_data[i])
                margin = tree.offsets[node] + float(
                    int(np.unpackbits(plane[:n_bytes])[bits == 1].sum())
                    - int(np.unpackbits(plane[n_bytes:])[bits == 1].sum())
                )
            else:
                margin = tree.offsets[node] + float(np.dot(plane, tree_data[i]))
            if abs(margin) < 1e-5:
                tied = True
                break
            node = tree.children[node, 0] if margin > 0 else tree.children[node, 1]
        if tied:
            continue
        checked += 1
        start, end = -tree.children[node, 0], -tree.children[node, 1]
        if i not in tree.indices[start:end]:
            misses += 1
    assert checked > 100
    # The builder may move a node's first point across the split to avoid an
    # empty side, so allow a few strays
    assert misses <= 0.02 * checked


def test_self_search_recall(fitted):
    _, train, _, nn_inds, nn_dists, index = fitted
    inds, dists = index.query(train, K, epsilon=0.1)
    assert inds.shape == (train.shape[0], K)
    # Found neighbours at least as close as knn_graph's K-th (tie-aware)
    recall = np.mean(dists <= nn_dists[:, -1:] + 1e-6)
    assert recall >= 0.95
    assert np.all(np.diff(dists, axis=1) >= 0)


def test_held_out_recall(fitted):
    dtype, train, test, _, _, index = fitted
    inds, dists = index.query(test, 10, epsilon=0.1, parallel=False)
    true_inds, true_dists = _brute_force(train, test, 10)
    # Count found neighbours at least as close as the true 10th (ties are common
    # for bit-packed and quantised data)
    recall = np.mean(dists <= true_dists[:, -1:] + 1e-6)
    assert recall >= (0.8 if dtype == "int8" else 0.9)


def test_distances_are_output_units(fitted):
    dtype, train, test, _, _, index = fitted
    inds, dists = index.query(test[:100], 5)
    data, inv = prepare_data(train)
    qdata, qinv = prepare_data(test[:100])
    expected = query_distances(qdata, qinv, data, inv, inds)
    np.testing.assert_allclose(dists, expected, rtol=1e-5, atol=1e-5)
    if dtype == "float":
        a = test[:100].astype(np.float64)
        b = train.astype(np.float64)[inds]
        cos = np.einsum("ij,ikj->ik", a, b) / (
            np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=2)
        )
        np.testing.assert_allclose(dists, np.maximum(-np.log2(cos), 0), atol=1e-5)


def test_pickle_round_trip(fitted):
    _, _, test, _, _, index = fitted
    inds, dists = index.query(test, 10)
    restored = pickle.loads(pickle.dumps(index))
    inds2, dists2 = restored.query(test, 10)
    np.testing.assert_array_equal(inds, inds2)
    np.testing.assert_array_equal(dists, dists2)


def test_parallel_matches_quality(fitted):
    dtype, train, test, nn_inds, nn_dists, _ = fitted
    index = prepare_search(
        train, nn_inds, nn_dists, random_state=42, parallel_batch_queries=True
    )
    inds, dists = index.query(test, 10)
    true_inds, true_dists = _brute_force(train, test, 10)
    recall = np.mean(dists <= true_dists[:, -1:] + 1e-6)
    assert recall >= (0.8 if dtype == "int8" else 0.9)


def test_input_validation(fitted):
    dtype, train, test, _, _, index = fitted
    with pytest.raises(ValueError):
        index.query(test[0], 5)
    with pytest.raises(ValueError):
        index.query(test[:, :-1], 5)
    wrong = test.astype(np.int16) if dtype != "float" else test.astype(np.int8)
    with pytest.raises(ValueError):
        index.query(wrong, 5)
    with pytest.raises(ValueError):
        index.query(test, 0)
    with pytest.raises(ValueError):
        index.query(test, train.shape[0] + 1)


def test_float_queries_any_float_dtype_and_zero_rows():
    X = _as_dtype(_blobs(1), "float")
    train, test = X[:N_TRAIN], X[N_TRAIN:]
    nn_inds, nn_dists = knn_graph(train, n_neighbors=K, random_state=0)
    index = SearchIndex(train, nn_inds, nn_dists, random_state=0)
    i32, d32 = index.query(test, 5)
    i64, d64 = index.query(test.astype(np.float64), 5)
    np.testing.assert_array_equal(i32, i64)
    test = test.copy()
    test[3] = 0.0
    inds, dists = index.query(test, 5)
    assert np.all(inds[3] == -1)
    assert np.all(np.isinf(dists[3]))
    assert np.all(inds[4] >= 0)
