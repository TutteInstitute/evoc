import numpy as np
import pytest
from sklearn.datasets import make_blobs

from evoc.knn_graph import knn_graph
from evoc.search import SearchIndex
from evoc.search.kernels import prepare_data, query_distances
from evoc.update import update_knn_graph

K = 15


def _data(dtype, n=3000, seed=0):
    X, y = make_blobs(
        n_samples=n,
        n_features=64,
        centers=8,
        cluster_std=2.0,
        center_box=(-10, 10),
        random_state=seed,
    )
    if dtype == "float":
        return X.astype(np.float32), y
    if dtype == "uint8":
        rng = np.random.RandomState(seed)
        prototypes = rng.rand(8, 256) < 0.3
        bits = prototypes[y] ^ (rng.rand(len(y), 256) < 0.15)
        return np.packbits(bits, axis=1), y
    return np.clip(np.round(X / np.abs(X).max() * 127), -127, 127).astype(np.int8), y


def _tie_aware_recall(nn_dists, data, inv, rows):
    all_inds = np.tile(np.arange(data.shape[0], dtype=np.int32), (len(rows), 1))
    inv_rows = inv[rows] if inv.shape[0] else inv
    full = query_distances(data[rows], inv_rows, data, inv, all_inds)
    kth = np.sort(full, axis=1)[:, K - 1]
    return np.mean(nn_dists[rows] <= kth[:, None] + 1e-6)


@pytest.mark.parametrize("dtype", ["float", "uint8", "int8"])
def test_update_knn_graph(dtype):
    X, _ = _data(dtype)
    n_old = 2500
    old, new = X[:n_old], X[n_old:]
    nn_inds, nn_dists = knn_graph(old, n_neighbors=K, random_state=0)
    index = SearchIndex(old, nn_inds, nn_dists, random_state=0)
    q_inds, q_dists = index.query(new, K - 1, epsilon=0.2)
    data, _ = prepare_data(old)
    new_data, _ = prepare_data(new)
    data_all, inds, dists = update_knn_graph(
        data,
        new_data,
        nn_inds,
        nn_dists,
        q_inds,
        q_dists,
        data.dtype.type,
        np.random.RandomState(0),
    )
    n = X.shape[0]
    assert inds.shape == (n, K) and dists.shape == (n, K)
    np.testing.assert_array_equal(inds[:, 0], np.arange(n))
    assert np.all(dists[:, 0] == 0)
    assert np.all(inds >= 0)
    for row in inds:
        assert len(np.unique(row)) == K
    assert np.all(np.diff(dists[:, 1:], axis=1) >= 0)
    # New points appear as neighbours of old points
    assert np.any(inds[:n_old, 1:] >= n_old)

    all_data, inv = prepare_data(X)
    scratch_inds, scratch_dists = knn_graph(X, n_neighbors=K, random_state=0)
    rng = np.random.RandomState(1)
    for rows in [rng.choice(n_old, 300, replace=False), np.arange(n_old, n)]:
        updated = _tie_aware_recall(dists, all_data, inv, rows)
        scratch = _tie_aware_recall(scratch_dists, all_data, inv, rows)
        assert updated >= scratch - 0.005
