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


# ------------------------------------------------------------------ EVoC.update
import pickle

from sklearn.metrics import adjusted_rand_score

from evoc import EVoC
from evoc.clustering_utilities import build_cluster_tree
from evoc.prediction import labels_from_hierarchy
from evoc.update import match_layers, stable_relabel

N_FIT = 2500


@pytest.fixture(scope="module", params=["float", "uint8", "int8"])
def update_case(request):
    # Seed chosen so that EVoC clusters the blobs cleanly for every dtype
    X, y = _data(request.param, seed=3)
    model = EVoC(random_state=0, prediction_data=True).fit(X[:N_FIT])
    old_layers = [l.copy() for l in model.cluster_layers_]
    old_labels = model.labels_.copy()
    old_embedding = model.prediction_data_.embedding.copy()
    blob = pickle.dumps(model)
    new_labels = model.update(X[N_FIT:])
    return dict(
        X=X,
        y=y,
        model=model,
        blob=blob,
        old_layers=old_layers,
        old_labels=old_labels,
        old_embedding=old_embedding,
        new_labels=new_labels,
    )


def test_update_requires_prediction_data():
    X, _ = _data("float", n=600)
    model = EVoC(random_state=0).fit(X[:500])
    with pytest.raises(ValueError, match="prediction_data=True"):
        model.update(X[500:])


def test_update_state_shapes(update_case):
    X, model = update_case["X"], update_case["model"]
    n = X.shape[0]
    assert update_case["new_labels"].shape == (n - N_FIT,)
    np.testing.assert_array_equal(update_case["new_labels"], model.labels_[N_FIT:])
    assert model.nn_inds_.shape == (n, K)
    np.testing.assert_array_equal(model.nn_inds_[:, 0], np.arange(n))
    assert np.all(model.nn_inds_ >= 0)
    for labels, strengths in zip(
        model.cluster_layers_, model.membership_strength_layers_
    ):
        assert labels.shape == (n,) and strengths.shape == (n,)
    pdata = model.prediction_data_
    assert pdata.data.shape[0] == n
    assert pdata.embedding.shape[0] == n
    assert pdata.sigmas.shape == (n,) and pdata.core_distances.shape == (n,)
    assert pdata.hierarchy.n_samples == n
    assert len(model.update_layer_matches_) == len(model.cluster_layers_)
    # The stored hierarchy reproduces the relabelled layers
    for layer, labels in enumerate(model.cluster_layers_):
        np.testing.assert_array_equal(
            labels_from_hierarchy(pdata.hierarchy, layer)[0], labels
        )


def test_update_cluster_ids_stable(update_case):
    model = update_case["model"]
    old_labels = update_case["old_labels"]
    X, y = update_case["X"], update_case["y"]
    # Refitting on everything with another seed is the alternative to update
    refit = EVoC(random_state=1).fit(X)
    refit_layer = match_layers([old_labels], refit.cluster_layers_, N_FIT)
    refit_stability = max(
        adjusted_rand_score(old_labels, layer[:N_FIT])
        for layer in refit.cluster_layers_
    )
    best = None
    for layer, match in zip(model.cluster_layers_, model.update_layer_matches_):
        stability = adjusted_rand_score(old_labels, layer[:N_FIT])
        if best is None or stability > best[0]:
            best = (stability, layer)
    stability, layer = best
    assert stability >= min(refit_stability, 0.95) - 0.02
    # Matched clusters keep their ids
    clustered = (old_labels >= 0) & (layer[:N_FIT] >= 0)
    assert np.mean(layer[:N_FIT][clustered] == old_labels[clustered]) >= 0.9
    # New samples are clustered consistently with the ground truth
    new = layer[N_FIT:]
    assert adjusted_rand_score(y[N_FIT:][new >= 0], new[new >= 0]) >= 0.9


def test_update_predict_and_pickle_after(update_case):
    model, X = update_case["model"], update_case["X"]
    labels = model.predict(X[:50])
    assert labels.shape == (50,)
    labels_e = model.predict(X[:50], method="embedding")
    assert labels_e.shape == (50,)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.predict(X[:50]), labels)
    _ = model.cluster_tree_


def test_update_without_reclustering(update_case):
    model = pickle.loads(update_case["blob"])
    X = update_case["X"]
    new_labels = model.update(X[N_FIT:], recluster=False)
    assert new_labels.shape == (X.shape[0] - N_FIT,)
    for old, new in zip(update_case["old_layers"], model.cluster_layers_):
        np.testing.assert_array_equal(new[:N_FIT], old)
    np.testing.assert_array_equal(
        model.prediction_data_.embedding[:N_FIT], update_case["old_embedding"]
    )
    assert model.prediction_data_.hierarchy.n_samples == N_FIT
    expected = model.predict(X[N_FIT:], method="embedding")
    assert np.mean(expected == new_labels) >= 0.9
    # A further update still works, in either mode
    model.update(X[:20], recluster=False)
    assert model.labels_.shape == (X.shape[0] + 20,)
    model.update(X[20:40])
    assert model.labels_.shape == (X.shape[0] + 40,)


def test_update_without_relabelling(update_case):
    model = pickle.loads(update_case["blob"])
    model.update(update_case["X"][N_FIT:], relabel=False)
    assert model.update_layer_matches_ is None
    for labels in model.cluster_layers_:
        present = np.unique(labels[labels >= 0])
        np.testing.assert_array_equal(present, np.arange(present.shape[0]))


def test_stable_relabel():
    old = np.array([0, 0, 1, 1, 2, 2, -1, -1])
    new = np.array([1, 1, 0, 0, 2, -1, 2, 3, 3, 3])
    relabelled, mapping = stable_relabel(old, new)
    np.testing.assert_array_equal(relabelled[:6], [0, 0, 1, 1, 2, -1])
    assert mapping[3] == 3
    # An old cluster that disappears leaves a gap rather than being reused
    old = np.array([0, 0, 1, 1, 2, 2])
    new = np.array([0, 0, 0, 0, 1, 1, 2])
    relabelled, mapping = stable_relabel(old, new)
    assert relabelled[4] == 2 and relabelled[6] == 3


def test_cluster_tree_with_label_gaps():
    lower = np.array([0, 0, 3, 3, 5, 5, -1])
    upper = np.array([2, 2, 2, 2, 7, 7, -1])
    tree = build_cluster_tree([lower, upper])
    assert sorted(tree[(1, 2)]) == [(0, 0), (0, 3)]
    assert tree[(1, 7)] == [(0, 5)]
    assert sorted(tree[(2, 0)]) == [(1, 2), (1, 7)]


def test_update_optimize_options(update_case):
    X = update_case["X"]
    model = pickle.loads(update_case["blob"])
    with pytest.raises(ValueError, match="optimize"):
        model.update(X[N_FIT:], optimize="old")
    model.update(X[N_FIT:])
    # optimize="new" leaves existing positions unchanged
    np.testing.assert_array_equal(
        model.prediction_data_.embedding[:N_FIT], update_case["old_embedding"]
    )
    model = pickle.loads(update_case["blob"])
    model.update(X[N_FIT:], optimize="all")
    assert model.labels_.shape == (X.shape[0],)
    assert not np.array_equal(
        model.prediction_data_.embedding[:N_FIT], update_case["old_embedding"]
    )
