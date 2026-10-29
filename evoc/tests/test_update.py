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


# --------------------------------------------------------------- expiring data
N_WINDOW = 2500
STEP = 500


@pytest.fixture(scope="module", params=["float", "uint8", "int8"])
def window_case(request):
    X, y = _data(request.param, n=N_WINDOW + 3 * STEP, seed=3)
    model = EVoC(random_state=0, prediction_data=True).fit(X[:N_WINDOW])
    return dict(X=X, y=y, blob=pickle.dumps(model), dtype=request.param)


def test_fit_sets_sample_ids(window_case):
    model = pickle.loads(window_case["blob"])
    np.testing.assert_array_equal(model.sample_ids_, np.arange(N_WINDOW))


def test_expire_forms_are_equivalent(window_case):
    X = window_case["X"]
    expired = np.arange(STEP)
    results = []
    for X_expired in (
        expired,
        np.isin(np.arange(N_WINDOW), expired),
        X[expired],
    ):
        model = pickle.loads(window_case["blob"])
        new_labels = model.update(X[N_WINDOW : N_WINDOW + STEP], X_expired=X_expired)
        results.append(model)
        assert new_labels.shape == (STEP,)
        np.testing.assert_array_equal(
            model.sample_ids_, np.arange(STEP, N_WINDOW + STEP)
        )
    for model in results[1:]:
        np.testing.assert_array_equal(model.nn_inds_, results[0].nn_inds_)
        np.testing.assert_array_equal(model.labels_, results[0].labels_)


def test_expire_graph_and_state(window_case):
    X = window_case["X"]
    model = pickle.loads(window_case["blob"])
    model.update(X[N_WINDOW : N_WINDOW + STEP], X_expired=np.arange(STEP))
    window = X[STEP : N_WINDOW + STEP]
    n = window.shape[0]
    assert model.nn_inds_.shape == (n, K)
    np.testing.assert_array_equal(model.nn_inds_[:, 0], np.arange(n))
    assert np.all((model.nn_inds_ >= 0) & (model.nn_inds_ < n))
    for row in model.nn_inds_[::13]:
        assert len(np.unique(row)) == K
    pdata = model.prediction_data_
    data, inv = prepare_data(window)
    np.testing.assert_allclose(pdata.data, data, atol=1e-6)
    assert pdata.embedding.shape[0] == n and pdata.hierarchy.n_samples == n
    for labels in model.cluster_layers_:
        assert labels.shape == (n,)
    # k-NN recall is as good as building the window's graph from scratch
    _, scratch_dists = knn_graph(window, n_neighbors=K, random_state=0)
    rows = np.arange(0, n, 7)
    updated = _tie_aware_recall(model.nn_dists_, data, inv, rows)
    scratch = _tie_aware_recall(scratch_dists, data, inv, rows)
    assert updated >= scratch - 0.005
    # Prediction state is consistent with the new samples
    assert model.predict(X[-20:]).shape == (20,)
    assert model.predict(X[-20:], method="embedding").shape == (20,)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(restored.sample_ids_, model.sample_ids_)


def test_moving_window(window_case):
    X, y = window_case["X"], window_case["y"]
    model = pickle.loads(window_case["blob"])
    seen_max = list(model._layer_max_labels)
    for step in range(3):
        start, stop = (step + 1) * STEP, N_WINDOW + (step + 1) * STEP
        before = [l.copy() for l in model.cluster_layers_]
        model.update(X[stop - STEP : stop], X_expired=np.arange(STEP))
        np.testing.assert_array_equal(model.sample_ids_, np.arange(start, stop))
        window_labels = y[start:stop]
        labels = model.labels_
        clustered = labels >= 0
        assert adjusted_rand_score(window_labels[clustered], labels[clustered]) >= 0.9
        # Labels that were not in the matched old layer are fresh ids, above
        # every id that layer has ever used
        for layer, match in zip(model.cluster_layers_, model.update_layer_matches_):
            fresh = np.setdiff1d(np.unique(layer[layer >= 0]), np.unique(before[match]))
            assert np.all(fresh > seen_max[match])
        seen_max = list(model._layer_max_labels)


def test_expire_only(window_case):
    X = window_case["X"]
    model = pickle.loads(window_case["blob"])
    new_labels = model.update(X_expired=np.arange(STEP))
    assert new_labels.shape == (0,)
    assert model.labels_.shape == (N_WINDOW - STEP,)
    np.testing.assert_array_equal(model.sample_ids_, np.arange(STEP, N_WINDOW))
    model = pickle.loads(window_case["blob"])
    model.update(None, X_expired=np.arange(N_WINDOW) >= N_WINDOW - STEP)
    np.testing.assert_array_equal(model.sample_ids_, np.arange(N_WINDOW - STEP))


def test_expire_errors(window_case):
    X = window_case["X"]
    model = pickle.loads(window_case["blob"])
    new = X[N_WINDOW : N_WINDOW + STEP]
    with pytest.raises(ValueError, match="recluster"):
        model.update(new, X_expired=np.arange(10), recluster=False)
    with pytest.raises(ValueError, match="X_new, X_expired"):
        model.update()
    with pytest.raises(ValueError, match="row positions"):
        model.update(new, X_expired=np.array([0, N_WINDOW]))
    with pytest.raises(ValueError, match="repeat"):
        model.update(new, X_expired=np.array([3, 3]))
    with pytest.raises(ValueError, match="mask"):
        model.update(new, X_expired=np.ones(10, dtype=bool))
    with pytest.raises(ValueError, match="does not match"):
        model.update(new, X_expired=X[-1:])
    with pytest.raises(ValueError):
        model.update(new, X_expired=X[:5, :-1])
    with pytest.raises(ValueError, match="n_neighbors"):
        model.update(X_expired=np.arange(N_WINDOW - 5))
    with pytest.raises(ValueError, match="retain at least one existing sample"):
        model.update(new, X_expired=np.arange(N_WINDOW))
    # A failed update leaves the model unchanged
    np.testing.assert_array_equal(model.sample_ids_, np.arange(N_WINDOW))


def test_expire_duplicate_rows_oldest_first():
    X, _ = _data("float", n=2000, seed=3)
    X = np.vstack([X, X[10:11], X[10:11]])
    model = EVoC(random_state=0, prediction_data=True).fit(X)
    n = X.shape[0]
    model.update(X_expired=X[10:11])
    assert 10 not in model.sample_ids_
    assert n - 2 in model.sample_ids_ and n - 1 in model.sample_ids_
    model.update(X_expired=X[10:12])
    assert n - 2 not in model.sample_ids_ and n - 1 in model.sample_ids_


def test_expire_whole_cluster(window_case):
    X, y = window_case["X"], window_case["y"]
    model = pickle.loads(window_case["blob"])
    gone = y[:N_WINDOW] == 0
    model.update(X[N_WINDOW:][y[N_WINDOW:] != 0][:STEP], X_expired=gone)
    assert model.labels_.shape == (N_WINDOW - gone.sum() + STEP,)
    assert np.all(model.nn_inds_ < model.labels_.shape[0])


# ------------------------------------------------------- repair distance kernels
import numba

from evoc.float_nndescent import fast_cosine
from evoc.int8_nndescent import fast_int_cosine_dissimilarity, int8_inverse_norms
from evoc.search.kernels import four_internal_distances_to, internal_distance_to
from evoc.uint8_nndescent import fast_bit_jaccard


@numba.njit
def _internal_single_and_four(data, inv_norms, a, candidates):
    inv_a = inv_norms[a] if inv_norms.shape[0] > 0 else np.float32(1.0)
    single = np.empty(candidates.shape[0], dtype=np.float32)
    for i in range(candidates.shape[0]):
        single[i] = internal_distance_to(data[a], inv_a, data, inv_norms, candidates[i])
    four = np.empty(candidates.shape[0], dtype=np.float32)
    out = np.empty(4, dtype=np.float32)
    for q in range(0, candidates.shape[0], 4):
        four_internal_distances_to(data[a], inv_a, data, inv_norms, candidates[q : q + 4], out)
        four[q : q + 4] = out
    return single, four


@pytest.mark.parametrize("dtype", ["float", "uint8", "int8"])
def test_internal_distance_kernels_match_nndescent(dtype):
    X, _ = _data(dtype, n=200)
    data, inv_norms = prepare_data(X)
    candidates = np.arange(1, 41, dtype=np.int32)
    single, four = _internal_single_and_four(data, inv_norms, 0, candidates)
    if dtype == "float":
        expected = [fast_cosine(data[0], data[c]) for c in candidates]
    elif dtype == "int8":
        inv = int8_inverse_norms(data)
        expected = [
            fast_int_cosine_dissimilarity(data[0], data[c], inv[0], inv[c])
            for c in candidates
        ]
    else:
        expected = [fast_bit_jaccard(data[0], data[c]) for c in candidates]
    np.testing.assert_allclose(single, expected, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(four, expected, rtol=1e-5, atol=1e-7)
