import pickle

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from evoc import EVoC
from evoc.graph_construction import smooth_knn_dist
from evoc.prediction import labels_from_hierarchy

N_SAMPLES = 2000


def _blob_data(dtype, seed=0, n_samples=N_SAMPLES, n_test=0):
    X, y = make_blobs(
        n_samples=n_samples + n_test,
        n_features=64,
        centers=8,
        cluster_std=2.0,
        center_box=(-10, 10),
        random_state=seed,
    )
    if dtype == "float":
        X = X.astype(np.float32)
    elif dtype == "uint8":
        # Noisy copies of a sparse binary prototype per cluster
        rng = np.random.RandomState(seed)
        prototypes = rng.rand(8, 256) < 0.3
        bits = prototypes[y] ^ (rng.rand(len(y), 256) < 0.15)
        X = np.packbits(bits, axis=1)
    else:
        X = np.clip(np.round(X / np.abs(X).max() * 127), -127, 127).astype(np.int8)
    return X, y


@pytest.fixture(scope="module", params=["float", "uint8", "int8"])
def dtype(request):
    return request.param


@pytest.fixture(scope="module")
def fitted_pair(dtype):
    X, _ = _blob_data(dtype)
    plain = EVoC(random_state=7).fit(X)
    with_data = EVoC(random_state=7, prediction_data=True).fit(X)
    return X, plain, with_data


def test_default_has_no_prediction_data(fitted_pair):
    _, plain, _ = fitted_pair
    assert not hasattr(plain, "prediction_data_")
    with pytest.raises(ValueError, match="prediction_data=True"):
        plain.prepare_prediction()


def test_prediction_data_does_not_change_results(fitted_pair):
    _, plain, with_data = fitted_pair
    assert len(plain.cluster_layers_) == len(with_data.cluster_layers_)
    for a, b in zip(plain.cluster_layers_, with_data.cluster_layers_):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(
        plain.membership_strength_layers_, with_data.membership_strength_layers_
    ):
        np.testing.assert_array_equal(a, b)
    assert plain.persistence_scores_ == with_data.persistence_scores_
    np.testing.assert_array_equal(plain.nn_inds_, with_data.nn_inds_)


@pytest.mark.parametrize(
    "kwargs", [{}, {"base_n_clusters": 6}, {"approx_n_clusters": 5}]
)
def test_hierarchy_reproduces_layers(kwargs):
    X, _ = _blob_data("float", seed=3)
    model = EVoC(random_state=1, prediction_data=True, **kwargs).fit(X)
    hierarchy = model.prediction_data_.hierarchy
    assert len(hierarchy.layer_clusters) == len(model.cluster_layers_)
    for layer in range(len(model.cluster_layers_)):
        labels, strengths = labels_from_hierarchy(hierarchy, layer)
        np.testing.assert_array_equal(labels, model.cluster_layers_[layer])
        np.testing.assert_array_equal(
            strengths, model.membership_strength_layers_[layer]
        )


def test_prediction_state_contents(fitted_pair):
    X, _, model = fitted_pair
    pdata = model.prediction_data_
    n = X.shape[0]
    assert pdata.data.shape[0] == n
    assert pdata.embedding.shape == (n, pdata.params["n_components"])
    assert pdata.sigmas.shape == (n,) and pdata.rhos.shape == (n,)
    sigmas, rhos = smooth_knn_dist(
        model.nn_dists_.astype(np.float32),
        float(pdata.params["neighbor_scale"] * pdata.params["n_neighbors"]),
    )
    np.testing.assert_array_equal(sigmas, pdata.sigmas)
    np.testing.assert_array_equal(rhos, pdata.rhos)
    # Training mean reproduces the batch fallback scale
    sigmas2, _ = smooth_knn_dist(
        model.nn_dists_.astype(np.float32),
        float(pdata.params["neighbor_scale"] * pdata.params["n_neighbors"]),
        mean_distance=pdata.mean_knn_distance,
    )
    np.testing.assert_allclose(sigmas2, pdata.sigmas, rtol=1e-6)
    # Core distances: distance to the min_samples-th nearest other point
    emb = pdata.embedding.astype(np.float64)
    sample = np.arange(0, n, 37)
    d = np.sqrt(((emb[sample, None, :] - emb[None, :, :]) ** 2).sum(-1))
    d[np.arange(len(sample)), sample] = np.inf
    expected = np.sort(d, axis=1)[:, pdata.hierarchy.min_samples - 1]
    np.testing.assert_allclose(pdata.core_distances[sample], expected, rtol=1e-4)


def test_pickle_round_trip(fitted_pair):
    _, _, model = fitted_pair
    model.prepare_prediction()
    assert model._search_index is not None
    restored = pickle.loads(pickle.dumps(model))
    assert not hasattr(restored, "_search_index")
    np.testing.assert_array_equal(
        restored.prediction_data_.embedding, model.prediction_data_.embedding
    )
    for a, b in zip(
        restored.prediction_data_.hierarchy.layer_clusters,
        model.prediction_data_.hierarchy.layer_clusters,
    ):
        np.testing.assert_array_equal(a, b)
    restored.prepare_prediction()


# ---------------------------------------------------------------- predict: vote
from sklearn.metrics import adjusted_rand_score

from evoc.prediction import (
    apply_thresholds,
    calibrate_vote_thresholds,
    membership_weights,
    stack_layers,
    vote_scores,
)


@pytest.fixture(scope="module")
def held_out(dtype):
    X, y = _blob_data(dtype, seed=11, n_samples=N_SAMPLES, n_test=400)
    model = EVoC(random_state=3, prediction_data=True).fit(X[:N_SAMPLES])
    return model, X[N_SAMPLES:], y[:N_SAMPLES], y[N_SAMPLES:]


def test_predict_requires_prediction_data(fitted_pair):
    X, plain, _ = fitted_pair
    with pytest.raises(ValueError, match="prediction_data=True"):
        plain.predict(X[:5])
    with pytest.raises(ValueError, match="prediction_data=True"):
        plain.predict_layers(X[:5])


def test_predict_input_checks(held_out):
    model, X_test, _, _ = held_out
    with pytest.raises(ValueError):
        model.predict(X_test[:, :-1])
    with pytest.raises(ValueError):
        model.predict(X_test[0])
    with pytest.raises(ValueError):
        model.predict(X_test, method="nearest")
    with pytest.raises(ValueError):
        model.predict(X_test, layer=len(model.cluster_layers_))
    wrong = X_test.astype(np.int16)
    with pytest.raises(ValueError):
        model.predict(wrong)


def test_predict_vote_shapes_and_default_layer(held_out):
    model, X_test, _, _ = held_out
    labels, strengths = model.predict_layers(X_test)
    assert len(labels) == len(model.cluster_layers_)
    for layer, (lab, st) in enumerate(zip(labels, strengths)):
        assert lab.shape == (X_test.shape[0],)
        assert lab.max() <= model.cluster_layers_[layer].max()
        assert np.all((st >= 0) & (st <= 1))
        assert np.all(st[lab < 0] == 0)
    best = model.predict(X_test)
    np.testing.assert_array_equal(best, labels[model._labels_layer()])
    lab0, st0 = model.predict(X_test, layer=0, return_strengths=True)
    np.testing.assert_array_equal(lab0, labels[0])
    np.testing.assert_array_equal(st0, strengths[0])


def test_predict_vote_leave_one_out(held_out):
    model, _, _, _ = held_out
    pdata = model.prediction_data_
    thresholds, _ = calibrate_vote_thresholds(
        model.nn_inds_,
        model.nn_dists_,
        pdata,
        model.cluster_layers_,
        model.membership_strength_layers_,
    )
    indices = np.ascontiguousarray(model.nn_inds_[:, 1:])
    distances = np.ascontiguousarray(model.nn_dists_[:, 1:])
    weights = membership_weights(indices, distances, pdata.sigmas, pdata.rhos)
    best_labels, best_scores = vote_scores(
        indices,
        weights,
        *stack_layers(model.cluster_layers_, model.membership_strength_layers_),
    )
    labels, _ = apply_thresholds(best_labels, best_scores, thresholds)
    layer = model._labels_layer()
    assert np.mean(labels[layer] == model.cluster_layers_[layer]) >= 0.95
    # Calibration reproduces each layer's training noise fraction
    for i, train_labels in enumerate(model.cluster_layers_):
        assert abs(np.mean(labels[i] < 0) - np.mean(train_labels < 0)) < 0.02


def test_predict_vote_held_out_agrees_with_truth(held_out):
    model, X_test, y_train, y_test = held_out
    layer = model._labels_layer()
    train_labels = model.cluster_layers_[layer]
    test_labels = model.predict(X_test)
    # Clustered points are assigned consistently with the ground truth ...
    train_ari = adjusted_rand_score(
        y_train[train_labels >= 0], train_labels[train_labels >= 0]
    )
    test_ari = adjusted_rand_score(
        y_test[test_labels >= 0], test_labels[test_labels >= 0]
    )
    assert test_ari >= min(0.9, train_ari - 0.05)
    # ... and the noise rate is close to the training noise rate
    assert abs(np.mean(test_labels < 0) - np.mean(train_labels < 0)) < 0.1


def test_predict_noise_threshold_override(held_out):
    model, X_test, _, _ = held_out
    none = model.predict(X_test, noise_threshold=0.0)
    all_noise = model.predict(X_test, noise_threshold=1.01)
    assert np.all(all_noise == -1)
    assert np.mean(none < 0) <= np.mean(model.predict(X_test) < 0)
    with pytest.raises(ValueError):
        model.predict_layers(X_test, noise_threshold=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1])


# ------------------------------------------------------------ predict: embedding
from evoc.prediction import predict_from_positions


@pytest.mark.parametrize(
    "kwargs", [{}, {"base_n_clusters": 6}, {"approx_n_clusters": 5}]
)
def test_embedding_self_prediction(kwargs):
    X, _ = _blob_data("float", seed=5)
    model = EVoC(random_state=2, prediction_data=True, **kwargs).fit(X)
    pdata = model.prediction_data_
    labels, strengths = predict_from_positions(
        pdata.embedding,
        pdata,
        model._get_embedding_tree(),
        model._get_tree_arrays(),
        exclude=np.arange(X.shape[0]),
    )
    for layer, train_labels in enumerate(model.cluster_layers_):
        assert np.mean(labels[layer] == train_labels) >= 0.98
        clustered = train_labels >= 0
        np.testing.assert_allclose(
            strengths[layer][clustered & (labels[layer] == train_labels)],
            model.membership_strength_layers_[layer][
                clustered & (labels[layer] == train_labels)
            ],
            atol=0.05,
        )


def test_predict_embedding_held_out(held_out):
    model, X_test, y_train, y_test = held_out
    train_labels = model.labels_
    labels, strengths = model.predict(
        X_test, method="embedding", return_strengths=True
    )
    assert labels.shape == (X_test.shape[0],)
    assert np.all((strengths >= 0) & (strengths <= 1))
    assert np.all(strengths[labels < 0] == 0)
    clustered = labels >= 0
    test_ari = adjusted_rand_score(y_test[clustered], labels[clustered])
    train_ari = adjusted_rand_score(
        y_train[train_labels >= 0], train_labels[train_labels >= 0]
    )
    assert test_ari >= min(0.9, train_ari - 0.05)
    assert abs(np.mean(labels < 0) - np.mean(train_labels < 0)) < 0.1
    # Vote and embedding mostly agree on clustered points
    vote = model.predict(X_test)
    both = (vote >= 0) & clustered
    assert np.mean(vote[both] == labels[both]) >= 0.9


def test_predict_embedding_refine_and_embed(held_out):
    model, X_test, _, _ = held_out
    positions = model.embed(X_test, refine_epochs=0)
    assert positions.shape == (X_test.shape[0], model.prediction_data_.embedding.shape[1])
    refined = model.embed(X_test, refine_epochs=5)
    assert np.all(np.isfinite(refined))
    assert not np.allclose(refined, positions)
    labels = model.predict(X_test, method="embedding", refine_epochs=5)
    assert labels.shape == (X_test.shape[0],)
    with pytest.raises(ValueError):
        model.predict(X_test, method="embedding", noise_threshold=0.5)
