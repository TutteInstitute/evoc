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
        n_features=32,
        centers=8,
        cluster_std=1.5,
        center_box=(-8, 8),
        random_state=seed,
    )
    if dtype == "float":
        X = X.astype(np.float32)
    elif dtype == "uint8":
        projection = np.random.RandomState(0).normal(size=(X.shape[1], 256))
        X = np.packbits(X @ projection > 0, axis=1)
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
