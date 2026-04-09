"""Test suite for GPU-accelerated EVoC clustering.

Tests are automatically skipped if no GPU backend (CUDA/MPS) is available.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from evoc.gpu import gpu_available

# Skip entire module if no GPU
pytestmark = pytest.mark.skipif(
    not gpu_available(), reason="No GPU backend available (CUDA/MPS)"
)


@pytest.fixture
def embedding_data():
    """Create normalized 512-dim embedding-like data with 4 clusters."""
    X, y = make_blobs(
        n_samples=800, centers=4, n_features=512, cluster_std=0.8, random_state=42
    )
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32), y


@pytest.fixture
def int8_data():
    """Create int8 quantized embedding data."""
    X, y = make_blobs(
        n_samples=500, centers=3, n_features=128, cluster_std=1.0, random_state=42
    )
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    X = (X * 127).clip(-128, 127).astype(np.int8)
    return X, y


@pytest.fixture
def uint8_data():
    """Create uint8 binary embedding data."""
    rng = np.random.RandomState(42)
    n_samples = 400
    n_bytes = 64
    # 3 clusters of random binary vectors with shared patterns
    centers = rng.randint(0, 256, size=(3, n_bytes), dtype=np.uint8)
    labels = rng.choice(3, size=n_samples)
    X = np.empty((n_samples, n_bytes), dtype=np.uint8)
    for i in range(n_samples):
        # Flip ~20% of bits from center
        flip_mask = rng.randint(0, 256, size=n_bytes, dtype=np.uint8)
        flip_mask = (flip_mask < 50).astype(np.uint8) * 255
        X[i] = centers[labels[i]] ^ flip_mask
    return X, labels


# ============================================================
# GPU KNN tests
# ============================================================


class TestGPUKNN:
    """Tests for GPU brute-force KNN."""

    def test_knn_float_shape(self, embedding_data):
        from evoc.gpu.knn_gpu import knn_graph_gpu

        X, _ = embedding_data
        inds, dists = knn_graph_gpu(X, n_neighbors=15)
        assert inds.shape == (X.shape[0], 15)
        assert dists.shape == (X.shape[0], 15)
        assert inds.dtype == np.int32
        assert dists.dtype == np.float32

    def test_knn_float_no_self_neighbors(self, embedding_data):
        from evoc.gpu.knn_gpu import knn_graph_gpu

        X, _ = embedding_data
        inds, _ = knn_graph_gpu(X, n_neighbors=10)
        for i in range(X.shape[0]):
            assert i not in inds[i], f"Point {i} is its own neighbor"

    def test_knn_float_distances_nonnegative(self, embedding_data):
        from evoc.gpu.knn_gpu import knn_graph_gpu

        X, _ = embedding_data
        _, dists = knn_graph_gpu(X, n_neighbors=15)
        assert np.all(dists >= 0), "Some distances are negative"

    def test_knn_float_sorted(self, embedding_data):
        from evoc.gpu.knn_gpu import knn_graph_gpu

        X, _ = embedding_data
        _, dists = knn_graph_gpu(X, n_neighbors=15)
        # Distances should be non-decreasing per row (nearest first)
        for i in range(min(50, X.shape[0])):
            assert np.all(
                np.diff(dists[i]) >= -1e-6
            ), f"Row {i} distances not sorted"

    def test_knn_int8_shape(self, int8_data):
        from evoc.gpu.knn_gpu import knn_graph_gpu

        X, _ = int8_data
        inds, dists = knn_graph_gpu(X, n_neighbors=10)
        assert inds.shape == (X.shape[0], 10)
        assert dists.shape == (X.shape[0], 10)

    def test_knn_uint8_shape(self, uint8_data):
        from evoc.gpu.knn_gpu import knn_graph_gpu

        X, _ = uint8_data
        inds, dists = knn_graph_gpu(X, n_neighbors=10)
        assert inds.shape == (X.shape[0], 10)
        assert dists.shape == (X.shape[0], 10)

    def test_knn_batch_sizes(self, embedding_data):
        from evoc.gpu.knn_gpu import knn_graph_gpu

        X, _ = embedding_data
        # Small batch size forces multiple batches
        inds1, dists1 = knn_graph_gpu(X, n_neighbors=10, batch_size=128)
        inds2, dists2 = knn_graph_gpu(X, n_neighbors=10, batch_size=X.shape[0])
        np.testing.assert_array_equal(inds1, inds2)
        np.testing.assert_allclose(dists1, dists2, atol=1e-5)


# ============================================================
# GPU graph construction tests
# ============================================================


class TestGPUGraphConstruction:
    """Tests for GPU-accelerated graph construction."""

    def test_graph_shape(self, embedding_data):
        from evoc.gpu.knn_gpu import knn_graph_gpu
        from evoc.gpu.graph_construction_gpu import neighbor_graph_matrix_gpu

        X, _ = embedding_data
        inds, dists = knn_graph_gpu(X, n_neighbors=15)
        graph = neighbor_graph_matrix_gpu(15.0, inds, dists)
        assert graph.shape == (X.shape[0], X.shape[0])

    def test_graph_symmetric(self, embedding_data):
        from evoc.gpu.knn_gpu import knn_graph_gpu
        from evoc.gpu.graph_construction_gpu import neighbor_graph_matrix_gpu

        X, _ = embedding_data
        inds, dists = knn_graph_gpu(X, n_neighbors=15)
        graph = neighbor_graph_matrix_gpu(15.0, inds, dists, symmetrize=True)
        diff = abs(graph - graph.T)
        assert diff.max() < 1e-5, "Symmetrized graph is not symmetric"

    def test_graph_weights_range(self, embedding_data):
        from evoc.gpu.knn_gpu import knn_graph_gpu
        from evoc.gpu.graph_construction_gpu import neighbor_graph_matrix_gpu

        X, _ = embedding_data
        inds, dists = knn_graph_gpu(X, n_neighbors=15)
        graph = neighbor_graph_matrix_gpu(15.0, inds, dists)
        assert graph.data.min() >= 0, "Negative weights in graph"
        assert graph.data.max() <= 2.0, "Unexpectedly large weights"


# ============================================================
# GPU node embedding tests
# ============================================================


class TestGPUNodeEmbedding:
    """Tests for GPU-accelerated node embedding."""

    def test_embedding_shape(self, embedding_data):
        from evoc.gpu.knn_gpu import knn_graph_gpu
        from evoc.gpu.graph_construction_gpu import neighbor_graph_matrix_gpu
        from evoc.gpu.node_embedding_gpu import node_embedding_gpu

        X, _ = embedding_data
        inds, dists = knn_graph_gpu(X, n_neighbors=15)
        graph = neighbor_graph_matrix_gpu(15.0, inds, dists)
        emb = node_embedding_gpu(graph, n_components=8, n_epochs=10)
        assert emb.shape == (X.shape[0], 8)
        assert emb.dtype == np.float32

    def test_embedding_finite(self, embedding_data):
        from evoc.gpu.knn_gpu import knn_graph_gpu
        from evoc.gpu.graph_construction_gpu import neighbor_graph_matrix_gpu
        from evoc.gpu.node_embedding_gpu import node_embedding_gpu

        X, _ = embedding_data
        inds, dists = knn_graph_gpu(X, n_neighbors=15)
        graph = neighbor_graph_matrix_gpu(15.0, inds, dists)
        emb = node_embedding_gpu(graph, n_components=8, n_epochs=20)
        assert np.all(np.isfinite(emb)), "Embedding contains nan/inf"


# ============================================================
# End-to-end GPU clustering tests
# ============================================================


class TestGPUFullPipeline:
    """End-to-end tests of EVoC with GPU acceleration."""

    def test_evoc_clusters_gpu(self, embedding_data):
        from evoc import evoc_clusters

        X, y = embedding_data
        result = evoc_clusters(
            X,
            n_neighbors=15,
            n_epochs=20,
            random_state=np.random.RandomState(42),
            use_gpu=True,
        )
        labels = result[0][0]
        assert labels.shape == (X.shape[0],)
        # Should find some clusters (not all noise)
        assert labels.max() >= 1

    def test_evoc_class_gpu(self, embedding_data):
        from evoc import EVoC

        X, y = embedding_data
        model = EVoC(
            n_neighbors=15,
            n_epochs=20,
            random_state=42,
            use_gpu=True,
        )
        labels = model.fit_predict(X)
        assert labels.shape == (X.shape[0],)
        assert labels.max() >= 1

    def test_gpu_auto_mode(self, embedding_data):
        """use_gpu='auto' should use GPU when available."""
        from evoc import evoc_clusters

        X, _ = embedding_data
        result = evoc_clusters(
            X,
            n_neighbors=15,
            n_epochs=10,
            random_state=np.random.RandomState(42),
            use_gpu="auto",
        )
        labels = result[0][0]
        assert labels.shape == (X.shape[0],)

    def test_gpu_clustering_quality(self, embedding_data):
        """GPU clustering should achieve reasonable quality on well-separated data."""
        from evoc import evoc_clusters

        X, y = embedding_data
        result = evoc_clusters(
            X,
            n_neighbors=15,
            n_epochs=30,
            random_state=np.random.RandomState(42),
            use_gpu=True,
        )
        labels = result[0][0]
        # Allow noise points, compute ARI on non-noise
        mask = labels >= 0
        if mask.sum() > 50:
            ari = adjusted_rand_score(y[mask], labels[mask])
            assert ari > 0.3, f"GPU clustering quality too low: ARI={ari:.3f}"

    def test_gpu_approx_n_clusters(self, embedding_data):
        from evoc import evoc_clusters

        X, _ = embedding_data
        result = evoc_clusters(
            X,
            n_neighbors=15,
            n_epochs=20,
            approx_n_clusters=4,
            random_state=np.random.RandomState(42),
            use_gpu=True,
        )
        labels = result[0][0]
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        assert 2 <= n_clusters <= 8, f"Expected ~4 clusters, got {n_clusters}"
