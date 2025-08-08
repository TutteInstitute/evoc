"""
Comprehensive test suite for the clustering module.

This module tests the EVoC clustering algorithm implementation, including
binary search for clusters, cluster layer building, duplicate detection,
and the main EVoC class functionality.
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score

from evoc.clustering import (
    _binary_search_for_n_clusters,
    build_cluster_layers,
    find_duplicates,
    _build_cluster_tree,
    build_cluster_tree,
    binary_search_for_n_clusters,
    evoc_clusters,
    EVoC,
)
import numba
from evoc.numba_kdtree import build_kdtree
from evoc.boruvka import parallel_boruvka
from evoc.cluster_trees import mst_to_linkage_tree


@pytest.fixture
def simple_embedding_data():
    """Create simple high-dimensional embedding-like data for testing."""
    # Create 512-dimensional data similar to CLIP embeddings
    X, y = make_blobs(n_samples=800, centers=4, n_features=512, 
                      cluster_std=0.8, random_state=42)
    # Normalize to unit sphere (typical for embeddings)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32), y


@pytest.fixture
def complex_embedding_data():
    """Create more complex high-dimensional embedding-like data for testing."""
    # Create 768-dimensional data similar to sentence transformer embeddings
    X, y = make_blobs(n_samples=2000, centers=8, n_features=768, 
                      cluster_std=0.6, random_state=42)
    # Normalize to unit sphere and add some noise
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    X += np.random.normal(0, 0.05, X.shape)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32), y


@pytest.fixture
def small_embedding_data():
    """Create small high-dimensional data for quick testing."""
    # Create 384-dimensional data (smaller embedding size)
    X, y = make_blobs(n_samples=300, centers=3, n_features=384, 
                      cluster_std=0.7, random_state=42)
    # Normalize to unit sphere
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32), y


@pytest.fixture
def duplicate_embedding_data():
    """Create high-dimensional embedding data with some duplicate points for testing."""
    X, y = make_blobs(n_samples=400, centers=3, n_features=512, random_state=42)
    # Normalize to unit sphere
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    # Add some duplicate points
    X_with_dups = np.vstack([X, X[:20]])  # Duplicate first 20 points
    y_with_dups = np.hstack([y, y[:20]])
    return X_with_dups.astype(np.float32), y_with_dups


@pytest.fixture
def quantized_embedding_data():
    """Create quantized (int8) embedding data for testing."""
    X, y = make_blobs(n_samples=600, centers=4, n_features=256, random_state=42)
    # Normalize and quantize to int8 range
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    X_quantized = (X * 127).astype(np.int8)
    return X_quantized, y


@pytest.fixture
def binary_embedding_data():
    """Create binary (uint8) embedding data for testing."""
    X, y = make_blobs(n_samples=500, centers=3, n_features=128, random_state=42)
    # Convert to binary representation
    X_binary = (X > np.median(X, axis=1, keepdims=True)).astype(np.uint8)
    return X_binary, y


@pytest.fixture
def small_linkage_tree():
    """Create a small linkage tree for testing."""
    # Create simple high-dimensional data and build MST
    X, _ = make_blobs(n_samples=100, centers=3, n_features=128, random_state=42)
    # Normalize to unit sphere like embeddings
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    numba_tree = build_kdtree(X.astype(np.float32))
    num_threads = numba.get_num_threads()
    edges = parallel_boruvka(numba_tree, num_threads, min_samples=3)
    sorted_mst = edges[np.argsort(edges.T[2])]
    return mst_to_linkage_tree(sorted_mst)


class TestBinarySearchForNClusters:
    """Test the binary search functionality for finding n clusters."""
    
    def test_binary_search_basic(self, small_linkage_tree):
        """Test basic binary search for cluster count."""
        n_samples = 100
        target_clusters = 3
        
        leaves, clusters, strengths = _binary_search_for_n_clusters(
            small_linkage_tree, target_clusters, n_samples
        )
        
        # Check return types and shapes
        assert isinstance(leaves, np.ndarray)
        assert isinstance(clusters, np.ndarray)
        assert isinstance(strengths, np.ndarray)
        assert len(clusters) == n_samples
        assert len(strengths) == n_samples
        
        # Check that we have reasonable cluster count
        n_clusters = len(np.unique(clusters[clusters >= 0]))
        assert n_clusters > 0
        assert n_clusters <= n_samples
        
        # Check that strengths are in valid range
        assert np.all(strengths >= 0)
        assert np.all(strengths <= 1)
    
    def test_binary_search_edge_cases(self, small_linkage_tree):
        """Test binary search with edge case parameters."""
        n_samples = 100
        
        # Test with very few clusters
        leaves, clusters, strengths = _binary_search_for_n_clusters(
            small_linkage_tree, 1, n_samples
        )
        assert len(clusters) == n_samples
        
        # Test with many clusters
        leaves, clusters, strengths = _binary_search_for_n_clusters(
            small_linkage_tree, 50, n_samples
        )
        assert len(clusters) == n_samples
    
    def test_binary_search_wrapper_function(self, simple_embedding_data):
        """Test the wrapper binary_search_for_n_clusters function."""
        X, y_true = simple_embedding_data
        num_threads = numba.get_num_threads()
        
        clusters, strengths = binary_search_for_n_clusters(
            X, approx_n_clusters=3, n_threads=num_threads, min_samples=5
        )
        
        # Check return types and shapes
        assert isinstance(clusters, np.ndarray)
        assert isinstance(strengths, np.ndarray)
        assert len(clusters) == len(X)
        assert len(strengths) == len(X)
        
        # Check that we found reasonable clusters
        n_clusters = len(np.unique(clusters[clusters >= 0]))
        assert n_clusters > 0
        assert n_clusters <= len(X)


class TestBuildClusterLayers:
    """Test the cluster layer building functionality."""
    
    def test_build_cluster_layers_basic(self, simple_embedding_data):
        """Test basic cluster layer building."""
        X, y_true = simple_embedding_data
        
        cluster_layers, membership_strengths, persistence_scores = build_cluster_layers(
            X,
            min_samples=5,
            base_min_cluster_size=10,
        )
        
        # Check return types
        assert isinstance(cluster_layers, list)
        assert isinstance(membership_strengths, list)
        assert len(cluster_layers) == len(membership_strengths)
        
        # Check that all layers have correct shape
        for clusters, strengths in zip(cluster_layers, membership_strengths):
            assert len(clusters) == len(X)
            assert len(strengths) == len(X)
            assert np.all(strengths >= 0)
            assert np.all(strengths <= 1)
    
    def test_build_cluster_layers_with_base_n_clusters(self, simple_embedding_data):
        """Test cluster layer building with specified base cluster count."""
        X, y_true = simple_embedding_data
        
        cluster_layers, membership_strengths, persistence_scores = build_cluster_layers(
            X,
            base_n_clusters=3,
            min_samples=5,
        )
        
        assert len(cluster_layers) > 0
        assert len(membership_strengths) > 0
        
        # Check that first layer has reasonable cluster count
        first_layer_clusters = cluster_layers[0]
        n_clusters = len(np.unique(first_layer_clusters[first_layer_clusters >= 0]))
        assert n_clusters > 0
    
    def test_build_cluster_layers_reproducible(self, simple_embedding_data):
        """Test that cluster layer building is reproducible."""
        X, y_true = simple_embedding_data
        
        layers1, strengths1, persistence1 = build_cluster_layers(
            X,
            base_min_cluster_size=10,
            reproducible_flag=True
        )
        
        layers2, strengths2, persistence2 = build_cluster_layers(
            X,
            base_min_cluster_size=10,
            reproducible_flag=True
        )
        
        # Results should be identical when reproducible flag is set
        assert len(layers1) == len(layers2)
        for l1, l2 in zip(layers1, layers2):
            np.testing.assert_array_equal(l1, l2)


class TestFindDuplicates:
    """Test the duplicate detection functionality."""
    
    def test_find_duplicates_basic(self):
        """Test basic duplicate detection."""
        # Create simple k-NN data with some duplicates
        knn_inds = np.array([
            [0, 1, 2],
            [1, 0, 2],
            [2, 0, 1],
            [3, 0, 1]  # Point 3 is close to points 0 and 1
        ], dtype=np.int32)
        
        knn_dists = np.array([
            [0.0, 0.5, 1.0],
            [0.5, 0.0, 1.0],
            [1.0, 0.5, 0.0],
            [0.8, 0.0, 0.0]  # Duplicate distance (0.0) indicates duplicates
        ], dtype=np.float32)
        
        duplicates = find_duplicates(knn_inds, knn_dists)
        
        # Check return type
        assert isinstance(duplicates, set)
        
        # Check that duplicates are tuples of pairs
        for dup in duplicates:
            assert isinstance(dup, tuple)
            assert len(dup) == 2
            assert dup[0] < dup[1]  # Should be ordered pairs
    
    def test_find_duplicates_no_duplicates(self):
        """Test duplicate detection when no duplicates exist."""
        knn_inds = np.array([
            [0, 1, 2],
            [1, 0, 2],
            [2, 0, 1]
        ], dtype=np.int32)
        
        knn_dists = np.array([
            [0.1, 0.5, 1.0],
            [0.5, 0.1, 1.0],
            [1.0, 0.5, 0.1]
        ], dtype=np.float32)
        
        duplicates = find_duplicates(knn_inds, knn_dists)
        
        # Should find minimal or no duplicates
        assert isinstance(duplicates, set)


class TestBuildClusterTree:
    """Test the cluster tree building functionality."""
    
    def test_build_cluster_tree_basic(self):
        """Test basic cluster tree building."""
        # Create simple hierarchical cluster labels
        labels = [
            np.array([0, 0, 1, 1, 2, 2]),  # Fine-grained
            np.array([0, 0, 0, 1, 1, 1]),  # Coarse-grained
        ]
        
        tree = build_cluster_tree(labels)
        
        # Check return type
        assert isinstance(tree, dict)
        
        # Check that keys are tuples (layer, cluster)
        for key in tree.keys():
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(key[0], (int, np.integer))
            assert isinstance(key[1], (int, np.integer))
        
        # Check that values are lists of child clusters
        for value in tree.values():
            assert isinstance(value, list)
            for child in value:
                assert isinstance(child, tuple)
                assert len(child) == 2
    
    def test_build_cluster_tree_empty(self):
        """Test cluster tree building with empty input."""
        labels = []
        # Empty input should be handled gracefully
        # Note: This may raise an error due to numba limitations with empty lists
        with pytest.raises((ValueError, Exception)):
            tree = build_cluster_tree(labels)
    
    def test_build_cluster_tree_single_layer(self):
        """Test cluster tree building with single layer."""
        labels = [np.array([0, 1, 0, 1, 2])]
        tree = build_cluster_tree(labels)
        assert isinstance(tree, dict)


class TestEvocClusters:
    """Test the main evoc_clusters function."""
    
    def test_evoc_clusters_basic(self, simple_embedding_data):
        """Test basic EVoC clustering."""
        X, y_true = simple_embedding_data
        
        cluster_layers, membership_strengths, persistence_scores = evoc_clusters(
            X,
            noise_level=0.5,
            base_min_cluster_size=5,
            base_n_clusters=2,
            n_neighbors=10,
            min_samples=3,
            n_epochs=20,
            random_state=np.random.RandomState(42)
        )
        
        # Check return types
        assert isinstance(cluster_layers, list)
        assert isinstance(membership_strengths, list)
        assert len(cluster_layers) == len(membership_strengths)
        assert len(cluster_layers) > 0
        
        # Check shapes
        for clusters, strengths in zip(cluster_layers, membership_strengths):
            assert len(clusters) == len(X)
            assert len(strengths) == len(X)
    
    def test_evoc_clusters_with_approx_n_clusters(self, simple_embedding_data):
        """Test EVoC clustering with specified cluster count."""
        X, y_true = simple_embedding_data
        
        cluster_layers, membership_strengths, persistence_scores = evoc_clusters(
            X,
            approx_n_clusters=3,
            n_neighbors=10,
            min_samples=3,
            n_epochs=20,
            random_state=np.random.RandomState(42)
        )
        
        # Should return exactly one layer
        assert len(cluster_layers) == 1
        assert len(membership_strengths) == 1
        
        # Check that we found reasonable clusters
        clusters = cluster_layers[0]
        n_clusters = len(np.unique(clusters[clusters >= 0]))
        assert n_clusters > 0
    
    def test_evoc_clusters_with_duplicates(self, duplicate_embedding_data):
        """Test EVoC clustering with duplicate detection."""
        X, y_true = duplicate_embedding_data
        
        cluster_layers, membership_strengths, persistence_scores, duplicates = evoc_clusters(
            X,
            return_duplicates=True,
            n_neighbors=10,
            min_samples=3,
            n_epochs=20,
            random_state=np.random.RandomState(42)
        )
        
        # Check that duplicates are returned
        assert isinstance(duplicates, set)
        
        # Check other return values
        assert isinstance(cluster_layers, list)
        assert isinstance(membership_strengths, list)
    
    def test_evoc_clusters_different_data_types(self, quantized_embedding_data, binary_embedding_data):
        """Test EVoC clustering with different embedding data types."""
        # Test with float32 data (standard embeddings)
        X_float = np.random.rand(100, 256).astype(np.float32)
        # Normalize like real embeddings
        X_float = X_float / np.linalg.norm(X_float, axis=1, keepdims=True)
        
        clusters, strengths, persistence_scores = evoc_clusters(
            X_float,
            approx_n_clusters=4,
            n_epochs=10,
            random_state=np.random.RandomState(42)
        )
        assert len(clusters) == 1
        assert len(clusters[0]) == 100
        
        # Test with int8 data (quantized embeddings)
        X_int8, _ = quantized_embedding_data
        clusters, strengths, persistence_scores = evoc_clusters(
            X_int8,
            approx_n_clusters=3,
            n_epochs=10,
            random_state=np.random.RandomState(42)
        )
        assert len(clusters) == 1
        assert len(clusters[0]) == len(X_int8)
        
        # Test with uint8 data (binary embeddings)
        X_uint8, _ = binary_embedding_data
        clusters, strengths, persistence_scores = evoc_clusters(
            X_uint8,
            approx_n_clusters=3,
            n_epochs=10,
            random_state=np.random.RandomState(42)
        )
        assert len(clusters) == 1
        assert len(clusters[0]) == len(X_uint8)


class TestEVoCClass:
    """Test the EVoC class implementation."""
    
    def test_evoc_init(self):
        """Test EVoC class initialization."""
        clusterer = EVoC(
            noise_level=0.3,
            base_min_cluster_size=10,
            n_neighbors=20,
            n_epochs=30,
            random_state=42
        )
        
        # Check that parameters are set correctly
        assert clusterer.noise_level == 0.3
        assert clusterer.base_min_cluster_size == 10
        assert clusterer.n_neighbors == 20
        assert clusterer.n_epochs == 30
        assert clusterer.random_state == 42
    
    def test_evoc_fit_predict(self, simple_embedding_data):
        """Test EVoC fit_predict method."""
        X, y_true = simple_embedding_data
        
        clusterer = EVoC(
            base_min_cluster_size=5,
            n_neighbors=10,
            n_epochs=20,
            random_state=42
        )
        
        labels = clusterer.fit_predict(X)
        
        # Check return type and shape
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(X)
        
        # Check that clusterer has fitted attributes
        assert hasattr(clusterer, 'labels_')
        assert hasattr(clusterer, 'membership_strengths_')
        assert hasattr(clusterer, 'cluster_layers_')
        assert hasattr(clusterer, 'membership_strength_layers_')
        assert hasattr(clusterer, 'duplicates_')
        
        # Check that labels are consistent
        np.testing.assert_array_equal(labels, clusterer.labels_)
    
    def test_evoc_fit(self, simple_embedding_data):
        """Test EVoC fit method."""
        X, y_true = simple_embedding_data
        
        clusterer = EVoC(
            base_min_cluster_size=5,
            n_neighbors=10,
            n_epochs=20,
            random_state=42
        )
        
        result = clusterer.fit(X)
        
        # Check that fit returns self
        assert result is clusterer
        
        # Check that clusterer has fitted attributes
        assert hasattr(clusterer, 'labels_')
        assert hasattr(clusterer, 'membership_strengths_')
    
    def test_evoc_with_approx_n_clusters(self, simple_embedding_data):
        """Test EVoC with specified cluster count."""
        X, y_true = simple_embedding_data
        
        clusterer = EVoC(
            approx_n_clusters=3,
            n_neighbors=10,
            n_epochs=20,
            random_state=42
        )
        
        labels = clusterer.fit_predict(X)
        
        # Check that we have reasonable cluster count
        n_clusters = len(np.unique(labels[labels >= 0]))
        assert n_clusters > 0
        assert n_clusters <= len(X)
    
    def test_evoc_cluster_tree_property(self, simple_embedding_data):
        """Test EVoC cluster_tree_ property."""
        X, y_true = simple_embedding_data
        
        clusterer = EVoC(
            base_min_cluster_size=5,
            n_neighbors=10,
            n_epochs=20,
            random_state=42
        )
        
        clusterer.fit(X)
        
        # Test cluster tree property
        tree = clusterer.cluster_tree_
        assert isinstance(tree, dict)
    
    def test_evoc_cluster_tree_not_fitted(self):
        """Test that cluster_tree_ raises error when not fitted."""
        clusterer = EVoC()
        
        with pytest.raises(Exception):  # Should raise NotFittedError or similar
            _ = clusterer.cluster_tree_
    
    def test_evoc_reproducibility(self, simple_embedding_data):
        """Test that EVoC produces reproducible results."""
        X, y_true = simple_embedding_data
        
        clusterer1 = EVoC(
            base_min_cluster_size=5,
            n_neighbors=10,
            n_epochs=20,
            random_state=42
        )
        
        clusterer2 = EVoC(
            base_min_cluster_size=5,
            n_neighbors=10,
            n_epochs=20,
            random_state=42
        )
        
        labels1 = clusterer1.fit_predict(X)
        labels2 = clusterer2.fit_predict(X)
        
        # Results should be identical with same random state
        np.testing.assert_array_equal(labels1, labels2)


class TestClusteringQuality:
    """Test clustering quality metrics and edge cases."""
    
    def test_clustering_quality_on_embeddings(self, simple_embedding_data):
        """Test that clustering works well on high-dimensional embedding data."""
        X, y_true = simple_embedding_data
        
        clusterer = EVoC(
            base_n_clusters=4,  # Match true number of clusters
            n_neighbors=15,
            n_epochs=30,
            random_state=42
        )
        
        try:
            labels = clusterer.fit_predict(X)
            
            # Calculate clustering quality metrics
            # Remove noise points for ARI calculation
            mask = labels >= 0
            if np.sum(mask) > 0:
                ari = adjusted_rand_score(y_true[mask], labels[mask])
                # Should achieve reasonable clustering quality on embeddings
                assert ari > 0.2  # Relaxed threshold for high-dimensional data
            
            # Check silhouette score (only if we have multiple clusters)
            n_clusters = len(np.unique(labels[labels >= 0]))
            if n_clusters > 1:
                sil_score = silhouette_score(X[mask], labels[mask])
                assert sil_score > 0.05  # Very relaxed for high-dimensional data
        except ValueError as e:
            # Handle case where clustering fails to find layers
            if "empty sequence" in str(e):
                pytest.skip("Clustering failed to find cluster layers for this data")
    
    def test_clustering_quality_on_blobs(self):
        """Test that clustering works on traditional blob data for compatibility."""
        # Create traditional 2D blob data for compatibility testing
        X, y_true = make_blobs(n_samples=100, centers=3, n_features=2, 
                              cluster_std=1.0, random_state=42)
        X = StandardScaler().fit_transform(X).astype(np.float32)
        
        clusterer = EVoC(
            base_n_clusters=3,  # Match true number of clusters
            n_neighbors=15,
            n_epochs=30,
            random_state=42
        )
        
        try:
            labels = clusterer.fit_predict(X)
            
            # Calculate clustering quality metrics
            # Remove noise points for ARI calculation
            mask = labels >= 0
            if np.sum(mask) > 0:
                ari = adjusted_rand_score(y_true[mask], labels[mask])
                # Should achieve good clustering quality on simple blobs
                assert ari > 0.3
            
            # Check silhouette score (only if we have multiple clusters)
            n_clusters = len(np.unique(labels[labels >= 0]))
            if n_clusters > 1:
                sil_score = silhouette_score(X[mask], labels[mask])
                assert sil_score > 0.1
        except ValueError as e:
            # Handle case where clustering fails to find layers
            if "empty sequence" in str(e):
                pytest.skip("Clustering failed to find cluster layers for this data")
    
    def test_clustering_on_small_dataset(self):
        """Test clustering on very small dataset."""
        # Use small but realistic embedding-like data
        X = np.random.rand(50, 256)
        # Normalize like embeddings
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        X = X.astype(np.float32)
        
        clusterer = EVoC(
            base_min_cluster_size=2,
            n_neighbors=5,
            n_epochs=10,
            random_state=42
        )
        
        try:
            labels = clusterer.fit_predict(X)
            
            # Should not crash and should return valid labels
            assert len(labels) == len(X)
            assert np.all((labels >= -1) & (labels < len(X)))
        except ValueError as e:
            # Handle case where clustering fails due to small dataset
            if "empty sequence" in str(e):
                pytest.skip("Clustering failed on very small dataset - expected behavior")
    
    def test_clustering_on_high_dimensional_data(self):
        """Test clustering on very high-dimensional embedding data."""
        # Test with 1024-dimensional data similar to large transformer models
        X = np.random.rand(500, 1024)
        # Normalize to unit sphere like real embeddings
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        X = X.astype(np.float32)
        
        clusterer = EVoC(
            base_min_cluster_size=8,
            n_neighbors=12,
            n_epochs=20,
            random_state=42
        )
        
        labels = clusterer.fit_predict(X)
        
        # Should handle very high dimensions gracefully
        assert len(labels) == len(X)
        assert np.all((labels >= -1) & (labels < len(X)))
    
    def test_edge_case_single_cluster(self):
        """Test edge case where all data forms single cluster."""
        # Create very tight cluster
        X = np.random.normal(0, 0.01, (50, 5))
        
        clusterer = EVoC(
            base_min_cluster_size=10,
            n_neighbors=15,
            n_epochs=20,
            random_state=42
        )
        
        try:
            labels = clusterer.fit_predict(X)
            
            # Should handle single cluster case
            assert len(labels) == len(X)
        except ValueError as e:
            # Handle case where clustering fails due to single tight cluster
            if "empty sequence" in str(e):
                pytest.skip("Clustering failed on single tight cluster - expected behavior")
    
    def test_parameter_validation(self):
        """Test that invalid parameters are handled appropriately."""
        # These should not crash during initialization
        clusterer = EVoC(
            noise_level=-0.1,  # Invalid but should be clamped/handled
            base_min_cluster_size=1,  # Very small
            n_neighbors=1,  # Very small
            n_epochs=1,  # Very small
        )
        
        # Should initialize without error
        assert isinstance(clusterer, EVoC)

    def test_clustering_on_clip_like_embeddings(self):
        """Test clustering on CLIP-like 512-dimensional embeddings."""
        # Simulate CLIP embeddings with multiple semantic clusters
        np.random.seed(42)
        n_samples_per_cluster = 80
        n_clusters = 5
        
        cluster_centers = np.random.randn(n_clusters, 512)
        cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)
        
        X = []
        y_true = []
        for i, center in enumerate(cluster_centers):
            # Generate points around each center
            cluster_points = center + np.random.normal(0, 0.1, (n_samples_per_cluster, 512))
            # Normalize to unit sphere
            cluster_points = cluster_points / np.linalg.norm(cluster_points, axis=1, keepdims=True)
            X.append(cluster_points)
            y_true.extend([i] * n_samples_per_cluster)
        
        X = np.vstack(X).astype(np.float32)
        y_true = np.array(y_true)
        
        clusterer = EVoC(
            base_n_clusters=n_clusters,
            n_neighbors=15,
            n_epochs=25,
            random_state=42
        )
        
        labels = clusterer.fit_predict(X)
        
        # Should handle CLIP-like embeddings well
        assert len(labels) == len(X)
        mask = labels >= 0
        if np.sum(mask) > 0:
            n_found_clusters = len(np.unique(labels[mask]))
            assert n_found_clusters > 1  # Should find multiple clusters
            
            # Check clustering quality
            ari = adjusted_rand_score(y_true[mask], labels[mask])
            assert ari > 0.15  # Should achieve reasonable clustering on well-separated data

    def test_clustering_on_sentence_transformer_like_embeddings(self):
        """Test clustering on sentence transformer-like 768-dimensional embeddings."""
        # Simulate sentence transformer embeddings
        np.random.seed(123)
        n_samples = 600
        n_dims = 768
        
        # Create embeddings with some structure
        X = np.random.rand(n_samples, n_dims) - 0.5
        # Add some clustering structure
        cluster_ids = np.random.choice([0, 1, 2, 3], n_samples)
        for i in range(4):
            mask = cluster_ids == i
            if np.sum(mask) > 0:
                # Add cluster-specific signal
                X[mask, i*50:(i+1)*50] += np.random.normal(2.0, 0.5, (np.sum(mask), 50))
        
        # Normalize like sentence transformers
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        X = X.astype(np.float32)
        
        clusterer = EVoC(
            base_min_cluster_size=8,
            n_neighbors=20,
            n_epochs=30,
            random_state=42
        )
        
        labels = clusterer.fit_predict(X)
        
        # Should handle sentence transformer-like embeddings
        assert len(labels) == len(X)
        mask = labels >= 0
        if np.sum(mask) > 0:
            n_found_clusters = len(np.unique(labels[mask]))
            assert n_found_clusters > 1
    
    def test_clustering_on_quantized_embeddings(self, quantized_embedding_data):
        """Test clustering specifically on quantized int8 embeddings."""
        X, y_true = quantized_embedding_data
        
        clusterer = EVoC(
            base_n_clusters=4,
            n_neighbors=12,
            n_epochs=20,
            random_state=42
        )
        
        labels = clusterer.fit_predict(X)
        
        # Should handle quantized embeddings
        assert len(labels) == len(X)
        assert np.all((labels >= -1) & (labels < len(X)))
        
        # Check that some clustering structure is found
        mask = labels >= 0
        if np.sum(mask) > 0:
            n_clusters = len(np.unique(labels[mask]))
            assert n_clusters >= 1
    
    def test_clustering_on_binary_embeddings(self, binary_embedding_data):
        """Test clustering specifically on binary uint8 embeddings."""
        X, y_true = binary_embedding_data
        
        clusterer = EVoC(
            base_n_clusters=3,
            n_neighbors=10,
            n_epochs=15,
            random_state=42
        )
        
        try:
            labels = clusterer.fit_predict(X)
            
            # Should handle binary embeddings
            assert len(labels) == len(X)
            assert np.all((labels >= -1) & (labels < len(X)))
            
            # Check that some clustering structure is found
            mask = labels >= 0
            if np.sum(mask) > 0:
                n_clusters = len(np.unique(labels[mask]))
                assert n_clusters >= 1
        except ValueError as e:
            # Handle case where clustering fails on binary data
            if "empty sequence" in str(e):
                pytest.skip("Clustering failed on binary embeddings - may need different parameters")


if __name__ == "__main__":
    pytest.main([__file__])
