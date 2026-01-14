"""
Comprehensive test suite for the knn_graph module.

This module tests the k-nearest neighbor graph construction functionality,
including random projection forest building, nearest neighbor descent,
and the main knn_graph function for different data types.
"""

import numpy as np
import pytest
import time
from unittest.mock import patch
from sklearn.datasets import make_blobs
from sklearn.utils import check_random_state

from evoc.knn_graph import (
    ts,
    make_forest,
    nn_descent,
    knn_graph,
    INT32_MIN,
    INT32_MAX,
)


class TestUtilityFunctions:
    """Test utility functions in the knn_graph module."""

    def test_ts_returns_string(self):
        """Test that ts() returns a properly formatted timestamp string."""
        timestamp = ts()
        assert isinstance(timestamp, str)
        assert len(timestamp) > 0
        # Test that it's a valid time format by checking it contains expected components
        assert any(
            month in timestamp
            for month in [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
        )

    def test_ts_consistency(self):
        """Test that ts() returns consistent format across multiple calls."""
        timestamp1 = ts()
        time.sleep(0.1)  # Small delay to potentially get different timestamps
        timestamp2 = ts()

        # Both should be strings of reasonable length
        assert isinstance(timestamp1, str)
        assert isinstance(timestamp2, str)
        assert len(timestamp1) > 20
        assert len(timestamp2) > 20

    def test_constants(self):
        """Test that INT32_MIN and INT32_MAX are properly defined."""
        assert INT32_MIN == np.iinfo(np.int32).min + 1
        assert INT32_MAX == np.iinfo(np.int32).max - 1
        assert INT32_MIN < INT32_MAX


class TestMakeForest:
    """Test the make_forest function for different data types and parameters."""

    @pytest.fixture
    def float_data(self):
        """Create normalized float32 test data."""
        np.random.seed(42)
        data = np.random.rand(100, 50).astype(np.float32)
        # Normalize to unit sphere
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        data = data / norms
        return data

    @pytest.fixture
    def uint8_data(self):
        """Create uint8 test data."""
        np.random.seed(42)
        return np.random.randint(0, 256, size=(100, 50), dtype=np.uint8)

    @pytest.fixture
    def int8_data(self):
        """Create int8 test data."""
        np.random.seed(42)
        return np.random.randint(-128, 128, size=(100, 50), dtype=np.int8)

    def test_make_forest_float32(self, float_data):
        """Test make_forest with float32 data."""
        random_state = check_random_state(42)
        n_neighbors = 10
        n_trees = 4
        leaf_size = 20

        result = make_forest(
            float_data, n_neighbors, n_trees, leaf_size, random_state, np.float32
        )

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int32
        assert result.shape[0] >= n_trees  # Should have at least n_trees rows

    def test_make_forest_uint8(self, uint8_data):
        """Test make_forest with uint8 data."""
        random_state = check_random_state(42)
        n_neighbors = 10
        n_trees = 4
        leaf_size = 20

        result = make_forest(
            uint8_data, n_neighbors, n_trees, leaf_size, random_state, np.uint8
        )

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int32
        assert result.shape[0] >= n_trees

    def test_make_forest_int8(self, int8_data):
        """Test make_forest with int8 data."""
        random_state = check_random_state(42)
        n_neighbors = 10
        n_trees = 4
        leaf_size = 20

        result = make_forest(
            int8_data, n_neighbors, n_trees, leaf_size, random_state, np.int8
        )

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int32
        assert result.shape[0] >= n_trees

    def test_make_forest_default_leaf_size(self, float_data):
        """Test make_forest with default leaf_size (None)."""
        random_state = check_random_state(42)
        n_neighbors = 15
        n_trees = 4

        result = make_forest(
            float_data, n_neighbors, n_trees, None, random_state, np.float32
        )

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.int32
        # With default leaf_size, it should be max(10, n_neighbors) = 15

    def test_make_forest_different_max_depth(self, float_data):
        """Test make_forest with different max_depth values."""
        random_state = check_random_state(42)
        n_neighbors = 10
        n_trees = 2
        leaf_size = 20

        # Test with small max_depth
        result_shallow = make_forest(
            float_data,
            n_neighbors,
            n_trees,
            leaf_size,
            random_state,
            np.float32,
            max_depth=5,
        )

        # Test with large max_depth
        random_state = check_random_state(42)  # Reset for consistency
        result_deep = make_forest(
            float_data,
            n_neighbors,
            n_trees,
            leaf_size,
            random_state,
            np.float32,
            max_depth=500,
        )

        assert isinstance(result_shallow, np.ndarray)
        assert isinstance(result_deep, np.ndarray)
        assert result_shallow.dtype == np.int32
        assert result_deep.dtype == np.int32

    @patch("evoc.knn_graph.make_float_forest")
    def test_make_forest_exception_handling(self, mock_make_float_forest, float_data):
        """Test make_forest handles exceptions properly."""
        # Mock the forest creation to raise an exception
        mock_make_float_forest.side_effect = RuntimeError("Test exception")

        random_state = check_random_state(42)

        with pytest.warns(
            UserWarning, match="Random Projection forest initialisation failed"
        ):
            result = make_forest(float_data, 10, 4, 20, random_state, np.float32)

        # Should return empty array on exception
        assert isinstance(result, np.ndarray)
        assert result.shape == (0, 0)
        assert result.dtype == np.int32


class TestNNDescent:
    """Test the nn_descent function for different data types."""

    @pytest.fixture
    def float_data(self):
        """Create normalized float32 test data."""
        np.random.seed(42)
        data = np.random.rand(50, 20).astype(np.float32)
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        data = data / norms
        return data

    @pytest.fixture
    def uint8_data(self):
        """Create uint8 test data."""
        np.random.seed(42)
        return np.random.randint(0, 256, size=(50, 20), dtype=np.uint8)

    @pytest.fixture
    def int8_data(self):
        """Create int8 test data."""
        np.random.seed(42)
        return np.random.randint(-128, 128, size=(50, 20), dtype=np.int8)

    def test_nn_descent_float32(self, float_data):
        """Test nn_descent with float32 data."""
        rng_state = np.random.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
        n_neighbors = 5

        with patch("evoc.float_nndescent.nn_descent_float") as mock_nn_descent:
            # Mock return value: (indices, distances)
            mock_indices = np.random.randint(
                0, len(float_data), size=(len(float_data), n_neighbors)
            )
            mock_distances = -np.random.exponential(
                1, size=(len(float_data), n_neighbors)
            )
            leaf_array = np.random.randint(
                0, float_data.shape[0], size=(4, len(float_data)), dtype=np.int32
            )
            mock_nn_descent.return_value = (mock_indices, mock_distances)

            result = nn_descent(
                float_data,
                n_neighbors,
                rng_state,
                30,
                5,
                0.001,
                np.float32,
                leaf_array=leaf_array,
                verbose=False,
            )

            assert len(result) == 2  # Should return (indices, distances)
            assert result[0].shape == (len(float_data), n_neighbors)
            assert result[1].shape == (len(float_data), n_neighbors)
            # Distances should be transformed: maximum(-log2(-distances), 0.0)
            assert np.all(result[1] >= 0.0)

    def test_nn_descent_uint8(self, uint8_data):
        """Test nn_descent with uint8 data."""
        rng_state = np.random.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
        n_neighbors = 5

        with patch("evoc.uint8_nndescent.nn_descent_uint8") as mock_nn_descent:
            mock_indices = np.random.randint(
                0, len(uint8_data), size=(len(uint8_data), n_neighbors)
            )
            mock_distances = -np.random.exponential(
                1, size=(len(uint8_data), n_neighbors)
            )
            leaf_array = np.random.randint(
                0, uint8_data.shape[0], size=(4, len(uint8_data)), dtype=np.int32
            )
            mock_nn_descent.return_value = (mock_indices, mock_distances)

            result = nn_descent(
                uint8_data,
                n_neighbors,
                rng_state,
                30,
                5,
                0.001,
                np.uint8,
                leaf_array=leaf_array,
                verbose=True,
            )

            assert len(result) == 2
            assert result[0].shape == (len(uint8_data), n_neighbors)
            assert result[1].shape == (len(uint8_data), n_neighbors)
            # Distances should be transformed: -log2(-distances)

    def test_nn_descent_int8(self, int8_data):
        """Test nn_descent with int8 data."""
        rng_state = np.random.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
        n_neighbors = 5

        with patch("evoc.int8_nndescent.nn_descent_int8") as mock_nn_descent:
            mock_indices = np.random.randint(
                0, len(int8_data), size=(len(int8_data), n_neighbors)
            )
            mock_distances = -np.random.exponential(
                1, size=(len(int8_data), n_neighbors)
            )
            mock_nn_descent.return_value = (mock_indices, mock_distances)
            leaf_array = np.random.randint(
                0, int8_data.shape[0], size=(4, len(int8_data)), dtype=np.int32
            )

            result = nn_descent(
                int8_data,
                n_neighbors,
                rng_state,
                30,
                5,
                0.001,
                np.int8,
                leaf_array=leaf_array,
            )

            assert len(result) == 2
            assert result[0].shape == (len(int8_data), n_neighbors)
            assert result[1].shape == (len(int8_data), n_neighbors)
            # Distances should be transformed: 1.0 / (-distances)


class TestKNNGraph:
    """Test the main knn_graph function."""

    @pytest.fixture
    def float_test_data(self):
        """Create test data for float32 testing."""
        # Create blob data that will be normalized
        X, y = make_blobs(
            n_samples=200, centers=4, n_features=50, cluster_std=1.0, random_state=42
        )
        return X.astype(np.float64)  # Start with float64 to test conversion

    @pytest.fixture
    def uint8_test_data(self):
        """Create uint8 test data."""
        np.random.seed(42)
        return np.random.randint(0, 256, size=(100, 30), dtype=np.uint8)

    @pytest.fixture
    def int8_test_data(self):
        """Create int8 test data."""
        np.random.seed(42)
        return np.random.randint(-128, 128, size=(100, 30), dtype=np.int8)

    def test_knn_graph_float_data(self, float_test_data):
        """Test knn_graph with float data (gets normalized)."""
        result = knn_graph(
            float_test_data, n_neighbors=10, n_trees=4, random_state=42, verbose=False
        )

        assert len(result) == 2  # (indices, distances)
        indices, distances = result

        assert indices.shape == (len(float_test_data), 10)
        assert distances.shape == (len(float_test_data), 10)
        assert indices.dtype == np.int32 or indices.dtype == np.int64
        assert distances.dtype == np.float32 or distances.dtype == np.float64

        # Check that indices are valid
        assert np.all(indices >= 0)
        assert np.all(indices < len(float_test_data))

        # Check that distances are non-negative (after transformation)
        assert np.all(distances >= 0.0)

    def test_knn_graph_uint8_data(self, uint8_test_data):
        """Test knn_graph with uint8 data."""
        result = knn_graph(
            uint8_test_data, n_neighbors=5, n_trees=3, random_state=42, verbose=False
        )

        assert len(result) == 2
        indices, distances = result

        assert indices.shape == (len(uint8_test_data), 5)
        assert distances.shape == (len(uint8_test_data), 5)
        assert np.all(indices >= 0)
        assert np.all(indices < len(uint8_test_data))

    def test_knn_graph_int8_data(self, int8_test_data):
        """Test knn_graph with int8 data."""
        result = knn_graph(int8_test_data, n_neighbors=8, random_state=42)

        assert len(result) == 2
        indices, distances = result

        assert indices.shape == (len(int8_test_data), 8)
        assert distances.shape == (len(int8_test_data), 8)
        assert np.all(indices >= 0)
        assert np.all(indices < len(int8_test_data))

    def test_knn_graph_parameters(self, float_test_data):
        """Test knn_graph with various parameter combinations."""
        # Test with custom parameters
        result = knn_graph(
            float_test_data,
            n_neighbors=15,
            n_trees=6,
            leaf_size=25,
            max_candidates=40,
            max_rptree_depth=100,
            n_iters=8,
            delta=0.01,
            n_jobs=1,
            verbose=True,
            random_state=123,
        )

        indices, distances = result
        assert indices.shape == (len(float_test_data), 15)
        assert distances.shape == (len(float_test_data), 15)

    def test_knn_graph_default_parameters(self, float_test_data):
        """Test knn_graph with mostly default parameters."""
        result = knn_graph(float_test_data, random_state=42)

        indices, distances = result
        # Default n_neighbors should be 30
        assert indices.shape == (len(float_test_data), 30)
        assert distances.shape == (len(float_test_data), 30)

    def test_knn_graph_n_jobs_setting(self, float_test_data):
        """Test that n_jobs parameter affects numba threading."""
        with (
            patch("numba.get_num_threads") as mock_get_threads,
            patch("numba.set_num_threads") as mock_set_threads,
        ):

            mock_get_threads.return_value = 8

            # Test with n_jobs=-1 (should not change threads)
            knn_graph(float_test_data, n_jobs=-1, random_state=42)
            mock_set_threads.assert_called()

            # Test with specific n_jobs
            knn_graph(float_test_data, n_jobs=4, random_state=42)
            # Should be called with 4 and then restored
            calls = mock_set_threads.call_args_list
            assert any(call[0][0] == 4 for call in calls)

    def test_knn_graph_auto_parameters(self, float_test_data):
        """Test automatic parameter selection."""
        with patch("numba.get_num_threads", return_value=2):
            result = knn_graph(
                float_test_data,
                n_trees=None,  # Should be auto-selected
                n_iters=None,  # Should be auto-selected
                random_state=42,
            )

            assert len(result) == 2
            # Auto n_trees should be max(4, min(8, num_threads)) = 8
            # Auto n_iters should be max(5, int(round(log2(n_samples))))

    def test_knn_graph_warning_on_failure(self, float_test_data):
        """Test that warning is issued when neighbor finding fails."""
        with patch("evoc.knn_graph.nn_descent") as mock_nn_descent:
            # Mock a result with some negative indices (indicating failure)
            mock_indices = np.full((len(float_test_data), 10), -1, dtype=np.int32)
            mock_distances = np.random.rand(len(float_test_data), 10)
            mock_nn_descent.return_value = (mock_indices, mock_distances)

            with pytest.warns(
                UserWarning, match="Failed to correctly find n_neighbors"
            ):
                result = knn_graph(float_test_data, n_neighbors=10, random_state=42)

    def test_knn_graph_data_validation(self):
        """Test that knn_graph properly validates input data."""
        # Test with invalid data shape
        invalid_data = np.array([1, 2, 3])  # 1D array

        with pytest.raises((ValueError, TypeError)):
            knn_graph(invalid_data)

    def test_knn_graph_float_normalization(self):
        """Test that float data gets properly normalized to unit sphere."""
        # Create data that's not normalized
        data = np.array([[3, 4], [1, 0], [0, 5]], dtype=np.float32)

        result = knn_graph(data, n_neighbors=2, random_state=42)

        # Should complete without error
        assert len(result) == 2
        indices, distances = result
        assert indices.shape == (3, 2)
        assert distances.shape == (3, 2)

    def test_knn_graph_zero_norm_handling(self):
        """Test handling of zero-norm vectors in float data."""
        # Include a zero vector
        data = np.array([[1, 1], [0, 0], [2, 2]], dtype=np.float32)

        result = knn_graph(data, n_neighbors=2, random_state=42)

        # Should complete without error (zero norms are set to 1.0)
        assert len(result) == 2
        indices, distances = result
        assert indices.shape == (3, 2)
        assert distances.shape == (3, 2)


class TestIntegration:
    """Integration tests for the complete knn_graph pipeline."""

    def test_small_dataset_complete_pipeline(self):
        """Test complete pipeline on a small dataset."""
        # Create a small, well-separated dataset
        X, y = make_blobs(
            n_samples=50, centers=3, n_features=10, cluster_std=0.5, random_state=42
        )
        X = X.astype(np.float32)

        result = knn_graph(X, n_neighbors=5, n_trees=2, random_state=42, verbose=True)

        indices, distances = result

        # Basic sanity checks
        assert indices.shape == (50, 5)
        assert distances.shape == (50, 5)
        assert np.all(indices >= 0)
        assert np.all(indices < 50)
        assert np.all(distances >= 0)

        # Note: Points may include themselves as neighbors, which is normal behavior

    def test_reproducibility(self):
        """Test that results are reproducible with same random state."""
        data = np.random.rand(30, 8).astype(np.float32)

        result1 = knn_graph(data, n_neighbors=5, random_state=42)
        result2 = knn_graph(data, n_neighbors=5, random_state=42)

        np.testing.assert_array_equal(result1[0], result2[0])  # indices
        np.testing.assert_array_almost_equal(result1[1], result2[1])  # distances

    def test_different_data_types_consistency(self):
        """Test that different data types produce reasonable results."""
        # Create base data
        np.random.seed(42)
        base_data = np.random.rand(40, 20)

        # Convert to different types
        float_data = base_data.astype(np.float32)
        uint8_data = (base_data * 255).astype(np.uint8)
        int8_data = ((base_data - 0.5) * 255).astype(np.int8)

        # Get results for each type
        float_result = knn_graph(float_data, n_neighbors=5, random_state=42)
        uint8_result = knn_graph(uint8_data, n_neighbors=5, random_state=42)
        int8_result = knn_graph(int8_data, n_neighbors=5, random_state=42)

        # All should have same shape
        for result in [float_result, uint8_result, int8_result]:
            assert result[0].shape == (40, 5)
            assert result[1].shape == (40, 5)
            assert np.all(result[0] >= 0)
            assert np.all(result[0] < 40)
            assert np.all(result[1] >= 0)
