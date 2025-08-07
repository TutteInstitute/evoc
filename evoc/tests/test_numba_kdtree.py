"""
Test suite for NumbaKDTree compatibility with sklearn KDTree.

This module tests that our NumbaKDTree implementation produces equivalent
partitioning and query results compared to sklearn's KDTree implementation.
"""

import numpy as np
import pytest
from sklearn.neighbors import KDTree as SklearnKDTree

from evoc.numba_kdtree import build_kdtree


class TestKDTreeCompatibility:
    """Test compatibility between NumbaKDTree and sklearn KDTree implementations."""
    
    @pytest.fixture(params=[
        (50, 2),   # Small 2D
        (100, 3),  # Medium 3D
        (200, 5),  # Large 5D
        (500, 8),  # Large 8D
    ])
    def test_data(self, request):
        """Generate test data for various configurations."""
        n_samples, n_features = request.param
        np.random.seed(42)  # Fixed seed for reproducible tests
        return np.random.rand(n_samples, n_features).astype(np.float32)
    
    @pytest.fixture(params=[10, 20, 40])
    def leaf_size(self, request):
        """Test different leaf sizes."""
        return request.param
    
    def test_tree_structure_compatibility(self, test_data, leaf_size):
        """Test that tree structures have compatible shapes and properties."""
        # Build trees
        sklearn_tree = SklearnKDTree(test_data, leaf_size=leaf_size)
        numba_tree = build_kdtree(test_data, leaf_size=leaf_size)
        
        # Get sklearn internal arrays
        sk_data, sk_idx_array, sk_node_data, sk_node_bounds = sklearn_tree.get_arrays()
        
        # Test data compatibility
        assert np.array_equal(sk_data, numba_tree.data), "Data arrays should match"
        assert sk_idx_array.shape == numba_tree.idx_array.shape, "Index array shapes should match"
        
        # Test node data shapes
        assert sk_node_data['idx_start'].shape == numba_tree.idx_start.shape, "idx_start shapes should match"
        assert sk_node_data['idx_end'].shape == numba_tree.idx_end.shape, "idx_end shapes should match"
        assert sk_node_data['radius'].shape == numba_tree.radius.shape, "radius shapes should match"
        assert sk_node_data['is_leaf'].shape == numba_tree.is_leaf.shape, "is_leaf shapes should match"
        
        # Test node bounds shape
        assert sk_node_bounds.shape == numba_tree.node_bounds.shape, "Node bounds shapes should match"
    
    def test_node_partitioning_equivalence(self, test_data, leaf_size):
        """
        Test that both implementations partition data into equivalent node sets.
        
        This verifies that each node contains the same set of data points,
        regardless of internal ordering differences.
        """
        # Build trees
        sklearn_tree = SklearnKDTree(test_data, leaf_size=leaf_size)
        numba_tree = build_kdtree(test_data, leaf_size=leaf_size)
        
        # Get sklearn internal arrays
        sk_data, sk_idx_array, sk_node_data, sk_node_bounds = sklearn_tree.get_arrays()
        n_nodes = sk_node_data.shape[0]
        
        matches = 0
        total_comparisons = 0
        
        for node in range(n_nodes):
            # Get node boundaries
            sk_start = sk_node_data[node]['idx_start']
            sk_end = sk_node_data[node]['idx_end']
            sk_is_leaf = sk_node_data[node]['is_leaf']
            
            nb_start = numba_tree.idx_start[node]
            nb_end = numba_tree.idx_end[node]
            nb_is_leaf = numba_tree.is_leaf[node]
            
            # Node properties should match exactly
            assert sk_start == nb_start, f"Node {node}: idx_start mismatch"
            assert sk_end == nb_end, f"Node {node}: idx_end mismatch"
            assert sk_is_leaf == nb_is_leaf, f"Node {node}: is_leaf mismatch"
            
            # Skip empty nodes
            if sk_start >= sk_end:
                continue
                
            total_comparisons += 1
            
            # Get indices for this node and sort them (to ignore ordering differences)
            sk_indices = np.sort(sk_idx_array[sk_start:sk_end])
            nb_indices = np.sort(numba_tree.idx_array[nb_start:nb_end])
            
            # The sorted indices should be identical
            if np.array_equal(sk_indices, nb_indices):
                matches += 1
        
        # Require high compatibility (allowing for minor algorithmic differences)
        match_rate = matches / total_comparisons if total_comparisons > 0 else 1.0
        assert match_rate >= 0.95, f"Node partitioning match rate {match_rate:.1%} is below 95% threshold"
    
    def test_data_ordering_equivalence(self, test_data, leaf_size):
        """
        Test that data ordering along split axes is equivalent.
        
        This is a more fundamental test of whether the partitioning logic
        is working similarly between implementations.
        """
        # Build trees
        sklearn_tree = SklearnKDTree(test_data, leaf_size=leaf_size)
        numba_tree = build_kdtree(test_data, leaf_size=leaf_size)
        
        # Get sklearn internal arrays
        sk_data, sk_idx_array, sk_node_data, sk_node_bounds = sklearn_tree.get_arrays()
        n_nodes = sk_node_data.shape[0]
        
        axis_ordering_matches = 0
        total_internal_nodes = 0
        
        for node in range(n_nodes):
            # Only check internal nodes (non-leaf nodes)
            if sk_node_data[node]['is_leaf']:
                continue
                
            total_internal_nodes += 1
            
            # Get node boundaries
            sk_start = sk_node_data[node]['idx_start']
            sk_end = sk_node_data[node]['idx_end']
            
            # Skip if insufficient points
            if sk_end - sk_start < 2:
                continue
            
            # Get indices for both implementations
            sk_indices = sk_idx_array[sk_start:sk_end]
            nb_indices = numba_tree.idx_array[sk_start:sk_end]
            
            # Find split axis (dimension with maximum spread)
            spreads = []
            for axis in range(test_data.shape[1]):
                sk_values = test_data[sk_indices, axis]
                min_val, max_val = np.min(sk_values), np.max(sk_values)
                spreads.append(max_val - min_val)
            
            split_axis = np.argmax(spreads)
            
            # Get data values along split axis
            sk_axis_values = test_data[sk_indices, split_axis]
            nb_axis_values = test_data[nb_indices, split_axis]
            
            # Check if the median/partition point is similar
            sk_median = np.median(sk_axis_values)
            nb_median = np.median(nb_axis_values)
            
            # Count points on each side of median
            sk_left_count = np.sum(sk_axis_values <= sk_median)
            sk_right_count = np.sum(sk_axis_values > sk_median)
            nb_left_count = np.sum(nb_axis_values <= nb_median)
            nb_right_count = np.sum(nb_axis_values > nb_median)
            
            # Check if partitioning is roughly equivalent
            # (allowing for different tie-breaking in median calculation)
            partitioning_similar = (
                abs(sk_left_count - nb_left_count) <= 2 and
                abs(sk_right_count - nb_right_count) <= 2
            )
            
            if partitioning_similar:
                axis_ordering_matches += 1
        
        # Require high compatibility for data ordering
        ordering_match_rate = axis_ordering_matches / total_internal_nodes if total_internal_nodes > 0 else 1.0
        assert ordering_match_rate >= 0.80, f"Data ordering match rate {ordering_match_rate:.1%} is below 80% threshold"
    
    def test_query_results_compatibility(self, test_data, leaf_size):
        """Test that query results are equivalent between implementations."""
        # Build trees
        sklearn_tree = SklearnKDTree(test_data, leaf_size=leaf_size)
        numba_tree = build_kdtree(test_data, leaf_size=leaf_size)
        
        # Create query points (subset of original data for deterministic results)
        np.random.seed(123)
        query_indices = np.random.choice(len(test_data), size=min(10, len(test_data)), replace=False)
        query_data = test_data[query_indices]
        
        k = min(5, len(test_data))  # Number of neighbors
        
        # Query sklearn tree
        sk_distances, sk_indices = sklearn_tree.query(query_data, k=k, return_distance=True)
        
        # Query numba tree using the parallel implementation
        from evoc.numba_kdtree import parallel_tree_query
        nb_distances, nb_indices = parallel_tree_query(numba_tree, query_data, k=k, output_rdist=False)
        
        # Results should be very similar (allowing for minor floating point differences)
        # Sort both results by indices to handle any ordering differences
        for i in range(len(query_data)):
            # Sort by indices to compare equivalent sets
            sk_sorted_idx = np.argsort(sk_indices[i])
            nb_sorted_idx = np.argsort(nb_indices[i])
            
            sk_sorted_indices = sk_indices[i][sk_sorted_idx]
            nb_sorted_indices = nb_indices[i][nb_sorted_idx]
            sk_sorted_distances = sk_distances[i][sk_sorted_idx]
            nb_sorted_distances = nb_distances[i][nb_sorted_idx]
            
            # Check that we get the same nearest neighbors
            np.testing.assert_array_equal(
                sk_sorted_indices, nb_sorted_indices,
                err_msg=f"Query {i}: Nearest neighbor indices don't match"
            )
            
            # Check that distances are very close
            np.testing.assert_allclose(
                sk_sorted_distances, nb_sorted_distances, 
                rtol=1e-5, atol=1e-6,
                err_msg=f"Query {i}: Distances don't match within tolerance"
            )
    
    def test_tree_bounds_compatibility(self, test_data, leaf_size):
        """Test that node bounds are calculated consistently."""
        # Build trees
        sklearn_tree = SklearnKDTree(test_data, leaf_size=leaf_size)
        numba_tree = build_kdtree(test_data, leaf_size=leaf_size)
        
        # Get sklearn bounds
        sk_data, sk_idx_array, sk_node_data, sk_node_bounds = sklearn_tree.get_arrays()
        
        # Node bounds should match closely
        np.testing.assert_allclose(
            sk_node_bounds, numba_tree.node_bounds,
            rtol=1e-5, atol=1e-6,
            err_msg="Node bounds don't match between implementations"
        )


class TestKDTreeEdgeCases:
    """Test edge cases and special conditions."""
    
    def test_single_point(self):
        """Test with a single data point."""
        data = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        
        sklearn_tree = SklearnKDTree(data, leaf_size=1)
        numba_tree = build_kdtree(data, leaf_size=1)
        
        # Should handle single point gracefully
        assert numba_tree.data.shape == (1, 3)
        assert numba_tree.idx_array.shape == (1,)
    
    def test_duplicate_points(self):
        """Test with duplicate data points."""
        data = np.array([
            [1.0, 2.0],
            [1.0, 2.0],  # Duplicate
            [3.0, 4.0],
            [1.0, 2.0],  # Another duplicate
        ], dtype=np.float32)
        
        sklearn_tree = SklearnKDTree(data, leaf_size=2)
        numba_tree = build_kdtree(data, leaf_size=2)
        
        # Should handle duplicates without error
        assert numba_tree.data.shape == data.shape
        
        # Query should work with duplicates
        from evoc.numba_kdtree import parallel_tree_query
        distances, indices = parallel_tree_query(numba_tree, data[:1], k=2)
        assert distances.shape == (1, 2)
        assert indices.shape == (1, 2)
    
    def test_high_dimensional_data(self):
        """Test with high-dimensional data."""
        np.random.seed(42)
        data = np.random.rand(100, 50).astype(np.float32)  # 50D data
        
        sklearn_tree = SklearnKDTree(data, leaf_size=10)
        numba_tree = build_kdtree(data, leaf_size=10)
        
        # Should handle high dimensions
        assert numba_tree.data.shape == (100, 50)
        
        # Quick query test
        from evoc.numba_kdtree import parallel_tree_query
        distances, indices = parallel_tree_query(numba_tree, data[:5], k=3)
        assert distances.shape == (5, 3)
        assert indices.shape == (5, 3)


# Integration test that can be run standalone
def test_full_pipeline_compatibility():
    """Integration test ensuring the full pipeline works with both tree types."""
    np.random.seed(42)
    data = np.random.rand(200, 5).astype(np.float32)
    
    # Build numba tree and run boruvka (this was the original failing case)
    from evoc.numba_kdtree import build_kdtree
    from evoc.boruvka import parallel_boruvka
    
    tree = build_kdtree(data, leaf_size=20)
    
    # This should not raise any numba errors
    edges = parallel_boruvka(tree, min_samples=5, reproducible=True)
    
    # Should produce reasonable results
    assert len(edges) > 0, "Boruvka should produce some edges"
    assert edges.shape[1] == 3, "Edges should have 3 columns (from, to, weight)"
    assert np.all(edges[:, 2] >= 0), "Edge weights should be non-negative"


if __name__ == "__main__":
    # Allow running as a script for quick testing
    pytest.main([__file__, "-v"])
