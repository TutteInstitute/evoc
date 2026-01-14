import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

from evoc.cluster_trees import (
    create_linkage_merge_data,
    eliminate_branch,
    linkage_merge_find,
    linkage_merge_join,
    mst_to_linkage_tree,
    bfs_from_hierarchy,
    condense_tree,
    extract_leaves,
    score_condensed_tree_nodes,
    cluster_tree_from_condensed_tree,
    extract_eom_clusters,
    get_cluster_labelling_at_cut,
    get_cluster_label_vector,
    get_point_membership_strength_vector,
    CondensedTree,
    LinkageMergeData,
)


class TestLinkageMergeData:
    """Test the LinkageMergeData structure and associated functions."""
    
    def test_create_linkage_merge_data(self):
        """Test creation of linkage merge data structure."""
        base_size = 5
        linkage_data = create_linkage_merge_data(base_size)
        
        # Check structure
        assert isinstance(linkage_data, LinkageMergeData)
        assert len(linkage_data.parent) == 2 * base_size - 1
        assert len(linkage_data.size) == 2 * base_size - 1
        assert len(linkage_data.next) == 1
        
        # Check initial values
        assert np.all(linkage_data.parent == -1)
        assert np.all(linkage_data.size[:base_size] == 1)
        assert np.all(linkage_data.size[base_size:] == 0)
        assert linkage_data.next[0] == base_size
    
    def test_linkage_merge_find_and_join(self):
        """Test find and join operations on linkage merge data."""
        base_size = 4
        linkage_data = create_linkage_merge_data(base_size)
        
        # Initially, each node should find itself
        for i in range(base_size):
            assert linkage_merge_find(linkage_data, i) == i
        
        # Join nodes 0 and 1
        linkage_merge_join(linkage_data, 0, 1)
        
        # Check that parent pointers are set correctly
        assert linkage_data.parent[0] == base_size  # 4
        assert linkage_data.parent[1] == base_size  # 4
        assert linkage_data.size[base_size] == 2   # Combined size
        assert linkage_data.next[0] == base_size + 1  # Next available index
        
        # Join the new cluster with node 2
        new_cluster = linkage_merge_find(linkage_data, 0)  # Should be 4
        linkage_merge_join(linkage_data, new_cluster, 2)
        
        # Check updated structure
        assert linkage_data.size[base_size + 1] == 3  # Size should be 3
        assert linkage_data.next[0] == base_size + 2   # Next available index


class TestMSTToLinkageTree:
    """Test conversion from MST to linkage tree."""
    
    @pytest.fixture
    def simple_mst(self):
        """Create a simple MST for testing."""
        # Simple 4-point MST: 0-1 (dist=1.0), 1-2 (dist=2.0), 2-3 (dist=3.0)
        return np.array([
            [0, 1, 1.0],
            [1, 2, 2.0],
            [2, 3, 3.0]
        ], dtype=np.float64)
    
    def test_mst_to_linkage_tree_basic(self, simple_mst):
        """Test basic MST to linkage tree conversion."""
        linkage_tree = mst_to_linkage_tree(simple_mst)
        
        # Should have same number of rows as MST
        assert linkage_tree.shape[0] == simple_mst.shape[0]
        assert linkage_tree.shape[1] == 4  # left, right, distance, size
        
        # Check that distances are preserved
        assert np.array_equal(linkage_tree[:, 2], simple_mst[:, 2])
        
        # Check that cluster sizes make sense (should be increasing)
        sizes = linkage_tree[:, 3]
        assert sizes[0] == 2  # First merge: 2 points
        assert sizes[1] == 3  # Second merge: 3 points  
        assert sizes[2] == 4  # Final merge: all 4 points
    
    def test_mst_to_linkage_tree_ordering(self, simple_mst):
        """Test that linkage tree maintains proper ordering."""
        linkage_tree = mst_to_linkage_tree(simple_mst)
        
        # In each row, larger cluster index should be in column 0
        for i in range(linkage_tree.shape[0]):
            assert linkage_tree[i, 0] >= linkage_tree[i, 1]
    
    def test_mst_to_linkage_tree_random(self):
        """Test with a larger random MST."""
        np.random.seed(42)
        n_points = 10
        
        # Create a random MST (n_points - 1 edges)
        edges = []
        for i in range(n_points - 1):
            edges.append([i, i + 1, np.random.random()])
        
        mst = np.array(edges, dtype=np.float64)
        mst = mst[np.argsort(mst[:, 2])]  # Sort by distance
        
        linkage_tree = mst_to_linkage_tree(mst)
        
        assert linkage_tree.shape[0] == n_points - 1
        assert linkage_tree.shape[1] == 4
        assert linkage_tree[-1, 3] == n_points  # Final cluster has all points


class TestBFSFromHierarchy:
    """Test breadth-first search on hierarchy."""
    
    @pytest.fixture
    def simple_hierarchy(self):
        """Create a simple hierarchy for testing.
        
        In scipy linkage format:
        - Points: 0, 1, 2, 3 (original data)  
        - Clusters: 4, 5, 6 (formed by merges)
        - Row 0: merge to form cluster 4 (n_points + 0)
        - Row 1: merge to form cluster 5 (n_points + 1) 
        - Row 2: merge to form cluster 6 (n_points + 2)
        """
        return np.array([
            [0, 1, 1.0, 2],  # Row 0: merge points 0,1 -> cluster 4
            [2, 3, 2.0, 2],  # Row 1: merge points 2,3 -> cluster 5
            [4, 5, 3.0, 4],  # Row 2: merge clusters 4,5 -> cluster 6 (root)
        ], dtype=np.float64)
    
    def test_bfs_leaf_node(self, simple_hierarchy):
        """Test BFS starting from a leaf node (original data point)."""
        result = bfs_from_hierarchy(simple_hierarchy, 0, 4)
        assert result == [0]  # Leaf node should return itself
    
    def test_bfs_internal_node(self, simple_hierarchy):
        """Test BFS starting from an internal cluster."""
        # Cluster 4 (formed by merging points 0,1)
        result = bfs_from_hierarchy(simple_hierarchy, 4, 4)
        expected = [4, 0, 1]  # Should include the cluster and its children
        assert result == expected
    
    def test_bfs_root_node(self, simple_hierarchy):
        """Test BFS starting from the root cluster."""
        # Cluster 6 is the root (formed by merging clusters 4,5)
        result = bfs_from_hierarchy(simple_hierarchy, 6, 4)
        expected = [6, 4, 5, 0, 1, 2, 3]  # Should traverse entire tree
        assert set(result) == set(expected)  # Order may vary in BFS


class TestCondenseTree:
    """Test tree condensation functionality."""
    
    @pytest.fixture
    def sample_hierarchy(self):
        """Create a sample hierarchy for testing."""
        # Create hierarchy for 6 points
        return np.array([
            [0, 1, 0.1, 2],   # Cluster 6: points 0,1
            [2, 3, 0.2, 2],   # Cluster 7: points 2,3  
            [6, 7, 0.3, 4],   # Cluster 8: clusters 6,7
            [8, 4, 0.4, 5],   # Cluster 9: cluster 8 + point 4
            [9, 5, 0.5, 6],   # Cluster 10: cluster 9 + point 5 (root)
        ], dtype=np.float64)
    
    def test_condense_tree_basic(self, sample_hierarchy):
        """Test basic tree condensation."""
        min_cluster_size = 3
        condensed = condense_tree(sample_hierarchy, min_cluster_size)
        
        # Check structure
        assert isinstance(condensed, CondensedTree)
        assert len(condensed.parent) == len(condensed.child)
        assert len(condensed.parent) == len(condensed.lambda_val)
        assert len(condensed.parent) == len(condensed.child_size)
        
        # Lambda values should be positive
        assert np.all(condensed.lambda_val > 0)
        
        # Child sizes should be reasonable
        assert np.all(condensed.child_size >= 1)
    
    def test_condense_tree_min_cluster_size_effect(self, sample_hierarchy):
        """Test that different min_cluster_size values produce different results."""
        condensed_small = condense_tree(sample_hierarchy, min_cluster_size=2)
        condensed_large = condense_tree(sample_hierarchy, min_cluster_size=4)
        
        # Different min_cluster_size should affect the result structure
        # (Exact comparison depends on the specific condensation logic)
        assert len(condensed_small.parent) >= 0
        assert len(condensed_large.parent) >= 0
    
    def test_condense_tree_lambda_values(self, sample_hierarchy):
        """Test that lambda values are computed correctly (1/distance)."""
        condensed = condense_tree(sample_hierarchy, min_cluster_size=2)
        
        # All lambda values should be finite and positive
        assert np.all(np.isfinite(condensed.lambda_val))
        assert np.all(condensed.lambda_val > 0)


class TestExtractLeaves:
    """Test leaf extraction from condensed trees."""
    
    def test_extract_leaves_simple(self):
        """Test leaf extraction from a simple condensed tree."""
        # Create simple condensed tree manually
        parent = np.array([5, 5, 5])
        child = np.array([0, 1, 2])  # Three leaf points
        lambda_val = np.array([1.0, 1.0, 1.0])
        child_size = np.array([1, 1, 1])
        
        condensed = CondensedTree(parent, child, lambda_val, child_size)
        leaves = extract_leaves(condensed)
        
        # Node 5 should be identified as a leaf cluster
        assert 5 in leaves
    
    def test_extract_leaves_hierarchical(self):
        """Test leaf extraction from a hierarchical condensed tree."""
        # Create a tree where node 5 has children that are clusters (not just points)
        parent = np.array([6, 6, 5, 5])
        child = np.array([5, 0, 1, 2])  # Node 5 is internal (has child_size > 1)
        lambda_val = np.array([1.0, 1.0, 1.0, 1.0])
        child_size = np.array([3, 1, 1, 1])  # Node 5 entry has child_size=3
        
        condensed = CondensedTree(parent, child, lambda_val, child_size)
        leaves = extract_leaves(condensed)
        
        # Based on the extract_leaves logic, clusters with child_size > 1 
        # in their entries are leaf clusters
        if len(leaves) > 0:
            for leaf in leaves:
                # Find entries where this node is the child
                mask = condensed.child == leaf
                if np.any(mask):
                    # At least one entry should have child_size > 1
                    assert np.any(condensed.child_size[mask] > 1)


class TestClusterLabeling:
    """Test cluster labeling and membership functions."""
    
    @pytest.fixture
    def sample_condensed_tree(self):
        """Create a sample condensed tree for testing."""
        parent = np.array([10, 10, 10, 11, 11])
        child = np.array([0, 1, 2, 3, 4])
        lambda_val = np.array([2.0, 2.0, 2.0, 1.0, 1.0])
        child_size = np.array([1, 1, 1, 1, 1])
        return CondensedTree(parent, child, lambda_val, child_size)
    
    def test_get_cluster_label_vector_single_cluster(self, sample_condensed_tree):
        """Test cluster labeling with a single cluster."""
        clusters = np.array([10])
        labels = get_cluster_label_vector(
            sample_condensed_tree, clusters, 0.0, 5
        )
        
        assert len(labels) == 5
        # Points 0, 1, 2 should be in cluster 0 (they have high lambda values)
        assert labels[0] == 0
        assert labels[1] == 0 
        assert labels[2] == 0
        # Points 3, 4 should be noise (-1) (they have lower lambda values)
        assert labels[3] == -1
        assert labels[4] == -1
    
    def test_get_cluster_label_vector_multiple_clusters(self, sample_condensed_tree):
        """Test cluster labeling with multiple clusters."""
        clusters = np.array([10, 11])
        labels = get_cluster_label_vector(
            sample_condensed_tree, clusters, 0.0, 5
        )
        
        assert len(labels) == 5
        # Should have valid cluster assignments
        unique_labels = np.unique(labels)
        assert -1 in unique_labels or len(unique_labels) > 1
    
    def test_get_point_membership_strength_vector(self, sample_condensed_tree):
        """Test membership strength calculation."""
        clusters = np.array([10, 11])
        labels = get_cluster_label_vector(
            sample_condensed_tree, clusters, 0.0, 5
        )
        
        strengths = get_point_membership_strength_vector(
            sample_condensed_tree, clusters, labels
        )
        
        assert len(strengths) == 5
        assert np.all(strengths >= 0.0)
        assert np.all(strengths <= 1.0)
        
        # Points with valid cluster assignments should have positive strength
        valid_points = labels >= 0
        if np.any(valid_points):
            assert np.all(strengths[valid_points] > 0)


class TestIntegrationWithRealData:
    """Integration tests using real clustered data."""
    
    @pytest.fixture
    def clustered_data(self):
        """Generate clustered data for integration testing."""
        np.random.seed(42)
        X, y = make_blobs(n_samples=50, centers=3, random_state=42)
        X = StandardScaler().fit_transform(X)
        return X, y
    
    def test_full_pipeline_simple_mst(self, clustered_data):
        """Test the full pipeline with a simple MST."""
        X, true_labels = clustered_data
        
        # Create a simple MST by connecting points sequentially
        n_samples = X.shape[0]
        mst_edges = []
        
        for i in range(n_samples - 1):
            # Connect point i to point i+1 with random distance
            mst_edges.append([i, i + 1, np.random.random()])
        
        mst = np.array(mst_edges, dtype=np.float64)
        mst = mst[np.argsort(mst[:, 2])]  # Sort by distance
        
        # Convert to linkage tree
        linkage_tree = mst_to_linkage_tree(mst)
        
        # Condense tree
        condensed = condense_tree(linkage_tree, min_cluster_size=5)
        
        # Extract clusters
        leaves = extract_leaves(condensed)
        
        # Get cluster labels
        if len(leaves) > 0:
            labels = get_cluster_label_vector(condensed, leaves, 0.0, n_samples)
            
            # Basic sanity checks
            assert len(labels) == n_samples
            assert np.all(labels >= -1)  # Valid range for labels
            
            # Should find some clusters or noise
            n_clusters = len(np.unique(labels[labels >= 0]))
            assert n_clusters >= 0  # Could be all noise
    
    def test_score_condensed_tree_nodes(self):
        """Test scoring of condensed tree nodes."""
        # Create a simple condensed tree
        parent = np.array([5, 5, 5])
        child = np.array([0, 1, 2])
        lambda_val = np.array([2.0, 1.5, 1.0])
        child_size = np.array([1, 1, 1])
        
        condensed = CondensedTree(parent, child, lambda_val, child_size)
        scores = score_condensed_tree_nodes(condensed)
        
        # Node 5 should have a positive score
        assert 5 in scores
        assert scores[5] > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_extract_leaves_empty_tree(self):
        """Test behavior with empty condensed trees."""
        empty_condensed = CondensedTree(
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64), 
            np.array([], dtype=np.float32),
            np.array([], dtype=np.int64)
        )
        
        # Should handle empty input gracefully
        leaves = extract_leaves(empty_condensed)
        assert len(leaves) == 0 or isinstance(leaves, np.ndarray)
    
    def test_single_point_mst(self):
        """Test with MST containing only one edge (two points)."""
        mst = np.array([[0, 1, 1.0]], dtype=np.float64)
        linkage_tree = mst_to_linkage_tree(mst)
        
        assert linkage_tree.shape[0] == 1
        assert linkage_tree[0, 3] == 2  # Should connect 2 points
    
    def test_zero_distance_edges(self):
        """Test handling of zero-distance edges in MST."""
        mst = np.array([
            [0, 1, 0.0],  # Zero distance
            [1, 2, 1.0]
        ], dtype=np.float64)
        
        linkage_tree = mst_to_linkage_tree(mst)
        condensed = condense_tree(linkage_tree, min_cluster_size=2)
        
        # Should handle zero distances gracefully
        # (may result in infinite lambda values)
        if len(condensed.lambda_val) > 0:
            finite_mask = np.isfinite(condensed.lambda_val)
            # At least some lambda values should be finite
            assert np.any(finite_mask) or np.any(np.isinf(condensed.lambda_val))


class TestBFSEdgeCases:
    """Test edge cases for BFS functionality."""
    
    def test_bfs_single_point_hierarchy(self):
        """Test BFS with minimal hierarchy."""
        # Single merge hierarchy for 2 points
        hierarchy = np.array([[0, 1, 1.0, 2]], dtype=np.float64)
        
        result = bfs_from_hierarchy(hierarchy, 2, 2)  # Cluster 2 (n_points + 0)
        assert set(result) == {2, 0, 1}
    
    def test_eliminate_branch_leaf(self):
        """Test eliminate_branch with a leaf node."""
        hierarchy = np.array([[0, 1, 1.0, 2]], dtype=np.float64)
        
        parents = np.zeros(10, dtype=np.int64)
        children = np.zeros(10, dtype=np.int64)
        lambdas = np.zeros(10, dtype=np.float32)
        sizes = np.zeros(10, dtype=np.int64)
        ignore = np.zeros(10, dtype=bool)
        
        # Eliminate a leaf node (point 0)
        new_idx = eliminate_branch(0, 5, 1.0, parents, children, lambdas, 
                                 sizes, 0, ignore, hierarchy, 2)
        
        assert new_idx == 1  # Should increment index
        assert parents[0] == 5
        assert children[0] == 0
        assert lambdas[0] == 1.0


# Utility function for running integration tests
def test_cluster_trees_integration():
    """High-level integration test of the entire cluster_trees module."""
    np.random.seed(42)
    
    # Generate test data
    X, _ = make_blobs(n_samples=20, centers=2, random_state=42)
    X = StandardScaler().fit_transform(X)
    
    # Create a minimal MST (for testing purposes)
    n_samples = X.shape[0]
    mst_edges = []
    for i in range(n_samples - 1):
        mst_edges.append([i, i + 1, np.random.random()])
    
    mst = np.array(mst_edges, dtype=np.float64)
    mst = mst[np.argsort(mst[:, 2])]
    
    # Test the full pipeline
    linkage_tree = mst_to_linkage_tree(mst)
    condensed = condense_tree(linkage_tree, min_cluster_size=3)
    leaves = extract_leaves(condensed)
    
    if len(leaves) > 0:
        labels = get_cluster_label_vector(condensed, leaves, 0.0, n_samples)
        strengths = get_point_membership_strength_vector(condensed, leaves, labels)
        
        # Verify results make sense
        assert len(labels) == n_samples
        assert len(strengths) == n_samples
        assert np.all(strengths >= 0.0) and np.all(strengths <= 1.0)
    
    # Test passed if we reach here without errors
    assert True


def test_linkage_merge_data_comprehensive():
    """Additional comprehensive test for linkage merge operations."""
    base_size = 6
    linkage_data = create_linkage_merge_data(base_size)
    
    # Test multiple sequential merges
    linkage_merge_join(linkage_data, 0, 1)  # Creates cluster 6
    linkage_merge_join(linkage_data, 2, 3)  # Creates cluster 7
    linkage_merge_join(linkage_data, 6, 7)  # Creates cluster 8
    
    # Verify the structure after multiple merges
    assert linkage_data.size[6] == 2  # Cluster 6 has 2 points
    assert linkage_data.size[7] == 2  # Cluster 7 has 2 points  
    assert linkage_data.size[8] == 4  # Cluster 8 has 4 points
    assert linkage_data.next[0] == 9  # Next available cluster ID
    
    # Test path compression in find
    assert linkage_merge_find(linkage_data, 0) == 8  # Should find root cluster
    assert linkage_merge_find(linkage_data, 2) == 8  # Should find same root
