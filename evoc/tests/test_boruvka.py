import numpy as np
import pytest
from evoc.boruvka import (
    merge_components,
    update_component_vectors,
    boruvka_tree_query,
    parallel_boruvka,
    Tree,
    NodeInfo,
)


def test_merge_components():
    # Test case 1: Empty input
    disjoint_set = np.array([], dtype=np.int32)
    candidate_neighbors = np.array([], dtype=np.int64)
    candidate_neighbor_distances = np.array([], dtype=np.float32)
    point_components = np.array([], dtype=np.int64)
    expected_result = np.empty((0, 3), dtype=np.float64)
    assert np.array_equal(
        merge_components(
            disjoint_set,
            candidate_neighbors,
            candidate_neighbor_distances,
            point_components,
        ),
        expected_result,
    )

    # Test case 2: Single component with one edge
    disjoint_set = np.array([0, 0], dtype=np.int32)
    candidate_neighbors = np.array([1], dtype=np.int64)
    candidate_neighbor_distances = np.array([0.5], dtype=np.float32)
    point_components = np.array([0], dtype=np.int64)
    expected_result = np.array([[0.0, 1.0, 0.5]], dtype=np.float64)
    assert np.array_equal(
        merge_components(
            disjoint_set,
            candidate_neighbors,
            candidate_neighbor_distances,
            point_components,
        ),
        expected_result,
    )

    # Test case 3: Multiple components with multiple edges
    disjoint_set = np.array([0, 0, 2, 2, 4, 4], dtype=np.int32)
    candidate_neighbors = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    candidate_neighbor_distances = np.array([0.5, 0.3, 0.8, 0.2, 0.6], dtype=np.float32)
    point_components = np.array([0, 0, 2, 2, 4], dtype=np.int64)
    expected_result = np.array(
        [[0.0, 1.0, 0.5], [2.0, 3.0, 0.2], [4.0, 5.0, 0.6]], dtype=np.float64
    )
    assert np.array_equal(
        merge_components(
            disjoint_set,
            candidate_neighbors,
            candidate_neighbor_distances,
            point_components,
        ),
        expected_result,
    )

    # Test case 4: All components already merged
    disjoint_set = np.array([0, 0, 0, 0], dtype=np.int32)
    candidate_neighbors = np.array([1, 2, 3], dtype=np.int64)
    candidate_neighbor_distances = np.array([0.5, 0.3, 0.8], dtype=np.float32)
    point_components = np.array([0, 0, 0], dtype=np.int64)
    expected_result = np.empty((0, 3), dtype=np.float64)
    assert np.array_equal(
        merge_components(
            disjoint_set,
            candidate_neighbors,
            candidate_neighbor_distances,
            point_components,
        ),
        expected_result,
    )


def test_update_component_vectors():
    # Test case 1: Leaf node with same component
    tree = Tree()
    tree.node_data = np.array(
        [
            NodeInfo(is_leaf=True, idx_start=0, idx_end=3),
            NodeInfo(is_leaf=True, idx_start=3, idx_end=6),
            NodeInfo(is_leaf=True, idx_start=6, idx_end=9),
            NodeInfo(is_leaf=True, idx_start=9, idx_end=12),
        ]
    )
    disjoint_set = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int32)
    node_components = np.zeros(4, dtype=np.int32)
    point_components = np.zeros(12, dtype=np.int32)
    expected_result = np.array([0, 1, 2, 3], dtype=np.int32)
    update_component_vectors(tree, disjoint_set, node_components, point_components)
    assert np.array_equal(node_components, expected_result)

    # Test case 2: Leaf node with different components
    tree = Tree()
    tree.node_data = np.array(
        [
            NodeInfo(is_leaf=True, idx_start=0, idx_end=3),
            NodeInfo(is_leaf=True, idx_start=3, idx_end=6),
            NodeInfo(is_leaf=True, idx_start=6, idx_end=9),
            NodeInfo(is_leaf=True, idx_start=9, idx_end=12),
        ]
    )
    disjoint_set = np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3], dtype=np.int32)
    node_components = np.zeros(4, dtype=np.int32)
    point_components = np.zeros(12, dtype=np.int32)
    expected_result = np.array([0, 0, 0, 0], dtype=np.int32)
    update_component_vectors(tree, disjoint_set, node_components, point_components)
    assert np.array_equal(node_components, expected_result)

    # Test case 3: Non-leaf node with same component
    tree = Tree()
    tree.node_data = np.array(
        [
            NodeInfo(is_leaf=False, idx_start=0, idx_end=3),
            NodeInfo(is_leaf=False, idx_start=3, idx_end=6),
            NodeInfo(is_leaf=True, idx_start=6, idx_end=9),
            NodeInfo(is_leaf=True, idx_start=9, idx_end=12),
        ]
    )
    disjoint_set = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.int32)
    node_components = np.zeros(4, dtype=np.int32)
    point_components = np.zeros(12, dtype=np.int32)
    expected_result = np.array([0, 0, 2, 3], dtype=np.int32)
    update_component_vectors(tree, disjoint_set, node_components, point_components)
    assert np.array_equal(node_components, expected_result)

    # Test case 4: Non-leaf node with different components
    tree = Tree()
    tree.node_data = np.array(
        [
            NodeInfo(is_leaf=False, idx_start=0, idx_end=3),
            NodeInfo(is_leaf=False, idx_start=3, idx_end=6),
            NodeInfo(is_leaf=True, idx_start=6, idx_end=9),
            NodeInfo(is_leaf=True, idx_start=9, idx_end=12),
        ]
    )
    disjoint_set = np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3], dtype=np.int32)
    node_components = np.zeros(4, dtype=np.int32)
    point_components = np.zeros(12, dtype=np.int32)
    expected_result = np.array([0, 0, 0, 0], dtype=np.int32)
    update_component_vectors(tree, disjoint_set, node_components, point_components)
    assert np.array_equal(node_components, expected_result)

    print("All test cases passed!")


def test_boruvka_tree_query():
    # Test case 1: Empty tree
    tree = np.empty((0, 2), dtype=np.float32)
    node_components = np.array([], dtype=np.int32)
    point_components = np.array([], dtype=np.int32)
    core_distances = np.array([], dtype=np.float32)
    expected_candidate_distances = np.empty((0,), dtype=np.float32)
    expected_candidate_indices = np.empty((0,), dtype=np.int32)
    assert np.array_equal(
        boruvka_tree_query(tree, node_components, point_components, core_distances),
        (expected_candidate_distances, expected_candidate_indices),
    )

    # Test case 2: Single node tree
    tree = np.array([[0.0, 0.0]], dtype=np.float32)
    node_components = np.array([0], dtype=np.int32)
    point_components = np.array([0], dtype=np.int32)
    core_distances = np.array([0.5], dtype=np.float32)
    expected_candidate_distances = np.array([np.inf], dtype=np.float32)
    expected_candidate_indices = np.array([-1], dtype=np.int32)
    assert np.array_equal(
        boruvka_tree_query(tree, node_components, point_components, core_distances),
        (expected_candidate_distances, expected_candidate_indices),
    )

    # Test case 3: Multiple nodes tree
    tree = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    node_components = np.array([0, 1, 2], dtype=np.int32)
    point_components = np.array([0, 1, 2], dtype=np.int32)
    core_distances = np.array([0.5, 0.3, 0.8], dtype=np.float32)
    expected_candidate_distances = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
    expected_candidate_indices = np.array([-1, -1, -1], dtype=np.int32)
    assert np.array_equal(
        boruvka_tree_query(tree, node_components, point_components, core_distances),
        (expected_candidate_distances, expected_candidate_indices),
    )

    print("All test cases passed!")


def test_parallel_boruvka():
    # Test case 1: Minimum samples = 10
    tree = create_test_tree()
    min_samples = 10
    expected_result = create_expected_result()
    assert np.array_equal(parallel_boruvka(tree, min_samples), expected_result)

    # Test case 2: Minimum samples = 1
    tree = create_test_tree()
    min_samples = 1
    expected_result = create_expected_result()
    assert np.array_equal(parallel_boruvka(tree, min_samples), expected_result)

    # Test case 3: Empty tree
    tree = create_empty_tree()
    min_samples = 10
    expected_result = np.empty((0, 3), dtype=np.float64)
    assert np.array_equal(parallel_boruvka(tree, min_samples), expected_result)

    print("All test cases passed!")


def create_test_tree():
    # Create a test tree
    # Replace with your own implementation
    pass


def create_expected_result():
    # Create the expected result for the test cases
    # Replace with your own implementation
    pass


def create_empty_tree():
    # Create an empty tree
    # Replace with your own implementation
    pass
