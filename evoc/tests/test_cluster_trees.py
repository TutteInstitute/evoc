import numpy as np
from evoc.cluster_trees import (
    create_linkage_merge_data,
    eliminate_branch,
    linkage_merge_find,
    linkage_merge_join,
    mst_to_linkage_tree,
)


def test_create_linkage_merge_data():
    # Test case 1: Base size = 1
    base_size = 1
    expected_parent = np.array([-1], dtype=np.intp)
    expected_size = np.array([1], dtype=np.intp)
    expected_next_parent = np.array([1], dtype=np.intp)
    result = create_linkage_merge_data(base_size)
    assert np.array_equal(result.parent, expected_parent)
    assert np.array_equal(result.size, expected_size)
    assert np.array_equal(result.next_parent, expected_next_parent)

    # Test case 2: Base size = 3
    base_size = 3
    expected_parent = np.array([-1, -1, -1, -1, -1], dtype=np.intp)
    expected_size = np.array([1, 1, 1, 0, 0], dtype=np.intp)
    expected_next_parent = np.array([3], dtype=np.intp)
    result = create_linkage_merge_data(base_size)
    assert np.array_equal(result.parent, expected_parent)
    assert np.array_equal(result.size, expected_size)
    assert np.array_equal(result.next_parent, expected_next_parent)

    # Test case 3: Base size = 5
    base_size = 5
    expected_parent = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=np.intp)
    expected_size = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0], dtype=np.intp)
    expected_next_parent = np.array([5], dtype=np.intp)
    result = create_linkage_merge_data(base_size)
    assert np.array_equal(result.parent, expected_parent)
    assert np.array_equal(result.size, expected_size)
    assert np.array_equal(result.next_parent, expected_next_parent)

    print("All test cases passed!")


def test_linkage_merge_find():
    # Test case 1: Single node
    linkage_merge = create_linkage_merge_data(1)
    node = 0
    expected_result = 0
    result = linkage_merge_find(linkage_merge, node)
    assert result == expected_result

    # Test case 2: Multiple nodes
    linkage_merge = create_linkage_merge_data(5)
    node = 3
    expected_result = 3
    result = linkage_merge_find(linkage_merge, node)
    assert result == expected_result

    # Test case 3: Node with parent
    linkage_merge = create_linkage_merge_data(5)
    linkage_merge.parent[3] = 2
    node = 3
    expected_result = 2
    result = linkage_merge_find(linkage_merge, node)
    assert result == expected_result

    print("All test cases passed!")


def test_linkage_merge_join():
    # Test case 1
    linkage_merge = create_linkage_merge_data(3)
    left = 0
    right = 1
    expected_size = np.array([2, 1, 1, 0, 0], dtype=np.intp)
    expected_parent = np.array([3, -1, -1, -1, -1], dtype=np.intp)
    expected_next_parent = np.array([4], dtype=np.intp)
    linkage_merge_join(linkage_merge, left, right)
    assert np.array_equal(linkage_merge.size, expected_size)
    assert np.array_equal(linkage_merge.parent, expected_parent)
    assert np.array_equal(linkage_merge.next_parent, expected_next_parent)

    # Test case 2
    linkage_merge = create_linkage_merge_data(5)
    left = 2
    right = 4
    expected_size = np.array([1, 1, 2, 1, 2, 0, 0, 0, 0], dtype=np.intp)
    expected_parent = np.array([-1, -1, 5, -1, 5, -1, -1, -1, -1], dtype=np.intp)
    expected_next_parent = np.array([6], dtype=np.intp)
    linkage_merge_join(linkage_merge, left, right)
    assert np.array_equal(linkage_merge.size, expected_size)
    assert np.array_equal(linkage_merge.parent, expected_parent)
    assert np.array_equal(linkage_merge.next_parent, expected_next_parent)

    # Test case 3
    linkage_merge = create_linkage_merge_data(7)
    left = 3
    right = 6
    expected_size = np.array(
        [1, 1, 1, 2, 1, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0], dtype=np.intp
    )
    expected_parent = np.array(
        [-1, -1, -1, 7, -1, -1, 7, -1, 7, -1, -1, -1, -1, -1, -1, -1], dtype=np.intp
    )
    expected_next_parent = np.array([8], dtype=np.intp)
    linkage_merge_join(linkage_merge, left, right)
    assert np.array_equal(linkage_merge.size, expected_size)
    assert np.array_equal(linkage_merge.parent, expected_parent)
    assert np.array_equal(linkage_merge.next_parent, expected_next_parent)

    print("All test cases passed!")


def test_mst_to_linkage_tree():
    # Test case 1
    sorted_mst = np.array([[0, 1, 0.5], [1, 2, 0.7], [2, 3, 0.9]])
    expected_result = np.array([[0, 1, 0.5, 2], [1, 2, 0.7, 3], [2, 3, 0.9, 4]])
    result = mst_to_linkage_tree(sorted_mst)
    assert np.array_equal(result, expected_result)

    # Test case 2
    sorted_mst = np.array([[0, 1, 0.2], [1, 3, 0.4], [2, 4, 0.6]])
    expected_result = np.array([[0, 1, 0.2, 2], [1, 3, 0.4, 3], [2, 4, 0.6, 3]])
    result = mst_to_linkage_tree(sorted_mst)
    assert np.array_equal(result, expected_result)

    # Test case 3
    sorted_mst = np.array([[0, 2, 0.1], [1, 4, 0.3], [3, 5, 0.5]])
    expected_result = np.array([[0, 2, 0.1, 2], [1, 4, 0.3, 2], [3, 5, 0.5, 2]])
    result = mst_to_linkage_tree(sorted_mst)
    assert np.array_equal(result, expected_result)

    print("All test cases passed!")


import numpy as np
from evoc.cluster_trees import create_linkage_merge_data, bfs_from_hierarchy


def test_bfs_from_hierarchy():
    # Test case 1: Single node hierarchy
    hierarchy = np.array([[0, 0]], dtype=np.intp)
    bfs_root = 0
    num_points = 1
    expected_result = [0]
    result = bfs_from_hierarchy(hierarchy, bfs_root, num_points)
    assert result == expected_result

    # Test case 2: Hierarchy with multiple nodes
    hierarchy = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.intp)
    bfs_root = 0
    num_points = 7
    expected_result = [0, 1, 2, 3, 4, 5, 6]
    result = bfs_from_hierarchy(hierarchy, bfs_root, num_points)
    assert result == expected_result

    # Test case 3: Hierarchy with multiple levels
    hierarchy = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.intp)
    bfs_root = 0
    num_points = 9
    expected_result = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    result = bfs_from_hierarchy(hierarchy, bfs_root, num_points)
    assert result == expected_result

    print("All test cases passed!")


import numpy as np


def test_eliminate_branch():
    # Test case 1: branch_node < num_points
    branch_node = 3
    parent_node = 2
    lambda_value = 0.5
    parents = np.zeros(10, dtype=np.intp)
    children = np.zeros(10, dtype=np.intp)
    lambdas = np.zeros(10)
    sizes = np.zeros(10, dtype=np.intp)
    idx = 0
    ignore = np.zeros(10, dtype=bool)
    hierarchy = np.zeros((10, 10), dtype=bool)
    num_points = 5

    expected_parents = np.array([2, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.intp)
    expected_children = np.array([0, 0, 0, 3, 0, 0, 0, 0, 0, 0], dtype=np.intp)
    expected_lambdas = np.array([0, 0, 0, 0.5, 0, 0, 0, 0, 0, 0])
    expected_idx = 1

    result_idx = eliminate_branch(
        branch_node,
        parent_node,
        lambda_value,
        parents,
        children,
        lambdas,
        sizes,
        idx,
        ignore,
        hierarchy,
        num_points,
    )
    assert np.array_equal(parents, expected_parents)
    assert np.array_equal(children, expected_children)
    assert np.array_equal(lambdas, expected_lambdas)
    assert result_idx == expected_idx

    # Test case 2: branch_node >= num_points
    branch_node = 6
    parent_node = 4
    lambda_value = 0.8
    parents = np.zeros(10, dtype=np.intp)
    children = np.zeros(10, dtype=np.intp)
    lambdas = np.zeros(10)
    sizes = np.zeros(10, dtype=np.intp)
    idx = 0
    ignore = np.zeros(10, dtype=bool)
    hierarchy = np.zeros((10, 10), dtype=bool)
    num_points = 5

    expected_parents = np.array([0, 0, 0, 0, 4, 4, 4, 4, 4, 4], dtype=np.intp)
    expected_children = np.array([0, 0, 0, 0, 6, 0, 0, 0, 0, 0], dtype=np.intp)
    expected_lambdas = np.array([0, 0, 0, 0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
    expected_idx = 5

    result_idx = eliminate_branch(
        branch_node,
        parent_node,
        lambda_value,
        parents,
        children,
        lambdas,
        sizes,
        idx,
        ignore,
        hierarchy,
        num_points,
    )
    assert np.array_equal(parents, expected_parents)
    assert np.array_equal(children, expected_children)
    assert np.array_equal(lambdas, expected_lambdas)
    assert result_idx == expected_idx

    print("All test cases passed!")


test_eliminate_branch()
