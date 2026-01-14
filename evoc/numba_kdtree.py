import numba
import numpy as np

from collections import namedtuple

NumbaKDTree = namedtuple("KDTree", ["data", "idx_array", "idx_start", "idx_end", "radius", "is_leaf", "node_bounds"])
NodeData = namedtuple("NodeData", ["idx_start", "idx_end", "radius", "is_leaf"])

NodeDataType = numba.types.NamedTuple([
    numba.types.intp[::1],
    numba.types.intp[::1],
    numba.types.float32[::1],
    numba.types.bool_[::1],
], NodeData)
NumbaKDTreeType = numba.types.NamedTuple([
    numba.types.float32[:, ::1],
    numba.types.intp[::1],
    numba.types.intp[::1],
    numba.types.intp[::1],
    numba.types.float32[::1],
    numba.types.bool_[::1],
    numba.types.float32[:, :, ::1],
], NumbaKDTree)

def kdtree_to_numba(sklearn_kdtree):
    data, idx_array, node_data, node_bounds = sklearn_kdtree.get_arrays()
    return NumbaKDTree(data, idx_array, node_data.idx_start, node_data.idx_end, node_data.radius, node_data.is_leaf, node_bounds)

@numba.njit(
    cache=True,
    fastmath=True,
    locals={
        "n_features": numba.types.intp,
        "lower_bounds": numba.types.float32[::1],
        "upper_bounds": numba.types.float32[::1],
        "radius": numba.types.float32,
        "diff": numba.types.float32,
        "data_row": numba.types.float32[::1],
    }
)
def _init_node(data, node_bounds, idx_array, idx_start_array, idx_end_array, radius_array, is_leaf_array, node, idx_start, idx_end):

    n_features = data.shape[1]
    lower_bounds = node_bounds[0, node, :]
    upper_bounds = node_bounds[1, node, :]


    # determine Node bounds
    for j in range(n_features):
        lower_bounds[j] = np.inf
        upper_bounds[j] = -np.inf

    for i in range(idx_start, idx_end):
        data_row = data[idx_array[i]]
        for j in range(n_features):
            lower_bounds[j] = min(lower_bounds[j], data_row[j])
            upper_bounds[j] = max(upper_bounds[j], data_row[j])

    radius = 0.0
    for j in range(n_features):
        diff = abs(upper_bounds[j] - lower_bounds[j]) * 0.5
        radius += diff * diff

    idx_start_array[node] = idx_start
    idx_end_array[node] = idx_end

    radius_array[node] = np.sqrt(radius)


@numba.njit(
    "intp(float32[:,::1], intp[::1], intp, intp)",
    cache=True,
    locals={
        "n_features": numba.types.intp,
        "result": numba.types.intp,
        "max_spread": numba.types.float32,
        "j": numba.types.intp,
        "i": numba.types.intp,
        "max_val": numba.types.float32,
        "min_val": numba.types.float32,
        "val": numba.types.float32,
        "spread": numba.types.float32,
    }
)
def _find_node_split_dim(data, idx_array, idx_start, idx_end):
    n_features = data.shape[1]
    result = 0
    max_spread = 0

    for j in range(n_features):
        max_val = data[idx_array[idx_start], j]
        min_val = max_val
        for i in range(idx_start + 1, idx_end):
            val = data[idx_array[i], j]
            max_val = max(max_val, val)
            min_val = min(min_val, val)

        spread = max_val - min_val

        if spread > max_spread:
            max_spread = spread
            result = j

    return result

@numba.njit(
    "int8(float32[:,::1], intp, intp, intp)",
    fastmath=True,
    cache=True,
    locals={
        "val1": numba.types.float32,
        "val2": numba.types.float32,
    }
)
def _compare_indices(data, axis, idx1, idx2):
    val1 = data[idx1, axis]
    val2 = data[idx2, axis]
    
    if val1 < val2:
        return -1
    elif val1 > val2:
        return 1
    else:
        # Break ties using original index values (like sklearn)
        if idx1 < idx2:
            return -1
        elif idx1 > idx2:
            return 1
        else:
            return 0


@numba.njit(
    "void(float32[:,::1], intp[::1], intp, intp, intp)",
    fastmath=True,
    cache=True,
    locals={
        "i": numba.types.intp,
        "key_idx": numba.types.intp,
        "j": numba.types.intp,
    }
)
def _insertion_sort_indices(data, idx_array, axis, left, right):
    for i in range(left + 1, right):
        key_idx = idx_array[i]
        j = i - 1
        
        while j >= left and _compare_indices(data, axis, idx_array[j], key_idx) > 0:
            idx_array[j + 1] = idx_array[j]
            j -= 1
        
        idx_array[j + 1] = key_idx


@numba.njit(
    "void(float32[:,::1], intp[::1], intp, intp, intp, intp)",
    fastmath=True,
    cache=True,
    locals={
        "root": numba.types.intp,
        "child": numba.types.intp,
        "swap": numba.types.intp,
    }
)
def _sift_down_indices(data, idx_array, axis, offset, start, end):
    root = start
    
    while root * 2 + 1 < end:
        child = root * 2 + 1
        swap = root
        
        if _compare_indices(data, axis, idx_array[offset + swap], idx_array[offset + child]) < 0:
            swap = child
        
        if child + 1 < end and _compare_indices(data, axis, idx_array[offset + swap], idx_array[offset + child + 1]) < 0:
            swap = child + 1
        
        if swap == root:
            return
        
        idx_array[offset + root], idx_array[offset + swap] = idx_array[offset + swap], idx_array[offset + root]
        root = swap


@numba.njit(
    "void(float32[:,::1], intp[::1], intp, intp, intp)",
    cache=True,
    locals={
        "size": numba.types.intp,
        "i": numba.types.intp,
    }
)
def _heapsort_indices(data, idx_array, axis, left, right):
    size = right - left
    
    # Build heap
    for i in range(size // 2 - 1, -1, -1):
        _sift_down_indices(data, idx_array, axis, left, i, size)
    
    # Extract elements
    for i in range(size - 1, 0, -1):
        idx_array[left], idx_array[left + i] = idx_array[left + i], idx_array[left]
        _sift_down_indices(data, idx_array, axis, left, 0, i)


@numba.njit(
    "intp(float32[:,::1], intp[::1], intp, intp, intp)",
    fastmath=True,
    cache=True,
    locals={
        "mid": numba.types.intp,
        "idx_left": numba.types.intp,
        "idx_mid": numba.types.intp,
        "idx_right": numba.types.intp,
    }
)
def _median_of_three_pivot(data, idx_array, axis, left, right):
    mid = (left + right - 1) // 2
    
    idx_left = idx_array[left]
    idx_mid = idx_array[mid]
    idx_right = idx_array[right - 1]
    
    # Sort the three candidates
    if _compare_indices(data, axis, idx_left, idx_mid) > 0:
        idx_array[left], idx_array[mid] = idx_array[mid], idx_array[left]
        idx_left, idx_mid = idx_mid, idx_left
    
    if _compare_indices(data, axis, idx_mid, idx_right) > 0:
        idx_array[mid], idx_array[right - 1] = idx_array[right - 1], idx_array[mid]
        idx_mid, idx_right = idx_right, idx_mid
        
        if _compare_indices(data, axis, idx_left, idx_mid) > 0:
            idx_array[left], idx_array[mid] = idx_array[mid], idx_array[left]
    
    return mid


@numba.njit(
    "intp(float32[:,::1], intp[::1], intp, intp, intp, intp)",
    fastmath=True,
    cache=True,
    locals={
        "pivot_value": numba.types.float32,
        "pivot_original_idx": numba.types.intp,
        "i": numba.types.intp,
        "j": numba.types.intp,
    }
)
def _partition_indices(data, idx_array, axis, left, right, pivot_idx):
    # Move pivot to end
    idx_array[pivot_idx], idx_array[right - 1] = idx_array[right - 1], idx_array[pivot_idx]
    pivot_value = data[idx_array[right - 1], axis]
    pivot_original_idx = idx_array[right - 1]
    
    i = left
    j = right - 2
    
    while True:
        # Find element from left that should be on right
        while i <= j and _compare_indices(data, axis, idx_array[i], pivot_original_idx) < 0:
            i += 1
        
        # Find element from right that should be on left
        while i <= j and _compare_indices(data, axis, idx_array[j], pivot_original_idx) >= 0:
            j -= 1
        
        if i >= j:
            break
            
        # Swap elements
        idx_array[i], idx_array[j] = idx_array[j], idx_array[i]
        i += 1
        j -= 1
    
    # Move pivot to final position
    idx_array[i], idx_array[right - 1] = idx_array[right - 1], idx_array[i]
    return i


@numba.njit(
    "void(float32[:,::1], intp[::1], intp, intp, intp, intp, intp)",
    cache=True,
    locals={
        "pivot_idx": numba.types.intp,
        "pivot_pos": numba.types.intp,
    }
)
def _introselect_impl(data, idx_array, axis, left, right, nth, depth_limit):
    while right - left > 16:
        if depth_limit == 0:
            # Fall back to heapsort when recursion gets too deep
            _heapsort_indices(data, idx_array, axis, left, right)
            return
        
        depth_limit -= 1
        
        # Choose pivot using median-of-three
        pivot_idx = _median_of_three_pivot(data, idx_array, axis, left, right)
        
        # Partition around pivot
        pivot_pos = _partition_indices(data, idx_array, axis, left, right, pivot_idx)
        
        # Recurse on the appropriate side
        if nth < pivot_pos:
            right = pivot_pos
        elif nth > pivot_pos:
            left = pivot_pos + 1
        else:
            # Found the nth element
            return
    
    # Use insertion sort for small subarrays
    _insertion_sort_indices(data, idx_array, axis, left, right)


@numba.njit(
    "void(float32[:,::1], intp[::1], intp, intp, intp, intp)",
    cache=True,
    locals={
        "size": numba.types.intp,
        "max_depth": numba.types.intp,
    }
)
def _introselect(data, idx_array, axis, left, right, nth):
    size = right - left
    
    # Use heapsort for small arrays or when recursion depth is too high
    if size <= 16:
        _insertion_sort_indices(data, idx_array, axis, left, right)
        return
    
    # Calculate maximum recursion depth (2 * log2(size))
    max_depth = 2 * int(np.log2(size))
    _introselect_impl(data, idx_array, axis, left, right, nth, max_depth)


@numba.njit(
    'void(float32[:, ::1], intp[::1], intp[::1], intp[::1], float32[::1], bool_[::1], float32[:, :, ::1], intp, intp, intp)',
    cache=True,
)
def _recursive_build_tree(
    data,
    idx_array,
    idx_start_array,
    idx_end_array,
    radius_array,
    is_leaf_array,
    node_bounds,
    idx_start,
    idx_end,
    node,
):
    n_points = idx_end - idx_start
    n_mid = n_points // 2

    _init_node(data, node_bounds, idx_array, idx_start_array, idx_end_array, radius_array, is_leaf_array, node, idx_start, idx_end)

    if 2 * node + 1 >= is_leaf_array.shape[0]:
        is_leaf_array[node] = True
    elif idx_end - idx_start < 2:
        is_leaf_array[node] = True
    else:
        is_leaf_array[node] = False
        axis = _find_node_split_dim(data, idx_array, idx_start, idx_end)
        _introselect(data, idx_array, axis, idx_start, idx_end, idx_start + n_mid)
        _recursive_build_tree(data, idx_array, idx_start_array, idx_end_array, radius_array, is_leaf_array, node_bounds, idx_start, idx_start + n_mid, 2 * node + 1)
        _recursive_build_tree(data, idx_array, idx_start_array, idx_end_array, radius_array, is_leaf_array, node_bounds, idx_start + n_mid, idx_end, 2 * node + 2)

    return

def build_kdtree(data, leaf_size=40):
        n_samples = data.shape[0]
        n_features = data.shape[1]

        if leaf_size < 1:
            raise ValueError("leaf_size must be greater than or equal to 1")

        # determine number of levels in the tree, and from this
        # the number of nodes in the tree.  This results in leaf nodes
        # with numbers of points between leaf_size and 2 * leaf_size
        n_levels = int(
            np.log2(max(1, (n_samples - 1) / leaf_size)) + 1)
        n_nodes = np.int32((2 ** n_levels) - 1)

        # allocate arrays for storage
        idx_array = np.arange(n_samples, dtype=np.intp)
        idx_start_array = np.zeros(n_nodes, dtype=np.intp)
        idx_end_array = np.zeros(n_nodes, dtype=np.intp)
        radius_array = np.zeros(n_nodes, dtype=np.float32)
        is_leaf_array = np.zeros(n_nodes, dtype=np.bool_)
        node_bounds = np.zeros((2, n_nodes, n_features), dtype=np.float32)

        _recursive_build_tree(
            data,
            idx_array,
            idx_start_array,
            idx_end_array,
            radius_array,
            is_leaf_array,
            node_bounds,
            0,
            n_samples,
            0,
        )
        
        return NumbaKDTree(data, idx_array, idx_start_array, idx_end_array, radius_array, is_leaf_array, node_bounds)

@numba.njit(
    [
        "f4(f4[::1],f4[::1])",
        "f8(f8[::1],f8[::1])",
        "f8(f4[::1],f8[::1])",
    ],
    fastmath=True,
    cache=True,
    locals={
        "dim": numba.types.intp,
        "i": numba.types.uint16,
        "diff": numba.types.float32,
        "result": numba.types.float32,
    }
)
def rdist(x, y):
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff

    return result


@numba.njit(
    [
        "f4(f4[::1],f4[::1],f4[::1])",
        "f4(f8[::1],f8[::1],f4[::1])",
        "f4(f8[::1],f8[::1],f8[::1])",
    ],
    fastmath=True,
    cache=True,
    locals={
        "dim": numba.types.intp,
        "i": numba.types.uint16,
        "d_lo": numba.types.float32,
        "d_hi": numba.types.float32,
        "d": numba.types.float32,
        "result": numba.types.float32,
    }
)
def point_to_node_lower_bound_rdist(upper, lower, pt):
    result = 0.0
    dim = pt.shape[0]
    for i in range(dim):
        d_lo = upper[i] - pt[i] if upper[i] > pt[i] else 0.0
        d_hi = pt[i] - lower[i] if pt[i] > lower[i] else 0.0
        d = d_lo + d_hi
        result += d * d

    return result


@numba.njit(
    [
        "i4(f4[::1],i4[::1],f4,i4)",
        "i4(f8[::1],i4[::1],f8,i4)",
    ],
    fastmath=True,
    locals={
        "size": numba.types.intp,
        "i": numba.types.uint16,
        "ic1": numba.types.uint16,
        "ic2": numba.types.uint16,
        "i_swap": numba.types.uint16,
    },
    cache=True
)
def simple_heap_push(priorities, indices, p, n):
    if p >= priorities[0]:
        return 0

    size = priorities.shape[0]

    # insert val at position zero
    priorities[0] = p
    indices[0] = n

    # descend the heap, swapping values until the max heap criterion is met
    i = 0
    while True:
        ic1 = 2 * i + 1
        ic2 = ic1 + 1

        if ic1 >= size:
            break
        elif ic2 >= size:
            if priorities[ic1] > p:
                i_swap = ic1
            else:
                break
        elif priorities[ic1] >= priorities[ic2]:
            if p < priorities[ic1]:
                i_swap = ic1
            else:
                break
        else:
            if p < priorities[ic2]:
                i_swap = ic2
            else:
                break

        priorities[i] = priorities[i_swap]
        indices[i] = indices[i_swap]

        i = i_swap

    priorities[i] = p
    indices[i] = n

    return 1


@numba.njit(
    fastmath=True,
    cache=True,
    locals={
        "left_child": numba.types.intp,
        "right_child": numba.types.intp,
        "swap": numba.types.intp,
    }
)
def siftdown(heap1, heap2, elt):
    while elt * 2 + 1 < heap1.shape[0]:
        left_child = elt * 2 + 1
        right_child = left_child + 1
        swap = elt

        if heap1[swap] < heap1[left_child]:
            swap = left_child

        if right_child < heap1.shape[0] and heap1[swap] < heap1[right_child]:
            swap = right_child

        if swap == elt:
            break
        else:
            heap1[elt], heap1[swap] = heap1[swap], heap1[elt]
            heap2[elt], heap2[swap] = heap2[swap], heap2[elt]
            elt = swap


@numba.njit(parallel=True, cache=True)
def deheap_sort(distances, indices):
    for i in numba.prange(indices.shape[0]):
        # starting from the end of the array and moving back
        for j in range(indices.shape[1] - 1, 0, -1):
            indices[i, 0], indices[i, j] = indices[i, j], indices[i, 0]
            distances[i, 0], distances[i, j] = distances[i, j], distances[i, 0]

            siftdown(distances[i, :j], indices[i, :j], 0)

    return distances, indices


@numba.njit(
    fastmath=True,
    cache=True,
    locals={
        "node": numba.types.intp,
        "left": numba.types.intp,
        "right": numba.types.intp,
        "d": numba.types.float32,
        "idx": numba.types.uint32,
        "idx_start": numba.types.intp,
        "idx_end": numba.types.intp,
        "is_leaf": numba.types.boolean,
        "i": numba.types.intp,
        "dist_lower_bound_left": numba.types.float32,
        "dist_lower_bound_right": numba.types.float32,
    }
)
def tree_query_recursion(
        tree,
        node,
        point,
        heap_p,
        heap_i,
        dist_lower_bound,
):
    # Get node information
    idx_start = tree.idx_start[node]
    idx_end = tree.idx_end[node]
    is_leaf = tree.is_leaf[node]

    # ------------------------------------------------------------
    # Case 1: query point is outside node radius:
    #         trim it from the query
    if dist_lower_bound > heap_p[0]:
        return

    # ------------------------------------------------------------
    # Case 2: this is a leaf node.  Update set of nearby points
    elif is_leaf:
        for i in range(idx_start, idx_end):
            idx = tree.idx_array[i]
            d = rdist(point, tree.data[idx])
            if d < heap_p[0]:
                simple_heap_push(heap_p, heap_i, d, idx)

    # ------------------------------------------------------------
    # Case 3: Node is not a leaf.  Recursively query subnodes
    #         starting with the closest
    else:
        left = 2 * node + 1
        right = left + 1
        dist_lower_bound_left = point_to_node_lower_bound_rdist(tree.node_bounds[0, left], tree.node_bounds[1, left],
                                                                point)
        dist_lower_bound_right = point_to_node_lower_bound_rdist(tree.node_bounds[0, right], tree.node_bounds[1, right],
                                                                 point)

        # recursively query subnodes
        if dist_lower_bound_left <= dist_lower_bound_right:
            tree_query_recursion(tree, left, point, heap_p, heap_i, dist_lower_bound_left)
            tree_query_recursion(tree, right, point, heap_p, heap_i, dist_lower_bound_right)
        else:
            tree_query_recursion(tree, right, point, heap_p, heap_i, dist_lower_bound_right)
            tree_query_recursion(tree, left, point, heap_p, heap_i, dist_lower_bound_left)

    return


@numba.njit(
    parallel=True,
    fastmath=True,
    cache=True,
    locals={
        "i": numba.types.intp,
        "distance_lower_bound": numba.types.float32,
    }
)
def parallel_tree_query(tree, data, k=10, output_rdist=False):
    result = (np.full((data.shape[0], k), np.inf, dtype=np.float32), np.full((data.shape[0], k), -1, dtype=np.int32))

    for i in numba.prange(data.shape[0]):
        distance_lower_bound = point_to_node_lower_bound_rdist(tree.node_bounds[0, 0], tree.node_bounds[1, 0], data[i])
        heap_priorities, heap_indices = result[0][i], result[1][i]
        tree_query_recursion(tree, 0, data[i], heap_priorities, heap_indices, distance_lower_bound)

    if output_rdist:
        return deheap_sort(result[0], result[1])
    else:
        return deheap_sort(np.sqrt(result[0]), result[1])