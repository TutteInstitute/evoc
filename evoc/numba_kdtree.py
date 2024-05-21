import numba
import numpy as np

from collections import namedtuple

NumbaKDTree = namedtuple("KDTree", ["data", "idx_array", "node_data", "node_bounds"])

def kdtree_to_numba(sklearn_kdtree):
    data, idx_array, node_data, node_bounds = sklearn_kdtree.get_arrays()
    return NumbaKDTree(data, idx_array, node_data, node_bounds)


@numba.njit(
    [
        "f4(f4[::1],f4[::1])",
        "f8(f8[::1],f8[::1])",
        "f8(f4[::1],f8[::1])",
    ],
    fastmath=True,
    locals={
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
    cache=True
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
    locals={
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
    cache=True
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


@numba.njit(cache=True)
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
    locals={
        "node": numba.types.intp,
        "left": numba.types.intp,
        "right": numba.types.intp,
        "d": numba.types.float32,
        "idx": numba.types.uint32,
    },
    cache=True
)
def tree_query_recursion(
        tree,
        node,
        point,
        heap_p,
        heap_i,
        dist_lower_bound,
):
    node_info = tree.node_data[node]

    # ------------------------------------------------------------
    # Case 1: query point is outside node radius:
    #         trim it from the query
    if dist_lower_bound > heap_p[0]:
        return

    # ------------------------------------------------------------
    # Case 2: this is a leaf node.  Update set of nearby points
    elif node_info.is_leaf:
        for i in range(node_info.idx_start, node_info.idx_end):
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


@numba.njit(parallel=True, cache=True)
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