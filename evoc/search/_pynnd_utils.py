"""Helpers copied from pynndescent for the search code in this package.

Source: https://github.com/lmcinnes/pynndescent at 1f460b2
(pynndescent/utils.py, rp_trees.py, sparse.py and pynndescent_.py). They are
copied unchanged, apart from importing ``popcnt_u8`` from EVoC, so that later
fixes upstream can be carried across.
"""
from collections import namedtuple

import numba
import numpy as np

from ..uint8_nndescent import popcnt_u8

@numba.njit("i4(i8[:])", cache=True)
def tau_rand_int(state):
    """A fast (pseudo)-random number generator.

    Parameters
    ----------
    state: array of int64, shape (3,)
        The internal state of the rng

    Returns
    -------
    A (pseudo)-random int32 value
    """
    state[0] = (((state[0] & 4294967294) << 12) & 0xFFFFFFFF) ^ (
        (((state[0] << 13) & 0xFFFFFFFF) ^ state[0]) >> 19
    )
    state[1] = (((state[1] & 4294967288) << 4) & 0xFFFFFFFF) ^ (
        (((state[1] << 2) & 0xFFFFFFFF) ^ state[1]) >> 25
    )
    state[2] = (((state[2] & 4294967280) << 17) & 0xFFFFFFFF) ^ (
        (((state[2] << 3) & 0xFFFFFFFF) ^ state[2]) >> 11
    )

    return state[0] ^ state[1] ^ state[2]


@numba.njit(inline="always", cache=True)
def _splitmix64(x, v):
    x = x ^ np.uint64(v)
    x = x + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


@numba.njit(inline="always", cache=True)
def hash_uniform(seed, a, b, c):
    """A stateless uniform float in [0, 1) derived from (seed, a, b, c).

    Unlike tau_rand there is no shared state, so it can be used from parallel
    loops and gives the same result for any number of threads.
    """
    x = _splitmix64(np.uint64(seed), a)
    x = _splitmix64(x, b)
    x = _splitmix64(x, c)
    return np.float64(x >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@numba.njit(cache=True)
def make_heap(n_points, size):
    """Constructor for the numba enabled heap objects. The heaps are used
    for approximate nearest neighbor search, maintaining a list of potential
    neighbors sorted by their distance. We also flag if potential neighbors
    are newly added to the list or not. Internally this is stored as
    a single ndarray; the first axis determines whether we are looking at the
    array of candidate graph_indices, the array of distances, or the flag array for
    whether elements are new or not. Each of these arrays are of shape
    (``n_points``, ``size``)

    Parameters
    ----------
    n_points: int
        The number of graph_data points to track in the heap.

    size: int
        The number of items to keep on the heap for each graph_data point.

    Returns
    -------
    heap: An ndarray suitable for passing to other numba enabled heap functions.
    """
    indices = np.full((int(n_points), int(size)), -1, dtype=np.int32)
    distances = np.full((int(n_points), int(size)), np.inf, dtype=np.float32)
    flags = np.zeros((int(n_points), int(size)), dtype=np.uint8)
    result = (indices, distances, flags)

    return result


@numba.njit(
    "i4(f4[::1],i4[::1],f4,i4)",
    fastmath=True,
    locals={
        "size": numba.types.intp,
        "i": numba.types.uint16,
        "ic1": numba.types.uint16,
        "ic2": numba.types.uint16,
        "i_swap": numba.types.uint16,
    },
    cache=True,
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
    """Restore the heap property for a heap with an out of place element
    at position ``elt``. This works with a heap pair where heap1 carries
    the weights and heap2 holds the corresponding elements."""
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


@numba.njit("b1(u1[::1],i4)", cache=True)
def check_and_mark_visited(table, candidate):
    """Check if candidate was visited and mark it as visited in one operation.

    Returns True if the candidate was already visited, False otherwise.
    More efficient than separate has_been_visited + mark_visited calls.
    """
    loc = candidate >> 3
    mask = numba.uint8(1 << (candidate & 7))
    was_visited = table[loc] & mask
    table[loc] |= mask
    return was_visited


@numba.njit("void(u1[::1],i4)", cache=True)
def mark_visited(table, candidate):
    loc = candidate >> 3
    mask = 1 << (candidate & 7)
    table[loc] |= mask
    return


@numba.njit(
    [
        "f4(f4[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True)
        ),
    ],
    locals={
        "dim": numba.types.intp,
        "i": numba.types.uint32,
        # "result": numba.types.float32, # This provides speed, but causes errors in corner cases
    },
    fastmath=True,
    cache=True,
)
def norm(vec):
    """Compute the (standard l2) norm of a vector.

    Parameters
    ----------
    vec: array of shape (dim,)

    Returns
    -------
    The l2 norm of vec.
    """
    result = 0.0
    dim = vec.shape[0]
    for i in range(dim):
        result += vec[i] * vec[i]
    return np.sqrt(result)


EPS = 1e-8


FlatTree = namedtuple(
    "FlatTree", ["hyperplanes", "offsets", "children", "indices", "leaf_size"]
)


@numba.njit(
    numba.types.Array(numba.types.int32, 1, "C")(
        numba.types.Array(numba.types.int32, 2, "C")
    ),
    fastmath=True,
    nogil=True,
    cache=True,
)
def compute_global_degrees(neighbor_indices):
    """Compute global in-degree for all points in the graph.

    In-degree of a point is how many times it appears as a neighbor of other points.
    This is computed once and reused throughout tree construction.

    Parameters
    ----------
    neighbor_indices : array of shape (n_samples, n_neighbors)
        The neighbor graph indices.

    Returns
    -------
    global_degrees : array of shape (n_samples,)
        The in-degree of each point.
    """
    n_points = neighbor_indices.shape[0]
    global_degrees = np.zeros(n_points, dtype=np.int32)

    for i in range(n_points):
        for j in range(neighbor_indices.shape[1]):
            neighbor = neighbor_indices[i, j]
            if neighbor >= 0 and neighbor < n_points:
                global_degrees[neighbor] += 1

    return global_degrees


@numba.njit(
    numba.types.Array(numba.types.int32, 1, "C")(
        numba.types.Array(numba.types.int32, 1, "C"),
        numba.types.Array(numba.types.int32, 1, "C"),
        numba.types.int64,
    ),
    fastmath=True,
    nogil=True,
    cache=True,
)
def get_top_k_hub_indices(indices, global_degrees, k=5):
    """Get the indices of the top k highest-degree points from a subset.

    Uses an efficient O(n) selection for small k by maintaining a min-heap of k elements.

    Parameters
    ----------
    indices : array of shape (n,)
        The point indices in the current split.
    global_degrees : array of shape (n_total,)
        Precomputed global degrees for all points.
    k : int
        Number of top hubs to return.

    Returns
    -------
    top_hubs : array of shape (min(k, n),)
        The actual point indices (not positions) of the top k hubs.
    """
    n_points = indices.shape[0]
    actual_k = min(k, n_points)

    # For small k, use a simple insertion-sort-based approach which is O(n*k)
    # but with very low constants for small k
    top_degrees = np.full(actual_k, np.int32(-1), dtype=np.int32)
    top_indices = np.empty(actual_k, dtype=np.int32)

    for i in range(n_points):
        deg = global_degrees[indices[i]]

        # Check if this degree is larger than the smallest in our top-k
        if deg > top_degrees[actual_k - 1]:
            # Find insertion point (sorted descending)
            insert_pos = actual_k - 1
            while insert_pos > 0 and deg > top_degrees[insert_pos - 1]:
                insert_pos -= 1

            # Shift elements down
            for j in range(actual_k - 1, insert_pos, -1):
                top_degrees[j] = top_degrees[j - 1]
                top_indices[j] = top_indices[j - 1]

            # Insert new element
            top_degrees[insert_pos] = deg
            top_indices[insert_pos] = indices[i]

    return top_indices


MIN_SPLIT_BALANCE = 0.1


@numba.njit(
    [
        "b1(f4[::1],f4,f4[::1],i8[::1])",
        numba.types.boolean(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.float32,
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.int64, 1, "C", readonly=False),
        ),
    ],
    fastmath=True,
    locals={
        "margin": numba.types.float32,
        "dim": numba.types.intp,
    },
    cache=True,
)
def select_side(hyperplane, offset, point, rng_state):
    margin = offset
    dim = point.shape[0]
    for d in range(dim):
        margin += hyperplane[d] * point[d]

    if abs(margin) < EPS:
        side = np.abs(tau_rand_int(rng_state)) % 2
        if side == 0:
            return 0
        else:
            return 1
    elif margin > 0:
        return 0
    else:
        return 1


@numba.njit(
    [
        "b1(u1[::1],f4,u1[::1],i8[::1])",
        numba.types.boolean(
            numba.types.Array(numba.types.uint8, 1, "C", readonly=True),
            numba.types.float32,
            numba.types.Array(numba.types.uint8, 1, "C", readonly=True),
            numba.types.Array(numba.types.int64, 1, "C", readonly=False),
        ),
    ],
    fastmath=True,
    locals={
        "margin": numba.types.float32,
        "dim": numba.types.intp,
    },
    cache=True,
)
def select_side_bit(hyperplane, offset, point, rng_state):
    margin = offset
    dim = point.shape[0]
    for d in range(dim):
        margin += np.float32(popcnt_u8(hyperplane[d] & point[d]))
        margin -= np.float32(popcnt_u8(hyperplane[dim + d] & point[d]))

    if abs(margin) < EPS:
        side = np.abs(tau_rand_int(rng_state)) % 2
        if side == 0:
            return 0
        else:
            return 1
    elif margin > 0:
        return 0
    else:
        return 1


@numba.njit(
    fastmath=True,
    locals={
        "result_ind": numba.types.int32[::1],
        "result_data": numba.types.float32[::1],
        "val": numba.types.float32,
        "i1": numba.types.int32,
        "i2": numba.types.int32,
        "j1": numba.types.int32,
        "j2": numba.types.int32,
    },
    cache=True,
)
def sparse_sum(ind1, data1, ind2, data2):
    result_size = ind1.shape[0] + ind2.shape[0]
    result_ind = np.zeros(result_size, dtype=np.int32)
    result_data = np.zeros(result_size, dtype=np.float32)

    i1 = 0
    i2 = 0
    nnz = 0

    # pass through both index lists
    while i1 < ind1.shape[0] and i2 < ind2.shape[0]:
        j1 = ind1[i1]
        j2 = ind2[i2]

        if j1 == j2:
            val = data1[i1] + data2[i2]
            if val != 0:
                result_ind[nnz] = j1
                result_data[nnz] = val
                nnz += 1
            i1 += 1
            i2 += 1
        elif j1 < j2:
            val = data1[i1]
            if val != 0:
                result_ind[nnz] = j1
                result_data[nnz] = val
                nnz += 1
            i1 += 1
        else:
            val = data2[i2]
            if val != 0:
                result_ind[nnz] = j2
                result_data[nnz] = val
                nnz += 1
            i2 += 1

    # pass over the tails
    while i1 < ind1.shape[0]:
        j1 = ind1[i1]
        val = data1[i1]
        if val != 0:
            result_ind[nnz] = j1
            result_data[nnz] = val
            nnz += 1
        i1 += 1

    while i2 < ind2.shape[0]:
        j2 = ind2[i2]
        val = data2[i2]
        if val != 0:
            result_ind[nnz] = j2
            result_data[nnz] = val
            nnz += 1
        i2 += 1

    # truncate to the correct length in case there were zeros created
    result_ind = result_ind[:nnz]
    result_data = result_data[:nnz]

    return result_ind, result_data


@numba.njit(cache=True)
def sparse_diff(ind1, data1, ind2, data2):
    return sparse_sum(ind1, data1, ind2, -data2)


@numba.njit(
    fastmath=True,
    locals={
        "val": numba.types.float32,
        "i1": numba.types.int32,
        "i2": numba.types.int32,
        "j1": numba.types.int32,
        "j2": numba.types.int32,
    },
    cache=True,
)
def sparse_mul(ind1, data1, ind2, data2):
    result_ind = numba.typed.List.empty_list(numba.types.int32)
    result_data = numba.typed.List.empty_list(numba.types.float32)

    i1 = 0
    i2 = 0

    # pass through both index lists
    while i1 < ind1.shape[0] and i2 < ind2.shape[0]:
        j1 = ind1[i1]
        j2 = ind2[i2]

        if j1 == j2:
            val = data1[i1] * data2[i2]
            if val != 0:
                result_ind.append(j1)
                result_data.append(val)
            i1 += 1
            i2 += 1
        elif j1 < j2:
            i1 += 1
        else:
            i2 += 1

    return result_ind, result_data


@numba.njit()
def sparse_dot_accumulate(start, ind1, data1, ind2, data2):
    """Return ``start`` plus the dot product of two sorted sparse vectors.

    The float32 products are added to ``start`` one at a time in index order,
    which is exactly what summing the output of :func:`sparse_mul` onto
    ``start`` computes, without allocating the product vector.
    """
    i1 = 0
    i2 = 0
    n1 = ind1.shape[0]
    n2 = ind2.shape[0]
    result = start
    while i1 < n1 and i2 < n2:
        j1 = ind1[i1]
        j2 = ind2[i2]
        if j1 == j2:
            result += np.float32(data1[i1] * data2[i2])
            i1 += 1
            i2 += 1
        elif j1 < j2:
            i1 += 1
        else:
            i2 += 1
    return result


def resort_tree_indices(tree, tree_order):
    """Given a new data indexing, resort the tree indices to match"""
    new_tree = FlatTree(
        tree.hyperplanes,
        tree.offsets,
        tree.children,
        tree.indices[tree_order].astype(np.int32, order="C"),
        tree.leaf_size,
    )
    return new_tree
