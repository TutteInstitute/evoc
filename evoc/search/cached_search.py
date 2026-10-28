"""Disk-cached graph search over a fitted EVoC model's training data.

Adapted from pynndescent's ``search.py`` (pynndescent 1f460b2,
https://github.com/lmcinnes/pynndescent). The design is the same:

* the index's arrays and settings travel as a single structref, typed
  read-only, so that a call converts one pointer rather than every array, and
  LLVM can assume the result heaps and visited bitset never alias them;
* the search functions are compiled once per argument types and cached on
  disk, so building an index compiles nothing after the first-ever use;
* queries call the compiled entry point for the one exact signature directly,
  skipping the dispatcher's per-call type resolution;
* the seed heap is pushed to directly rather than through a helper, which
  would add reference counting to every push.

Differences from pynndescent:

* distances are EVoC's output units, dispatched on the data dtype at compile
  time (:mod:`evoc.search.kernels`) rather than by a kernel id, so float32,
  int8 and bit-packed uint8 data share one search function;
* there is a single search tree and one leaf per query (no leaf spill);
* queries arrive already prepared (float queries normalised) and come with a
  separate array for descending the tree (float32 for float and int8 data,
  uint8 for bit-packed data) and their inverse norms (int8 only).
"""
import functools
import heapq

import numba
import numpy as np
from numba import types
from numba.experimental import structref
from numba.extending import overload

from ._pynnd_utils import (
    check_and_mark_visited,
    select_side,
    select_side_bit,
    simple_heap_push,
    siftdown,
    tau_rand_int,
)
from .kernels import distance_to, four_distances_to


def _array(dtype, ndim, readonly=False):
    return types.Array(dtype, ndim, "C", readonly=readonly)


_HEAP_TYPE = types.Tuple(
    (_array(types.int32, 2), _array(types.float32, 2), _array(types.uint8, 2))
)


@structref.register
class SearchStructType(types.StructRef):
    def preprocess_fields(self, fields):
        return tuple((name, types.unliteral(typ)) for name, typ in fields)


class SearchStruct(structref.StructRefProxy):
    pass


_SEARCH_STRUCT_FIELDS = [
    "data",
    "inv_norms",
    "indptr",
    "indices",
    "tree_hyperplanes",
    "tree_offsets",
    "tree_children",
    "tree_indices",
    "n_neighbors",
    "min_distance",
]
structref.define_proxy(SearchStruct, SearchStructType, _SEARCH_STRUCT_FIELDS)


def _select_side_any(hyperplane, offset, point, rng_state):
    raise NotImplementedError("only available in numba code")


@overload(_select_side_any)
def _ov_select_side_any(hyperplane, offset, point, rng_state):
    if hyperplane.dtype == types.uint8:

        def impl(hyperplane, offset, point, rng_state):
            return select_side_bit(hyperplane, offset, point, rng_state)

    else:

        def impl(hyperplane, offset, point, rng_state):
            return select_side(hyperplane, offset, point, rng_state)

    return impl


@numba.njit(inline="always", locals={"node": types.uint32})
def _tree_leaf(point, tree_hyperplanes, tree_offsets, tree_children, rng_state):
    node = 0
    while tree_children[node, 0] > 0:
        side = _select_side_any(
            tree_hyperplanes[node], tree_offsets[node], point, rng_state
        )
        if side == 0:
            node = tree_children[node, 0]
        else:
            node = tree_children[node, 1]
    return -tree_children[node, 0], -tree_children[node, 1]


@numba.njit(
    inline="always",
    fastmath=True,
    locals={
        "candidate": types.int32,
        "vertex": types.int32,
        "d": types.float32,
        "d_vertex": types.float32,
        "distance_bound": types.float32,
    },
)
def _search_one(
    current_query,
    tree_current_query,
    query_inv_norm,
    heap_priorities,
    heap_indices,
    k,
    epsilon,
    visited,
    rng_state,
    data,
    inv_norms,
    indptr,
    indices,
    tree_hyperplanes,
    tree_offsets,
    tree_children,
    tree_indices,
    n_neighbors,
    min_distance,
):
    # seed_set is pushed to directly throughout: passing the list to a helper
    # (even an inlined one) adds reference counting to every push.
    seed_set = [(np.float32(np.inf), np.int32(-1)) for j in range(0)]

    # Initialise from the search tree leaf containing the query
    n_initial_points = 0
    leaf_start, leaf_end = _tree_leaf(
        tree_current_query, tree_hyperplanes, tree_offsets, tree_children, rng_state
    )
    for j in range(leaf_start, leaf_end):
        candidate = tree_indices[j]
        if check_and_mark_visited(visited, candidate) == 0:
            d = np.float32(
                distance_to(current_query, query_inv_norm, data, inv_norms, candidate)
            )
            simple_heap_push(heap_priorities, heap_indices, d, candidate)
            heapq.heappush(seed_set, (d, candidate))
            n_initial_points += 1

    # Random samples if needed
    n_random_samples = min(k, n_neighbors) - n_initial_points
    if n_random_samples > 0:
        for j in range(n_random_samples):
            candidate = np.int32(np.abs(tau_rand_int(rng_state)) % data.shape[0])
            if check_and_mark_visited(visited, candidate) == 0:
                d = np.float32(
                    distance_to(
                        current_query, query_inv_norm, data, inv_norms, candidate
                    )
                )
                simple_heap_push(heap_priorities, heap_indices, d, candidate)
                heapq.heappush(seed_set, (d, candidate))

    if len(seed_set) == 0:
        return

    # Graph search
    distance_bound = heap_priorities[0] + (
        epsilon * (heap_priorities[0] - min_distance)
    )
    d_vertex, vertex = heapq.heappop(seed_set)
    pending = np.empty(4, dtype=np.int32)
    pending_d = np.empty(4, dtype=np.float32)

    while d_vertex < distance_bound:
        # Unvisited neighbours are compared four at a time and then considered
        # in their original order, so the search path is unchanged.
        n_pending = 0
        for j in range(indptr[vertex], indptr[vertex + 1]):
            candidate = indices[j]
            if check_and_mark_visited(visited, candidate) == 0:
                pending[n_pending] = candidate
                n_pending += 1
                if n_pending == 4:
                    four_distances_to(
                        current_query, query_inv_norm, data, inv_norms, pending,
                        pending_d,
                    )
                    for r in range(4):
                        d = pending_d[r]
                        if d < distance_bound:
                            simple_heap_push(
                                heap_priorities, heap_indices, d, pending[r]
                            )
                            heapq.heappush(seed_set, (d, pending[r]))
                            distance_bound = heap_priorities[0] + (
                                epsilon * (heap_priorities[0] - min_distance)
                            )
                    n_pending = 0
        for r in range(n_pending):
            d = np.float32(
                distance_to(current_query, query_inv_norm, data, inv_norms, pending[r])
            )
            if d < distance_bound:
                simple_heap_push(heap_priorities, heap_indices, d, pending[r])
                heapq.heappush(seed_set, (d, pending[r]))
                distance_bound = heap_priorities[0] + (
                    epsilon * (heap_priorities[0] - min_distance)
                )

        if len(seed_set) == 0:
            break
        d_vertex, vertex = heapq.heappop(seed_set)


@numba.njit(inline="always")
def _query_inv_norm(query_inv_norms, i):
    if query_inv_norms.shape[0] > 0:
        return query_inv_norms[i]
    return np.float32(1.0)


# Arguments are in the order SearchIndex.query passes them, after the index
# struct, so that functools.partial(entry_point, struct) can be called directly.
@numba.njit(fastmath=True, cache=True)
def search_serial(
    index, query_points, tree_query_points, query_inv_norms, result, k, epsilon,
    visited, rng_state,
):
    data = index.data
    inv_norms = index.inv_norms
    indptr = index.indptr
    indices = index.indices
    tree_hyperplanes = index.tree_hyperplanes
    tree_offsets = index.tree_offsets
    tree_children = index.tree_children
    tree_indices = index.tree_indices
    n_neighbors = index.n_neighbors
    min_distance = index.min_distance
    internal_rng_state = np.copy(rng_state)
    for i in range(query_points.shape[0]):
        visited[:] = 0
        heap_priorities = result[1][i]
        heap_indices = result[0][i]
        heap_priorities[:] = np.inf
        heap_indices[:] = -1
        _search_one(
            query_points[i], tree_query_points[i],
            _query_inv_norm(query_inv_norms, i), heap_priorities, heap_indices, k,
            epsilon, visited, internal_rng_state, data, inv_norms, indptr, indices,
            tree_hyperplanes, tree_offsets, tree_children, tree_indices,
            n_neighbors, min_distance,
        )
    return result


@numba.njit(fastmath=True, cache=True, parallel=True)
def search_parallel(
    index, query_points, tree_query_points, query_inv_norms, result, k, epsilon,
    visited, rng_state,
):
    data = index.data
    inv_norms = index.inv_norms
    indptr = index.indptr
    indices = index.indices
    tree_hyperplanes = index.tree_hyperplanes
    tree_offsets = index.tree_offsets
    tree_children = index.tree_children
    tree_indices = index.tree_indices
    n_neighbors = index.n_neighbors
    min_distance = index.min_distance
    for i in numba.prange(query_points.shape[0]):
        # Each query needs its own visited set and rng state in parallel
        visited_nodes = np.zeros_like(visited)
        query_rng_state = np.copy(rng_state)
        query_rng_state[0] += i
        heap_priorities = result[1][i]
        heap_indices = result[0][i]
        heap_priorities[:] = np.inf
        heap_indices[:] = -1
        _search_one(
            query_points[i], tree_query_points[i],
            _query_inv_norm(query_inv_norms, i), heap_priorities, heap_indices, k,
            epsilon, visited_nodes, query_rng_state, data, inv_norms, indptr,
            indices, tree_hyperplanes, tree_offsets, tree_children, tree_indices,
            n_neighbors, min_distance,
        )
    return result


@numba.njit(cache=True)
def _new_search_struct(
    data, inv_norms, indptr, indices, tree_hyperplanes, tree_offsets,
    tree_children, tree_indices, n_neighbors, min_distance,
):
    # Constructing the structref in a cached function avoids compiling its
    # constructor in every process.
    return SearchStruct(
        data, inv_norms, indptr, indices, tree_hyperplanes, tree_offsets,
        tree_children, tree_indices, n_neighbors, min_distance,
    )


def _readonly(array, dtype):
    view = np.ascontiguousarray(array, dtype=dtype).view()
    view.flags.writeable = False
    return view


def make_search_struct(
    data, inv_norms, indptr, indices, tree, n_neighbors, min_distance
):
    """Build the structref holding an index's arrays (all read-only)."""
    return _new_search_struct(
        _readonly(data, data.dtype),
        _readonly(inv_norms, np.float32),
        _readonly(indptr, np.int32),
        _readonly(indices, np.int32),
        _readonly(tree.hyperplanes, tree.hyperplanes.dtype),
        _readonly(tree.offsets, np.float32),
        _readonly(tree.children, np.int32),
        _readonly(tree.indices, np.int32),
        np.int64(n_neighbors),
        np.float32(min_distance),
    )


def make_cached_search_function(struct, query_dtype, tree_query_dtype, parallel):
    """Bind an index struct to the cached search function.

    Returns a callable ``search(query_points, tree_query_points,
    query_inv_norms, result, k, epsilon, visited, rng_state) -> result``. When
    the compiled entry point is available it is called directly; it only checks
    dtypes, so callers must pass C-contiguous 2-D queries of the right dtypes
    and a result heap from ``make_heap`` (``SearchIndex.query`` does).
    """
    dispatcher = search_parallel if parallel else search_serial
    signature = (
        numba.typeof(struct),
        _array(numba.from_dtype(np.dtype(query_dtype)), 2),
        _array(numba.from_dtype(np.dtype(tree_query_dtype)), 2),
        _array(types.float32, 1),
        _HEAP_TYPE,
        types.int64,
        types.float32,
        _array(types.uint8, 1),
        _array(types.int64, 1),
    )

    entry_point = None
    try:
        dispatcher.compile(signature)
        entry_point = dispatcher.overloads[signature].entry_point
    except (AttributeError, KeyError, TypeError):
        entry_point = None

    if entry_point is not None:
        return functools.partial(entry_point, struct)

    def search(
        query_points, tree_query_points, query_inv_norms, result, k, epsilon,
        visited, rng_state,
    ):
        return dispatcher(
            struct, query_points, tree_query_points, query_inv_norms, result,
            np.int64(k), np.float32(epsilon), visited, rng_state,
        )

    return search


# deheap_sort variants with on-disk caching
@numba.njit(cache=True)
def deheap_sort_serial(indices, distances):
    for i in range(indices.shape[0]):
        for j in range(indices.shape[1] - 1, 0, -1):
            indices[i, 0], indices[i, j] = indices[i, j], indices[i, 0]
            distances[i, 0], distances[i, j] = distances[i, j], distances[i, 0]
            siftdown(distances[i, :j], indices[i, :j], 0)
    return indices, distances


@numba.njit(parallel=True, cache=True)
def deheap_sort_parallel(indices, distances):
    for i in numba.prange(indices.shape[0]):
        for j in range(indices.shape[1] - 1, 0, -1):
            indices[i, 0], indices[i, j] = indices[i, j], indices[i, 0]
            distances[i, 0], distances[i, j] = distances[i, j], distances[i, 0]
            siftdown(distances[i, :j], indices[i, :j], 0)
    return indices, distances
