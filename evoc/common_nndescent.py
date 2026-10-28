import numpy as np
import numba



# ---------------------------------------------------------------------------
# Graph update buffers are float32 arrays of (p, q, d) records. The vertex ids p
# and q are stored as the raw bits of an int32 rather than converted to float32
# values, which cannot represent ids above 2**24 exactly. These intrinsics
# reinterpret the bits (an LLVM bitcast, which costs nothing). Never perform
# arithmetic or comparisons on the float32 form of an id.
# ---------------------------------------------------------------------------
from numba import types as _nb_types
from numba.extending import intrinsic as _nb_intrinsic
from llvmlite import ir as _llvm_ir


@_nb_intrinsic
def int32_bits_as_float32(typingctx, x):
    """Store an integer vertex id in a float32 slot without changing its bits."""
    if not isinstance(x, _nb_types.Integer):
        return None

    def codegen(context, builder, sig, args):
        val = context.cast(builder, args[0], sig.args[0], _nb_types.int32)
        return builder.bitcast(val, _llvm_ir.FloatType())

    return _nb_types.float32(x), codegen


@_nb_intrinsic
def float32_bits_as_int32(typingctx, x):
    """Recover a vertex id stored with int32_bits_as_float32."""
    if not isinstance(x, _nb_types.Float) or x.bitwidth != 32:
        return None

    def codegen(context, builder, sig, args):
        return builder.bitcast(args[0], _llvm_ir.IntType(32))

    return _nb_types.int32(x), codegen


@numba.njit("void(i8[:], i8)", cache=True)
def seed(rng_state, seed):
    """Seed the random number generator with a given seed."""
    rng_state.fill(seed + 0xFFFF)


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


@numba.njit("f4(i8[:])", cache=True)
def tau_rand(state):
    """A fast (pseudo)-random number generator for floats in the range [0,1]

    Parameters
    ----------
    state: array of int64, shape (3,)
        The internal state of the rng

    Returns
    -------
    A (pseudo)-random float32 in the interval [0, 1]
    """
    integer = tau_rand_int(state)
    return abs(float(integer) / 0x7FFFFFFF)


# @numba.njit(cache=True)
def make_heap(n_points, size):
    indices = np.full((int(n_points), int(size)), -1, dtype=np.int32)
    distances = np.full((int(n_points), int(size)), np.inf, dtype=np.float32)
    flags = np.zeros((int(n_points), int(size)), dtype=np.uint8)
    result = (indices, distances, flags)

    return result


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


@numba.njit(parallel=True, cache=True)
def deheap_sort(indices, distances):
    """Given two arrays representing a heap (indices and distances), reorder the
     arrays by increasing distance. This is effectively just the second half of
     heap sort (the first half not being required since we already have the
     graph_data in a heap).

     Note that this is done in-place.

    Parameters
    ----------
    indices : array of shape (n_samples, n_neighbors)
        The graph indices to sort by distance.
    distances : array of shape (n_samples, n_neighbors)
        The corresponding edge distance.

    Returns
    -------
    indices, distances: arrays of shape (n_samples, n_neighbors)
        The indices and distances sorted by increasing distance.
    """
    for i in numba.prange(indices.shape[0]):
        # starting from the end of the array and moving back
        for j in range(indices.shape[1] - 1, 0, -1):
            indices[i, 0], indices[i, j] = indices[i, j], indices[i, 0]
            distances[i, 0], distances[i, j] = distances[i, j], distances[i, 0]

            siftdown(distances[i, :j], indices[i, :j], 0)

    return indices, distances


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
def build_candidates_heap_push(priorities, indices, p, n):
    if p >= priorities[0]:
        return 0

    size = priorities.shape[0]

    # break if we already have this element.
    # Duplicate check. For wider heaps a branch-free reduction (no early exit)
    # lets LLVM vectorise the scan.
    if size >= 16:
        found = False
        for j in range(size):
            found |= indices[j] == n
        if found:
            return 0
    else:
        for j in range(size):
            if n == indices[j]:
                return 0

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


@numba.njit(parallel=True, locals={"idx": numba.types.int64}, cache=True)
def build_candidates(current_graph, max_candidates, rng_state, n_threads):
    """Build a heap of candidate neighbors for nearest neighbor descent. For
    each vertex the candidate neighbors are any current neighbors, and any
    vertices that have the vertex as one of their nearest neighbors.

    Parameters
    ----------
    current_graph: heap
        The current state of the graph for nearest neighbor descent.

    max_candidates: int
        The maximum number of new candidate neighbors.

    rng_state: array of int64, shape (3,)
        The internal state of the rng

    Returns
    -------
    candidate_neighbors: A heap with an array of (randomly sorted) candidate
    neighbors for each vertex in the graph.
    """
    current_indices = current_graph[0]
    current_flags = current_graph[2]

    n_vertices = current_indices.shape[0]
    n_neighbors = current_indices.shape[1]

    new_candidate_indices = np.full((n_vertices, max_candidates), -1, dtype=np.int32)
    new_candidate_priority = np.full(
        (n_vertices, max_candidates), np.inf, dtype=np.float32
    )

    old_candidate_indices = np.full((n_vertices, max_candidates), -1, dtype=np.int32)
    old_candidate_priority = np.full(
        (n_vertices, max_candidates), np.inf, dtype=np.float32
    )

    block_size = n_vertices // n_threads + 1

    for n in numba.prange(n_threads):
        local_rng_state = rng_state + n
        block_start = n * block_size
        block_end = min(block_start + block_size, n_vertices)

        for i in range(n_vertices):
            for j in range(n_neighbors):
                idx = current_indices[i, j]

                if idx >= 0 and (
                    (i >= block_start and i < block_end)
                    or (idx >= block_start and idx < block_end)
                ):
                    isn = current_flags[i, j]
                    d = tau_rand(local_rng_state)

                    if isn:
                        if i >= block_start and i < block_end:
                            build_candidates_heap_push(
                                new_candidate_priority[i],
                                new_candidate_indices[i],
                                d,
                                idx,
                            )
                        if idx >= block_start and idx < block_end:
                            build_candidates_heap_push(
                                new_candidate_priority[idx],
                                new_candidate_indices[idx],
                                d,
                                i,
                            )
                    else:
                        if i >= block_start and i < block_end:
                            build_candidates_heap_push(
                                old_candidate_priority[i],
                                old_candidate_indices[i],
                                d,
                                idx,
                            )
                        if idx >= block_start and idx < block_end:
                            build_candidates_heap_push(
                                old_candidate_priority[idx],
                                old_candidate_indices[idx],
                                d,
                                i,
                            )

    indices = current_graph[0]
    flags = current_graph[2]

    for i in numba.prange(n_vertices):
        for j in range(n_neighbors):
            idx = indices[i, j]

            for k in range(max_candidates):
                if new_candidate_indices[i, k] == idx:
                    flags[i, j] = 0
                    break

    return new_candidate_indices, old_candidate_indices


@numba.njit(
    "i4(f4[::1],i4[::1],u1[::1],f4,i4)",
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
def flagged_heap_push(priorities, indices, flags, p, n):
    if p >= priorities[0]:
        return 0

    size = priorities.shape[0]

    # break if we already have this element.
    # Duplicate check. For wider heaps a branch-free reduction (no early exit)
    # lets LLVM vectorise the scan.
    if size >= 16:
        found = False
        for j in range(size):
            found |= indices[j] == n
        if found:
            return 0
    else:
        for j in range(size):
            if n == indices[j]:
                return 0

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
        flags[i] = flags[i_swap]

        i = i_swap

    priorities[i] = p
    indices[i] = n
    flags[i] = 1

    return 1


@numba.njit(
    numba.uint32(
        numba.types.Tuple(
            (numba.int32[:, ::1], numba.float32[:, ::1], numba.uint8[:, ::1])
        ),
        numba.float32[:, :, ::1],
        numba.int32[::1],
        numba.int64,
    ),
    parallel=True,
    locals={
        "p": numba.int32,
        "q": numba.int32,
        "d": numba.float32,
        "added": numba.uint8,
        "n": numba.uint32,
        "i": numba.uint32,
        "j": numba.uint32,
        "priorities": numba.float32[:, ::1],
        "indices": numba.int32[:, ::1],
        "flags": numba.uint8[:, ::1],
    },
    cache=True,
)
def apply_graph_update_array(
    current_graph, update_array, n_updates_per_thread, n_threads
):

    n_changes = 0
    priorities = current_graph[1]
    indices = current_graph[0]
    flags = current_graph[2]

    n_vertices = priorities.shape[0]
    block_size = n_vertices // n_threads + 1

    for n in numba.prange(n_threads):
        block_start = n * block_size
        block_end = min(block_start + block_size, n_vertices)

        for i in range(update_array.shape[0]):
            for j in range(n_updates_per_thread[i]):
                p = float32_bits_as_int32(update_array[i, j, 0])

                if p == -1:
                    break

                q = float32_bits_as_int32(update_array[i, j, 1])
                d = np.float32(update_array[i, j, 2])

                if p >= block_start and p < block_end:
                    added = flagged_heap_push(priorities[p], indices[p], flags[p], d, q)
                    n_changes += added

                if q >= block_start and q < block_end:
                    added = flagged_heap_push(priorities[q], indices[q], flags[q], d, p)
                    n_changes += added

    return n_changes


@numba.njit(
    numba.uint32(
        numba.types.Tuple(
            (numba.int32[:, ::1], numba.float32[:, ::1], numba.uint8[:, ::1])
        ),
        numba.float32[:, :, ::1],
        numba.int32[:, ::1],
        numba.int64,
    ),
    parallel=True,
    cache=True,
    locals={
        "p": numba.int32,
        "q": numba.int32,
        "d": numba.float32,
        "added": numba.uint8,
        "n": numba.uint32,
        "t": numba.uint32,
        "j": numba.uint32,
        "priorities": numba.float32[:, ::1],
        "indices": numba.int32[:, ::1],
        "flags": numba.uint8[:, ::1],
    },
)
def apply_sorted_graph_updates(
    current_graph, update_array, n_updates_per_block, n_threads
):
    """
    Apply pre-sorted graph updates where updates are bucketed by target block.

    Each thread processes only its own bucket, avoiding the need to scan all updates.
    This provides O(updates_per_block) work per thread instead of O(total_updates).
    """
    n_changes = 0
    priorities = current_graph[1]
    indices = current_graph[0]
    flags = current_graph[2]

    n_vertices = priorities.shape[0]
    vertex_block_size = n_vertices // n_threads + 1
    max_updates_per_thread = update_array.shape[1] // n_threads

    for n in numba.prange(n_threads):
        block_start = n * vertex_block_size
        block_end = min(block_start + vertex_block_size, n_vertices)

        # Process all updates in this block's bucket
        # Updates were written by each thread at offset t * max_updates_per_thread
        for t in range(n_threads):
            thread_start = t * max_updates_per_thread
            thread_count = n_updates_per_block[n, t + 1]

            for j in range(thread_count):
                idx = thread_start + j
                p = float32_bits_as_int32(update_array[n, idx, 0])
                q = float32_bits_as_int32(update_array[n, idx, 1])
                d = np.float32(update_array[n, idx, 2])

                # Apply update to p if it's in this block
                if p >= block_start and p < block_end:
                    added = flagged_heap_push(priorities[p], indices[p], flags[p], d, q)
                    n_changes += added

                # Apply update to q if it's in this block
                if q >= block_start and q < block_end:
                    added = flagged_heap_push(priorities[q], indices[q], flags[q], d, p)
                    n_changes += added

    return n_changes


@numba.njit(inline="always")
def _push_candidate(priorities, indices, row, d, n):
    build_candidates_heap_push(priorities[row], indices[row], d, n)


@numba.njit(
    parallel=True,
    cache=False,
    locals={"idx": numba.types.int64, "owner": numba.types.int64, "k": numba.types.int64},
)
def build_candidates_owner(
    current_graph,
    max_candidates,
    rng_state,
    n_threads,
    new_candidate_indices,
    new_candidate_priority,
    old_candidate_indices,
    old_candidate_priority,
    active,
    reverse_dst,
    reverse_src,
    bucket_bounds,
):
    """Owner-computes candidate construction with an active set.

    Each thread scans only the rows of its own vertex block. Forward edges and
    reverse edges whose endpoint is in the same block are pushed directly;
    other reverse edges are counting-sorted into buckets addressed to the
    owning thread, which drains them in a second parallel pass. Per-thread work
    is therefore O(n_vertices * n_neighbors / n_threads), rather than a scan of
    the whole graph per thread.

    New candidates are built first. Old candidates are then built only for
    vertices that received at least one new candidate, since the local join
    never reads the old list of a vertex with no new candidates. The candidate
    arrays are a workspace reused across calls; only rows written by the
    previous call (``active``) are reset. On the first call the workspace must
    be (-1, inf) and ``active`` all ones.

    ``reverse_dst`` and ``reverse_src`` need room for n_vertices * n_neighbors
    entries; ``bucket_bounds`` has shape (n_threads, n_threads + 1).
    """
    current_indices = current_graph[0]
    current_flags = current_graph[2]
    n_vertices = current_indices.shape[0]
    n_neighbors = current_indices.shape[1]
    block_size = n_vertices // n_threads + 1

    for i in numba.prange(n_vertices):
        if active[i]:
            for k in range(max_candidates):
                new_candidate_indices[i, k] = -1
                new_candidate_priority[i, k] = np.inf
                old_candidate_indices[i, k] = -1
                old_candidate_priority[i, k] = np.inf

    for phase in range(2):
        # phase 0 gathers new candidates (flag 1); phase 1 gathers old ones (flag 0)
        # for active vertices only.
        want_flag = 1 - phase
        if phase == 0:
            cand_idx = new_candidate_indices
            cand_pri = new_candidate_priority
        else:
            cand_idx = old_candidate_indices
            cand_pri = old_candidate_priority

        # 1. count cross-block reverse edges per destination block
        for t in numba.prange(n_threads):
            for b in range(n_threads + 1):
                bucket_bounds[t, b] = 0
            row_start = t * block_size
            row_end = min(row_start + block_size, n_vertices)
            for i in range(row_start, row_end):
                for j in range(n_neighbors):
                    idx = current_indices[i, j]
                    if idx >= 0 and current_flags[i, j] == want_flag:
                        owner = idx // block_size
                        if owner != t and (phase == 0 or active[idx]):
                            bucket_bounds[t, owner + 1] += 1

        # 2. prefix sum: bucket (t, b) occupies bucket_bounds[t, b]:bucket_bounds[t, b + 1]
        total = 0
        for t in range(n_threads):
            for b in range(n_threads):
                count = bucket_bounds[t, b + 1]
                bucket_bounds[t, b] = total
                total += count
            bucket_bounds[t, n_threads] = total

        # 3. scan own rows: push local edges, scatter cross-block reverse edges
        for t in numba.prange(n_threads):
            local_rng_state = rng_state + (phase * n_threads + t)
            cursor = bucket_bounds[t, :n_threads].copy()
            row_start = t * block_size
            row_end = min(row_start + block_size, n_vertices)
            for i in range(row_start, row_end):
                for j in range(n_neighbors):
                    idx = current_indices[i, j]
                    if idx < 0 or current_flags[i, j] != want_flag:
                        continue
                    if phase == 0 or active[i]:
                        d = tau_rand(local_rng_state)
                        _push_candidate(cand_pri, cand_idx, i, d, idx)
                    if phase == 0 or active[idx]:
                        owner = idx // block_size
                        if owner == t:
                            d = tau_rand(local_rng_state)
                            _push_candidate(cand_pri, cand_idx, idx, d, i)
                        else:
                            k = cursor[owner]
                            reverse_dst[k] = idx
                            reverse_src[k] = i
                            cursor[owner] = k + 1

        # 4. each owner drains the buckets addressed to it, in producer order
        for b in numba.prange(n_threads):
            local_rng_state = rng_state + ((2 + phase) * n_threads + b)
            for t in range(n_threads):
                end = bucket_bounds[t, b + 1]
                for k in range(bucket_bounds[t, b], end):
                    d = tau_rand(local_rng_state)
                    _push_candidate(cand_pri, cand_idx, reverse_dst[k], d, reverse_src[k])

        if phase == 0:
            for i in numba.prange(n_vertices):
                has_new = 0
                for k in range(max_candidates):
                    if new_candidate_indices[i, k] >= 0:
                        has_new = 1
                active[i] = has_new

    for i in numba.prange(n_vertices):
        if active[i]:
            for j in range(n_neighbors):
                idx = current_indices[i, j]
                for k in range(max_candidates):
                    if new_candidate_indices[i, k] == idx:
                        current_flags[i, j] = 0
                        break

    return new_candidate_indices, old_candidate_indices
