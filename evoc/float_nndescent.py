import numba
import numpy as np

from .common_nndescent import (
    int32_bits_as_float32,
    float32_bits_as_int32,
    tau_rand_int,
    make_heap,
    deheap_sort,
    flagged_heap_push,
    build_candidates,
    build_candidates_owner,
    apply_graph_update_array,
    apply_sorted_graph_updates,
)
from .nested_parallelism import ENABLE_NESTED_PARALLELISM

# Used for a floating point "nearly zero" comparison
EPS = 1e-8
INF = np.finfo(np.float32).max
EXP_NEG_INF = np.finfo(np.float32).tiny
INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

point_indices_type = numba.int32[::1]


@numba.njit(
    [
        "f4(f4[::1],f4[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
            numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    locals={
        "result": numba.types.float32,
        "dim": numba.types.intp,
    },
    boundscheck=False,
    nogil=True,
    cache=True,
)
def fast_cosine(x, y):
    """
    Calculates the cosine similarity between two vectors.

    Args:
        x (numpy.ndarray): The first vector.
        y (numpy.ndarray): The second vector.

    Returns:
        float: The cosine similarity between x and y.
    """
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        result += x[i] * y[i]

    if result > 0.0:
        return -result
    else:
        return -EXP_NEG_INF


@numba.njit(
    numba.types.Tuple((numba.int32[::1], numba.int32[::1]))(
        numba.types.Array(numba.types.float32, 2, "C", readonly=True),
        numba.int32[::1],
        numba.int64[::1],
    ),
    locals={
        "n_left": numba.uint64,
        "n_right": numba.uint64,
        "left_data": numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        "right_data": numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        "test_data": numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        "hyperplane_vector": numba.float32[::1],
        "hyperplane_norm": numba.float32,
        "margin": numba.float32,
        "d": numba.uint32,
        "left_index": numba.uint32,
        "right_index": numba.uint32,
        "point_idx": numba.int32,
        "classification": numba.int8,
        "max_size": numba.uint32,
        "temp_left": numba.int32[::1],
        "temp_right": numba.int32[::1],
        "indices_size": numba.int32,
    },
    fastmath=True,
    nogil=True,
    cache=True,
    boundscheck=False,
)
def float_random_projection_split(data, indices, rng_state):
    """Given a set of ``graph_indices`` for graph_data points from ``graph_data``, create
    a random hyperplane to split the graph_data, returning two arrays graph_indices
    that fall on either side of the hyperplane. This is the basis for a
    random projection tree, which simply uses this splitting recursively.
    This particular split uses cosine distance to determine the hyperplane
    and which side each graph_data sample falls on.
    Parameters
    ----------
    data: array of shape (n_samples, n_features)
        The original graph_data to be split
    indices: array of shape (tree_node_size,)
        The graph_indices of the elements in the ``graph_data`` array that are to
        be split in the current operation.
    rng_state: array of int64, shape (3,)
        The internal state of the rng
    Returns
    -------
    indices_left: array
        The elements of ``graph_indices`` that fall on the "left" side of the
        random hyperplane.
    indices_right: array
        The elements of ``graph_indices`` that fall on the "left" side of the
        random hyperplane.
    """
    dim = data.shape[1]

    # Select two random points, set the hyperplane between them
    indices_size = np.int32(indices.shape[0])
    left_index = tau_rand_int(rng_state) % indices_size
    right_index = tau_rand_int(rng_state) % indices_size
    right_index += left_index == right_index
    right_index = right_index % indices_size
    left = indices[left_index]
    right = indices[right_index]
    left_data = data[left]
    right_data = data[right]

    # Compute the normal vector to the hyperplane (the vector between
    # the two points)
    hyperplane_vector = np.empty(dim, dtype=np.float32)
    hyperplane_norm = 0.0

    for d in range(dim):
        hyperplane_vector[d] = left_data[d] - right_data[d]
        hyperplane_norm += hyperplane_vector[d] * hyperplane_vector[d]

    hyperplane_norm = np.sqrt(hyperplane_norm)
    if abs(hyperplane_norm) < EPS:
        hyperplane_norm = 1.0

    # Normalize in the same vector (avoiding second loop when possible)
    for d in range(dim):
        hyperplane_vector[d] /= hyperplane_norm

    # Use temporary arrays sized for worst case, then trim
    max_size = np.uint32(indices.shape[0])
    temp_left = np.empty(max_size, dtype=np.int32)
    temp_right = np.empty(max_size, dtype=np.int32)
    n_left = 0
    n_right = 0

    # Single pass: classify points and directly populate result arrays
    for idx in range(indices.shape[0]):
        local_rng_state = rng_state + idx
        point_idx = indices[idx]
        test_data = data[point_idx]
        margin = 0.0

        # Compute margin (dot product with hyperplane normal)
        for d in range(dim):
            margin += hyperplane_vector[d] * test_data[d]

        # Classify point and directly assign to appropriate array
        if abs(margin) < EPS:
            classification = tau_rand_int(local_rng_state) % 2
        else:
            classification = 0 if margin > 0 else 1

        if classification == 0:
            temp_left[n_left] = point_idx
            n_left += 1
        else:
            temp_right[n_right] = point_idx
            n_right += 1

    # Handle degenerate case where all points end up on one side
    if n_left == 0 or n_right == 0:
        n_left = 0
        n_right = 0
        # Reassign randomly
        for idx in range(indices.shape[0]):
            point_idx = indices[idx]
            classification = tau_rand_int(rng_state) % 2
            if classification == 0:
                temp_left[n_left] = point_idx
                n_left += 1
            else:
                temp_right[n_right] = point_idx
                n_right += 1

    # Create final arrays with exact sizes (copy only what we need)
    indices_left = np.empty(n_left, dtype=np.int32)
    indices_right = np.empty(n_right, dtype=np.int32)

    for i in range(n_left):
        indices_left[i] = temp_left[i]
    for j in range(n_right):
        indices_right[j] = temp_right[j]

    return indices_left, indices_right


@numba.njit(
    numba.void(
        numba.types.Array(numba.types.float32, 2, "C", readonly=True),
        numba.int32[::1],
        numba.types.ListType(numba.int32[::1]),
        numba.int64[::1],
        numba.int64,
        numba.int64,
    ),
    nogil=True,
    locals={"left_indices": numba.int32[::1], "right_indices": numba.int32[::1]},
    cache=True,
)
def make_float_tree(
    data,
    indices,
    point_indices,
    rng_state,
    leaf_size=30,
    max_depth=200,
):
    """
    Recursively constructs a float tree for nearest neighbor descent.

    Args:
        data: The input data.
        indices: The indices of the data points to consider.
        point_indices: A list to store the indices of the points in each leaf node.
        rng_state: The random number generator state.
        leaf_size: The maximum number of points in a leaf node (default: 30).
        max_depth: The maximum depth of the tree (default: 200).

    Returns:
        None
    """
    if indices.shape[0] > leaf_size and max_depth > 0:
        (
            left_indices,
            right_indices,
        ) = float_random_projection_split(data, indices, rng_state)

        make_float_tree(
            data,
            left_indices,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        )

        make_float_tree(
            data,
            right_indices,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        )
    else:
        point_indices.append(indices)

    return


@numba.njit(
    numba.int32[:, ::1](
        numba.types.Array(numba.types.float32, 2, "C", readonly=True),
        numba.int64[::1],
        numba.int64,
        numba.int64,
    ),
    nogil=True,
    locals={
        "points": numba.int32[::1],
    },
    parallel=True,
    cache=True,
)
def make_float_leaf_array_parallel(data, rng_state, leaf_size=30, max_depth=200):
    indices = np.arange(data.shape[0]).astype(np.int32)

    point_indices = numba.typed.List.empty_list(numba.int32[::1])

    make_float_tree(
        data,
        indices,
        point_indices,
        rng_state,
        leaf_size,
        max_depth=max_depth,
    )

    n_leaves = numba.int64(len(point_indices))
    max_leaf_size = numba.int32(leaf_size)

    for i in numba.prange(n_leaves):
        points = point_indices[numba.int64(i)]
        max_leaf_size = max(max_leaf_size, numba.int32(len(points)))

    result = np.full((n_leaves, max_leaf_size), -1, dtype=np.int32)
    for i in numba.prange(n_leaves):
        points = point_indices[numba.int64(i)]
        n_points = numba.int32(len(points))
        result[i, :n_points] = points

    return result


@numba.njit(
    numba.int32[:, ::1](
        numba.types.Array(numba.types.float32, 2, "C", readonly=True),
        numba.int64[::1],
        numba.int64,
        numba.int64,
    ),
    nogil=True,
    locals={
        "points": numba.int32[::1],
    },
    parallel=False,
    cache=True,
)
def make_float_leaf_array_serial(data, rng_state, leaf_size=30, max_depth=200):
    indices = np.arange(data.shape[0]).astype(np.int32)

    point_indices = numba.typed.List.empty_list(numba.int32[::1])

    make_float_tree(
        data,
        indices,
        point_indices,
        rng_state,
        leaf_size,
        max_depth=max_depth,
    )

    n_leaves = numba.int64(len(point_indices))

    max_leaf_size = numba.int32(leaf_size)

    for i in range(n_leaves):
        points = point_indices[numba.int64(i)]
        max_leaf_size = max(max_leaf_size, numba.int32(len(points)))

    result = np.full((n_leaves, max_leaf_size), -1, dtype=np.int32)
    for i in range(n_leaves):
        points = point_indices[numba.int64(i)]
        n_points = numba.int32(len(points))
        result[i, :n_points] = points

    return result


@numba.njit(
    numba.types.List(numba.int32[:, ::1])(
        numba.types.Array(numba.types.float32, 2, "C", readonly=True),
        numba.int64[:, ::1],
        numba.uint64,
        numba.uint64,
    ),
    parallel=True,
    cache=True,
)
def make_float_forest_no_nested_parallelism(data, rng_states, leaf_size, max_depth):
    result = [np.empty((1, 1), dtype=np.int32)] * rng_states.shape[0]
    for i in numba.prange(len(result)):
        result[i] = make_float_leaf_array_serial(
            data, rng_states[i], leaf_size, max_depth=max_depth
        )
    return result


@numba.njit(
    numba.types.List(numba.int32[:, ::1])(
        numba.types.Array(numba.types.float32, 2, "C", readonly=True),
        numba.int64[:, ::1],
        numba.uint64,
        numba.uint64,
    ),
    parallel=True,
    cache=True,
)
def make_float_forest_with_nested_parallelism(data, rng_states, leaf_size, max_depth):
    result = [np.empty((1, 1), dtype=np.int32)] * rng_states.shape[0]
    for i in numba.prange(len(result)):
        result[i] = make_float_leaf_array_parallel(
            data, rng_states[i], leaf_size, max_depth=max_depth
        )
    return result


def make_float_forest(data, rng_states, leaf_size=30, max_depth=200):
    if ENABLE_NESTED_PARALLELISM:
        return make_float_forest_with_nested_parallelism(
            data, rng_states, leaf_size, max_depth
        )
    else:
        return make_float_forest_no_nested_parallelism(
            data, rng_states, leaf_size, max_depth
        )


@numba.njit(
    numba.float32[:, :, ::1](
        numba.float32[:, :, ::1],
        numba.int32[::1],
        numba.types.Array(numba.types.int32, 2, "C", readonly=True),
        numba.float32[:],
        numba.types.Array(numba.types.float32, 2, "C", readonly=True),
        numba.int64,
    ),
    parallel=True,
    locals={
        "d": numba.float32,
        "p": numba.int32,
        "q": numba.int32,
        "t": numba.uint16,
        "r": numba.uint32,
        "n": numba.uint32,
        "idx": numba.uint32,
        "data_p": numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        "max_threshold": numba.float32,
    },
    cache=True,
)
def generate_leaf_updates_float(
    updates, n_updates_per_thread, leaf_block, dist_thresholds, data, n_threads
):
    block_size = leaf_block.shape[0]
    rows_per_thread = (block_size // n_threads) + 1

    max_updates = updates.shape[1]
    for t in numba.prange(n_threads):
        idx = 0
        for r in range(rows_per_thread):
            n = t * rows_per_thread + r
            if n >= block_size:
                break

            for i in range(leaf_block.shape[1]):
                p = leaf_block[n, i]
                if p < 0:
                    break
                data_p = data[p]
                if idx < max_updates:
                    updates[t, idx, 0] = int32_bits_as_float32(p)
                    updates[t, idx, 1] = int32_bits_as_float32(p)
                    updates[t, idx, 2] = -1.0
                    idx += 1

                for j in range(
                    i + 1, leaf_block.shape[1]
                ):  # Start from i+1 to skip self-comparison
                    q = leaf_block[n, j]
                    if q < 0:
                        break

                    d = fast_cosine(data_p, data[q])
                    # Use max for better branch prediction than OR condition
                    max_threshold = max(dist_thresholds[p], dist_thresholds[q])
                    if d < max_threshold:
                        if idx < max_updates:
                            updates[t, idx, 0] = int32_bits_as_float32(p)
                            updates[t, idx, 1] = int32_bits_as_float32(q)
                            updates[t, idx, 2] = d
                            idx += 1

        n_updates_per_thread[t] = idx

    return updates


@numba.njit(
    [
        numba.void(
            numba.types.Array(numba.types.float32, 2, "C", readonly=True),
            numba.types.Tuple(
                (numba.int32[:, ::1], numba.float32[:, ::1], numba.uint8[:, ::1])
            ),
            numba.types.optional(
                numba.types.Array(numba.types.int32, 2, "C", readonly=True)
            ),
            numba.types.int32,
        ),
    ],
    locals={
        "d": numba.float32,
        "p": numba.int32,
        "q": numba.int32,
        "i": numba.uint16,
        "updates": numba.float32[:, :, ::1],
        "n_updates_per_thread": numba.int32[::1],
    },
    parallel=True,
    cache=True,
)
def init_rp_tree_float(data, current_graph, leaf_array, n_threads):
    n_leaves = leaf_array.shape[0]
    block_size = n_threads * 64
    n_blocks = n_leaves // block_size

    max_leaf_size = leaf_array.shape[1]
    updates_per_thread = (
        int(block_size * max_leaf_size * (max_leaf_size - 1) / (2 * n_threads)) + 1
    )
    updates = np.zeros((n_threads, updates_per_thread, 3), dtype=np.float32)
    n_updates_per_thread = np.zeros(n_threads, dtype=np.int32)
    n_vertices = current_graph[0].shape[0]
    vertex_block_size = n_vertices // n_threads + 1

    for i in range(n_blocks + 1):
        block_start = i * block_size
        block_end = min(n_leaves, (i + 1) * block_size)

        leaf_block = leaf_array[block_start:block_end]
        dist_thresholds = current_graph[1][:, 0]

        updates = generate_leaf_updates_float(
            updates, n_updates_per_thread, leaf_block, dist_thresholds, data, n_threads
        )

        for t in numba.prange(n_threads):
            block_start = t * vertex_block_size
            block_end = min(block_start + vertex_block_size, n_vertices)

            for j in range(n_threads):
                for k in range(n_updates_per_thread[j]):
                    p = float32_bits_as_int32(updates[j, k, 0])

                    if p == -1:
                        continue

                    q = float32_bits_as_int32(updates[j, k, 1])
                    d = np.float32(updates[j, k, 2])

                    if p >= block_start and p < block_end:
                        flagged_heap_push(
                            current_graph[1][p],
                            current_graph[0][p],
                            current_graph[2][p],
                            d,
                            q,
                        )
                    if q >= block_start and q < block_end:
                        flagged_heap_push(
                            current_graph[1][q],
                            current_graph[0][q],
                            current_graph[2][q],
                            d,
                            p,
                        )


@numba.njit(
    numba.types.void(
        numba.int32,
        numba.types.Array(numba.types.float32, 2, "C", readonly=True),
        numba.types.Tuple(
            (numba.int32[:, ::1], numba.float32[:, ::1], numba.uint8[:, ::1])
        ),
        numba.int64[::1],
    ),
    fastmath=True,
    parallel=True,
    locals={"d": numba.float32, "idx": numba.int32, "i": numba.int32},
    cache=True,
)
def init_random_float(n_neighbors, data, heap, rng_state):
    for i in numba.prange(data.shape[0]):
        local_rng_state = rng_state + i
        if heap[0][i, 0] < 0.0:
            for j in range(n_neighbors - np.sum(heap[0][i] >= 0.0)):
                idx = np.abs(tau_rand_int(local_rng_state)) % data.shape[0]
                if idx in heap[0][i]:
                    continue
                d = fast_cosine(data[idx], data[i])
                flagged_heap_push(heap[1][i], heap[0][i], heap[2][i], d, idx)

    return


@numba.njit(
    numba.types.void(
        numba.float32[:, :, ::1],
        numba.int32[::1],
        numba.int32[:, ::1],
        numba.int32[:, ::1],
        numba.float32[:],
        numba.types.Array(numba.types.float32, 2, "C", readonly=True),
        numba.int64,
    ),
    locals={
        "data_p": numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        "dist_thresh_p": numba.float32,
        "dist_thresh_q": numba.float32,
        "p": numba.int32,
        "q": numba.int32,
        "d": numba.float32,
        "max_updates": numba.int32,
        "threshold_check": numba.boolean,
        "max_threshold": numba.float32,
    },
    parallel=True,
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def generate_graph_update_array_float_basic(
    update_array,
    n_updates_per_thread,
    new_candidate_block,
    old_candidate_block,
    dist_thresholds,
    data,
    n_threads,
):
    """
    Basic optimized version with aggressive optimizations but without cache-specific enhancements.
    Kept for comparison and benchmarking purposes.
    """
    block_size = new_candidate_block.shape[0]
    max_new_candidates = new_candidate_block.shape[1]
    max_old_candidates = old_candidate_block.shape[1]
    rows_per_thread = (block_size // n_threads) + 1

    for t in numba.prange(n_threads):
        idx = 0
        max_updates = update_array.shape[1]

        for r in range(rows_per_thread):
            i = t * rows_per_thread + r
            if i >= block_size or idx >= max_updates:
                break

            for j in range(max_new_candidates):
                if idx >= max_updates:
                    break

                p = new_candidate_block[i, j]
                if p < 0:
                    continue
                data_p = data[p]
                dist_thresh_p = dist_thresholds[p]

                for k in range(j + 1, max_new_candidates):
                    if idx >= max_updates:
                        break

                    q = new_candidate_block[i, k]
                    if q < 0:
                        continue

                    # Compute distance once
                    d = fast_cosine(data_p, data[q])

                    # Use max for better branch prediction than OR condition
                    dist_thresh_q = dist_thresholds[q]
                    max_threshold = max(dist_thresh_p, dist_thresh_q)
                    threshold_check = d <= max_threshold

                    if threshold_check:
                        update_array[t, idx, 0] = int32_bits_as_float32(p)
                        update_array[t, idx, 1] = int32_bits_as_float32(q)
                        update_array[t, idx, 2] = d
                        idx += 1

                for k in range(max_old_candidates):
                    if idx >= max_updates:
                        break

                    q = old_candidate_block[i, k]
                    if q < 0:
                        continue

                    d = fast_cosine(data_p, data[q])
                    dist_thresh_q = dist_thresholds[q]
                    max_threshold = max(dist_thresh_p, dist_thresh_q)
                    threshold_check = d <= max_threshold

                    if threshold_check:
                        update_array[t, idx, 0] = int32_bits_as_float32(p)
                        update_array[t, idx, 1] = int32_bits_as_float32(q)
                        update_array[t, idx, 2] = d
                        idx += 1

        n_updates_per_thread[t] = idx


@numba.njit(
    numba.void(
        numba.float32[:, :, ::1],
        numba.int32[::1],
        numba.int32[:, ::1],
        numba.int32[:, ::1],
        numba.float32[:],
        numba.types.Array(numba.types.float32, 2, "C", readonly=True),
        numba.int64,
    ),
    locals={
        "data_p": numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        "dist_thresh_p": numba.float32,
        "dist_thresh_q": numba.float32,
        "p": numba.int32,
        "q": numba.int32,
        "d": numba.float32,
        "max_updates": numba.int32,
        "threshold_check": numba.boolean,
        "working_set_size": numba.int32,
        "batch_start": numba.int32,
        "batch_end": numba.int32,
        "max_threshold": numba.float32,
    },
    parallel=True,
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def generate_graph_update_array_float(
    update_array,
    n_updates_per_thread,
    new_candidate_block,
    old_candidate_block,
    dist_thresholds,
    data,
    n_threads,
):
    """
    Optimized version using working set approach that processes candidates in small groups
    that fit well in CPU cache. This reduces cache misses by keeping frequently
    accessed data vectors in cache longer, providing the best performance for typical workloads.
    """
    block_size = new_candidate_block.shape[0]
    max_new_candidates = new_candidate_block.shape[1]
    max_old_candidates = old_candidate_block.shape[1]
    rows_per_thread = (block_size // n_threads) + 1

    # Working set size - process this many candidates at a time
    # Tuned for typical L1/L2 cache sizes (adjust based on data dimensionality)
    working_set_size = 8

    for t in numba.prange(n_threads):
        idx = 0
        max_updates = update_array.shape[1]

        for r in range(rows_per_thread):
            i = t * rows_per_thread + r
            if i >= block_size or idx >= max_updates:
                break

            # Process new candidates in working set chunks
            new_start = 0
            while new_start < max_new_candidates and idx < max_updates:
                new_end = min(new_start + working_set_size, max_new_candidates)

                # Process pairs within this working set
                for j in range(new_start, new_end):
                    if idx >= max_updates:
                        break

                    p = new_candidate_block[i, j]
                    if p < 0:
                        continue

                    data_p = data[p]
                    dist_thresh_p = dist_thresholds[p]

                    # Compare with other candidates in the same working set
                    for k in range(j + 1, new_end):
                        if idx >= max_updates:
                            break

                        q = new_candidate_block[i, k]
                        if q < 0:
                            continue

                        d = fast_cosine(data_p, data[q])
                        dist_thresh_q = dist_thresholds[q]
                        max_threshold = max(dist_thresh_p, dist_thresh_q)
                        threshold_check = d <= max_threshold

                        if threshold_check:
                            update_array[t, idx, 0] = int32_bits_as_float32(p)
                            update_array[t, idx, 1] = int32_bits_as_float32(q)
                            update_array[t, idx, 2] = d
                            idx += 1

                    # Compare with candidates in future working sets
                    for k in range(new_end, max_new_candidates):
                        if idx >= max_updates:
                            break

                        q = new_candidate_block[i, k]
                        if q < 0:
                            continue

                        d = fast_cosine(data_p, data[q])
                        dist_thresh_q = dist_thresholds[q]
                        max_threshold = max(dist_thresh_p, dist_thresh_q)
                        threshold_check = d <= max_threshold

                        if threshold_check:
                            update_array[t, idx, 0] = int32_bits_as_float32(p)
                            update_array[t, idx, 1] = int32_bits_as_float32(q)
                            update_array[t, idx, 2] = d
                            idx += 1

                    # Compare with old candidates in working set chunks
                    old_start = 0
                    while old_start < max_old_candidates and idx < max_updates:
                        old_end = min(old_start + working_set_size, max_old_candidates)

                        for k in range(old_start, old_end):
                            if idx >= max_updates:
                                break

                            q = old_candidate_block[i, k]
                            if q < 0:
                                continue

                            d = fast_cosine(data_p, data[q])
                            dist_thresh_q = dist_thresholds[q]
                            max_threshold = max(dist_thresh_p, dist_thresh_q)
                            threshold_check = d <= max_threshold

                            if threshold_check:
                                update_array[t, idx, 0] = int32_bits_as_float32(p)
                                update_array[t, idx, 1] = int32_bits_as_float32(q)
                                update_array[t, idx, 2] = d
                                idx += 1

                        old_start = old_end

                new_start = new_end

        n_updates_per_thread[t] = idx


@numba.njit(
    numba.void(
        numba.float32[:, :, ::1],
        numba.int32[:, ::1],
        numba.int32[:, ::1],
        numba.int32[:, ::1],
        numba.float32[:],
        numba.types.Array(numba.types.float32, 2, "C", readonly=True),
        numba.int64,
    ),
    locals={
        "data_p": numba.types.Array(numba.types.float32, 1, "C", readonly=True),
        "dist_thresh_p": numba.float32,
        "dist_thresh_q": numba.float32,
        "p": numba.int32,
        "q": numba.int32,
        "d": numba.float32,
        "max_updates": numba.intp,
        "threshold_check": numba.boolean,
        "max_threshold": numba.float32,
        "p_block": numba.int32,
        "q_block": numba.int32,
        "p_idx": numba.int32,
        "q_idx": numba.int32,
    },
    parallel=True,
    cache=True,
    fastmath=True,
    boundscheck=False,
)
def generate_sorted_graph_update_array_float(
    update_array,
    n_updates_per_block,
    new_candidate_block,
    old_candidate_block,
    dist_thresholds,
    data,
    n_threads,
):
    """Generate graph updates pre-sorted by target block.

    Updates are bucketed by their target vertex block so that
    apply_sorted_graph_updates can process each bucket with perfect data
    locality. Each update (p, q, d) is placed in both p's bucket and q's bucket
    (if different). update_array has shape (n_threads, max_updates_per_block, 3);
    each source thread writes its own segment of every bucket.

    Each new candidate p is compared with the later new candidates and then the
    old candidates, in that order. Distances are computed four at a time,
    sharing the loads of p, and the updates are then emitted in the same order
    as comparing one at a time.
    """
    block_size_candidates = new_candidate_block.shape[0]
    max_new_candidates = new_candidate_block.shape[1]
    max_old_candidates = old_candidate_block.shape[1]
    rows_per_thread = (block_size_candidates // n_threads) + 1
    n_vertices = data.shape[0]
    vertex_block_size = n_vertices // n_threads + 1
    max_updates = update_array.shape[1]
    max_updates_per_src_thread = max_updates // n_threads
    dim = data.shape[1]

    for b in numba.prange(n_threads):
        for t in range(n_threads + 1):
            n_updates_per_block[b, t] = 0

    for t in numba.prange(n_threads):
        local_counts = np.zeros(n_threads, dtype=np.int32)
        partners = np.empty(4, dtype=np.int32)
        partner_dists = np.empty(4, dtype=np.float32)

        for r in range(rows_per_thread):
            i = t * rows_per_thread + r
            if i >= block_size_candidates:
                break

            for j in range(max_new_candidates):
                p = new_candidate_block[i, j]
                if p < 0:
                    continue

                data_p = data[p]
                dist_thresh_p = dist_thresholds[p]
                p_block = p // vertex_block_size
                if p_block >= n_threads:
                    p_block = n_threads - 1

                # Partners: later new candidates, then old candidates
                n_new_after = max_new_candidates - j - 1
                n_partners = n_new_after + max_old_candidates
                c = 0
                while c < n_partners:
                    n_found = 0
                    while c < n_partners and n_found < 4:
                        if c < n_new_after:
                            q = new_candidate_block[i, j + 1 + c]
                        else:
                            q = old_candidate_block[i, c - n_new_after]
                        c += 1
                        if q >= 0:
                            partners[n_found] = q
                            n_found += 1

                    if n_found == 4:
                        y0 = data[partners[0]]
                        y1 = data[partners[1]]
                        y2 = data[partners[2]]
                        y3 = data[partners[3]]
                        s0 = np.float32(0.0)
                        s1 = np.float32(0.0)
                        s2 = np.float32(0.0)
                        s3 = np.float32(0.0)
                        for dd in range(dim):
                            xv = data_p[dd]
                            s0 += xv * y0[dd]
                            s1 += xv * y1[dd]
                            s2 += xv * y2[dd]
                            s3 += xv * y3[dd]
                        # Same transform as fast_cosine
                        partner_dists[0] = -s0 if s0 > 0.0 else -EXP_NEG_INF
                        partner_dists[1] = -s1 if s1 > 0.0 else -EXP_NEG_INF
                        partner_dists[2] = -s2 if s2 > 0.0 else -EXP_NEG_INF
                        partner_dists[3] = -s3 if s3 > 0.0 else -EXP_NEG_INF
                    else:
                        for rr in range(n_found):
                            partner_dists[rr] = fast_cosine(data_p, data[partners[rr]])

                    for rr in range(n_found):
                        q = partners[rr]
                        d = partner_dists[rr]
                        dist_thresh_q = dist_thresholds[q]
                        max_threshold = max(dist_thresh_p, dist_thresh_q)

                        if d <= max_threshold:
                            q_block = q // vertex_block_size
                            if q_block >= n_threads:
                                q_block = n_threads - 1

                            bucket_idx = local_counts[p_block]
                            write_idx = t * max_updates_per_src_thread + bucket_idx
                            if bucket_idx < max_updates_per_src_thread:
                                update_array[p_block, write_idx, 0] = int32_bits_as_float32(p)
                                update_array[p_block, write_idx, 1] = int32_bits_as_float32(q)
                                update_array[p_block, write_idx, 2] = d
                                local_counts[p_block] += 1

                            # If q is in a different block, also place in q's bucket
                            if q_block != p_block:
                                bucket_idx = local_counts[q_block]
                                write_idx = t * max_updates_per_src_thread + bucket_idx
                                if bucket_idx < max_updates_per_src_thread:
                                    update_array[q_block, write_idx, 0] = int32_bits_as_float32(p)
                                    update_array[q_block, write_idx, 1] = int32_bits_as_float32(q)
                                    update_array[q_block, write_idx, 2] = d
                                    local_counts[q_block] += 1

        # Record total updates generated by this thread for each bucket
        for b in range(n_threads):
            n_updates_per_block[b, t + 1] = local_counts[b]


def nn_descent_float(
    data,
    n_neighbors,
    rng_state,
    max_candidates=50,
    n_iters=10,
    delta=0.001,
    delta_improv=None,
    leaf_array=None,
    verbose=False,
):
    """
    Perform approximate nearest neighbor descent algorithm using float data.

    Parameters:
    - data: The input data array.
    - n_neighbors: The number of nearest neighbors to search for.
    - rng_state: The random number generator state.
    - max_candidates: The maximum number of candidates to consider during the search. Default is 50.
    - n_iters: The number of iterations to perform. Default is 10.
    - delta: The stopping threshold based on update count. Default is 0.001.
    - delta_improv: Optional stopping threshold based on relative improvement in total
        graph distance. When set (e.g., 0.001 for 0.1%), the algorithm will also
        terminate when the relative improvement in sum of all distances drops below
        this threshold. This can provide earlier termination on data with good
        structure, adapting to the intrinsic difficulty of the dataset. Default is None
        (disabled).
    - leaf_array: The array representing the leaf structure of the RP-tree. Default is None.
    - verbose: Whether to print progress information. Default is False.

    Returns:
    - The sorted nearest neighbor graph.
    """
    n_threads = numba.get_num_threads()
    current_graph = make_heap(data.shape[0], n_neighbors)
    init_rp_tree_float(data, current_graph, leaf_array, n_threads)
    init_random_float(n_neighbors, data, current_graph, rng_state)

    n_vertices = data.shape[0]
    n_threads = numba.get_num_threads()
    block_size = 65536 // n_threads
    n_blocks = n_vertices // block_size

    max_updates_per_thread = int(
        ((max_candidates**2 + max_candidates * (max_candidates - 1) / 2) * block_size)
    )
    update_array = np.empty((n_threads, max_updates_per_thread, 3), dtype=np.float32)
    n_updates_per_thread = np.zeros(n_threads, dtype=np.int32)

    # For distance-based termination
    prev_sum_dist = None

    # Candidate workspace, reused across iterations; build_candidates_owner only
    # resets the rows it wrote on the previous iteration.
    _n_vertices = current_graph[0].shape[0]
    _n_neighbors = current_graph[0].shape[1]
    new_candidate_indices = np.full((_n_vertices, max_candidates), -1, dtype=np.int32)
    new_candidate_priority = np.full(
        (_n_vertices, max_candidates), np.inf, dtype=np.float32
    )
    old_candidate_indices = np.full((_n_vertices, max_candidates), -1, dtype=np.int32)
    old_candidate_priority = np.full(
        (_n_vertices, max_candidates), np.inf, dtype=np.float32
    )
    candidate_active = np.ones(_n_vertices, dtype=np.uint8)
    # Cross-block reverse edges, bucketed by owning thread (at most one per edge)
    reverse_dst = np.empty(_n_vertices * _n_neighbors, dtype=np.int32)
    reverse_src = np.empty(_n_vertices * _n_neighbors, dtype=np.int32)
    bucket_bounds = np.zeros((n_threads, n_threads + 1), dtype=np.int64)
    for n in range(n_iters):
        if verbose:
            print("\t", n + 1, " / ", n_iters)

        (new_candidate_neighbors, old_candidate_neighbors) = build_candidates_owner(
            current_graph,
            max_candidates,
            rng_state,
            n_threads,
            new_candidate_indices,
            new_candidate_priority,
            old_candidate_indices,
            old_candidate_priority,
            candidate_active,
            reverse_dst,
            reverse_src,
            bucket_bounds,
        )

        c = 0
        n_vertices = new_candidate_neighbors.shape[0]
        for i in range(n_blocks + 1):
            block_start = i * block_size
            block_end = min(n_vertices, (i + 1) * block_size)

            new_candidate_block = new_candidate_neighbors[block_start:block_end]
            old_candidate_block = old_candidate_neighbors[block_start:block_end]

            dist_thresholds = current_graph[1][:, 0]

            generate_graph_update_array_float(
                update_array,
                n_updates_per_thread,
                new_candidate_block,
                old_candidate_block,
                dist_thresholds,
                data,
                n_threads,
            )

            c += apply_graph_update_array(
                current_graph, update_array, n_updates_per_thread, n_threads
            )

        # Check update count termination
        if c <= delta * n_neighbors * data.shape[0]:
            if verbose:
                print("\tStopping threshold met -- exiting after", n + 1, "iterations")
            return deheap_sort(current_graph[0], current_graph[1])

        # Check distance improvement termination (if enabled)
        if delta_improv is not None:
            all_distances = current_graph[1]
            valid_mask = all_distances < INF
            sum_dist = np.sum(all_distances[valid_mask])

            if prev_sum_dist is not None:
                rel_improv = abs(sum_dist - prev_sum_dist) / abs(prev_sum_dist)
                if rel_improv < delta_improv:
                    if verbose:
                        print(
                            f"\tDistance improvement threshold met ({rel_improv:.4%} < {delta_improv:.4%})"
                            f" -- exiting after {n + 1} iterations"
                        )
                    return deheap_sort(current_graph[0], current_graph[1])

            prev_sum_dist = sum_dist

        block_size = min(n_vertices, 2 * block_size)
        n_blocks = n_vertices // block_size

    return deheap_sort(current_graph[0], current_graph[1])


def nn_descent_float_sorted(
    data,
    n_neighbors,
    rng_state,
    max_candidates=50,
    n_iters=10,
    delta=0.001,
    delta_improv=None,
    leaf_array=None,
    verbose=False,
):
    """
    Perform approximate nearest neighbor descent algorithm using float data.

    This version uses pre-sorted updates bucketed by target block for potentially
    better performance when n_threads is large. Each thread only processes updates
    targeting its own vertex block.

    Parameters:
    - data: The input data array.
    - n_neighbors: The number of nearest neighbors to search for.
    - rng_state: The random number generator state.
    - max_candidates: The maximum number of candidates to consider during the search. Default is 50.
    - n_iters: The number of iterations to perform. Default is 10.
    - delta: The stopping threshold based on update count. Default is 0.001.
    - delta_improv: Optional stopping threshold based on relative improvement in total
        graph distance. When set (e.g., 0.001 for 0.1%), the algorithm will also
        terminate when the relative improvement in sum of all distances drops below
        this threshold. This can provide earlier termination on data with good
        structure, adapting to the intrinsic difficulty of the dataset. Default is None
        (disabled).
    - leaf_array: The array representing the leaf structure of the RP-tree. Default is None.
    - verbose: Whether to print progress information. Default is False.

    Returns:
    - The sorted nearest neighbor graph.
    """
    n_threads = numba.get_num_threads()
    current_graph = make_heap(data.shape[0], n_neighbors)
    init_rp_tree_float(data, current_graph, leaf_array, n_threads)
    init_random_float(n_neighbors, data, current_graph, rng_state)

    n_vertices = data.shape[0]
    n_threads = numba.get_num_threads()
    block_size = 65536 // n_threads
    n_blocks = n_vertices // block_size

    max_updates_per_thread = int(
        ((max_candidates**2 + max_candidates * (max_candidates - 1) / 2) * block_size)
    )
    # For sorted updates: shape is (n_threads, max_updates_per_block, 3)
    # Each bucket (first dim) holds updates targeting that block
    sorted_update_array = np.empty(
        (n_threads, max_updates_per_thread, 3), dtype=np.float32
    )
    # Track updates per block, with per-thread breakdown: (n_threads, n_threads + 1)
    # Column 0 is unused, columns 1..n_threads store count from each generating thread
    n_updates_per_block = np.zeros((n_threads, n_threads + 1), dtype=np.int32)

    # For distance-based termination
    prev_sum_dist = None

    # Candidate workspace, reused across iterations; build_candidates_owner only
    # resets the rows it wrote on the previous iteration.
    _n_vertices = current_graph[0].shape[0]
    _n_neighbors = current_graph[0].shape[1]
    new_candidate_indices = np.full((_n_vertices, max_candidates), -1, dtype=np.int32)
    new_candidate_priority = np.full(
        (_n_vertices, max_candidates), np.inf, dtype=np.float32
    )
    old_candidate_indices = np.full((_n_vertices, max_candidates), -1, dtype=np.int32)
    old_candidate_priority = np.full(
        (_n_vertices, max_candidates), np.inf, dtype=np.float32
    )
    candidate_active = np.ones(_n_vertices, dtype=np.uint8)
    # Cross-block reverse edges, bucketed by owning thread (at most one per edge)
    reverse_dst = np.empty(_n_vertices * _n_neighbors, dtype=np.int32)
    reverse_src = np.empty(_n_vertices * _n_neighbors, dtype=np.int32)
    bucket_bounds = np.zeros((n_threads, n_threads + 1), dtype=np.int64)
    for n in range(n_iters):
        if verbose:
            print("\t", n + 1, " / ", n_iters)

        (new_candidate_neighbors, old_candidate_neighbors) = build_candidates_owner(
            current_graph,
            max_candidates,
            rng_state,
            n_threads,
            new_candidate_indices,
            new_candidate_priority,
            old_candidate_indices,
            old_candidate_priority,
            candidate_active,
            reverse_dst,
            reverse_src,
            bucket_bounds,
        )

        c = 0
        n_vertices = new_candidate_neighbors.shape[0]
        for i in range(n_blocks + 1):
            block_start = i * block_size
            block_end = min(n_vertices, (i + 1) * block_size)

            new_candidate_block = new_candidate_neighbors[block_start:block_end]
            old_candidate_block = old_candidate_neighbors[block_start:block_end]

            dist_thresholds = current_graph[1][:, 0]

            # Reset update counts for this iteration
            n_updates_per_block.fill(0)

            generate_sorted_graph_update_array_float(
                sorted_update_array,
                n_updates_per_block,
                new_candidate_block,
                old_candidate_block,
                dist_thresholds,
                data,
                n_threads,
            )

            c += apply_sorted_graph_updates(
                current_graph, sorted_update_array, n_updates_per_block, n_threads
            )

        # Check update count termination
        if c <= delta * n_neighbors * data.shape[0]:
            if verbose:
                print("\tStopping threshold met -- exiting after", n + 1, "iterations")
            return deheap_sort(current_graph[0], current_graph[1])

        # Check distance improvement termination (if enabled)
        if delta_improv is not None:
            all_distances = current_graph[1]
            valid_mask = all_distances < INF
            sum_dist = np.sum(all_distances[valid_mask])

            if prev_sum_dist is not None:
                rel_improv = abs(sum_dist - prev_sum_dist) / abs(prev_sum_dist)
                if rel_improv < delta_improv:
                    if verbose:
                        print(
                            f"\tDistance improvement threshold met ({rel_improv:.4%} < {delta_improv:.4%})"
                            f" -- exiting after {n + 1} iterations"
                        )
                    return deheap_sort(current_graph[0], current_graph[1])

            prev_sum_dist = sum_dist

        block_size = min(n_vertices, 2 * block_size)
        n_blocks = n_vertices // block_size

    return deheap_sort(current_graph[0], current_graph[1])
