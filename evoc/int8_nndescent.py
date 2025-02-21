import numba
import numpy as np

from .common_nndescent import (
    tau_rand_int,
    make_heap,
    deheap_sort,
    flagged_heap_push,
    build_candidates,
    apply_graph_update_array,
)

# Used for a floating point "nearly zero" comparison
EPS = 1e-8
INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

point_indices_type = numba.int32[::1]


@numba.njit(
    [
        "f4(i1[::1],i1[::1])",
        numba.types.float32(
            numba.types.Array(numba.types.int8, 1, "C", readonly=True),
            numba.types.Array(numba.types.int8, 1, "C", readonly=True),
        ),
    ],
    fastmath=True,
    locals={
        "result": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.uint16,
    },
    cache=True,
)
def fast_int_inner_product_dissimilarity(x, y):
    result = 0.0
    dim = x.shape[0]

    for i in range(dim):
        result += x[i] * y[i]

    return -result


@numba.njit(
    numba.types.Tuple((numba.int32[::1], numba.int32[::1]))(
        numba.types.Array(numba.types.int8, 2, "C", readonly=True), numba.int32[::1], numba.int64[::1]
    ),
    locals={
        "n_left": numba.uint32,
        "n_right": numba.uint32,
        "left_data": numba.types.Array(numba.types.int8, 1, "C", readonly=True),
        "right_data": numba.types.Array(numba.types.int8, 1, "C", readonly=True),
        "test_data": numba.types.Array(numba.types.int8, 1, "C", readonly=True),
        "hyperplane_vector": numba.float32[::1],
        "margin": numba.float32,
        "d": numba.uint32,
        "i": numba.uint32,
        "left_index": numba.uint32,
        "right_index": numba.uint32,
    },
    fastmath=True,
    nogil=True,
    cache=False,
)
def int8_random_projection_split(data, indices, rng_state):
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
    left_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index = tau_rand_int(rng_state) % indices.shape[0]
    right_index += left_index == right_index
    right_index = right_index % indices.shape[0]
    left = indices[left_index]
    right = indices[right_index]

    left_data = data[left]
    right_data = data[right]

    left_norm = 0.0
    right_norm = 0.0
    for d in range(dim):
        left_norm += left_data[d] * left_data[d]
        right_norm += right_data[d] * right_data[d]

    left_norm = np.sqrt(left_norm)
    right_norm = np.sqrt(right_norm)

    # Compute the normal vector to the hyperplane (the vector between
    # the two points)
    hyperplane_vector = np.empty(dim, dtype=np.float32)
    hyperplane_norm = 0.0

    for d in range(dim):
        hyperplane_vector[d] = (left_data[d] / left_norm) - (right_data[d] / right_norm)
        hyperplane_norm += hyperplane_vector[d] * hyperplane_vector[d]
    hyperplane_norm = np.sqrt(hyperplane_norm)

    # hyperplane_norm = norm(hyperplane_vector)
    if abs(hyperplane_norm) < EPS:
        hyperplane_norm = 1.0

    for d in range(dim):
        hyperplane_vector[d] /= hyperplane_norm

    # For each point compute the margin (project into normal vector)
    # If we are on lower side of the hyperplane put in one pile, otherwise
    # put it in the other pile (if we hit hyperplane on the nose, flip a coin)
    n_left = 0
    n_right = 0
    side = np.empty(indices.shape[0], np.bool_)
    for i in range(indices.shape[0]):
        margin = 0.0
        local_rng_state = rng_state + np.int64(i)
        test_data = data[indices[i]]
        for d in range(dim):
            margin += hyperplane_vector[d] * test_data[d]

        if abs(margin) < EPS:
            side[i] = np.bool_(tau_rand_int(local_rng_state) % 2)
            if side[i] == 0:
                n_left += 1
            else:
                n_right += 1
        elif margin > 0:
            side[i] = 0
            n_left += 1
        else:
            side[i] = 1
            n_right += 1

    # If all points end up on one side, something went wrong numerically
    # In this case, assign points randomly; they are likely very close anyway
    if n_left == 0 or n_right == 0:
        n_left = 0
        n_right = 0
        for i in range(indices.shape[0]):
            side[i] = tau_rand_int(rng_state) % 2
            if side[i] == 0:
                n_left += 1
            else:
                n_right += 1

    # Now that we have the counts allocate arrays
    indices_left = np.empty(n_left, dtype=np.int32)
    indices_right = np.empty(n_right, dtype=np.int32)

    # Populate the arrays with graph_indices according to which side they fell on
    n_left = 0
    n_right = 0
    for i in range(side.shape[0]):
        if side[i] == 0:
            indices_left[n_left] = indices[i]
            n_left += 1
        else:
            indices_right[n_right] = indices[i]
            n_right += 1

    return indices_left, indices_right


@numba.njit(
    numba.void(
        numba.types.Array(numba.types.int8, 2, "C", readonly=True),
        numba.int32[::1],
        numba.types.ListType(numba.int32[::1]),
        numba.int64[::1],
        numba.int64,
        numba.int64,
    ),
    nogil=True,
    cache=False,
)
def make_int8_tree(
    data,
    indices,
    point_indices,
    rng_state,
    leaf_size=30,
    max_depth=200,
):
    if indices.shape[0] > leaf_size and max_depth > 0:
        (
            left_indices,
            right_indices,
        ) = int8_random_projection_split(data, indices, rng_state)

        make_int8_tree(
            data,
            left_indices,
            point_indices,
            rng_state,
            leaf_size,
            max_depth - 1,
        )

        make_int8_tree(
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
        numba.types.Array(numba.types.int8, 2, "C", readonly=True),
        numba.int64[::1],
        numba.int64,
        numba.int64,
    ),
    nogil=True,
    locals={"n_leaves": numba.uint32, "i": numba.uint32},
    parallel=True,
    cache=False,
)
def make_int8_leaf_array(data, rng_state, leaf_size=30, max_depth=200):
    indices = np.arange(data.shape[0]).astype(np.int32)

    point_indices = numba.typed.List.empty_list(point_indices_type)

    make_int8_tree(
        data,
        indices,
        point_indices,
        rng_state,
        leaf_size,
        max_depth=max_depth,
    )

    n_leaves = len(point_indices)

    max_leaf_size = leaf_size
    for i in numba.prange(n_leaves):
        points = point_indices[i]
        max_leaf_size = max(max_leaf_size, numba.int32(len(points)))

    result = np.full((n_leaves, max_leaf_size), -1, dtype=np.int32)
    for i in numba.prange(n_leaves):
        points = point_indices[i]
        leaf_size = numba.int32(len(points))
        result[i, :leaf_size] = points

    return result


@numba.njit(
    numba.types.List(numba.int32[:, ::1])(
        numba.types.Array(numba.types.int8, 2, "C", readonly=True),
        numba.int64[:, ::1],
        numba.int64,
        numba.int64,
    ),
    parallel=True,
    cache=False,
)
def make_int8_forest(data, rng_states, leaf_size, max_depth):
    result = [np.empty((1, 1), dtype=np.int32)] * rng_states.shape[0]
    for i in numba.prange(len(result)):
        result[i] = make_int8_leaf_array(
            data, rng_states[i], leaf_size, max_depth=max_depth
        )
    return result


@numba.njit(
    numba.float32[:, :, ::1](
        numba.float32[:, :, ::1],
        numba.int32[::1],
        numba.types.Array(numba.types.int32, 2, "C", readonly=True),
        numba.float32[:],
        numba.types.Array(numba.types.int8, 2, "C", readonly=True),
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
        "data_p": numba.types.Array(numba.types.int8, 1, "C", readonly=True),
    },
    cache=True,
)
def generate_leaf_updates_int8(
    updates, n_updates_per_thread, leaf_block, dist_thresholds, data, n_threads
):

    block_size = leaf_block.shape[0]
    rows_per_thread = (block_size // n_threads) + 1

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

                for j in range(i, leaf_block.shape[1]):
                    q = leaf_block[n, j]
                    if q < 0:
                        break

                    d = fast_int_inner_product_dissimilarity(data_p, data[q])
                    if d < dist_thresholds[p] or d < dist_thresholds[q]:
                        updates[t, idx, 0] = p
                        updates[t, idx, 1] = q
                        updates[t, idx, 2] = d
                        idx += 1

        n_updates_per_thread[t] = idx

    return updates


@numba.njit(
    numba.void(
        numba.types.Array(numba.types.int8, 2, "C", readonly=True),
        numba.types.Tuple(
            (numba.int32[:, ::1], numba.float32[:, ::1], numba.uint8[:, ::1])
        ),
        numba.types.Array(numba.types.int32, 2, "C", readonly=True),
        numba.types.int32,
    ),
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
def init_rp_tree_int8(data, current_graph, leaf_array, n_threads):

    n_leaves = leaf_array.shape[0]
    block_size = 64
    n_blocks = n_leaves // block_size

    max_leaf_size = leaf_array.shape[1]
    updates_per_thread = (
        int(block_size * max_leaf_size * (max_leaf_size - 1) / (2 * n_threads)) + 1
    )
    updates = np.zeros((n_threads, updates_per_thread, 3), dtype=np.float32)
    n_updates_per_thread = np.zeros(n_threads, dtype=np.int32)

    for i in range(n_blocks + 1):
        block_start = i * block_size
        block_end = min(n_leaves, (i + 1) * block_size)

        leaf_block = leaf_array[block_start:block_end]
        dist_thresholds = current_graph[1][:, 0]

        updates = generate_leaf_updates_int8(
            updates, n_updates_per_thread, leaf_block, dist_thresholds, data, n_threads
        )

        for t in numba.prange(n_threads):
            for j in range(n_threads):
                for k in range(n_updates_per_thread[j]):
                    p = np.int32(updates[j, k, 0])
                    q = np.int32(updates[j, k, 1])
                    d = np.float32(updates[j, k, 2])

                    if p == -1 or q == -1:
                        continue

                    if p % n_threads == t:
                        flagged_heap_push(
                            current_graph[1][p],
                            current_graph[0][p],
                            current_graph[2][p],
                            d,
                            q,
                        )
                    if q % n_threads == t:
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
        numba.types.Array(numba.types.int8, 2, "C", readonly=True),
        numba.types.Tuple(
            (numba.int32[:, ::1], numba.float32[:, ::1], numba.uint8[:, ::1])
        ),
        numba.int64[::1],
    ),
    fastmath=True,
    locals={"d": numba.float32, "idx": numba.int32, "i": numba.int32},
    cache=True,
)
def init_random_int8(n_neighbors, data, heap, rng_state):
    for i in range(data.shape[0]):
        if heap[0][i, 0] < 0.0:
            for j in range(n_neighbors - np.sum(heap[0][i] >= 0.0)):
                idx = np.abs(tau_rand_int(rng_state)) % data.shape[0]
                if idx in heap[0][i]:
                    continue
                d = fast_int_inner_product_dissimilarity(data[idx], data[i])
                flagged_heap_push(heap[1][i], heap[0][i], heap[2][i], d, idx)

    return


@numba.njit(
    numba.types.void(
        numba.float32[:, :, ::1],
        numba.int32[::1],
        numba.int32[:, ::1],
        numba.int32[:, ::1],
        numba.float32[:],
        numba.types.Array(numba.types.int8, 2, "C", readonly=True),
        numba.int64,
    ),
    locals={
        "data_p": numba.types.Array(numba.types.int8, 1, "C", readonly=True),
    },
    parallel=True,
    cache=True,
)
def generate_graph_update_array_int8(
    update_array,
    n_updates_per_thread,
    new_candidate_block,
    old_candidate_block,
    dist_thresholds,
    data,
    n_threads,
):

    block_size = new_candidate_block.shape[0]
    max_new_candidates = new_candidate_block.shape[1]
    max_old_candidates = old_candidate_block.shape[1]
    rows_per_thread = (block_size // n_threads) + 1

    for t in numba.prange(n_threads):
        idx = 0
        updates_are_full = False
        for r in range(rows_per_thread):
            i = t * rows_per_thread + r
            if i >= block_size:
                break

            for j in range(max_new_candidates):
                p = int(new_candidate_block[i, j])
                if p < 0:
                    continue
                data_p = data[p]

                for k in range(j, max_new_candidates):
                    q = int(new_candidate_block[i, k])
                    if q < 0:
                        continue

                    d = fast_int_inner_product_dissimilarity(data_p, data[q])
                    if d <= dist_thresholds[p] or d <= dist_thresholds[q]:
                        update_array[t, idx, 0] = p
                        update_array[t, idx, 1] = q
                        update_array[t, idx, 2] = d
                        idx += 1
                        if idx >= update_array.shape[1]:
                            updates_are_full = True
                            break

                if updates_are_full:
                    break

                for k in range(max_old_candidates):
                    q = int(old_candidate_block[i, k])
                    if q < 0:
                        continue

                    d = fast_int_inner_product_dissimilarity(data_p, data[q])
                    if d <= dist_thresholds[p] or d <= dist_thresholds[q]:
                        update_array[t, idx, 0] = p
                        update_array[t, idx, 1] = q
                        update_array[t, idx, 2] = d
                        idx += 1
                        if idx >= update_array.shape[1]:
                            updates_are_full = True
                            break

                if updates_are_full:
                    break

            if updates_are_full:
                break

        n_updates_per_thread[t] = idx


def nn_descent_int8(
    data,
    n_neighbors,
    rng_state,
    max_candidates=50,
    n_iters=10,
    delta=0.001,
    leaf_array=None,
    verbose=False,
):
    n_threads = numba.get_num_threads()
    current_graph = make_heap(data.shape[0], n_neighbors)
    init_rp_tree_int8(data, current_graph, leaf_array, n_threads)
    init_random_int8(n_neighbors, data, current_graph, rng_state)

    n_vertices = data.shape[0]
    n_threads = numba.get_num_threads()
    block_size = 65536 // n_threads
    n_blocks = n_vertices // block_size

    max_updates_per_thread = int(
        ((max_candidates ** 2 + max_candidates * (max_candidates - 1) / 2) * block_size)
    )
    update_array = np.empty((n_threads, max_updates_per_thread, 3), dtype=np.float32)
    n_updates_per_thread = np.zeros(n_threads, dtype=np.int32)

    for n in range(n_iters):
        if verbose:
            print("\t", n + 1, " / ", n_iters)

        (new_candidate_neighbors, old_candidate_neighbors) = build_candidates(
            current_graph, max_candidates, rng_state, n_threads
        )

        c = 0
        n_vertices = new_candidate_neighbors.shape[0]
        for i in range(n_blocks + 1):
            block_start = i * block_size
            block_end = min(n_vertices, (i + 1) * block_size)

            new_candidate_block = new_candidate_neighbors[block_start:block_end]
            old_candidate_block = old_candidate_neighbors[block_start:block_end]

            dist_thresholds = current_graph[1][:, 0]

            generate_graph_update_array_int8(
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

        if c <= delta * n_neighbors * data.shape[0]:
            if verbose:
                print("\tStopping threshold met -- exiting after", n + 1, "iterations")
            return deheap_sort(current_graph[0], current_graph[1])

        block_size = min(n_vertices, 2 * block_size)
        n_blocks = n_vertices // block_size

    return deheap_sort(current_graph[0], current_graph[1])
