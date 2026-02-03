import numpy as np
import numba

from tqdm import tqdm

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


def make_epochs_per_sample(weights, n_epochs):
    result = np.full(weights.shape[0], n_epochs, dtype=np.float32)
    n_samples = np.maximum(n_epochs * (weights / weights.max()), 1.0)
    result = float(n_epochs) / np.float32(n_samples)
    return result


@numba.njit(
    "f4(f4[::1],f4[::1])",
    fastmath=True,
    cache=True,
    locals={
        "result": numba.types.float32,
        "diff": numba.types.float32,
        "dim": numba.types.intp,
        "i": numba.types.intp,
    },
)
def rdist(x, y):
    result = 0.0
    dim = x.shape[0]
    for i in range(dim):
        diff = x[i] - y[i]
        result += diff * diff

    return result


@numba.njit(inline="always")
def clip(val, lo, hi):
    if val > hi:
        return hi
    elif val < lo:
        return lo
    else:
        return val


@numba.njit(
    "void(f4[:,::1],u4[::1],u4[::1],u4,f4[::1],u4,u1,f4,f4[::1],f4[::1],f4[::1],u1,f4)",
    fastmath=True,
    parallel=True,
    cache=True,
    locals={
        "i": numba.uint32,
        "j": numba.uint32,
        "k": numba.uint32,
        "di": numba.uint8,
        "p": numba.uint8,
        "n_neg_samples": numba.uint8,
        "dist_squared": numba.float32,
        "grad_coeff": numba.float32,
        "current": numba.float32[::1],
        "other": numba.float32[::1],
    },
)
def node_embedding_epoch(
    embedding,
    head,
    tail,
    n_vertices,
    epochs_per_sample,
    rng_state,
    dim,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
    noise_level,
):
    for i in numba.prange(epochs_per_sample.shape[0]):
        if epoch_of_next_sample[i] <= n:
            j = head[i]
            k = tail[i]

            current = embedding[j]
            other = embedding[k]

            dist_squared = rdist(current, other)

            if dist_squared > 0.0:
                dist = np.sqrt(dist_squared)
                grad_coeff = (-2.0 * noise_level * dist - 2.0) / (
                    2.0 * dist_squared - 0.5 * dist + 1.0
                )

                for di in range(dim):
                    grad_d = grad_coeff * (current[di] - other[di])

                    current[di] += grad_d * alpha
                    other[di] += -grad_d * alpha

            epoch_of_next_sample[i] += epochs_per_sample[i]
            n_neg_samples = int(
                (n - epoch_of_next_negative_sample[i]) / epochs_per_negative_sample[i]
            )

            for p in range(n_neg_samples):
                k = ((n + p) * i * rng_state) % n_vertices
                other = embedding[k]
                dist_squared = rdist(current, other)

                if dist_squared > 1e-2:
                    grad_coeff = 4.0 / ((1.0 + 0.25 * dist_squared) * dist_squared)

                    for di in range(dim):
                        grad_d = clip(grad_coeff * (current[di] - other[di]), -4, 4)
                        current[di] += grad_d * alpha

            epoch_of_next_negative_sample[i] += (
                n_neg_samples * epochs_per_negative_sample[i]
            )


@numba.njit(
    "void(f4[:, ::1], u4[::1], u4[::1], u4, f4[::1], u4, u1, f4, f4[::1], f4[::1], f4[::1], u1, f4, f4, f4[:, ::1], u4[::1], u4)",
    fastmath=True,
    parallel=True,
    locals={
        "updates": numba.types.float32[:, ::1],
        "from_node": numba.types.intp,
        "to_node": numba.types.intp,
        "raw_index": numba.types.intp,
        "dist_squared": numba.types.float32,
        "dist": numba.types.float32,
        "grad_coeff": numba.types.float32,
        "grad_d": numba.types.float32,
        "current": numba.types.float32[::1],
        "other": numba.types.float32[::1],
        "block_start": numba.types.intp,
        "block_end": numba.types.intp,
        "node_idx": numba.types.intp,
        "d": numba.types.uint8,
        "n": numba.types.uint8,
        "p": numba.types.uint8,
        "n_neg_samples": numba.types.uint8,
    },
)
def node_embedding_epoch_repr(
    embedding,
    csr_indptr,
    csr_indices,
    n_vertices,
    epochs_per_sample,
    rng_state,
    dim,
    alpha,
    epochs_per_negative_sample,
    epoch_of_next_negative_sample,
    epoch_of_next_sample,
    n,
    noise_level,
    gamma,
    updates,
    node_order,
    block_size=4096,
):
    for block_start in range(0, n_vertices, block_size):
        block_end = min(block_start + block_size, n_vertices)
        for node_idx in numba.prange(block_start, block_end):
            from_node = node_order[node_idx]
            current = embedding[from_node]

            for raw_index in range(csr_indptr[from_node], csr_indptr[from_node + 1]):
                if epoch_of_next_sample[raw_index] <= n:
                    to_node = csr_indices[raw_index]
                    other = embedding[to_node]

                    dist_squared = rdist(current, other)

                    if dist_squared > 0.0:
                        dist = np.sqrt(dist_squared)
                        grad_coeff = (-2.0 * noise_level * dist - 2.0) / (
                            2.0 * dist_squared - 0.5 * dist + 1.0
                        )
                        for d in range(dim):
                            grad_d = grad_coeff * (current[d] - other[d])
                            updates[from_node, d] += grad_d * alpha

                    epoch_of_next_sample[raw_index] += epochs_per_sample[raw_index]

                    n_neg_samples = int(
                        (n - epoch_of_next_negative_sample[raw_index])
                        / epochs_per_negative_sample[raw_index]
                    )

                    for p in range(n_neg_samples):
                        to_node = node_order[
                            (raw_index * (n + p + 1) * rng_state) % n_vertices
                        ]
                        other = embedding[to_node]

                        dist_squared = rdist(current, other)

                        if dist_squared > 1e-2:
                            grad_coeff = (
                                gamma
                                * 4.0
                                / ((1.0 + 0.25 * dist_squared) * dist_squared)
                            )
                            # grad_coeff /= n_neg_samples

                            if grad_coeff > 0.0:
                                for d in range(dim):
                                    grad_d = clip(
                                        grad_coeff * (current[d] - other[d]), -4, 4
                                    )
                                    updates[from_node, d] += grad_d * alpha

                    epoch_of_next_negative_sample[raw_index] += (
                        n_neg_samples * epochs_per_negative_sample[raw_index]
                    )

        for node_idx in numba.prange(block_start, block_end):
            from_node = node_order[node_idx]
            for d in range(dim):
                embedding[from_node, d] += updates[from_node, d]


def node_embedding(
    graph,
    n_components,
    n_epochs,
    initial_embedding=None,
    initial_alpha=0.5,
    negative_sample_rate=1.0,
    noise_level=0.5,
    random_state=None,
    reproducible_flag=True,
    verbose=False,
    tqdm_kwds={},
):
    """Learn a low-dimensional embedding of a graph using a UMAP-like algorithm.

    This function performs stochastic gradient descent optimization to learn a
    low-dimensional embedding of graph structure. It uses both positive (connected
    edges) and negative (random) samples to guide the optimization.

    Parameters
    ----------
    graph : scipy.sparse matrix, typically csr_matrix or csc_matrix
        A sparse adjacency matrix representing the graph. The weights in the matrix
        represent connection strengths between nodes.

    n_components : int
        The number of dimensions in the output embedding.

    n_epochs : int
        The number of epochs to train the embedding.

    initial_embedding : array-like of shape (n_vertices, n_components) or None, default=None
        An initial embedding to use as a starting point. If None, a random
        embedding is generated from a normal distribution with scale 0.25.

    initial_alpha : float, default=0.5
        The initial learning rate. The learning rate decays linearly over epochs.

    negative_sample_rate : float, default=1.0
        The rate at which negative samples are drawn relative to positive samples.
        Controls the ratio of negative to positive updates per epoch.

    noise_level : float, default=0.5
        Controls the strength of noise in the gradient computation. Higher values
        increase the tolerance for larger distances before penalizing in the
        embedding space.

    random_state : RandomState instance or None, default=None
        Random state for reproducibility. If None, uses system randomness.

    reproducible_flag : bool, default=True
        If True, uses a deterministic (but slower) update strategy that processes
        nodes in blocks for reproducibility. If False, uses a faster stochastic
        approach.

    verbose : bool, default=False
        If True, display a progress bar during training.

    tqdm_kwds : dict, default={}
        Additional keyword arguments to pass to tqdm for progress bar customization.

    Returns
    -------
    embedding : array-like of shape (n_vertices, n_components)
        The learned low-dimensional embedding of the graph vertices.
    """
    if random_state is None:
        random_state = np.random.RandomState()
    if initial_embedding is None:
        embedding = random_state.normal(
            scale=0.25, size=(graph.shape[0], n_components)
        ).astype(np.float32, order="C")
    else:
        embedding = initial_embedding

    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs).astype(
        np.float32, order="C"
    )
    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    if reproducible_flag:
        epochs_per_negative_sample *= 1.5
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    if tqdm_kwds is None:
        tqdm_kwds = {}

    if "disable" not in tqdm_kwds:
        tqdm_kwds["disable"] = not verbose

    rng_val = random_state.randint(INT32_MAX, size=n_epochs)

    coo_graph = graph.tocoo()
    head_u4 = coo_graph.row.astype(np.uint32)
    tail_u4 = coo_graph.col.astype(np.uint32)
    # New
    csr_indptr = graph.indptr.astype(np.uint32)
    csr_indices = graph.indices.astype(np.uint32)
    updates = np.zeros_like(embedding)
    node_order = np.arange(graph.shape[0], dtype=np.uint32)
    gamma_schedule = np.linspace(0.5, 1.5, n_epochs)
    # End new
    n_vertices = np.uint32(graph.shape[0])
    block_size = max(1024, n_vertices // 8)
    dim = np.uint8(embedding.shape[1])
    alpha = np.float32(initial_alpha)

    for n in tqdm(range(n_epochs), **tqdm_kwds):

        if not reproducible_flag:
            node_embedding_epoch(
                embedding,
                head_u4,
                tail_u4,
                n_vertices,
                epochs_per_sample,
                rng_val[n],
                dim,
                alpha,
                epochs_per_negative_sample,
                epoch_of_next_negative_sample,
                epoch_of_next_sample,
                n,
                noise_level,
            )
        else:
            node_embedding_epoch_repr(
                embedding,
                csr_indptr,
                csr_indices,
                n_vertices,
                epochs_per_sample,
                np.uint32(rng_val[n]),
                dim,
                alpha,
                epochs_per_negative_sample,
                epoch_of_next_negative_sample,
                epoch_of_next_sample,
                np.uint8(n),
                np.float32(noise_level),
                gamma_schedule[n],
                updates,
                node_order,
                np.uint32(block_size),
            )
            updates *= (1.0 - alpha) ** 2 * 0.5
            random_state.shuffle(node_order)
        alpha = np.float32(initial_alpha * (1.0 - (float(n) / float(n_epochs))))

    return embedding
