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


def node_embedding(
    graph,
    n_components,
    n_epochs,
    initial_embedding=None,
    initial_alpha=0.5,
    negative_sample_rate=1.0,
    noise_level=0.5,
    verbose=False,
    tqdm_kwds={},
):
    if initial_embedding is None:
        embedding = np.random.normal(
            scale=0.25, size=(graph.shape[0], n_components)
        ).astype(np.float32, order="C")
    else:
        embedding = initial_embedding

    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs).astype(
        np.float32, order="C"
    )
    epochs_per_negative_sample = epochs_per_sample / negative_sample_rate
    epoch_of_next_negative_sample = epochs_per_negative_sample.copy()
    epoch_of_next_sample = epochs_per_sample.copy()

    if tqdm_kwds is None:
        tqdm_kwds = {}

    if "disable" not in tqdm_kwds:
        tqdm_kwds["disable"] = not verbose

    rng_val = np.random.randint(INT32_MAX, size=n_epochs)
    head_u4 = graph.row.astype(np.uint32)
    tail_u4 = graph.col.astype(np.uint32)
    n_vertices = graph.shape[0]
    dim = embedding.shape[1]
    alpha = initial_alpha

    for n in tqdm(range(n_epochs), **tqdm_kwds):

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
        alpha = initial_alpha * (1.0 - (float(n) / float(n_epochs)))

    return embedding
