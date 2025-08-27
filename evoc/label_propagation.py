import numpy as np
import numba

from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding, MDS

from .node_embedding import node_embedding
from .common_nndescent import tau_rand, tau_rand_int

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


@numba.njit(fastmath=True, parallel=True, cache=True)
def label_prop_iteration(
    indptr,
    indices,
    data,
    labels,
    rng_state,
):
    n_rows = indptr.shape[0] - 1
    result = labels.copy()

    for i in numba.prange(n_rows):
        current_l = labels[i]
        if current_l >= 0:
            continue
        local_rng_state = rng_state + i
        votes = {}
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]
            l = labels[j]
            if l in votes:
                votes[l] += data[k]
            else:
                votes[l] = data[k]

        max_vote = 1
        tie_count = 1
        for l in votes:
            if l == -1:
                continue
            elif votes[l] > max_vote:
                max_vote = votes[l]
                result[i] = l
                tie_count = 1
            elif votes[l] == max_vote:
                tie_count += 1
                if current_l == -1:
                    result[i] = l
                elif tau_rand(local_rng_state) < 1.0 / tie_count:
                    result[i] = l
            else:
                continue

    return result


@numba.njit(fastmath=True, parallel=True, cache=True)
def original_label_prop_iteration(
    indptr,
    indices,
    data,
    labels,
    rng_state,
):
    n_rows = indptr.shape[0] - 1
    result = labels.copy()

    for i in numba.prange(n_rows):
        current_l = labels[i]
        local_rng_state = rng_state + i
        votes = {}
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]
            l = labels[j]
            if l in votes:
                votes[l] += data[k]
            else:
                votes[l] = data[k]

        max_vote = 1
        tie_count = 1
        for l in votes:
            if l == -1:
                continue
            elif votes[l] > max_vote:
                max_vote = votes[l]
                result[i] = l
                tie_count = 1
            elif votes[l] == max_vote:
                tie_count += 1
                if current_l == -1:
                    result[i] = l
                elif tau_rand(local_rng_state) < 1.0 / tie_count:
                    result[i] = l
            else:
                continue

    return result


@numba.njit(cache=True)
def label_outliers(indptr, indices, labels, rng_state):
    n_rows = indptr.shape[0] - 1
    max_label = labels.max()

    for i in numba.prange(n_rows):
        local_rng_state = rng_state + i
        if labels[i] < 0:

            node_queue = [i]
            unlabelled = True
            n_iter = 0

            while unlabelled and n_iter < 100:

                n_iter += 1
                current_node = node_queue.pop()
                for k in range(indptr[current_node], indptr[current_node + 1]):
                    j = indices[k]
                    if labels[j] >= 0:
                        labels[i] = labels[j]
                        unlabelled = False
                        break
                    else:
                        node_queue.append(j)

            if n_iter >= 100:
                labels[i] = tau_rand_int(local_rng_state) % (max_label + 1)

    return labels


@numba.njit(cache=True)
def remap_labels(labels):
    mapping = {}
    unique_labels = np.unique(labels)
    if unique_labels[0] == -1:
        unique_labels = unique_labels[1:]
    for i, l in enumerate(unique_labels):
        mapping[l] = i
    next_label = i + 1
    for i in range(labels.shape[0]):
        if labels[i] < 0:
            labels[i] = next_label
            next_label += 1
        else:
            labels[i] = mapping[labels[i]]

    return labels


def label_prop_loop(
    indptr, indices, data, labels, random_state, n_iter=20, approx_n_parts=2048
):
    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    for i in range(int(1.25 * approx_n_parts)):  # range(int(1.25 * approx_n_parts)):
        labels[random_state.randint(labels.shape[0])] = i

    for i in range(n_iter):
        new_labels = label_prop_iteration(indptr, indices, data, labels, rng_state)
        labels = new_labels

    labels = label_outliers(indptr, indices, labels, rng_state)
    return remap_labels(labels)


def original_label_prop_loop(
    indptr, indices, data, labels, random_state, n_iter=20, approx_n_parts=2048
):
    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    for i in range(int(1.25 * approx_n_parts)):
        labels[random_state.randint(labels.shape[0])] = i

    for i in range(n_iter):
        new_labels = original_label_prop_iteration(
            indptr, indices, data, labels, rng_state
        )
        labels = new_labels

    labels = label_outliers(indptr, indices, labels, rng_state)
    return remap_labels(labels)


import matplotlib.pyplot as plt


def label_propagation_init(
    graph,
    n_iter=20,
    approx_n_parts=512,
    n_components=2,
    scaling=0.1,
    random_scale=1.0,
    noise_level=0.5,
    random_state=None,
    data=None,
    recursive_init=True,
    base_init="pca",
    base_init_threshold=64,
    original_label_propagation=False,
    upscaling="partition_expander",
):
    if random_state is None:
        random_state = np.random.RandomState()

    # if recursive_init:
    #     approx_n_parts = graph.shape[0] // 4

    if graph.shape[0] < base_init_threshold:
        if base_init == "random":
            result = random_state.normal(
                loc=0.0, scale=1.0, size=(graph.shape[0], n_components)
            )
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = result / norms
            return result.astype(np.float32)
        elif base_init == "pca":
            result = (
                PCA(n_components=n_components, random_state=random_state)
                .fit_transform(data)
                .astype(np.float32, order="C")
            )
            result -= result.mean()
            result /= (result.max() - result.min()) / 2.0
            return result
        elif base_init == "spectral":
            result = (
                SpectralEmbedding(n_components=n_components, random_state=random_state)
                .fit_transform(data)
                .astype(np.float32, order="C")
            )
            result -= result.mean()
            result /= (result.max() - result.min()) / 2.0
            return result
        elif base_init == "mds":
            result = (
                MDS(
                    n_components=n_components,
                    random_state=random_state,
                    n_init=1,
                    max_iter=300,
                )
                .fit_transform(data)
                .astype(np.float32, order="C")
            )
            result -= result.mean()
            result /= (result.max() - result.min()) / 2.0
            return result
        else:
            raise ValueError(
                "Unknown base initialization method. Should be one of ['random', 'pca', 'spectral', 'mds']"
            )

    labels = np.full(graph.shape[0], -1, dtype=np.int64)
    if original_label_propagation:
        partition = original_label_prop_loop(
            graph.indptr,
            graph.indices,
            graph.data,
            labels,
            random_state,
            n_iter,
            approx_n_parts,
        )
    else:
        partition = label_prop_loop(
            graph.indptr,
            graph.indices,
            graph.data,
            labels,
            random_state,
            n_iter,
            approx_n_parts,
        )
    base_reduction_map = csr_matrix(
        (np.ones(partition.shape[0]), partition, np.arange(partition.shape[0] + 1)),
        shape=(partition.shape[0], partition.max() + 1),
    )
    normalized_reduction_map = normalize(base_reduction_map, axis=0, norm="l2")
    data_reducer = normalize(normalized_reduction_map.T, norm="l1")
    if data is not None:
        reduced_data = data_reducer @ data
    else:
        reduced_data = None

    reduced_graph = normalized_reduction_map.T * graph * base_reduction_map
    reduced_graph.data = np.clip(reduced_graph.data, 0.0, 1.0)

    if recursive_init:
        reduced_init = label_propagation_init(
            reduced_graph,
            n_iter=min(255, n_iter * 2),
            approx_n_parts=approx_n_parts // 4,
            n_components=n_components,
            scaling=scaling,
            random_scale=random_scale,
            noise_level=noise_level,
            random_state=random_state,
            data=reduced_data,
            recursive_init=True,
            original_label_propagation=original_label_propagation,
            upscaling=upscaling,
            base_init=base_init,
            base_init_threshold=base_init_threshold,
        )
    else:
        reduced_init = None

    reduced_layout = node_embedding(
        reduced_graph,
        n_components,
        n_iter,
        verbose=False,
        noise_level=noise_level,
        random_state=random_state,
        initial_embedding=reduced_init,
        initial_alpha=0.001 * n_iter,
    )

    if upscaling == "partition_expander":
        data_expander = normalize(
            (graph.multiply(graph.T)) @ normalized_reduction_map, norm="l1"
        )
        result = (
            data_expander @ reduced_layout
            + normalize(normalized_reduction_map, norm="l1") @ reduced_layout
        ) / 2.0
    elif upscaling == "jitter_expander":
        data_expander = normalize(
            (graph.multiply(graph.T)) @ normalized_reduction_map, norm="l1"
        )
        expanded = (
            data_expander @ reduced_layout
            + normalize(normalized_reduction_map, norm="l1") @ reduced_layout
        ) / 2.0
        jittered = reduced_layout[partition]
        jittered += random_state.normal(
            scale=random_scale / 4.0, size=(partition.shape[0], reduced_layout.shape[1])
        )
        result = (expanded + jittered) / 2.0
    else:
        result = reduced_layout[partition]
        result += random_state.normal(
            scale=random_scale, size=(partition.shape[0], reduced_layout.shape[1])
        )

    result = (scaling * (result - result.mean(axis=0))).astype(np.float32)
    # plt.scatter(result[:, 0], result[:, 1], c=partition, s=1)
    # plt.show()
    return result
