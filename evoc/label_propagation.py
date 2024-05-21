import numpy as np
import numba

from scipy.sparse import csr_matrix

from .node_embedding import node_embedding


@numba.njit(fastmath=True, parallel=True, cache=True)
def label_prop_iteration(
    indptr,
    indices,
    data,
    labels,
):
    n_rows = indptr.shape[0] - 1
    result = labels.copy()

    for i in numba.prange(n_rows):
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
        current_l = labels[i]
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
                elif np.random.rand() < 1.0 / tie_count:
                    result[i] = l
            else:
                continue

    return result


@numba.njit(cache=True)
def label_outliers(indptr, indices, labels):
    n_rows = indptr.shape[0] - 1

    for i in numba.prange(n_rows):
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
                labels[i] = 0

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


def label_prop_loop(indptr, indices, data, labels, n_iter=20, approx_n_parts=2048):
    for i in range(int(1.25 * approx_n_parts)):
        labels[np.random.randint(labels.shape[0])] = i

    for i in range(n_iter):
        new_labels = label_prop_iteration(indptr, indices, data, labels)
        labels = new_labels

    labels = label_outliers(indptr, indices, labels)
    return remap_labels(labels)


def label_propagation_init(
    graph,
    n_iter=20,
    approx_n_parts=512,
    n_components=2,
    scaling=0.1,
    random_scale=1.0,
    noise_level=0.5,
):
    labels = np.full(graph.shape[0], -1, dtype=np.int64)
    partition = label_prop_loop(
        graph.indptr, graph.indices, graph.data, labels, n_iter, approx_n_parts
    )
    reduction_map = csr_matrix(
        (np.ones(partition.shape[0]), partition, np.arange(partition.shape[0] + 1)),
        shape=(partition.shape[0], partition.max() + 1),
    )
    reduced_graph = (reduction_map.T * graph * reduction_map).tocoo()
    reduced_graph.data = np.clip(reduced_graph.data, 0.0, 1.0)
    reduced_layout = node_embedding(
        reduced_graph, n_components, 50, verbose=False, noise_level=noise_level
    )
    result = reduced_layout[partition]
    result += np.random.normal(
        scale=random_scale, size=(partition.shape[0], reduced_layout.shape[1])
    )
    result = (scaling * (result - result.mean(axis=0))).astype(np.float32)
    return result
