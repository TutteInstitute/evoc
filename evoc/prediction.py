"""State kept by a fitted EVoC model for ``predict`` and ``update``."""
from collections import namedtuple

import numba
import numpy as np

from .clustering_utilities import extract_clusters_by_id
from .graph_construction import smooth_knn_dist, smooth_knn_dist_serial
from .numba_kdtree import build_kdtree, parallel_tree_query

ClusterHierarchy = namedtuple(
    "ClusterHierarchy",
    [
        # list of CondensedTree; cluster ids are only comparable within a tree
        "condensed_trees",
        # int per layer: index into condensed_trees of the tree the layer uses
        "layer_tree",
        # list per layer of int64 arrays: the selected cluster ids. Labels are
        # the ranks of the ids in sorted order, as in get_cluster_label_vector
        "layer_clusters",
        "min_cluster_size",
        "min_samples",
        "n_samples",
        # list per layer of int64 arrays aligned with layer_clusters: the label
        # of each selected cluster. None means labels are the ranks (as after
        # fit); EVoC.update relabels layers to keep cluster ids stable.
        "layer_labels",
    ],
    defaults=(None,),
)
ClusterHierarchy.__doc__ = """The condensed trees behind a model's cluster layers.

Layers are in the same order as ``cluster_layers_``. Usually every layer uses
the same condensed tree; with ``base_n_clusters`` the base layer comes from a
tree condensed with a different minimum cluster size, and with
``approx_n_clusters`` the single layer comes from its own tree.
"""

PredictionData = namedtuple(
    "PredictionData",
    [
        # training data as the search code uses it: float data L2 normalised
        # float32, int8 and bit-packed uint8 data as given
        "data",
        # np.float32, np.int8 or np.uint8
        "input_dtype",
        # inverse row norms (int8 only; empty otherwise)
        "inv_norms",
        # smooth_knn_dist bandwidths of the k-NN graph (float32[n_samples])
        "sigmas",
        "rhos",
        # mean of the k-NN distances, the fallback scale in smooth_knn_dist
        "mean_knn_distance",
        # node embedding (float32[n_samples, n_components])
        "embedding",
        "hierarchy",
        # core distances in the embedding (float32[n_samples])
        "core_distances",
        # dict of the fit parameters predict and update need
        "params",
    ],
)
PredictionData.__doc__ = """Everything ``predict`` and ``update`` need from a fit."""


def as_cluster_ids(clusters):
    return np.sort(np.asarray(clusters, dtype=np.int64))


def layer_label_map(hierarchy, layer):
    """Label of each selected cluster of a layer (in sorted cluster id order)."""
    n_clusters = len(hierarchy.layer_clusters[layer])
    if hierarchy.layer_labels is None or hierarchy.layer_labels[layer] is None:
        return np.arange(n_clusters, dtype=np.int64)
    return np.asarray(hierarchy.layer_labels[layer], dtype=np.int64)


def labels_from_hierarchy(hierarchy, layer):
    """Labels and membership strengths of a layer, re-derived from the tree.

    Reproduces the layer's labels in ``cluster_layers_`` exactly.
    """
    tree = hierarchy.condensed_trees[hierarchy.layer_tree[layer]]
    clusters = hierarchy.layer_clusters[layer]
    if len(clusters) == 0 or len(tree.parent) == 0:
        n = hierarchy.n_samples
        return np.full(n, -1, dtype=np.intp), np.zeros(n, dtype=np.float32)
    ranks, strengths = extract_clusters_by_id(
        tree, np.asarray(clusters, dtype=np.int64)
    )
    label_map = layer_label_map(hierarchy, layer)
    labels = np.where(ranks >= 0, label_map[np.maximum(ranks, 0)], -1)
    return labels.astype(np.intp), strengths


def embedding_core_distances(embedding, min_samples, tree=None):
    """Core distances in the embedding, as ``parallel_boruvka`` computes them.

    The core distance of a training point is the Euclidean distance to its
    ``min_samples``-th nearest other point (0 if ``min_samples <= 1``).
    """
    embedding = np.ascontiguousarray(embedding, dtype=np.float32)
    if min_samples <= 1:
        return np.zeros(embedding.shape[0], dtype=np.float32)
    if tree is None:
        tree = build_kdtree(embedding)
    distances, _ = parallel_tree_query(
        tree, embedding, k=np.int64(min_samples + 1), output_rdist=True
    )
    return np.sqrt(distances[:, -1]).astype(np.float32)


# ------------------------------------------------------------------ vote method
# Batches smaller than this use single-threaded kernels: launching parallel
# kernels costs tens of microseconds per call, and more on a busy machine.
SERIAL_BATCH_SIZE = 64


@numba.njit(fastmath=True, inline="always")
def _membership_weights_row(indices, distances, sigmas, rhos, weights, i):
    rho = rhos[i]
    sigma = sigmas[i]
    for j in range(indices.shape[1]):
        if indices[i, j] < 0:
            continue
        d = distances[i, j]
        if (d - rho) <= 0.0 or sigma == 0.0:
            weights[i, j] = 1.0
        else:
            weights[i, j] = np.exp(-((d - rho) / sigma))


@numba.njit(parallel=True, fastmath=True, cache=True)
def _membership_weights_parallel(indices, distances, sigmas, rhos):
    weights = np.zeros(indices.shape, dtype=np.float32)
    for i in numba.prange(indices.shape[0]):
        _membership_weights_row(indices, distances, sigmas, rhos, weights, i)
    return weights


@numba.njit(fastmath=True, cache=True)
def _membership_weights_serial(indices, distances, sigmas, rhos):
    weights = np.zeros(indices.shape, dtype=np.float32)
    for i in range(indices.shape[0]):
        _membership_weights_row(indices, distances, sigmas, rhos, weights, i)
    return weights


def membership_weights(indices, distances, sigmas, rhos):
    """Edge weights from each query to its neighbours.

    Uses the same formula as ``compute_membership_strengths``; ``indices`` and
    ``distances`` exclude the query itself. Missing neighbours (-1) get 0.
    """
    if indices.shape[0] < SERIAL_BATCH_SIZE:
        return _membership_weights_serial(indices, distances, sigmas, rhos)
    return _membership_weights_parallel(indices, distances, sigmas, rhos)


def query_bandwidths(distances, pdata):
    """``smooth_knn_dist`` bandwidths for queries with neighbour distances
    ``distances`` (self excluded), computed as for the training points."""
    n = distances.shape[0]
    with_self = np.zeros((n, distances.shape[1] + 1), dtype=np.float32)
    with_self[:, 1:] = distances
    params = pdata.params
    function = smooth_knn_dist_serial if n < SERIAL_BATCH_SIZE else smooth_knn_dist
    return function(
        with_self,
        float(params["neighbor_scale"] * params["n_neighbors"]),
        mean_distance=np.float64(pdata.mean_knn_distance),
    )


@numba.njit(fastmath=True, inline="always")
def _vote_row(
    indices, weights, layer_labels, layer_strengths, best_labels, best_scores, i,
    cluster_ids, cluster_scores,
):
    k = indices.shape[1]
    total = np.float32(0.0)
    for j in range(k):
        if indices[i, j] >= 0:
            total += weights[i, j]
    if total <= 0.0:
        return
    for layer in range(layer_labels.shape[0]):
        n_found = 0
        for j in range(k):
            idx = indices[i, j]
            if idx < 0:
                continue
            c = layer_labels[layer, idx]
            if c < 0:
                continue
            s = weights[i, j] * layer_strengths[layer, idx]
            found = False
            for m in range(n_found):
                if cluster_ids[m] == c:
                    cluster_scores[m] += s
                    found = True
                    break
            if not found:
                cluster_ids[n_found] = c
                cluster_scores[n_found] = s
                n_found += 1
        if n_found == 0:
            continue
        best = 0
        for m in range(1, n_found):
            if cluster_scores[m] > cluster_scores[best] or (
                cluster_scores[m] == cluster_scores[best]
                and cluster_ids[m] < cluster_ids[best]
            ):
                best = m
        best_labels[layer, i] = cluster_ids[best]
        best_scores[layer, i] = cluster_scores[best] / total


@numba.njit(parallel=True, fastmath=True, cache=True)
def _vote_scores_parallel(indices, weights, layer_labels, layer_strengths):
    n, k = indices.shape
    n_layers = layer_labels.shape[0]
    best_labels = np.full((n_layers, n), -1, dtype=np.int64)
    best_scores = np.zeros((n_layers, n), dtype=np.float32)
    for i in numba.prange(n):
        cluster_ids = np.empty(k, dtype=np.int64)
        cluster_scores = np.empty(k, dtype=np.float32)
        _vote_row(
            indices, weights, layer_labels, layer_strengths, best_labels,
            best_scores, i, cluster_ids, cluster_scores,
        )
    return best_labels, best_scores


@numba.njit(fastmath=True, cache=True)
def _vote_scores_serial(indices, weights, layer_labels, layer_strengths):
    n, k = indices.shape
    n_layers = layer_labels.shape[0]
    best_labels = np.full((n_layers, n), -1, dtype=np.int64)
    best_scores = np.zeros((n_layers, n), dtype=np.float32)
    cluster_ids = np.empty(k, dtype=np.int64)
    cluster_scores = np.empty(k, dtype=np.float32)
    for i in range(n):
        _vote_row(
            indices, weights, layer_labels, layer_strengths, best_labels,
            best_scores, i, cluster_ids, cluster_scores,
        )
    return best_labels, best_scores


def vote_scores(indices, weights, layer_labels, layer_strengths):
    """Weighted neighbour vote for every layer.

    For each query and layer, the score of cluster c is the sum over neighbours
    labelled c of (edge weight * neighbour membership strength). Returns the
    best cluster (-1 if no neighbour is clustered) and its score divided by the
    total edge weight, each of shape (n_layers, n_queries).
    """
    if indices.shape[0] < SERIAL_BATCH_SIZE:
        return _vote_scores_serial(indices, weights, layer_labels, layer_strengths)
    return _vote_scores_parallel(indices, weights, layer_labels, layer_strengths)


def stack_layers(cluster_layers, strength_layers):
    labels = np.ascontiguousarray(np.vstack(cluster_layers), dtype=np.int64)
    strengths = np.ascontiguousarray(np.vstack(strength_layers), dtype=np.float32)
    return labels, strengths


def calibrate_vote_thresholds(nn_inds, nn_dists, pdata, cluster_layers, strength_layers):
    """Per-layer noise thresholds for the vote method.

    Each training point is predicted from its other k-NN graph neighbours
    (leave one out), and the threshold is the score quantile that makes the
    predicted noise fraction match the layer's training noise fraction.
    Returns (thresholds, leave-one-out labels of shape (n_layers, n_samples)).
    """
    indices = np.ascontiguousarray(nn_inds[:, 1:], dtype=np.int32)
    distances = np.ascontiguousarray(nn_dists[:, 1:], dtype=np.float32)
    weights = membership_weights(indices, distances, pdata.sigmas, pdata.rhos)
    labels, strengths = stack_layers(cluster_layers, strength_layers)
    best_labels, best_scores = vote_scores(indices, weights, labels, strengths)
    thresholds = np.zeros(labels.shape[0], dtype=np.float32)
    for layer in range(labels.shape[0]):
        noise_fraction = np.mean(labels[layer] < 0)
        scores = np.where(best_labels[layer] >= 0, best_scores[layer], 0.0)
        if noise_fraction > 0:
            thresholds[layer] = np.quantile(scores, noise_fraction)
    return thresholds, best_labels


def apply_thresholds(best_labels, best_scores, thresholds):
    labels = best_labels.copy()
    strengths = best_scores.copy()
    noise = (labels < 0) | (best_scores < thresholds[:, None])
    labels[noise] = -1
    strengths[noise] = 0.0
    return labels, strengths


# ------------------------------------------------------------- embedding method
TreeArrays = namedtuple(
    "TreeArrays",
    [
        "root",
        "point_parent",  # cluster each training point falls out of
        "point_lambda",  # ... and the lambda at which it does
        "cluster_parent",  # indexed by cluster - root; -1 for the root
        "cluster_birth",  # lambda at which each cluster appears (0 for root)
        "max_lambda",  # max lambda of points falling directly out of each cluster
        "max_child_lambda",  # max lambda of any child (point or cluster)
    ],
)


def tree_arrays(tree, n_samples):
    """Lookup arrays for placing new points in a condensed tree."""
    parent = np.asarray(tree.parent, dtype=np.int64)
    child = np.asarray(tree.child, dtype=np.int64)
    lambda_val = np.asarray(tree.lambda_val, dtype=np.float64)
    child_size = np.asarray(tree.child_size)
    if parent.shape[0] == 0:
        root = n_samples
        n_clusters = 1
    else:
        root = int(parent.min())
        n_clusters = int(max(parent.max(), child.max())) + 1 - root
    point_parent = np.full(n_samples, -1, dtype=np.int64)
    point_lambda = np.zeros(n_samples, dtype=np.float64)
    is_point = child < root
    point_parent[child[is_point]] = parent[is_point]
    point_lambda[child[is_point]] = lambda_val[is_point]
    cluster_parent = np.full(n_clusters, -1, dtype=np.int64)
    cluster_birth = np.zeros(n_clusters, dtype=np.float64)
    is_cluster = ~is_point
    cluster_parent[child[is_cluster] - root] = parent[is_cluster]
    cluster_birth[child[is_cluster] - root] = lambda_val[is_cluster]
    max_lambda = np.zeros(n_clusters, dtype=np.float64)
    direct = child_size == 1
    np.maximum.at(max_lambda, parent[direct] - root, lambda_val[direct])
    max_child_lambda = np.zeros(n_clusters, dtype=np.float64)
    np.maximum.at(max_child_lambda, parent - root, lambda_val)
    return TreeArrays(
        root,
        point_parent,
        point_lambda,
        cluster_parent,
        cluster_birth,
        max_lambda,
        max_child_lambda,
    )


@numba.njit(parallel=True, cache=True)
def weighted_positions(indices, weights, embedding):
    """Weighted mean of the neighbours' embedding positions (NaN if no weight)."""
    n, k = indices.shape
    dim = embedding.shape[1]
    result = np.full((n, dim), np.nan, dtype=np.float32)
    for i in numba.prange(n):
        total = 0.0
        acc = np.zeros(dim, dtype=np.float64)
        for j in range(k):
            idx = indices[i, j]
            if idx < 0 or weights[i, j] <= 0.0:
                continue
            total += weights[i, j]
            for d in range(dim):
                acc[d] += weights[i, j] * embedding[idx, d]
        if total > 0.0:
            for d in range(dim):
                result[i, d] = acc[d] / total
    return result


@numba.njit(inline="always")
def _xorshift(state):
    state ^= (state << np.uint64(13)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    state ^= state >> np.uint64(7)
    state ^= (state << np.uint64(17)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return state


@numba.njit(parallel=True, fastmath=True, cache=True)
def refine_positions(
    positions,
    indices,
    weights,
    embedding,
    n_epochs,
    initial_alpha,
    noise_level,
    negative_sample_rate,
    seed,
):
    """SGD on the given positions only, with the rest of the embedding fixed.

    Uses the attractive and repulsive gradients of ``node_embedding_epoch``:
    each epoch pulls a point towards each neighbour (scaled by the edge
    weight) and pushes it away from ``negative_sample_rate`` random rows of
    ``embedding`` per neighbour. ``positions`` may be a view of rows of
    ``embedding`` (to optimise those rows in place); each epoch reads the
    positions from the end of the previous epoch, so results do not depend on
    the number of threads. Rows of ``positions`` that are NaN are skipped.
    """
    n, k = indices.shape
    n_train = embedding.shape[0]
    dim = embedding.shape[1]
    n_neg = max(int(round(negative_sample_rate)), 0)
    states = np.empty(n, dtype=np.uint64)
    for i in range(n):
        states[i] = np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(i + 1)
    updated = positions.copy()
    for epoch in range(n_epochs):
        alpha = initial_alpha * (1.0 - epoch / n_epochs)
        for i in numba.prange(n):
            if np.isnan(positions[i, 0]):
                continue
            current = updated[i]
            state = states[i]
            for j in range(k):
                idx = indices[i, j]
                if idx < 0 or weights[i, j] <= 0.0:
                    continue
                other = embedding[idx]
                dist_squared = 0.0
                for d in range(dim):
                    diff = current[d] - other[d]
                    dist_squared += diff * diff
                if dist_squared > 0.0:
                    dist = np.sqrt(dist_squared)
                    grad_coeff = (-2.0 * noise_level * dist - 2.0) / (
                        2.0 * dist_squared - 0.5 * dist + 1.0
                    )
                    for d in range(dim):
                        current[d] += (
                            weights[i, j] * grad_coeff * (current[d] - other[d]) * alpha
                        )
                for p in range(n_neg):
                    state = _xorshift(state)
                    other = embedding[np.int64(state % np.uint64(n_train))]
                    dist_squared = 0.0
                    for d in range(dim):
                        diff = current[d] - other[d]
                        dist_squared += diff * diff
                    if dist_squared > 1e-2:
                        grad_coeff = 4.0 / ((1.0 + 0.25 * dist_squared) * dist_squared)
                        for d in range(dim):
                            grad_d = grad_coeff * (current[d] - other[d])
                            if grad_d > 4.0:
                                grad_d = 4.0
                            elif grad_d < -4.0:
                                grad_d = -4.0
                            current[d] += weights[i, j] * grad_d * alpha
            states[i] = state
        positions[:] = updated
    return positions


@numba.njit(parallel=True, cache=True)
def nearest_mutual_reachability(neighbor_indices, neighbor_distances, core_query, core_train):
    """The training point with the smallest mutual reachability distance to each
    query, and lambda = 1 / that distance (-1 and 0 if there is none)."""
    n, k = neighbor_indices.shape
    nearest = np.full(n, -1, dtype=np.int64)
    lambdas = np.zeros(n, dtype=np.float64)
    for i in numba.prange(n):
        best = np.inf
        for j in range(k):
            t = neighbor_indices[i, j]
            if t < 0:
                continue
            mr = max(core_query[i], core_train[t], neighbor_distances[i, j])
            if mr < best:
                best = mr
                nearest[i] = t
        if nearest[i] >= 0:
            lambdas[i] = np.inf if best <= 0.0 else 1.0 / best
    return nearest, lambdas


@numba.njit(parallel=True, cache=True)
def place_in_tree(nearest, lambdas, root, point_parent, point_lambda, cluster_parent, cluster_birth):
    """Attach each query to the condensed tree, as HDBSCAN's approximate_predict.

    The query falls out of the tree no later than its nearest (by mutual
    reachability) training point does, so its lambda is capped at that point's
    lambda; it attaches to the deepest cluster on the path from that point's
    cluster to the root that already exists at the query's lambda. Returns the
    cluster (-1 if no neighbour) and the (capped) lambda.
    """
    n = nearest.shape[0]
    attached = np.full(n, -1, dtype=np.int64)
    capped = np.zeros(n, dtype=np.float64)
    for i in numba.prange(n):
        t = nearest[i]
        if t < 0 or point_parent[t] < 0:
            continue
        lam = min(lambdas[i], point_lambda[t])
        c = point_parent[t]
        while c != root and cluster_birth[c - root] > lam:
            c = cluster_parent[c - root]
        attached[i] = c
        capped[i] = lam
    return attached, capped


@numba.njit(parallel=True, cache=True)
def layer_assignment(attached, lambdas, root, cluster_parent, max_lambda, selected_rank):
    """Label and membership strength of attached queries in one layer.

    Walks up from the attached cluster to the first selected cluster, whose
    rank among the selected ids is the label, mirroring get_cluster_label_vector
    and get_point_membership_strength_vector.
    """
    n = attached.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    strengths = np.zeros(n, dtype=np.float32)
    for i in numba.prange(n):
        c = attached[i]
        if c < 0:
            continue
        while c != root and selected_rank[c - root] < 0:
            c = cluster_parent[c - root]
        if c == root:
            continue
        labels[i] = selected_rank[c - root]
        m = max_lambda[c - root]
        lam = lambdas[i]
        if m == 0.0 or not np.isfinite(lam):
            strengths[i] = 1.0
        else:
            strengths[i] = min(lam, m) / m
    return labels, strengths


def embedding_layer_labels(attached_by_tree, hierarchy, arrays_by_tree):
    """Labels and strengths for every layer from per-tree attachments."""
    labels, strengths = [], []
    for layer, selected in enumerate(hierarchy.layer_clusters):
        tree_index = hierarchy.layer_tree[layer]
        arrays = arrays_by_tree[tree_index]
        attached, lambdas = attached_by_tree[tree_index]
        selected = np.asarray(selected, dtype=np.int64)
        n = attached.shape[0]
        if selected.shape[0] == 0:
            labels.append(np.full(n, -1, dtype=np.int64))
            strengths.append(np.zeros(n, dtype=np.float32))
            continue
        label_map = layer_label_map(hierarchy, layer)
        if selected.shape[0] == 1:
            # get_single_cluster_label_vector: label 0 iff the point falls out
            # at or after the densest point falling directly out of the cluster
            c = selected[0] - arrays.root
            lab = np.where(
                (attached >= 0) & (lambdas >= arrays.max_child_lambda[c]),
                label_map[0],
                -1,
            ).astype(np.int64)
            m = arrays.max_lambda[c]
            st = np.where(
                lab >= 0,
                np.where(
                    (m == 0) | ~np.isfinite(lambdas),
                    1.0,
                    np.minimum(lambdas, m) / np.where(m == 0, 1.0, m),
                ),
                0.0,
            ).astype(np.float32)
            labels.append(lab)
            strengths.append(st)
            continue
        selected_rank = np.full(arrays.cluster_parent.shape[0], -1, dtype=np.int64)
        selected_rank[selected - arrays.root] = label_map
        lab, st = layer_assignment(
            attached,
            lambdas,
            arrays.root,
            arrays.cluster_parent,
            arrays.max_lambda,
            selected_rank,
        )
        labels.append(lab)
        strengths.append(st)
    return labels, strengths


def predict_from_positions(positions, pdata, embedding_tree, arrays_by_tree, exclude=None):
    """Cluster labels for points at ``positions`` in the node embedding.

    ``exclude``, if given, is a training index per row to leave out of the
    neighbours (for predicting training points from their own positions).
    """
    hierarchy = pdata.hierarchy
    min_samples = max(int(hierarchy.min_samples), 1)
    n = positions.shape[0]
    valid = ~np.isnan(positions[:, 0])
    query_positions = np.ascontiguousarray(
        np.where(valid[:, None], positions, 0.0), dtype=np.float32
    )
    # The trees cover the first hierarchy.n_samples points of the embedding
    n_train = hierarchy.n_samples
    k = min(2 * min_samples + (1 if exclude is not None else 0), n_train)
    distances, indices = parallel_tree_query(
        embedding_tree, query_positions, k=np.int64(k), output_rdist=False
    )
    indices = indices.astype(np.int64)
    if exclude is not None:
        drop = indices == np.asarray(exclude)[:, None]
        keep_order = np.argsort(drop, axis=1, kind="stable")[:, : k - 1]
        indices = np.take_along_axis(indices, keep_order, axis=1)
        distances = np.take_along_axis(distances, keep_order, axis=1)
    indices[~valid] = -1
    if min_samples > 1:
        core_query = distances[:, min(min_samples, indices.shape[1]) - 1].astype(
            np.float64
        )
    else:
        core_query = np.zeros(n, dtype=np.float64)
    nearest, lambdas = nearest_mutual_reachability(
        indices,
        distances.astype(np.float64),
        core_query,
        pdata.core_distances.astype(np.float64),
    )
    attached_by_tree = [
        place_in_tree(
            nearest,
            lambdas,
            arrays.root,
            arrays.point_parent,
            arrays.point_lambda,
            arrays.cluster_parent,
            arrays.cluster_birth,
        )
        for arrays in arrays_by_tree
    ]
    return embedding_layer_labels(attached_by_tree, hierarchy, arrays_by_tree)
