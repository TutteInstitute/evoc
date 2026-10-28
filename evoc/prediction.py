"""State kept by a fitted EVoC model for ``predict`` and ``update``."""
from collections import namedtuple

import numba
import numpy as np

from .clustering_utilities import extract_clusters_by_id
from .graph_construction import smooth_knn_dist
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
    ],
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


def labels_from_hierarchy(hierarchy, layer):
    """Labels and membership strengths of a layer, re-derived from the tree.

    Reproduces the layer's labels in ``cluster_layers_`` exactly.
    """
    tree = hierarchy.condensed_trees[hierarchy.layer_tree[layer]]
    clusters = hierarchy.layer_clusters[layer]
    if len(clusters) == 0 or len(tree.parent) == 0:
        n = hierarchy.n_samples
        return np.full(n, -1, dtype=np.intp), np.zeros(n, dtype=np.float32)
    return extract_clusters_by_id(tree, np.asarray(clusters, dtype=np.int64))


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
@numba.njit(parallel=True, fastmath=True, cache=True)
def membership_weights(indices, distances, sigmas, rhos):
    """Edge weights from each query to its neighbours.

    Uses the same formula as ``compute_membership_strengths``; ``indices`` and
    ``distances`` exclude the query itself. Missing neighbours (-1) get 0.
    """
    n, k = indices.shape
    weights = np.zeros((n, k), dtype=np.float32)
    for i in numba.prange(n):
        rho = rhos[i]
        sigma = sigmas[i]
        for j in range(k):
            if indices[i, j] < 0:
                continue
            d = distances[i, j]
            if (d - rho) <= 0.0 or sigma == 0.0:
                weights[i, j] = 1.0
            else:
                weights[i, j] = np.exp(-((d - rho) / sigma))
    return weights


def query_bandwidths(distances, pdata):
    """``smooth_knn_dist`` bandwidths for queries with neighbour distances
    ``distances`` (self excluded), computed as for the training points."""
    n = distances.shape[0]
    with_self = np.zeros((n, distances.shape[1] + 1), dtype=np.float32)
    with_self[:, 1:] = distances
    params = pdata.params
    return smooth_knn_dist(
        with_self,
        float(params["neighbor_scale"] * params["n_neighbors"]),
        mean_distance=np.float64(pdata.mean_knn_distance),
    )


@numba.njit(parallel=True, fastmath=True, cache=True)
def vote_scores(indices, weights, layer_labels, layer_strengths):
    """Weighted neighbour vote for every layer.

    For each query and layer, the score of cluster c is the sum over neighbours
    labelled c of (edge weight * neighbour membership strength). Returns the
    best cluster (-1 if no neighbour is clustered) and its score divided by the
    total edge weight, each of shape (n_layers, n_queries).
    """
    n, k = indices.shape
    n_layers = layer_labels.shape[0]
    best_labels = np.full((n_layers, n), -1, dtype=np.int64)
    best_scores = np.zeros((n_layers, n), dtype=np.float32)
    for i in numba.prange(n):
        total = np.float32(0.0)
        for j in range(k):
            if indices[i, j] >= 0:
                total += weights[i, j]
        if total <= 0.0:
            continue
        cluster_ids = np.empty(k, dtype=np.int64)
        cluster_scores = np.empty(k, dtype=np.float32)
        for layer in range(n_layers):
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
    return best_labels, best_scores


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
