"""State kept by a fitted EVoC model for ``predict`` and ``update``."""
from collections import namedtuple

import numpy as np

from .clustering_utilities import extract_clusters_by_id
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
