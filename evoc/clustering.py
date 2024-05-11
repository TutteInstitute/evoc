import numpy as np
import numba

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from sklearn.neighbors import KDTree

from .numba_kdtree import kdtree_to_numba
from .boruvka import parallel_boruvka
from .cluster_trees import (
    mst_to_linkage_tree,
    condense_tree,
    extract_leaves,
    get_cluster_label_vector,
    get_point_membership_strength_vector,
)
from .knn_graph import knn_graph
from .label_propagation import label_propagation_init
from .node_embedding import node_embedding
from .graph_construction import neighbor_graph_matrix


def build_cluster_layers(
    data,
    *,
    min_clusters=3,
    min_samples=5,
    base_min_cluster_size=10,
    next_cluster_size_quantile=0.8,
):
    cluster_layers = []
    membership_strength_layers = []

    min_cluster_size = base_min_cluster_size

    sklearn_tree = KDTree(data)
    numba_tree = kdtree_to_numba(sklearn_tree)
    edges = parallel_boruvka(
        numba_tree, min_samples=min_cluster_size if min_samples is None else min_samples
    )
    sorted_mst = edges[np.argsort(edges.T[2])]
    uncondensed_tree = mst_to_linkage_tree(sorted_mst)
    new_tree = condense_tree(uncondensed_tree, base_min_cluster_size)
    leaves = extract_leaves(new_tree)
    clusters = get_cluster_label_vector(new_tree, leaves)
    strengths = get_point_membership_strength_vector(new_tree, leaves, clusters)
    n_clusters_in_layer = clusters.max() + 1

    while n_clusters_in_layer >= min_clusters:
        cluster_layers.append(clusters)
        membership_strength_layers.append(strengths)
        cluster_sizes = np.bincount(clusters[clusters >= 0])
        next_min_cluster_size = int(
            np.quantile(cluster_sizes, next_cluster_size_quantile)
        )
        if next_min_cluster_size <= min_cluster_size + 1:
            break
        else:
            min_cluster_size = next_min_cluster_size
        new_tree = condense_tree(uncondensed_tree, min_cluster_size)
        leaves = extract_leaves(new_tree)
        clusters = get_cluster_label_vector(new_tree, leaves)
        strengths = get_point_membership_strength_vector(new_tree, leaves, clusters)
        n_clusters_in_layer = clusters.max() + 1

    return cluster_layers, membership_strength_layers


@numba.njit()
def _build_cluster_tree(labels):
    mapping = [(-1, -1, -1, -1) for i in range(0)]
    found = [set([-1]) for i in range(len(labels))]
    mapping_idx = 0
    for upper_layer in range(1, len(labels)):
        upper_layer_unique_labels = np.unique(labels[upper_layer])
        for lower_layer in range(upper_layer - 1, -1, -1):
            upper_cluster_order = np.argsort(labels[upper_layer])
            cluster_groups = np.split(
                labels[lower_layer][upper_cluster_order],
                np.cumsum(np.bincount(labels[upper_layer] + 1))[:-1],
            )
            for i, label in enumerate(upper_layer_unique_labels):
                if label >= 0:
                    for child in cluster_groups[i]:
                        if child >= 0 and child not in found[lower_layer]:
                            mapping.append((upper_layer, label, lower_layer, child))
                            found[lower_layer].add(child)

    for lower_layer in range(len(labels) - 1, -1, -1):
        for child in range(labels[lower_layer].max() + 1):
            if child >= 0 and child not in found[lower_layer]:
                mapping.append((len(labels), 0, lower_layer, child))

    return mapping


def build_cluster_tree(labels):
    result = {}
    raw_mapping = _build_cluster_tree(labels)
    for parent_layer, parent_cluster, child_layer, child_cluster in raw_mapping:
        parent_name = (parent_layer, parent_cluster)
        if parent_name in result:
            result[parent_name].append((child_layer, child_cluster))
        else:
            result[parent_name] = [(child_layer, child_cluster)]
    return result


@numba.njit(cache=True)
def _binary_search_for_n_clusters(uncondensed_tree, approx_n_clusters, n_samples):
    lower_bound_min_cluster_size = 2
    upper_bound_min_cluster_size = n_samples // 2
    mid_min_cluster_size = int(
        round((lower_bound_min_cluster_size + upper_bound_min_cluster_size) / 2.0)
    )
    min_n_clusters = 0

    upper_tree = condense_tree(uncondensed_tree, upper_bound_min_cluster_size)
    leaves = extract_leaves(upper_tree)
    upper_n_clusters = len(leaves)

    lower_tree = condense_tree(uncondensed_tree, lower_bound_min_cluster_size)
    leaves = extract_leaves(lower_tree)
    lower_n_clusters = len(leaves)

    while upper_bound_min_cluster_size - lower_bound_min_cluster_size > 1:
        mid_min_cluster_size = int(
            round((lower_bound_min_cluster_size + upper_bound_min_cluster_size) / 2.0)
        )
        if (
            mid_min_cluster_size == lower_bound_min_cluster_size
            or mid_min_cluster_size == upper_bound_min_cluster_size
        ):
            break
        mid_tree = condense_tree(uncondensed_tree, mid_min_cluster_size)
        leaves = extract_leaves(mid_tree)
        mid_n_clusters = len(leaves)

        if mid_n_clusters < approx_n_clusters:
            upper_bound_min_cluster_size = mid_min_cluster_size
            upper_n_clusters = mid_n_clusters
        elif mid_n_clusters >= approx_n_clusters:
            lower_bound_min_cluster_size = mid_min_cluster_size
            lower_n_clusters = mid_n_clusters

    if abs(lower_n_clusters - approx_n_clusters) < abs(
        upper_n_clusters - approx_n_clusters
    ):
        lower_tree = condense_tree(uncondensed_tree, lower_bound_min_cluster_size)
        leaves = extract_leaves(lower_tree)
        clusters = get_cluster_label_vector(lower_tree, leaves)
        strengths = get_point_membership_strength_vector(lower_tree, leaves, clusters)
        return clusters, strengths
    elif abs(lower_n_clusters - approx_n_clusters) > abs(
        upper_n_clusters - approx_n_clusters
    ):
        upper_tree = condense_tree(uncondensed_tree, upper_bound_min_cluster_size)
        leaves = extract_leaves(upper_tree)
        clusters = get_cluster_label_vector(upper_tree, leaves)
        strengths = get_point_membership_strength_vector(upper_tree, leaves, clusters)
        return clusters, strengths
    else:
        lower_tree = condense_tree(uncondensed_tree, lower_bound_min_cluster_size)
        lower_leaves = extract_leaves(lower_tree)
        lower_clusters = get_cluster_label_vector(lower_tree, lower_leaves)
        upper_tree = condense_tree(uncondensed_tree, upper_bound_min_cluster_size)
        upper_leaves = extract_leaves(upper_tree)
        upper_clusters = get_cluster_label_vector(upper_tree, upper_leaves)

        if np.sum(lower_clusters >= 0) > np.sum(upper_clusters >= 0):
            strengths = get_point_membership_strength_vector(
                lower_tree, lower_leaves, lower_clusters
            )
            return lower_clusters, strengths
        else:
            strengths = get_point_membership_strength_vector(
                upper_tree, upper_leaves, upper_clusters
            )
            return upper_clusters, strengths


def binary_search_for_n_clusters(
    data,
    approx_n_clusters,
    *,
    min_samples=5,
):
    sklearn_tree = KDTree(data)
    numba_tree = kdtree_to_numba(sklearn_tree)
    edges = parallel_boruvka(numba_tree, min_samples=min_samples)
    sorted_mst = edges[np.argsort(edges.T[2])]
    uncondensed_tree = mst_to_linkage_tree(sorted_mst)

    n_samples = data.shape[0]

    return _binary_search_for_n_clusters(uncondensed_tree, approx_n_clusters, n_samples)


def evoc_clusters(
    data,
    n_neighbors=15,
    noise_level=0.5,
    base_min_cluster_size=5,
    min_num_clusters=4,
    approx_n_clusters=None,
    next_cluster_size_quantile=0.8,
    min_samples=5,
    n_epochs=50,
    node_embedding_init="label_prop",
    symmetrize_graph=True,
    node_embedding_dim=None,
    neighbor_scale=1.0,
):
    nn_inds, nn_dists = knn_graph(data, n_neighbors=n_neighbors)
    graph = neighbor_graph_matrix(
        neighbor_scale * n_neighbors, nn_inds, nn_dists, symmetrize_graph
    )
    if node_embedding_init == "label_prop":
        init_embedding = label_propagation_init(
            graph,
            n_components=node_embedding_dim or min(n_neighbors, 15),
            approx_n_parts=np.clip(int(np.sqrt(data.shape[0])), 100, 1024),
            random_scale=0.1,
            scaling=0.5,
            noise_level=noise_level,
        )
    elif node_embedding_init is None:
        init_embedding = None

    graph = graph.tocoo()
    embedding = node_embedding(
        graph,
        n_components=min(n_neighbors, 15),
        n_epochs=n_epochs,
        initial_embedding=init_embedding,
        negative_sample_rate=1.0,
        noise_level=noise_level,
        verbose=False,
    )

    if approx_n_clusters is not None:
        cluster_vector, strengths = binary_search_for_n_clusters(
            embedding,
            approx_n_clusters,
            min_samples=min_samples,
        )
        return [cluster_vector], [strengths]
    else:
        cluster_layers, membership_strengths = build_cluster_layers(
            embedding,
            min_clusters=min_num_clusters,
            min_samples=min_samples,
            base_min_cluster_size=base_min_cluster_size,
            next_cluster_size_quantile=next_cluster_size_quantile,
        )
        
        return cluster_layers, membership_strengths


class EVoC (BaseEstimator, ClusterMixin):
    def __init__(
        self,
        n_neighbors=15,
        noise_level=0.5,
        base_min_cluster_size=5,
        min_num_clusters=4,
        approx_n_clusters=None,
        next_cluster_size_quantile=0.8,
        min_samples=5,
        n_epochs=50,
        node_embedding_init="label_prop",
        symmetrize_graph=True,
        node_embedding_dim=None,
        neighbor_scale=1.0,
    ):
        self.n_neighbors = n_neighbors
        self.noise_level = noise_level
        self.base_min_cluster_size = base_min_cluster_size
        self.min_num_clusters = min_num_clusters
        self.approx_n_clusters = approx_n_clusters
        self.next_cluster_size_quantile = next_cluster_size_quantile
        self.min_samples = min_samples
        self.n_epochs = n_epochs
        self.node_embedding_init = node_embedding_init
        self.symmetrize_graph = symmetrize_graph
        self.node_embedding_dim = node_embedding_dim
        self.neighbor_scale = neighbor_scale

    def fit_predict(self, X, y=None, **fit_params):

        X = check_array(X)

        self.cluster_layers_, self.membership_strength_layers_ = evoc_clusters(
            X,
            n_neighbors=self.n_neighbors,
            noise_level=self.noise_level,
            base_min_cluster_size=self.base_min_cluster_size,
            min_num_clusters=self.min_num_clusters,
            approx_n_clusters=self.approx_n_clusters,
            next_cluster_size_quantile=self.next_cluster_size_quantile,
            min_samples=self.min_samples,
            n_epochs=self.n_epochs,
            node_embedding_init=self.node_embedding_init,
            symmetrize_graph=self.symmetrize_graph,
            node_embedding_dim=self.node_embedding_dim,
            neighbor_scale=self.neighbor_scale,
        )

        if len(self.cluster_layers_) == 1:
            self.labels_ = self.cluster_layers_[0]
            self.membership_strengths_ = self.membership_strength_layers_[0]
        else:
            n_points_clustered_per_layer = [
                np.sum(layer >= 0) for layer in self.cluster_layers_
            ]
            best_layer = np.argmax(n_points_clustered_per_layer)
            self.labels_ = self.cluster_layers_[best_layer]
            self.membership_strengths_ = self.membership_strength_layers_[best_layer]

        return self.labels_

    def fit(self, X, y=None, **fit_params):
        self.fit_predict(X, y, **fit_params)
        return self

