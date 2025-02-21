import numpy as np
import numba

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
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
    n_samples = data.shape[0]
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
    clusters = get_cluster_label_vector(new_tree, leaves, 0.0, n_samples)
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
        clusters = get_cluster_label_vector(new_tree, leaves, 0.0, n_samples)
        strengths = get_point_membership_strength_vector(new_tree, leaves, clusters)
        n_clusters_in_layer = clusters.max() + 1

    return cluster_layers, membership_strength_layers


@numba.njit(cache=True)
def find_duplicates(knn_inds, knn_dists):
    duplicate_distance = np.max(knn_dists.T[0])
    duplicates = set([(-1, -1) for i in range(0)])
    for i in range(knn_inds.shape[0]):
        for j in range(0, knn_inds.shape[1]):
            if knn_dists[i, j] <= duplicate_distance:
                k = knn_inds[i, j]
                if i < k:
                    duplicates.add((i, k))
                elif k < i:
                    duplicates.add((k, i))
                else:
                    continue

    return duplicates


@numba.njit(cache=True)
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
        clusters = get_cluster_label_vector(lower_tree, leaves, 0.0, n_samples)
        strengths = get_point_membership_strength_vector(lower_tree, leaves, clusters)
        return clusters, strengths
    elif abs(lower_n_clusters - approx_n_clusters) > abs(
        upper_n_clusters - approx_n_clusters
    ):
        upper_tree = condense_tree(uncondensed_tree, upper_bound_min_cluster_size)
        leaves = extract_leaves(upper_tree)
        clusters = get_cluster_label_vector(upper_tree, leaves, 0.0, n_samples)
        strengths = get_point_membership_strength_vector(upper_tree, leaves, clusters)
        return clusters, strengths
    else:
        lower_tree = condense_tree(uncondensed_tree, lower_bound_min_cluster_size)
        lower_leaves = extract_leaves(lower_tree)
        lower_clusters = get_cluster_label_vector(lower_tree, lower_leaves)
        upper_tree = condense_tree(uncondensed_tree, upper_bound_min_cluster_size)
        upper_leaves = extract_leaves(upper_tree)
        upper_clusters = get_cluster_label_vector(upper_tree, upper_leaves, 0.0, n_samples)

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
    noise_level=0.5,
    base_min_cluster_size=5,
    min_num_clusters=4,
    approx_n_clusters=None,
    n_neighbors=15,
    min_samples=5,
    next_cluster_size_quantile=0.8,
    n_epochs=50,
    node_embedding_init="label_prop",
    symmetrize_graph=True,
    return_duplicates=False,
    node_embedding_dim=None,
    neighbor_scale=1.0,
):
    """Cluster data using the EVoC algorithm.

    Parameters
    ----------

    data : array-like of shape (n_samples, n_features)
        The data to cluster. If the data is float valued then it is assumed to use
        cosine distance as a matric. If the data is int8 valued then it is assumed
        that a quantized embedding is being used and a quantized version of cosine
        distance is used. If the data is uint8 valued then it is assumed that a
        binary embedding is being used, and a bitwise Jaccard distance is used.

    noise_level : float, default=0.5
        The noise level expected in the data. A value of 0.0 will try to cluster
        more data, at the expense of getting less accurate clustering. A value of
        1.0 will try for accurate clusters, discarding more data as noise to do so.

    base_min_cluster_size : int, default=5
        The minimum number of points in a cluster at the base layer of the clustering.
        This gives the finest granularity clustering that will be returned, with less
        graularity at higher layers.

    min_num_clusters : int, default=4
        The minimum number of clusters in the least granular layer of the clustering.
        Once a layer produces this many clusters or less no further layers will be
        produced.

    approx_n_clusters : int, default=None
        If not None, the algorithm will attempt to find the granularity of
        clustering that will give exactly this many clusters. Since the actual
        number of clusters cannot be guaranteed this is only approximate, but
        usually the algorithm can manage to get this exact number, assuming a
        resonable clustering into ``approx_n_clusters`` exists. When not None
        only this granularity will be returned -- no other cluster layers
        will be produced.

    n_neighbors : int, default=15
        The number of neighbors to use in the nearest neighbor graph construction.

    min_samples : int, default=5
        The minimum number of samples to use in the density estimation when
        performing density based clustering on the node embedding.

    next_cluster_size_quantile : float, default=0.8
        The quantile of cluster sizes to use when determining the minimum cluster
        size for the next layer of clustering. This is used to determine the
        granularity of clustering at each layer.

    n_epochs : int, default=50
        The number of epochs to use when training the node embedding.

    node_embedding_init : str or None, default='label_prop'
        The method to use to initialize the node embedding. If None, no initialization
        will be used. If 'label_prop', the label propagation method will be used.

    symmetrize_graph : bool, default=True
        Whether to symmetrize the nearest neighbor graph before using it to
        construct the node embedding.

    return_duplicates : bool, default=False
        Whether to return a set of duplicate pairs of points in the data.

    node_embedding_dim : int or None, default=None
        The number of dimensions to use in the node embedding. If None, a default
        value of min(n_neighbors, 15) will be used.

    neighbor_scale : float, default=1.0
        The scale factor to use when constructing the nearest neighbor graph. This
        can be used to increase the number of neighbors used in the graph construction
        by scaling the number of neighbors by this factor.

    Returns
    -------

    cluster_layers : list of array-like of shape (n_samples,)
        The clustering of the data at each layer of the clustering. Each layer
        is a clustering of the data into a different number of clusters.

    membership_strengths : list of array-like of shape (n_samples,)
        The membership strengths of each point in the clustering at each layer.
        This gives a measure of how strongly each point belongs to each cluster.

    duplicates : set of tuple of int
        Only returned in ``return_duplicates`` is True. A set of pairs of indices of
        potential duplicate points in the data.
    """
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

    if return_duplicates:
        duplicates = find_duplicates(nn_inds, nn_dists)

    if approx_n_clusters is not None:
        cluster_vector, strengths = binary_search_for_n_clusters(
            embedding,
            approx_n_clusters,
            min_samples=min_samples,
        )
        if return_duplicates:
            return [cluster_vector], [strengths], duplicates
        else:
            return [cluster_vector], [strengths]
    else:
        cluster_layers, membership_strengths = build_cluster_layers(
            embedding,
            min_clusters=min_num_clusters,
            min_samples=min_samples,
            base_min_cluster_size=base_min_cluster_size,
            next_cluster_size_quantile=next_cluster_size_quantile,
        )
        if return_duplicates:
            return cluster_layers, membership_strengths, duplicates
        else:
            return cluster_layers, membership_strengths


class EVoC(BaseEstimator, ClusterMixin):
    """
    Embedding Vector Oriented Clustering for efficient clustering of high-dimensional
    embedding vectors such as CLIP-vectors, sentence-transformers output, etc. The
    clustering uses a combination of a node embedding of a nearest neighbour graph,
    related to UMAP, and a density based clustering approach related to HDBSCAN,
    improving upon those approaches in efficiency and quality for the specific case
    of high-dimensional embedding vectors.

    Parameters
    ----------

    noise_level : float, default=0.5
        The noise level expected in the data. A value of 0.0 will try to cluster
        more data, at the expense of getting less accurate clustering. A value of
        1.0 will try for accurate clusters, discarding more data as noise to do so.

    base_min_cluster_size : int, default=5
        The minimum number of points in a cluster at the base layer of the clustering.
        This gives the finest granularity clustering that will be returned, with less
        graularity at higher layers.

    min_num_clusters : int, default=4
        The minimum number of clusters in the least granular layer of the clustering.
        Once a layer produces this many clusters or less no further layers will be
        produced.

    approx_n_clusters : int, default=None
        If not None, the algorithm will attempt to find the granularity of
        clustering that will give exactly this many clusters. Since the actual
        number of clusters cannot be guaranteed this is only approximate, but
        usually the algorithm can manage to get this exact number, assuming a
        resonable clustering into ``approx_n_clusters`` exists. When not None
        only this granularity will be returned -- no other cluster layers
        will be produced.

    n_neighbors : int, default=15
        The number of neighbors to use in the nearest neighbor graph construction.

    min_samples : int, default=5
        The minimum number of samples to use in the density estimation when
        performing density based clustering on the node embedding.

    next_cluster_size_quantile : float, default=0.8
        The quantile of cluster sizes to use when determining the minimum cluster
        size for the next layer of clustering. This is used to determine the
        granularity of clustering at each layer.

    n_epochs : int, default=50
        The number of epochs to use when training the node embedding.

    node_embedding_init : str or None, default='label_prop'
        The method to use to initialize the node embedding. If None, no initialization
        will be used. If 'label_prop', the label propagation method will be used.

    symmetrize_graph : bool, default=True
        Whether to symmetrize the nearest neighbor graph before using it to
        construct the node embedding.

    node_embedding_dim : int or None, default=None
        The number of dimensions to use in the node embedding. If None, a default
        value of min(n_neighbors, 15) will be used.

    neighbor_scale : float, default=1.0
        The scale factor to use when constructing the nearest neighbor graph. This
        can be used to increase the number of neighbors used in the graph construction
        by scaling the number of neighbors by this factor.

    Attributes
    ----------

    labels_ : array-like of shape (n_samples,)
        An array of labels for the data samples; this is a integer array as per other scikit-learn
        clustering algorithms. A value of -1 indicates that a point is a noise point and
        not in any cluster.

    membership_strengths_ : array-like of shape (n_samples,)
        An array of membership strengths for the data samples; this gives a measure of how
        strongly each point belongs to each cluster. This is a floating point array with
        values between 0 and 1.

    cluster_layers_ : list of array-like of shape (n_samples,)
        The clustering of the data at each layer of the clustering. Each layer
        is a clustering of the data into a different number of clusters; the earlier the
        cluster vector is in this list the finer the granularity of clustering.

    membership_strength_layers_ : list of array-like of shape (n_samples,)
        The membership strengths of each point in the clustering at each layer.

    cluster_tree_ : dict
        A dictionary representing the hierarchical clustering of the data. The keys are
        tuples of (layer, cluster) and the values are lists of tuples of (layer, cluster)
        representing the children of the key cluster.

    duplicates_ : set of tuple of int
        A set of pairs of indices of potential duplicate points in the data.
    """

    def __init__(
        self,
        noise_level=0.5,
        base_min_cluster_size=5,
        min_num_clusters=4,
        approx_n_clusters=None,
        n_neighbors=15,
        min_samples=5,
        next_cluster_size_quantile=0.8,
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
        """Fit the model to the data and return the clustering labels.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            The data to cluster. If the data is float valued then it is assumed to use
            cosine distance as a matric. If the data is int8 valued then it is assumed
            that a quantized embedding is being used and a quantized version of cosine
            distance is used. If the data is uint8 valued then it is assumed that a
            binary embedding is being used, and a bitwise Jaccard distance is used.

        y : array-like of shape (n_samples,), default=None
            Ignored. This parameter exists only for compatibility with
            scikit-learn's fit_predict method.

        Returns
        -------

        labels_ : array-like of shape (n_samples,)
            An array of labels for the data samples; this is a integer array as per other scikit-learn
            clustering algorithms. A value of -1 indicates that a point is a noise point and
            not in any cluster.

        """

        X = check_array(X)

        self.cluster_layers_, self.membership_strength_layers_, self.duplicates_ = (
            evoc_clusters(
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
                return_duplicates=True,
                node_embedding_dim=self.node_embedding_dim,
                neighbor_scale=self.neighbor_scale,
            )
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
        """Fit the model to the data.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            The data to cluster. If the data is float valued then it is assumed to use
            cosine distance as a matric. If the data is int8 valued then it is assumed
            that a quantized embedding is being used and a quantized version of cosine
            distance is used. If the data is uint8 valued then it is assumed that a
            binary embedding is being used, and a bitwise Jaccard distance is used.

        y : array-like of shape (n_samples,), default=None
            Ignored. This parameter exists only for compatibility with
            scikit-learn's fit method.

        Returns
        -------

        self : sklearn Estimator
            Returns the instance itself.
        """
        self.fit_predict(X, y, **fit_params)
        return self

    @property
    def cluster_tree_(self):
        check_is_fitted(
            self,
            "cluster_layers_",
            msg="This %(name)s instance is not fitted yet, and 'cluster_tree_' is not available. "
            "Please call 'fit' with appropriate arguments before accessing this attribute.",
        )
        return build_cluster_tree(self.cluster_layers_)
