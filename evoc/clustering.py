from contextlib import contextmanager

import numpy as np
import numba

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted

from .numba_kdtree import build_kdtree
from .boruvka import parallel_boruvka
from .cluster_trees import (
    mst_to_linkage_tree,
    condense_tree,
    mask_condensed_tree,
    extract_leaves,
    get_cluster_label_vector,
    get_point_membership_strength_vector,
)
from .clustering_utilities import (
    find_peaks,
    _binary_search_for_n_clusters,
    binary_search_for_n_clusters,
    min_cluster_size_barcode,
    compute_total_persistence,
    extract_clusters_by_id,
    select_diverse_peaks,
    build_cluster_tree,
    find_duplicates,
)
from .knn_graph import knn_graph
from .label_propagation import label_propagation_init
from .node_embedding import node_embedding
from .graph_construction import neighbor_graph_matrix
from .prediction import (
    SERIAL_BATCH_SIZE,
    ClusterHierarchy,
    PredictionData,
    apply_thresholds,
    as_cluster_ids,
    calibrate_vote_thresholds,
    embedding_core_distances,
    membership_weights,
    predict_from_positions,
    query_bandwidths,
    refine_positions,
    stack_layers,
    tree_arrays,
    vote_scores,
    weighted_positions,
)
from .search.kernels import prepare_data


@contextmanager
def _single_threaded(enabled=True):
    """Run numba parallel code on the calling thread only (thread-local), which
    avoids the cost of waking the thread pool for small batches."""
    if not enabled:
        yield
        return
    previous = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        yield
    finally:
        numba.set_num_threads(previous)


def build_cluster_layers(
    data,
    *,
    min_samples=5,
    base_min_cluster_size=10,
    base_n_clusters=None,
    reproducible_flag=False,
    min_similarity_threshold=0.2,
    max_layers=10,
    return_tree=False,
):
    """Build hierarchical cluster layers from embedding data.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The embedding data to cluster. Typically the output of a node embedding
        algorithm.

    min_samples : int, default=5
        The minimum number of samples to use in the density estimation when
        performing density based clustering.

    base_min_cluster_size : int, default=10
        The minimum number of points in a cluster at the base layer of the clustering.
        This gives the finest granularity clustering that will be returned.

    base_n_clusters : int or None, default=None
        If not None, the algorithm will attempt to find the granularity of
        clustering that will give exactly this many clusters for the bottom-most layer
        of clustering. This affects the base layer computation and allows multiple
        layers to be built on top of this base.

    reproducible_flag : bool, default=False
        Whether to ensure reproducible results by using deterministic algorithms
        where possible.

    min_similarity_threshold : float, default=0.2
        The minimum similarity threshold for cluster layer selection. Peaks that result
        in clusterings with Jaccard similarity above this threshold will be filtered out
        to ensure diverse cluster layers.

    max_layers : int, default=10
        The maximum number of cluster layers to return. The algorithm will select up to
        this many diverse peaks based on persistence and similarity criteria.

    return_tree : bool, default=False
        If True, also return a :class:`evoc.prediction.ClusterHierarchy`
        recording the condensed tree(s) and the clusters selected for each
        layer, and the core distances of the points in ``data``.

    Returns
    -------
    cluster_layers : list of array-like of shape (n_samples,)
        The clustering of the data at each layer of the clustering. Each layer
        is a clustering of the data into a different number of clusters.

    membership_strength_layers : list of array-like of shape (n_samples,)
        The membership strengths of each point in the clustering at each layer.
        This gives a measure of how strongly each point belongs to each cluster.

    persistence_scores : list of float
        The persistence scores for each cluster layer, indicating the quality or
        stability of the clustering at that layer.

    hierarchy : ClusterHierarchy
        Only returned if ``return_tree`` is True.

    core_distances : array of shape (n_samples,)
        Only returned if ``return_tree`` is True.
    """
    n_samples = data.shape[0]
    min_cluster_size = base_min_cluster_size
    cluster_layers = []
    membership_strength_layers = []
    persistence_scores = []

    n_threads = numba.get_num_threads()

    effective_min_samples = min_cluster_size if min_samples is None else min_samples
    numba_tree = build_kdtree(data.astype(np.float32))
    edges = parallel_boruvka(
        numba_tree,
        n_threads,
        min_samples=effective_min_samples,
        reproducible=reproducible_flag,
    )
    sorted_mst = edges[np.argsort(edges.T[2])]
    uncondensed_tree = mst_to_linkage_tree(sorted_mst)
    if base_n_clusters is not None:
        leaves, clusters, strengths, base_tree = _binary_search_for_n_clusters(
            uncondensed_tree, base_n_clusters, n_samples=n_samples, return_tree=True
        )
        cluster_sizes = np.bincount(clusters[clusters >= 0])
        if len(cluster_sizes) > 0:
            min_cluster_size = max(1, np.min(cluster_sizes))
        else:
            min_cluster_size = base_min_cluster_size
        # Still need condensed tree for later processing
        condensed_tree = condense_tree(uncondensed_tree, min_cluster_size)
    else:
        condensed_tree = condense_tree(uncondensed_tree, base_min_cluster_size)
        leaves = extract_leaves(condensed_tree)
        clusters = get_cluster_label_vector(condensed_tree, leaves, 0.0, n_samples)
        strengths = get_point_membership_strength_vector(
            condensed_tree, leaves, clusters
        )
        base_tree = condensed_tree

    # Trees and selected clusters per layer, for return_tree
    condensed_trees = [condensed_tree]
    if base_tree is not condensed_tree:
        condensed_trees.append(base_tree)
    layer_tree = [len(condensed_trees) - 1]
    layer_clusters = [as_cluster_ids(leaves)]

    mask = condensed_tree.child >= n_samples
    cluster_tree = mask_condensed_tree(condensed_tree, mask)
    # points_tree = mask_condensed_tree(condensed_tree, ~mask)

    # Check if cluster_tree is valid before processing
    if len(cluster_tree.child) > 0 and cluster_tree.child[-1] >= n_samples:
        births, deaths, parents, lambda_deaths = min_cluster_size_barcode(
            cluster_tree, n_samples, min_cluster_size
        )
        sizes, total_persistence = compute_total_persistence(
            births, deaths, lambda_deaths
        )
        peaks = find_peaks(total_persistence)
    else:
        # Handle empty or invalid cluster tree
        births = np.array([])
        deaths = np.array([])
        parents = np.array([])
        lambda_deaths = np.array([])
        sizes = np.array([])
        total_persistence = np.array([])
        peaks = np.array([], dtype=np.int64)

    # Always include the base layer (from initial condensed tree)
    cluster_layers.append(clusters)
    membership_strength_layers.append(strengths)
    persistence_scores.append(0.0)  # Base layer gets 0 persistence score

    # Select diverse peaks using hierarchical selection
    selected_peaks = select_diverse_peaks(
        peaks,
        total_persistence,
        sizes,
        births,
        deaths,
        min_similarity_threshold=min_similarity_threshold,
        max_layers=max_layers - 1,  # Reserve one slot for base layer
    )

    for peak in selected_peaks:
        best_birth = sizes[peak]
        persistence = total_persistence[peak]
        selected_clusters = (
            np.where((births <= best_birth) & (deaths > best_birth))[0] + n_samples
        )
        labels, strengths = extract_clusters_by_id(condensed_tree, selected_clusters)
        layer_tree.append(0)
        layer_clusters.append(as_cluster_ids(selected_clusters))
        cluster_layers.append(labels)
        membership_strength_layers.append(strengths)
        persistence_scores.append(persistence)

    # Sort cluster layers by number of clusters (most clusters first)
    n_clusters_per_layer = [layer.max() + 1 for layer in cluster_layers]
    sorted_indices = np.argsort(n_clusters_per_layer)[::-1]  # Descending order

    cluster_layers = [cluster_layers[i] for i in sorted_indices]
    membership_strength_layers = [membership_strength_layers[i] for i in sorted_indices]
    persistence_scores = [persistence_scores[i] for i in sorted_indices]

    if return_tree:
        hierarchy = ClusterHierarchy(
            condensed_trees=condensed_trees,
            layer_tree=[layer_tree[i] for i in sorted_indices],
            layer_clusters=[layer_clusters[i] for i in sorted_indices],
            min_cluster_size=min_cluster_size,
            min_samples=effective_min_samples,
            n_samples=n_samples,
        )
        core_distances = embedding_core_distances(
            data, effective_min_samples, tree=numba_tree
        )
        return (
            cluster_layers,
            membership_strength_layers,
            persistence_scores,
            hierarchy,
            core_distances,
        )
    return cluster_layers, membership_strength_layers, persistence_scores


def cluster_embedding(
    embedding,
    *,
    approx_n_clusters=None,
    min_samples=5,
    base_min_cluster_size=5,
    base_n_clusters=None,
    reproducible_flag=True,
    min_similarity_threshold=0.2,
    max_layers=10,
    return_tree=False,
):
    """Cluster a node embedding into layers, as ``evoc_clusters`` does.

    Returns ``(cluster_layers, membership_strengths, persistence_scores,
    hierarchy, core_distances)``; the last two are None unless
    ``return_tree`` is True.
    """
    n_threads = numba.get_num_threads()
    hierarchy = None
    core_distances = None
    if approx_n_clusters is not None:
        if return_tree:
            cluster_vector, strengths, condensed_tree, leaves = (
                binary_search_for_n_clusters(
                    embedding,
                    approx_n_clusters,
                    n_threads,
                    min_samples=min_samples,
                    return_tree=True,
                )
            )
            hierarchy = ClusterHierarchy(
                condensed_trees=[condensed_tree],
                layer_tree=[0],
                layer_clusters=[as_cluster_ids(leaves)],
                min_cluster_size=None,
                min_samples=min_samples,
                n_samples=embedding.shape[0],
            )
            core_distances = embedding_core_distances(embedding, min_samples)
        else:
            cluster_vector, strengths = binary_search_for_n_clusters(
                embedding,
                approx_n_clusters,
                n_threads,
                min_samples=min_samples,
            )
        return [cluster_vector], [strengths], [0.0], hierarchy, core_distances

    layer_results = build_cluster_layers(
        embedding,
        min_samples=min_samples,
        base_min_cluster_size=base_min_cluster_size,
        base_n_clusters=base_n_clusters,
        reproducible_flag=reproducible_flag,
        min_similarity_threshold=min_similarity_threshold,
        max_layers=max_layers,
        return_tree=return_tree,
    )
    if return_tree:
        return layer_results
    return (*layer_results, None, None)


def evoc_clusters(
    data,
    noise_level=0.5,
    base_min_cluster_size=5,
    base_n_clusters=None,
    approx_n_clusters=None,
    n_neighbors=15,
    min_samples=5,
    n_epochs=50,
    node_embedding_init="label_prop",
    symmetrize_graph=True,
    return_duplicates=False,
    node_embedding_dim=None,
    neighbor_scale=1.0,
    random_state=None,
    reproducible_flag=True,
    min_similarity_threshold=0.2,
    max_layers=10,
    n_label_prop_iter=20,
    prediction_data=False,
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

    base_n_clusters : int, default=None
        If not None, the algorithm will attempt to find the granularity of
        clustering that will give exactly this many clusters for the bottom-most layer
        of clustering. This affects the base layer computation and allows multiple
        layers to be built on top of this base. Since the actual number of clusters
        cannot be guaranteed this is only approximate, but usually the algorithm can
        manage to get this exact number, assuming a reasonable clustering into
        ``base_n_clusters`` exists.

    approx_n_clusters : int, default=None
        If not None, the algorithm will attempt to find the granularity of
        clustering that will give exactly this many clusters as the final output.
        Unlike ``base_n_clusters``, when this parameter is set, only a single
        clustering layer will be returned -- no hierarchical layers will be produced.
        This is useful when you know the exact number of clusters you want and don't
        need the multi-layer analysis. Since the actual number of clusters cannot be
        guaranteed this is only approximate, but usually the algorithm can manage to
        get this exact number, assuming a reasonable clustering into ``approx_n_clusters``
        exists.

    n_neighbors : int, default=15
        The number of neighbors to use in the nearest neighbor graph construction.

    min_samples : int, default=5
        The minimum number of samples to use in the density estimation when
        performing density based clustering on the node embedding.

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
        value of min(max(n_neighbors // 4, 4), 15) will be used.

    neighbor_scale : float, default=1.0
        The scale factor to use when constructing the nearest neighbor graph. This
        multiplies the effective number of neighbors used in graph construction
        (neighbor_scale * n_neighbors). Values > 1.0 create denser graphs with more
        connectivity, potentially capturing more global structure but at increased
        computational cost. Values < 1.0 create sparser graphs focused on local
        structure.

    random_state : np.random.RandomState or None, default=None
        The random state to use for the random number generator. If None, the random
        number generator will not be seeded and will use the system time as the seed.

    reproducible_flag : bool, default=True
        Whether to ensure reproducible results by using deterministic algorithms
        where possible. When True, the clustering results should be consistent
        across runs with the same random_state.

    min_similarity_threshold : float, default=0.2
        The minimum similarity threshold for cluster layer selection. Peaks that result
        in clusterings with Jaccard similarity above this threshold will be filtered out
        to ensure diverse cluster layers.

    max_layers : int, default=10
        The maximum number of cluster layers to return. The algorithm will select up to
        this many diverse peaks based on persistence and similarity criteria.

    n_label_prop_iter : int, default=20
        The number of iterations to use in the label propagation algorithm when
        initializing the node embedding.

    prediction_data : bool, default=False
        If True, also return a :class:`evoc.prediction.PredictionData` record
        holding what ``EVoC.predict`` and ``EVoC.update`` need (the prepared
        training data, k-NN bandwidths, node embedding, condensed trees and
        core distances). Clustering results are unaffected.

    Returns
    -------

    cluster_layers : list of array-like of shape (n_samples,)
        The clustering of the data at each layer of the clustering. Each layer
        is a clustering of the data into a different number of clusters.

    membership_strengths : list of array-like of shape (n_samples,)
        The membership strengths of each point in the clustering at each layer.
        This gives a measure of how strongly each point belongs to each cluster.

    nn_inds : array-like of shape (n_samples, n_neighbors)
        Indices of nearest neighbors for each sample.

    nn_dists : array-like of shape (n_samples, n_neighbors)
        Distance from each sample to each nearest neighbor indexed by nn_inds

    duplicates : set of tuple of int
        Only returned in ``return_duplicates`` is True. A set of pairs of indices of
        potential duplicate points in the data.

    prediction_data : PredictionData
        Only returned if ``prediction_data`` is True; always the last element.
    """
    if node_embedding_init not in ("label_prop", None):
        raise ValueError(
            f"node_embedding_init should be 'label_prop' or None, not {node_embedding_init!r}"
        )

    if random_state is None:
        random_state = np.random.RandomState()

    nn_inds, nn_dists = knn_graph(
        data, n_neighbors=n_neighbors, random_state=random_state
    )
    if prediction_data:
        graph, sigmas, rhos = neighbor_graph_matrix(
            neighbor_scale * n_neighbors,
            nn_inds,
            nn_dists,
            symmetrize_graph,
            return_bandwidths=True,
        )
    else:
        graph = neighbor_graph_matrix(
            neighbor_scale * n_neighbors, nn_inds, nn_dists, symmetrize_graph
        )
    n_embedding_components = node_embedding_dim or min(max(n_neighbors // 4, 4), 15)
    if node_embedding_init == "label_prop":
        init_embedding = label_propagation_init(
            graph,
            n_components=n_embedding_components,
            approx_n_parts=np.clip(int(8 * np.sqrt(data.shape[0])), 256, 16384),
            random_scale=0.1,
            scaling=0.5,
            noise_level=noise_level,
            random_state=random_state,
            data=data,
            n_label_prop_iter=n_label_prop_iter,
        )
    elif node_embedding_init is None:
        init_embedding = None

    embedding = node_embedding(
        graph,
        n_components=n_embedding_components,
        n_epochs=n_epochs,
        initial_embedding=init_embedding,
        negative_sample_rate=1.0,
        noise_level=noise_level,
        random_state=random_state,
        verbose=False,
        reproducible_flag=reproducible_flag,
        initial_alpha=0.1,
    )

    if return_duplicates:
        duplicates = find_duplicates(nn_inds, nn_dists)

    (
        cluster_layers,
        membership_strengths,
        persistence_scores,
        hierarchy,
        core_distances,
    ) = cluster_embedding(
        embedding,
        approx_n_clusters=approx_n_clusters,
        min_samples=min_samples,
        base_min_cluster_size=base_min_cluster_size,
        base_n_clusters=base_n_clusters,
        reproducible_flag=reproducible_flag,
        min_similarity_threshold=min_similarity_threshold,
        max_layers=max_layers,
        return_tree=prediction_data,
    )

    result = [
        cluster_layers,
        membership_strengths,
        persistence_scores,
        nn_inds,
        nn_dists,
    ]
    if return_duplicates:
        result.append(duplicates)
    if prediction_data:
        prepared_data, inv_norms = prepare_data(data)
        if prepared_data is data:
            prepared_data = prepared_data.copy()
        result.append(
            PredictionData(
                data=prepared_data,
                input_dtype=prepared_data.dtype.type,
                inv_norms=inv_norms,
                sigmas=sigmas,
                rhos=rhos,
                mean_knn_distance=float(np.mean(nn_dists.astype(np.float32))),
                embedding=np.ascontiguousarray(embedding, dtype=np.float32),
                hierarchy=hierarchy,
                core_distances=core_distances,
                params=dict(
                    n_neighbors=n_neighbors,
                    neighbor_scale=neighbor_scale,
                    symmetrize_graph=symmetrize_graph,
                    n_components=n_embedding_components,
                    n_epochs=n_epochs,
                    noise_level=noise_level,
                    min_samples=min_samples,
                    base_min_cluster_size=base_min_cluster_size,
                    base_n_clusters=base_n_clusters,
                    approx_n_clusters=approx_n_clusters,
                    min_similarity_threshold=min_similarity_threshold,
                    max_layers=max_layers,
                    reproducible_flag=reproducible_flag,
                ),
            )
        )
    return tuple(result)


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

    base_n_clusters : int or None, default=None
        If not None, the algorithm will attempt to find the granularity of
        clustering that will give exactly this many clusters for the bottom-most layer
        of clustering. This affects the base layer computation and allows multiple
        layers to be built on top of this base. Since the actual number of clusters
        cannot be guaranteed this is only approximate, but usually the algorithm can
        manage to get this exact number, assuming a reasonable clustering into
        ``base_n_clusters`` exists.

    approx_n_clusters : int, default=None
        If not None, the algorithm will attempt to find the granularity of
        clustering that will give exactly this many clusters as the final output.
        Unlike ``base_n_clusters``, when this parameter is set, only a single
        clustering layer will be returned -- no hierarchical layers will be produced.
        This is useful when you know the exact number of clusters you want and don't
        need the multi-layer analysis. Since the actual number of clusters cannot be
        guaranteed this is only approximate, but usually the algorithm can manage to
        get this exact number, assuming a reasonable clustering into ``approx_n_clusters``
        exists.

    n_neighbors : int, default=15
        The number of neighbors to use in the nearest neighbor graph construction.

    min_samples : int, default=5
        The minimum number of samples to use in the density estimation when
        performing density based clustering on the node embedding.

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
        value of min(max(n_neighbors // 4, 4), 15) will be used.

    neighbor_scale : float, default=1.0
        The scale factor to use when constructing the nearest neighbor graph. This
        multiplies the effective number of neighbors used in graph construction
        (neighbor_scale * n_neighbors). Values > 1.0 create denser graphs with more
        connectivity, potentially capturing more global structure but at increased
        computational cost. Values < 1.0 create sparser graphs focused on local
        structure.

    random_state : int or None, default=None
        The random seed to use for the random number generator. If None, the random
        number generator will not be seeded and will use the system time as the seed.

    min_similarity_threshold : float, default=0.2
        The minimum similarity threshold for cluster layer selection. Peaks that result
        in clusterings with Jaccard similarity above this threshold will be filtered out
        to ensure diverse cluster layers.

    max_layers : int, default=10
        The maximum number of cluster layers to return. The algorithm will select up to
        this many diverse peaks based on persistence and similarity criteria.

    n_label_prop_iter : int, default=20
        The number of iterations to use in the label propagation algorithm when
        initializing the node embedding. This parameter controls how many steps
        the label propagation process takes to converge when node_embedding_init
        is set to 'label_prop'.

    prediction_data : bool, default=False
        Whether to keep the state needed by :meth:`predict`,
        :meth:`predict_layers` and :meth:`update` (see ``prediction_data_``).
        This keeps a copy of the training data (as float32 for float input),
        the node embedding and the condensed trees, so it costs memory roughly
        the size of the training data plus ``n_samples * (n_components + 4)``
        floats. Clustering results are the same either way.

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

    nn_inds_ : array-like of shape (n_samples, n_neighbors)
        Indices of nearest neighbors for each sample.

    nn_dists_ : array-like of shape (n_samples, n_neighbors)
        Distance from each sample to each nearest neighbor (indexed by nn_inds).

    duplicates_ : set of tuple of int
        A set of pairs of indices of potential duplicate points in the data.

    prediction_data_ : PredictionData
        Only if ``prediction_data=True``: the prepared training data, k-NN
        bandwidths, node embedding (``prediction_data_.embedding``), condensed
        trees and core distances used by :meth:`predict` and :meth:`update`.
    """

    def __init__(
        self,
        noise_level: float = 0.5,
        base_min_cluster_size: int = 5,
        base_n_clusters: int | None = None,
        approx_n_clusters: int | None = None,
        n_neighbors: int = 15,
        min_samples: int = 5,
        n_epochs: int = 50,
        node_embedding_init: str | None = "label_prop",
        symmetrize_graph: bool = True,
        node_embedding_dim: int | None = None,
        neighbor_scale: float = 1.0,
        random_state: int | None = None,
        min_similarity_threshold: float = 0.2,
        max_layers: int = 10,
        n_label_prop_iter=20,
        prediction_data: bool = False,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.noise_level = noise_level
        self.base_min_cluster_size = base_min_cluster_size
        self.base_n_clusters = base_n_clusters
        self.approx_n_clusters = approx_n_clusters
        self.min_samples = min_samples
        self.n_epochs = n_epochs
        self.node_embedding_init = node_embedding_init
        self.symmetrize_graph = symmetrize_graph
        self.node_embedding_dim = node_embedding_dim
        self.neighbor_scale = neighbor_scale
        self.random_state = random_state
        self.min_similarity_threshold = min_similarity_threshold
        self.max_layers = max_layers
        self.n_label_prop_iter = n_label_prop_iter
        self.prediction_data = prediction_data

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

        **fit_params : dict
            Additional fit parameters. Currently unused, included for compatibility
            with scikit-learn's fit_predict interface.

        Returns
        -------

        labels_ : array-like of shape (n_samples,)
            An array of labels for the data samples; this is a integer array as per other scikit-learn
            clustering algorithms. A value of -1 indicates that a point is a noise point and
            not in any cluster.

        """

        X = check_array(X)
        current_random_state = check_random_state(self.random_state)
        self._clear_prediction_caches()

        results = evoc_clusters(
            X,
            n_neighbors=self.n_neighbors,
            noise_level=self.noise_level,
            base_min_cluster_size=self.base_min_cluster_size,
            base_n_clusters=self.base_n_clusters,
            approx_n_clusters=self.approx_n_clusters,
            min_samples=self.min_samples,
            n_epochs=self.n_epochs,
            node_embedding_init=self.node_embedding_init,
            symmetrize_graph=self.symmetrize_graph,
            return_duplicates=True,
            node_embedding_dim=self.node_embedding_dim,
            neighbor_scale=self.neighbor_scale,
            random_state=current_random_state,
            reproducible_flag=self.random_state is not None,
            min_similarity_threshold=self.min_similarity_threshold,
            max_layers=self.max_layers,
            n_label_prop_iter=self.n_label_prop_iter,
            prediction_data=self.prediction_data,
        )
        (
            self.cluster_layers_,
            self.membership_strength_layers_,
            self.persistence_scores_,
            self.nn_inds_,
            self.nn_dists_,
            self.duplicates_,
        ) = results[:6]
        if self.prediction_data:
            self.prediction_data_ = results[6]
        elif hasattr(self, "prediction_data_"):
            del self.prediction_data_

        self._select_labels()
        return self.labels_

    def _select_labels(self):
        if len(self.cluster_layers_) == 1:
            self.labels_ = self.cluster_layers_[0]
            self.membership_strengths_ = self.membership_strength_layers_[0]
        else:
            best_layer = np.argmax(self.persistence_scores_)
            self.labels_ = self.cluster_layers_[best_layer]
            self.membership_strengths_ = self.membership_strength_layers_[best_layer]

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

        **fit_params : dict
            Additional fit parameters. Currently unused, included for compatibility
            with scikit-learn's fit interface.

        Returns
        -------

        self : sklearn Estimator
            Returns the instance itself.
        """
        self.fit_predict(X, y, **fit_params)
        return self

    @property
    def cluster_tree_(self):
        """dict
        A dictionary representing the hierarchical clustering of the data.

        The keys are tuples of (layer, cluster) and the values are lists of
        tuples of (layer, cluster) representing the children of the key cluster.
        This provides a tree structure showing how clusters at different layers
        relate to each other hierarchically.

        Only available after fitting the model.

        Returns
        -------
        dict
            Hierarchical tree structure with (layer, cluster) tuples as keys
            and lists of child (layer, cluster) tuples as values.

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.
        """
        check_is_fitted(
            self,
            "cluster_layers_",
            msg="This %(name)s instance is not fitted yet, and 'cluster_tree_' is not available. "
            "Please call 'fit' with appropriate arguments before accessing this attribute.",
        )
        if not hasattr(self, "_cluster_tree"):
            self._cluster_tree = build_cluster_tree(self.cluster_layers_)
        return self._cluster_tree

    # ------------------------------------------------------------ prediction
    _PREDICTION_CACHES = ("_search_index", "_embedding_tree", "_prediction_cache")

    def _clear_prediction_caches(self):
        for name in self._PREDICTION_CACHES + ("_cluster_tree",):
            self.__dict__.pop(name, None)

    def _check_prediction_data(self):
        check_is_fitted(self, "cluster_layers_")
        if not hasattr(self, "prediction_data_"):
            raise ValueError(
                "This EVoC model was fitted without prediction data. Refit with "
                "prediction_data=True to use predict, predict_layers or update."
            )

    def prepare_prediction(self):
        """Build the structures used by :meth:`predict` and :meth:`update` now.

        They are otherwise built on first use: a nearest-neighbour search
        index over the training data, a kd-tree over the node embedding, and
        per-layer calibration for ``method="vote"``. They are not pickled, and
        are rebuilt on first use after unpickling.

        Returns
        -------
        self : EVoC
        """
        self._check_prediction_data()
        self._get_search_index()
        self._get_vote_thresholds()
        self._get_embedding_tree()
        self._get_tree_arrays()
        return self

    def _get_embedding_tree(self):
        if getattr(self, "_embedding_tree", None) is None:
            pdata = self.prediction_data_
            # The condensed trees cover the first hierarchy.n_samples points
            n_tree = pdata.hierarchy.n_samples
            self._embedding_tree = build_kdtree(
                np.ascontiguousarray(pdata.embedding[:n_tree])
            )
        return self._embedding_tree

    def _get_tree_arrays(self):
        cache = self._cache()
        if "tree_arrays" not in cache:
            hierarchy = self.prediction_data_.hierarchy
            cache["tree_arrays"] = [
                tree_arrays(tree, hierarchy.n_samples)
                for tree in hierarchy.condensed_trees
            ]
        return cache["tree_arrays"]

    def _cache(self):
        if getattr(self, "_prediction_cache", None) is None:
            self._prediction_cache = {}
        return self._prediction_cache

    def _labels_layer(self):
        if len(self.cluster_layers_) == 1:
            return 0
        return int(np.argmax(self.persistence_scores_))

    def _get_vote_thresholds(self):
        cache = self._cache()
        if "vote_thresholds" not in cache:
            thresholds, _ = calibrate_vote_thresholds(
                self.nn_inds_,
                self.nn_dists_,
                self.prediction_data_,
                self.cluster_layers_,
                self.membership_strength_layers_,
            )
            cache["vote_thresholds"] = thresholds
        return cache["vote_thresholds"]

    def _get_stacked_layers(self):
        cache = self._cache()
        if "stacked_layers" not in cache:
            cache["stacked_layers"] = stack_layers(
                self.cluster_layers_, self.membership_strength_layers_
            )
        return cache["stacked_layers"]

    def _query_neighbors(self, X, k, epsilon):
        """Training-set neighbours of X: (indices, distances), self excluded."""
        if k is None:
            k = self.prediction_data_.params["n_neighbors"]
        k = int(k)
        if k < 2:
            raise ValueError("k must be at least 2")
        index = self._get_search_index()
        return index.query(X, min(k - 1, index.n_samples), epsilon=epsilon)

    def _resolve_thresholds(self, noise_threshold, n_layers):
        if noise_threshold is None:
            return self._get_vote_thresholds()
        thresholds = np.asarray(noise_threshold, dtype=np.float32)
        if thresholds.ndim == 0:
            return np.full(n_layers, thresholds, dtype=np.float32)
        if thresholds.shape != (n_layers,):
            raise ValueError(
                "noise_threshold must be a float or have one value per layer"
            )
        return thresholds

    def predict_layers(
        self,
        X,
        *,
        method="vote",
        k=None,
        epsilon=0.1,
        noise_threshold=None,
        refine_epochs=10,
    ):
        """Assign new samples to clusters in every cluster layer.

        Requires a model fitted with ``prediction_data=True``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New samples, of the same kind (float, int8 or bit-packed uint8) and
            number of features as the training data.

        method : {"vote", "embedding"}, default="vote"
            ``"vote"`` assigns each sample to the cluster with the most
            (edge-weight times membership-strength) support among its nearest
            training samples, using the same edge weights as the fuzzy k-NN
            graph built during fitting. ``"embedding"`` places each sample in
            the node embedding and then in the condensed cluster tree, like
            HDBSCAN's ``approximate_predict``.

        k : int or None, default=None
            Number of neighbours (counting the sample itself, as in
            ``n_neighbors``) used to place each sample. Defaults to the fitted
            ``n_neighbors``.

        epsilon : float, default=0.1
            Accuracy/speed trade-off of the nearest-neighbour search.

        noise_threshold : float, array of floats or None, default=None
            ``"vote"`` only. Samples whose winning score (as a fraction of the
            total edge weight) is below the threshold are labelled noise
            (-1). By default each layer uses a threshold calibrated so that
            predicting the training samples from their neighbours reproduces
            the layer's training noise fraction.

        refine_epochs : int, default=10
            ``"embedding"`` only. Samples are placed in the node embedding at
            the weighted mean of their neighbours' positions; this many epochs
            of the node embedding's SGD (on the new samples only, with the
            training embedding fixed) then refine their positions. Without
            refinement the mean position tends to sit inside clusters, so fewer
            samples are labelled noise than in training.

        Returns
        -------
        labels : list of arrays of shape (n_samples,)
            Labels for each layer, aligned with ``cluster_layers_``.

        strengths : list of arrays of shape (n_samples,)
            Membership strengths in [0, 1] for each layer (0 for noise).
        """
        self._check_prediction_data()
        if method == "vote":
            labels, strengths = self._predict_vote(X, k, epsilon, noise_threshold)
        elif method == "embedding":
            if noise_threshold is not None:
                raise ValueError("noise_threshold only applies to method='vote'")
            labels, strengths = self._predict_embedding(X, k, epsilon, refine_epochs)
        else:
            raise ValueError(
                f"method should be 'vote' or 'embedding', not {method!r}"
            )
        return [l for l in labels], [s for s in strengths]

    def predict(
        self,
        X,
        *,
        method="vote",
        layer=None,
        k=None,
        epsilon=0.1,
        noise_threshold=None,
        refine_epochs=10,
        return_strengths=False,
    ):
        """Assign new samples to clusters without refitting.

        Requires a model fitted with ``prediction_data=True``. See
        :meth:`predict_layers` for the parameters.

        Parameters
        ----------
        layer : int or None, default=None
            The cluster layer to predict; None gives the layer used for
            ``labels_``.

        return_strengths : bool, default=False
            Also return membership strengths.

        Returns
        -------
        labels : array of shape (n_samples,)
            Cluster labels, -1 for noise.

        strengths : array of shape (n_samples,)
            Only if ``return_strengths`` is True.
        """
        self._check_prediction_data()
        if layer is None:
            layer = self._labels_layer()
        n_layers = len(self.cluster_layers_)
        if not -n_layers <= layer < n_layers:
            raise ValueError(f"layer must be in [0, {n_layers}), got {layer}")
        labels, strengths = self.predict_layers(
            X,
            method=method,
            k=k,
            epsilon=epsilon,
            noise_threshold=noise_threshold,
            refine_epochs=refine_epochs,
        )
        if return_strengths:
            return labels[layer], strengths[layer]
        return labels[layer]

    def _predict_vote(self, X, k, epsilon, noise_threshold):
        pdata = self.prediction_data_
        indices, distances = self._query_neighbors(X, k, epsilon)
        sigmas, rhos = query_bandwidths(distances, pdata)
        weights = membership_weights(indices, distances, sigmas, rhos)
        layer_labels, layer_strengths = self._get_stacked_layers()
        best_labels, best_scores = vote_scores(
            indices, weights, layer_labels, layer_strengths
        )
        thresholds = self._resolve_thresholds(noise_threshold, layer_labels.shape[0])
        return apply_thresholds(best_labels, best_scores, thresholds)

    def embed(self, X, *, k=None, epsilon=0.1, refine_epochs=10):
        """Positions of new samples in the node embedding.

        Each sample is placed at the mean of its nearest training samples'
        embedding positions, weighted by fuzzy k-NN graph edge weights, and
        optionally refined (see ``refine_epochs`` in :meth:`predict_layers`).
        Samples with no neighbours get NaN positions.

        Returns
        -------
        positions : array of shape (n_samples, n_components)
        """
        self._check_prediction_data()
        pdata = self.prediction_data_
        indices, distances = self._query_neighbors(X, k, epsilon)
        sigmas, rhos = query_bandwidths(distances, pdata)
        weights = membership_weights(indices, distances, sigmas, rhos)
        positions = weighted_positions(indices, weights, pdata.embedding)
        if refine_epochs > 0:
            seed = check_random_state(self.random_state).randint(
                0, np.iinfo(np.int32).max
            )
            positions = refine_positions(
                positions,
                indices,
                weights,
                pdata.embedding,
                int(refine_epochs),
                0.1,
                float(pdata.params["noise_level"]),
                1.0,
                np.uint64(seed),
            )
        return positions

    def _predict_embedding(self, X, k, epsilon, refine_epochs):
        n_queries = np.shape(X)[0] if np.ndim(X) == 2 else 0
        with _single_threaded(n_queries < SERIAL_BATCH_SIZE):
            positions = self.embed(
                X, k=k, epsilon=epsilon, refine_epochs=refine_epochs
            )
            labels, strengths = predict_from_positions(
                positions,
                self.prediction_data_,
                self._get_embedding_tree(),
                self._get_tree_arrays(),
            )
        return labels, strengths

    def _get_search_index(self):
        if getattr(self, "_search_index", None) is None:
            from .search import prepare_search

            pdata = self.prediction_data_
            self._search_index = prepare_search(
                pdata.data,
                self.nn_inds_,
                self.nn_dists_,
                random_state=check_random_state(self.random_state).randint(
                    0, np.iinfo(np.int32).max
                ),
            )
        return self._search_index

    def __getstate__(self):
        state = super().__getstate__()
        for name in self._PREDICTION_CACHES:
            state.pop(name, None)
        return state

    # ---------------------------------------------------------------- update
    def update(
        self,
        X_new,
        *,
        recluster=True,
        relabel=True,
        optimize="new",
        n_epochs=None,
        initial_alpha=None,
        epsilon=0.2,
        refresh_index=True,
        random_state=None,
    ):
        """Add samples to a fitted model without refitting from scratch.

        Requires a model fitted with ``prediction_data=True``. The k-NN graph
        is extended incrementally and the fuzzy graph rebuilt. New samples are
        placed in the node embedding at the mean of their neighbours'
        positions, weighted by fuzzy graph edge weights. Then, by default,
        the embedding is re-clustered and each new layer is relabelled to
        agree with the corresponding old layer, so that cluster ids stay
        stable.

        Parameters
        ----------
        X_new : array-like of shape (n_new, n_features)
            New samples, of the same kind and number of features as the
            training data. They are appended after the existing samples.

        recluster : bool, default=True
            Re-cluster the updated embedding. If False, the existing labels are
            kept and the new samples are assigned with
            ``predict(method="embedding")`` (the condensed trees then still
            describe the original samples).

        relabel : bool, default=True
            Relabel re-clustered layers to keep cluster ids stable: each new
            layer is matched to the old layer with the most similar number of
            clusters, and its clusters to that layer's clusters by maximum
            overlap on the existing samples. Clusters with no counterpart get
            new ids above every existing id, so labels need not be contiguous.
            The old layer each layer was matched to is recorded in
            ``update_layer_matches_``. If False, layers are labelled as a fresh
            fit would label them.

        optimize : {"new", "all"}, default="new"
            Which embedding positions may move when re-clustering. With "new"
            existing samples keep their positions and only the new samples'
            positions are (optionally, see ``n_epochs``) optimised with the
            node embedding's gradients. "all" re-optimises every position,
            starting from the current embedding, which lets the embedding
            adapt to the new samples but moves existing samples. On MNIST
            (adding 10k samples to 60k) "new" agreed better with the previous
            layers than either "all" or a refit, at the same agreement with the
            digit labels.

        n_epochs : int or None, default=None
            Epochs of embedding optimisation. For ``optimize="new"`` the
            default is 0: new samples stay at the weighted mean of their
            neighbours, which gave the most stable and accurate clusterings in
            our measurements (optimising only the new samples against a fixed
            embedding made results depend strongly on the random negative
            samples). For ``optimize="all"`` the default is
            ``max(10, round(0.4 * n_epochs))`` of the fitted ``n_epochs``.

        initial_alpha : float or None, default=None
            Initial learning rate; defaults to 0.1 (as in fitting) for
            ``optimize="new"`` and 0.01 for ``optimize="all"``, where lower
            values move existing samples less.

        epsilon : float, default=0.2
            Accuracy/speed trade-off of the search for the new samples'
            neighbours.

        refresh_index : bool, default=True
            Rebuild the search index for later predict/update calls now rather
            than on first use.

        random_state : int, RandomState instance or None, default=None
            Randomness for this update; defaults to the model's
            ``random_state``.

        Returns
        -------
        labels : array of shape (n_new,)
            Labels of the new samples in the ``labels_`` layer.
        """
        from .update import (
            match_layers,
            stable_relabel,
            update_knn_graph,
            warm_start_positions,
        )

        self._check_prediction_data()
        pdata = self.prediction_data_
        params = pdata.params
        rng = check_random_state(
            self.random_state if random_state is None else random_state
        )
        k = params["n_neighbors"]
        n_old = pdata.data.shape[0]
        if optimize not in ("new", "all"):
            raise ValueError(f"optimize should be 'new' or 'all', not {optimize!r}")
        if initial_alpha is None:
            initial_alpha = 0.1 if optimize == "new" else 0.01

        index = self._get_search_index()
        q_inds, q_dists = index.query(X_new, min(k - 1, n_old), epsilon=epsilon)
        new_data, new_inv_norms = prepare_data(np.asarray(X_new))
        n_new = new_data.shape[0]
        data_all, nn_inds, nn_dists = update_knn_graph(
            pdata.data,
            new_data,
            self.nn_inds_,
            self.nn_dists_,
            q_inds,
            q_dists,
            pdata.input_dtype,
            rng,
        )
        inv_norms = (
            np.concatenate([pdata.inv_norms, new_inv_norms])
            if pdata.inv_norms.shape[0]
            else pdata.inv_norms
        )
        graph, sigmas, rhos = neighbor_graph_matrix(
            params["neighbor_scale"] * k,
            nn_inds,
            nn_dists,
            params["symmetrize_graph"],
            return_bandwidths=True,
        )
        new_weights = membership_weights(
            np.ascontiguousarray(nn_inds[n_old:, 1:], dtype=np.int32),
            np.ascontiguousarray(nn_dists[n_old:, 1:], dtype=np.float32),
            sigmas[n_old:],
            rhos[n_old:],
        )
        initial = warm_start_positions(
            pdata.embedding,
            np.ascontiguousarray(nn_inds[n_old:, 1:]),
            new_weights,
            n_old,
            rng,
        )

        old_layers = self.cluster_layers_
        old_strengths = self.membership_strength_layers_
        if recluster:
            if optimize == "new":
                # Only the new samples move; existing positions are kept
                embedding = np.ascontiguousarray(initial, dtype=np.float32)
                if n_epochs is None:
                    n_epochs = 0
                refine_positions(
                    embedding[n_old:],
                    np.ascontiguousarray(nn_inds[n_old:, 1:], dtype=np.int32),
                    new_weights,
                    embedding,
                    int(n_epochs),
                    float(initial_alpha),
                    float(params["noise_level"]),
                    1.0,
                    np.uint64(rng.randint(0, np.iinfo(np.int32).max)),
                )
            else:
                if n_epochs is None:
                    n_epochs = max(10, int(round(0.4 * params["n_epochs"])))
                embedding = node_embedding(
                    graph,
                    n_components=params["n_components"],
                    n_epochs=n_epochs,
                    initial_embedding=initial,
                    negative_sample_rate=1.0,
                    noise_level=params["noise_level"],
                    random_state=rng,
                    verbose=False,
                    reproducible_flag=params["reproducible_flag"],
                    initial_alpha=initial_alpha,
                )
            layers, strengths, persistence, hierarchy, core_distances = (
                cluster_embedding(
                    embedding,
                    approx_n_clusters=params["approx_n_clusters"],
                    min_samples=params["min_samples"],
                    base_min_cluster_size=params["base_min_cluster_size"],
                    base_n_clusters=params["base_n_clusters"],
                    reproducible_flag=params["reproducible_flag"],
                    min_similarity_threshold=params["min_similarity_threshold"],
                    max_layers=params["max_layers"],
                    return_tree=True,
                )
            )
            if relabel:
                matches = match_layers(old_layers, layers, n_old)
                layer_labels = []
                for i, match in enumerate(matches):
                    layers[i], mapping = stable_relabel(old_layers[match], layers[i])
                    layer_labels.append(mapping)
                hierarchy = hierarchy._replace(layer_labels=layer_labels)
            self.update_layer_matches_ = matches if relabel else None
        else:
            # Keep the existing samples' positions and labels; place the new
            # samples by refining their positions against the fixed embedding
            new_positions = refine_positions(
                np.ascontiguousarray(initial[n_old:]),
                np.ascontiguousarray(nn_inds[n_old:, 1:], dtype=np.int32),
                new_weights,
                np.ascontiguousarray(initial),
                10,
                0.1,
                float(params["noise_level"]),
                1.0,
                np.uint64(rng.randint(0, np.iinfo(np.int32).max)),
            )
            embedding = np.vstack([pdata.embedding, new_positions]).astype(
                np.float32
            )
            new_labels, new_strengths = predict_from_positions(
                new_positions,
                pdata,
                self._get_embedding_tree(),
                self._get_tree_arrays(),
            )
            layers = [
                np.concatenate([old, new.astype(old.dtype)])
                for old, new in zip(old_layers, new_labels)
            ]
            strengths = [
                np.concatenate([old, new.astype(old.dtype)])
                for old, new in zip(old_strengths, new_strengths)
            ]
            persistence = self.persistence_scores_
            hierarchy = pdata.hierarchy
            core_distances = pdata.core_distances
            self.update_layer_matches_ = None

        self.nn_inds_ = nn_inds
        self.nn_dists_ = nn_dists
        self.duplicates_ = find_duplicates(nn_inds, nn_dists)
        self.cluster_layers_ = layers
        self.membership_strength_layers_ = strengths
        self.persistence_scores_ = persistence
        self._select_labels()
        self.prediction_data_ = pdata._replace(
            data=data_all,
            inv_norms=inv_norms,
            sigmas=sigmas,
            rhos=rhos,
            mean_knn_distance=float(np.mean(nn_dists.astype(np.float32))),
            embedding=np.ascontiguousarray(embedding, dtype=np.float32),
            hierarchy=hierarchy,
            core_distances=core_distances,
        )
        self._clear_prediction_caches()
        if refresh_index:
            self._get_search_index()
        return self.labels_[n_old:]

