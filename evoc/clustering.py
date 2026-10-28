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
    ClusterHierarchy,
    PredictionData,
    as_cluster_ids,
    embedding_core_distances,
)
from .search.kernels import prepare_data


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

    n_threads = numba.get_num_threads()

    if approx_n_clusters is not None:
        if prediction_data:
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
                n_samples=data.shape[0],
            )
            core_distances = embedding_core_distances(embedding, min_samples)
        else:
            cluster_vector, strengths = binary_search_for_n_clusters(
                embedding,
                approx_n_clusters,
                n_threads,
                min_samples=min_samples,
            )
        cluster_layers = [cluster_vector]
        membership_strengths = [strengths]
        persistence_scores = [0.0]
    else:
        layer_results = build_cluster_layers(
            embedding,
            min_samples=min_samples,
            base_min_cluster_size=base_min_cluster_size,
            base_n_clusters=base_n_clusters,
            reproducible_flag=reproducible_flag,
            min_similarity_threshold=min_similarity_threshold,
            max_layers=max_layers,
            return_tree=prediction_data,
        )
        if prediction_data:
            (
                cluster_layers,
                membership_strengths,
                persistence_scores,
                hierarchy,
                core_distances,
            ) = layer_results
        else:
            cluster_layers, membership_strengths, persistence_scores = layer_results

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
        return self

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
