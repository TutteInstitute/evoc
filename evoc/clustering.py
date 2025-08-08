import numpy as np
import numba

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted
from scipy.signal import find_peaks

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
from .knn_graph import knn_graph
from .label_propagation import label_propagation_init
from .node_embedding import node_embedding
from .graph_construction import neighbor_graph_matrix


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
        return leaves, clusters, strengths
    elif abs(lower_n_clusters - approx_n_clusters) > abs(
        upper_n_clusters - approx_n_clusters
    ):
        upper_tree = condense_tree(uncondensed_tree, upper_bound_min_cluster_size)
        leaves = extract_leaves(upper_tree)
        clusters = get_cluster_label_vector(upper_tree, leaves, 0.0, n_samples)
        strengths = get_point_membership_strength_vector(upper_tree, leaves, clusters)
        return leaves, clusters, strengths
    else:
        lower_tree = condense_tree(uncondensed_tree, lower_bound_min_cluster_size)
        lower_leaves = extract_leaves(lower_tree)
        lower_clusters = get_cluster_label_vector(
            lower_tree, lower_leaves, 0.0, n_samples
        )
        upper_tree = condense_tree(uncondensed_tree, upper_bound_min_cluster_size)
        upper_leaves = extract_leaves(upper_tree)
        upper_clusters = get_cluster_label_vector(
            upper_tree, upper_leaves, 0.0, n_samples
        )

        if np.sum(lower_clusters >= 0) > np.sum(upper_clusters >= 0):
            strengths = get_point_membership_strength_vector(
                lower_tree, lower_leaves, lower_clusters
            )
            return lower_leaves, lower_clusters, strengths
        else:
            strengths = get_point_membership_strength_vector(
                upper_tree, upper_leaves, upper_clusters
            )
            return upper_leaves, upper_clusters, strengths


def binary_search_for_n_clusters(
    data,
    approx_n_clusters,
    n_threads,
    *,
    min_samples=5,
):
    numba_tree = build_kdtree(data.astype(np.float32))
    edges = parallel_boruvka(numba_tree, n_threads, min_samples=min_samples)
    sorted_mst = edges[np.argsort(edges.T[2])]
    uncondensed_tree = mst_to_linkage_tree(sorted_mst)

    n_samples = data.shape[0]

    leaves, clusters, strengths = _binary_search_for_n_clusters(
        uncondensed_tree, approx_n_clusters, n_samples
    )
    return clusters, strengths

@numba.njit(cache=True)
def min_cluster_size_barcode(cluster_tree, n_points, min_size):
    n_nodes = cluster_tree.child[-1] - n_points + 1
    parents = np.empty(n_nodes, dtype=np.int32)
    lambda_deaths = np.empty(n_nodes, dtype=np.float32)
    size_deaths = np.empty(n_nodes, dtype=np.float32)
    size_births = np.full(n_nodes, min_size, dtype=np.float32)
    lambda_deaths[0] = 0
    size_deaths[0] = n_points
    parents[0] = n_points

    # Iterate over row-pairs in reverse order
    n_rows = cluster_tree.child.shape[0]
    for idx in range(n_rows - 1, 0, -2):
        out_idx = cluster_tree.child[idx] - n_points
        parents[out_idx - 1 : out_idx + 1] = cluster_tree.parent[idx]
        lambda_deaths[out_idx - 1 : out_idx + 1] =  np.exp(-1/cluster_tree.lambda_val[idx])

        death_size = cluster_tree.child_size[idx - 1 : idx + 1].min()
        size_deaths[out_idx - 1 : out_idx + 1] = death_size
        size_births[cluster_tree.parent[idx] - n_points] = max(
            size_births[out_idx - 1], size_births[out_idx], death_size
        )

    return size_births, size_deaths, parents, lambda_deaths

@numba.njit(cache=True)
def compute_total_persistence(births, deaths, lambda_deaths):
    # maintain left-open (birth, death] interval!
    sizes = np.unique(births)
    total_persistence = np.zeros(sizes.shape[0], dtype=np.float32)
    
    for i in range(1, len(births)):
        birth = births[i]
        death = deaths[i]
        lambda_death = lambda_deaths[i]
        
        if death <= birth:
            continue
            
        # Manual binary search for birth_idx
        birth_idx = 0
        for j in range(len(sizes)):
            if sizes[j] >= birth:
                birth_idx = j
                break
                
        # Manual binary search for death_idx  
        death_idx = len(sizes)
        for j in range(len(sizes)):
            if sizes[j] >= death:
                death_idx = j
                break
                
        # Update persistence values
        for k in range(birth_idx, death_idx):
            total_persistence[k] += (death - birth) * lambda_death
            
    return sizes, total_persistence

@numba.njit(cache=True)
def extract_clusters_by_id(condensed_tree, selected_ids):
    labels = get_cluster_label_vector(
        condensed_tree,
        selected_ids,
        cluster_selection_epsilon=0.0,
        n_samples=condensed_tree.parent[0],
    )
    strengths = get_point_membership_strength_vector(condensed_tree, selected_ids, labels)
    return labels, strengths


@numba.njit(cache=True)
def jaccard_similarity(set_a_array, set_b_array):
    # Convert to sets for intersection/union operations
    intersection_count = 0
    union_set = set(set_a_array)
    
    for item in set_b_array:
        if item in union_set:
            intersection_count += 1
        else:
            union_set.add(item)
    
    union_count = len(union_set)
    return intersection_count / union_count if union_count > 0 else 0.0


@numba.njit(cache=True)
def estimate_cluster_similarity(births, deaths, birth_a, birth_b):
    # Find clusters active at birth_a
    clusters_a = np.empty(len(births), dtype=np.int64)
    count_a = 0
    for i in range(len(births)):
        if births[i] <= birth_a and deaths[i] > birth_a:
            clusters_a[count_a] = i
            count_a += 1
    
    # Find clusters active at birth_b  
    clusters_b = np.empty(len(births), dtype=np.int64)
    count_b = 0
    for i in range(len(births)):
        if births[i] <= birth_b and deaths[i] > birth_b:
            clusters_b[count_b] = i
            count_b += 1
    
    # Trim arrays to actual sizes
    active_a = clusters_a[:count_a]
    active_b = clusters_b[:count_b]
    
    return jaccard_similarity(active_a, active_b)


@numba.njit(cache=True)
def select_diverse_peaks(peaks, total_persistence, sizes, births, deaths, 
                              min_similarity_threshold=0.2, max_layers=10):
    if len(peaks) == 0:
        return np.empty(0, dtype=np.int64)
    
    # Sort peaks by persistence (highest first)
    peak_persistence = total_persistence[peaks]
    sorted_indices = np.argsort(peak_persistence)[::-1]
    sorted_peaks = peaks[sorted_indices]
    
    # Pre-allocate arrays for selected peaks and births
    selected_peaks = np.empty(max_layers, dtype=np.int64)
    selected_births = np.empty(max_layers, dtype=np.float64)
    n_selected = 0
    
    for i in range(len(sorted_peaks)):
        if n_selected >= max_layers:
            break
            
        peak = sorted_peaks[i]
        birth_size = sizes[peak]
        
        # Check similarity with already selected peaks
        is_diverse = True
        for j in range(n_selected):
            selected_birth = selected_births[j]
            similarity = estimate_cluster_similarity(births, deaths, birth_size, selected_birth)
            if similarity > min_similarity_threshold:
                is_diverse = False
                break
        
        if is_diverse:
            selected_peaks[n_selected] = peak
            selected_births[n_selected] = birth_size
            n_selected += 1
    
    return selected_peaks[:n_selected]


#@numba.njit(cache=True)
def build_cluster_layers(
    data,
    *,
    min_samples=5,
    base_min_cluster_size=10,
    base_n_clusters=None,
    reproducible_flag=False,
    min_similarity_threshold=0.2,
    max_layers=10,
):
    n_samples = data.shape[0]
    min_cluster_size = base_min_cluster_size
    cluster_layers = []
    membership_strength_layers = []
    persistence_scores = []

    n_threads = numba.get_num_threads()

    numba_tree = build_kdtree(data.astype(np.float32))
    edges = parallel_boruvka(
        numba_tree, n_threads, min_samples=min_cluster_size if min_samples is None else min_samples, reproducible=reproducible_flag
    )
    sorted_mst = edges[np.argsort(edges.T[2])]
    uncondensed_tree = mst_to_linkage_tree(sorted_mst)
    if base_n_clusters is not None:
        leaves, clusters, strengths = _binary_search_for_n_clusters(
            uncondensed_tree, base_n_clusters, n_samples=n_samples
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
        strengths = get_point_membership_strength_vector(condensed_tree, leaves, clusters)

    mask = condensed_tree.child >= n_samples
    cluster_tree = mask_condensed_tree(condensed_tree, mask)
    # points_tree = mask_condensed_tree(condensed_tree, ~mask)
    
    # Check if cluster_tree is valid before processing
    if len(cluster_tree.child) > 0 and cluster_tree.child[-1] >= n_samples:
        births, deaths, parents, lambda_deaths = min_cluster_size_barcode(cluster_tree, n_samples, min_cluster_size)
        sizes, total_persistence = compute_total_persistence(births, deaths, lambda_deaths)
        peaks, _ = find_peaks(total_persistence, distance=1)
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
        peaks, total_persistence, sizes, births, deaths,
        min_similarity_threshold=min_similarity_threshold, 
        max_layers=max_layers - 1  # Reserve one slot for base layer
    )
    
    for peak in selected_peaks:
        best_birth = sizes[peak]
        persistence = total_persistence[peak]
        selected_clusters = (
            np.where((births <= best_birth) & (deaths > best_birth))[0] + n_samples
        )
        labels, strengths = extract_clusters_by_id(condensed_tree, selected_clusters)
        cluster_layers.append(labels)
        membership_strength_layers.append(strengths)
        persistence_scores.append(persistence)

    # Sort cluster layers by number of clusters (most clusters first)
    n_clusters_per_layer = [layer.max() + 1 for layer in cluster_layers]
    sorted_indices = np.argsort(n_clusters_per_layer)[::-1]  # Descending order
    
    cluster_layers = [cluster_layers[i] for i in sorted_indices]
    membership_strength_layers = [membership_strength_layers[i] for i in sorted_indices] 
    persistence_scores = [persistence_scores[i] for i in sorted_indices]

    return cluster_layers, membership_strength_layers, persistence_scores

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
        of clustering. Since the actual number of clusters cannot be guaranteed this
        is only approximate, but usually the algorithm can manage to get this exact number,
        assuming a resonable clustering into ``base_n_clusters`` exists.

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

    random_state : np.random.RandomState or None, default=None
        The random state to use for the random number generator. If None, the random
        number generator will not be seeded and will use the system time as the seed.

    min_similarity_threshold : float, default=0.2
        The minimum similarity threshold for cluster layer selection. Peaks that result
        in clusterings with Jaccard similarity above this threshold will be filtered out
        to ensure diverse cluster layers.

    max_layers : int, default=10
        The maximum number of cluster layers to return. The algorithm will select up to
        this many diverse peaks based on persistence and similarity criteria.

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
    if random_state is None:
        random_state = np.random.RandomState()

    nn_inds, nn_dists = knn_graph(data, n_neighbors=n_neighbors, random_state=random_state)
    graph = neighbor_graph_matrix(
        neighbor_scale * n_neighbors, nn_inds, nn_dists, symmetrize_graph
    )
    n_embedding_components = node_embedding_dim or min(max(n_neighbors // 4, 4), 15)
    if node_embedding_init == "label_prop":
        init_embedding = label_propagation_init(
            graph,
            n_components=n_embedding_components,
            approx_n_parts=np.clip(int(np.sqrt(data.shape[0])), 100, 1024),
            random_scale=0.1,
            scaling=0.5,
            noise_level=noise_level,
            random_state=random_state,
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
    )

    if return_duplicates:
        duplicates = find_duplicates(nn_inds, nn_dists)

    n_threads = numba.get_num_threads()

    if approx_n_clusters is not None:
        cluster_vector, strengths = binary_search_for_n_clusters(
            embedding,
            approx_n_clusters,
            n_threads,
            min_samples=min_samples,
        )
        if return_duplicates:
            return [cluster_vector], [strengths], [0.0], duplicates
        else:
            return [cluster_vector], [strengths], [0.0]
    else:
        cluster_layers, membership_strengths, persistence_scores = build_cluster_layers(
            embedding,
            min_samples=min_samples,
            base_min_cluster_size=base_min_cluster_size,
            base_n_clusters=base_n_clusters,
            reproducible_flag=reproducible_flag,
            min_similarity_threshold=min_similarity_threshold,
            max_layers=max_layers,
        )

        
        if return_duplicates:
            return cluster_layers, membership_strengths, persistence_scores, duplicates
        else:
            return cluster_layers, membership_strengths, persistence_scores


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
        of clustering. Since the actual number of clusters cannot be guaranteed this
        is only approximate, but usually the algorithm can manage to get this exact
        number, assuming a resonable clustering into ``base_n_clusters`` exists.

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
        current_random_state = check_random_state(self.random_state)

        self.cluster_layers_, self.membership_strength_layers_, self.persistence_scores_, self.duplicates_ = (
            evoc_clusters(
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
            )
        )

        if len(self.cluster_layers_) == 1:
            self.labels_ = self.cluster_layers_[0]
            self.membership_strengths_ = self.membership_strength_layers_[0]
        else:
            best_layer = np.argmax(self.persistence_scores_)
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

