import numpy as np
import numba

from .numba_kdtree import build_kdtree
from .boruvka import parallel_boruvka
from .cluster_trees import (
    mst_to_linkage_tree,
    condense_tree,
    extract_leaves,
    get_cluster_label_vector,
    get_point_membership_strength_vector,
)

##############################################################
# Directly derived from scipy's find_peaks function:
# https://github.com/scipy/scipy/blob/bd66693b8aecc6f528ca9b1cfd6bb1f61477ca0f/scipy/signal/_peak_finding_utils.pyx#L20
##############################################################
@numba.njit(
    ['intp[:](float32[::1])', 'intp[:](float64[::1])'],
    locals={
        "midpoints": numba.types.intp[::1],
        "left_edges": numba.types.intp[::1],
        "right_edges": numba.types.intp[::1],
        "m": numba.types.uint32,
        "i": numba.types.uint32,
    },
    nogil=True,
    parallel=False,
    fastmath=True,
    cache=True,
)
def find_peaks(x):
    # Preallocate, there can't be more maxima than half the size of `x`
    midpoints = np.empty(x.shape[0] // 2, dtype=np.intp)
    left_edges = np.empty(x.shape[0] // 2, dtype=np.intp)
    right_edges = np.empty(x.shape[0] // 2, dtype=np.intp)
    m = 0  # Pointer to the end of valid area in allocated arrays

    i = 1  # Pointer to current sample, first one can't be maxima
    i_max = x.shape[0] - 1  # Last sample can't be maxima
    while i < i_max:
        # Test if previous sample is smaller
        if x[i - 1] < x[i]:
            i_ahead = i + 1  # Index to look ahead of current sample

            # Find next sample that is unequal to x[i]
            while i_ahead < i_max and x[i_ahead] == x[i]:
                i_ahead += 1

            # Maxima is found if next unequal sample is smaller than x[i]
            if x[i_ahead] < x[i]:
                left_edges[m] = i
                right_edges[m] = i_ahead - 1
                midpoints[m] = (left_edges[m] + right_edges[m]) // 2
                m += 1
                # Skip samples that can't be maximum
                i = i_ahead
        i += 1

    return midpoints[:m]

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


@numba.njit(cache=True)
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