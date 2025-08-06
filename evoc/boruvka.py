import numba
import numpy as np

from .disjoint_set import ds_rank_create, ds_find, ds_union_by_rank
from .numba_kdtree import parallel_tree_query, rdist, point_to_node_lower_bound_rdist, NumbaKDTree

@numba.njit(locals={"i": numba.types.int64}, cache=True)
def merge_components(disjoint_set, candidate_neighbors, candidate_neighbor_distances, point_components):
    component_edges = {np.int64(0): (np.int64(0), np.int64(1), np.float32(0.0)) for i in range(0)}

    # Find the best edges from each component
    for i in range(candidate_neighbors.shape[0]):
        from_component = np.int64(point_components[i])
        if from_component in component_edges:
            if candidate_neighbor_distances[i] < component_edges[from_component][2]:
                component_edges[from_component] = (np.int64(i), np.int64(candidate_neighbors[i]), candidate_neighbor_distances[i])
        else:
            component_edges[from_component] = (np.int64(i), np.int64(candidate_neighbors[i]), candidate_neighbor_distances[i])

    result = np.empty((len(component_edges), 3), dtype=np.float64)
    result_idx = 0

    # Add the best edges to the edge set and merge the relevant components
    for edge in component_edges.values():
        from_component = ds_find(disjoint_set, np.int32(edge[0]))
        to_component = ds_find(disjoint_set, np.int32(edge[1]))
        if from_component != to_component:
            result[result_idx] = (np.float64(edge[0]), np.float64(edge[1]), np.float64(edge[2]))
            result_idx += 1

            ds_union_by_rank(disjoint_set, from_component, to_component)

    return result[:result_idx]


@numba.njit(
    locals={
        "i": numba.types.int32,
        "j": numba.types.int32,
        "idx": numba.types.int32,
        "left": numba.types.int32,
        "right": numba.types.int32,
        "candidate_component": numba.types.int32
    }, 
    parallel=True, 
    cache=True, 
    fastmath=True
)
def update_component_vectors(tree, disjoint_set, node_components, point_components):
    for i in numba.prange(point_components.shape[0]):
        point_components[i] = ds_find(disjoint_set, np.int32(i))

    for i in range(tree.node_data.shape[0] - 1, -1, -1):
        node_info = tree.node_data[i]

        # Case 1:
        #    If the node is a leaf we need to check that every point
        #    in the node is of the same component
        if node_info.is_leaf:
            candidate_component = point_components[tree.idx_array[node_info.idx_start]]
            for j in range(node_info.idx_start + 1, node_info.idx_end):
                idx = tree.idx_array[j]
                if point_components[idx] != candidate_component:
                    break
            else:
                node_components[i] = candidate_component

        # Case 2:
        #    If the node is not a leaf we only need to check
        #    that both child nodes are in the same component
        else:
            left = 2 * i + 1
            right = left + 1

            if node_components[left] == node_components[right]:
                node_components[i] = node_components[left]


@numba.njit(
    locals={
        "i": numba.types.int32,
        "idx": numba.types.int32,
        "left": numba.types.int32,
        "right": numba.types.int32,
        "d": numba.types.float32,
        "dist_lower_bound_left": numba.types.float32,
        "dist_lower_bound_right": numba.types.float32,
    }, 
    cache=True, 
    fastmath=True
)
def component_aware_query_recursion(
        tree,
        node,
        point,
        heap_p,
        heap_i,
        current_core_distance,
        core_distances,
        current_component,
        node_components,
        point_components,
        dist_lower_bound,
        component_nearest_neighbor_dist,
):
    node_info = tree.node_data[node]

    # ------------------------------------------------------------
    # Case 1a: query point is outside node radius:
    #         trim it from the query
    if dist_lower_bound > heap_p[0]:
        return

    # ------------------------------------------------------------
    # Case 1b: we can't improve on the best distance for this component
    #         trim it from the query
    elif dist_lower_bound > component_nearest_neighbor_dist[0] or current_core_distance > \
            component_nearest_neighbor_dist[0]:
        return

    # ------------------------------------------------------------
    # Case 1c: node contains only points in same component as query
    #         trim it from the query
    elif node_components[node] == current_component:
        return

    # ------------------------------------------------------------
    # Case 2: this is a leaf node.  Update set of nearby points
    elif node_info.is_leaf:
        for i in range(node_info.idx_start, node_info.idx_end):
            idx = tree.idx_array[i]
            if point_components[idx] != current_component and core_distances[idx] < component_nearest_neighbor_dist[0]:
                d = max(rdist(point, tree.data[idx]), current_core_distance, core_distances[idx])
                if d < heap_p[0]:
                    heap_p[0] = d
                    heap_i[0] = idx
                    if d < component_nearest_neighbor_dist[0]:
                        component_nearest_neighbor_dist[0] = d

    # ------------------------------------------------------------
    # Case 3: Node is not a leaf.  Recursively query subnodes
    #         starting with the closest
    else:
        left = 2 * node + 1
        right = left + 1
        dist_lower_bound_left = point_to_node_lower_bound_rdist(tree.node_bounds[0, left], tree.node_bounds[1, left],
                                                                point)
        dist_lower_bound_right = point_to_node_lower_bound_rdist(tree.node_bounds[0, right], tree.node_bounds[1, right],
                                                                 point)

        # recursively query subnodes
        if dist_lower_bound_left <= dist_lower_bound_right:
            component_aware_query_recursion(
                tree,
                left,
                point,
                heap_p,
                heap_i,
                current_core_distance,
                core_distances,
                current_component,
                node_components,
                point_components,
                dist_lower_bound_left,
                component_nearest_neighbor_dist
            )
            component_aware_query_recursion(
                tree,
                right,
                point,
                heap_p,
                heap_i,
                current_core_distance,
                core_distances,
                current_component,
                node_components,
                point_components,
                dist_lower_bound_right,
                component_nearest_neighbor_dist
            )
        else:
            component_aware_query_recursion(
                tree,
                right,
                point,
                heap_p,
                heap_i,
                current_core_distance,
                core_distances,
                current_component,
                node_components,
                point_components,
                dist_lower_bound_right,
                component_nearest_neighbor_dist
            )
            component_aware_query_recursion(
                tree,
                left,
                point,
                heap_p,
                heap_i,
                current_core_distance,
                core_distances,
                current_component,
                node_components,
                point_components,
                dist_lower_bound_left,
                component_nearest_neighbor_dist
            )

    return


@numba.njit(
    locals={
        "i": numba.types.int32,
        "distance_lower_bound": numba.types.float32,
        "current_component": numba.types.int32
    }, 
    parallel=True, 
    cache=True, 
    fastmath=True
)
def boruvka_tree_query(tree, node_components, point_components, core_distances):
    candidate_distances = np.full(tree.data.shape[0], np.inf, dtype=np.float32)
    candidate_indices = np.full(tree.data.shape[0], -1, dtype=np.int32)
    component_nearest_neighbor_dist = np.full(tree.data.shape[0], np.inf, dtype=np.float32)

    data = tree.data.astype(np.float32)

    for i in numba.prange(tree.data.shape[0]):
        distance_lower_bound = point_to_node_lower_bound_rdist(tree.node_bounds[0, 0], tree.node_bounds[1, 0],
                                                               tree.data[i])
        heap_p, heap_i = candidate_distances[i:i + 1], candidate_indices[i:i + 1]
        component_aware_query_recursion(
            tree,
            0,
            data[i],
            heap_p,
            heap_i,
            core_distances[i],
            core_distances,
            point_components[i],
            node_components,
            point_components,
            distance_lower_bound,
            component_nearest_neighbor_dist[point_components[i]:point_components[i] + 1]
        )

    return candidate_distances, candidate_indices

def calculate_block_size(n_components, n_points, num_threads):
    """Calculate adaptive block size based on component sizes."""
    if n_components == 0:
        points_per_component = n_points
    else:
        points_per_component = n_points / n_components
    
    if points_per_component < 10:
        block_size = num_threads * 512    # Weak pruning, large blocks
    elif points_per_component < 100:
        block_size = num_threads * 128    # Moderate pruning
    elif points_per_component < 1000:
        block_size = num_threads * 32    # Good pruning
    else:
        block_size = num_threads * 8     # Excellent pruning, small blocks

    # Ensure reasonable bounds
    block_size = max(num_threads, min(block_size, n_points // 4 + 1))
    return int(block_size)


@numba.njit([
    'void(float32[:], float32[:], int32[:], int32, int32)',
    'void(float64[:], float64[:], int64[:], int64, int64)'
], locals={
    "i": numba.types.int32,
    "component": numba.types.int32,
    "block_bound": numba.types.float32
}, cache=True, fastmath=True, inline='always')
def update_component_bounds_from_block(component_nearest_neighbor_dist, block_component_bounds, point_components, block_start, block_end):
    """Update global component bounds from block results."""
    for i in range(block_start, block_end):
        component = point_components[i]
        block_bound = block_component_bounds[i - block_start]
        if block_bound < component_nearest_neighbor_dist[component]:
            component_nearest_neighbor_dist[component] = block_bound


@numba.njit(
    locals={
        "block_start": numba.types.int32,
        "block_end": numba.types.int32,
        "block_size_actual": numba.types.int32,
        "i": numba.types.int32,
        "distance_lower_bound": numba.types.float32,
        "current_component": numba.types.int32
    }, 
    parallel=True, 
    cache=True, 
    fastmath=True
)
def boruvka_tree_query_reproducible(tree, node_components, point_components, core_distances, block_size):
    """Reproducible version using block-based processing to avoid race conditions."""
    candidate_distances = np.full(tree.data.shape[0], np.inf, dtype=np.float32)
    candidate_indices = np.full(tree.data.shape[0], -1, dtype=np.int32)
    component_nearest_neighbor_dist = np.full(tree.data.shape[0], np.inf, dtype=np.float32)
    
    data = tree.data.astype(np.float32)
    
    # Reusable buffer for block component bounds (allocate once, reuse)
    max_block_component_bounds = np.full(block_size, np.inf, dtype=np.float32)
    
    # Process points in blocks
    for block_start in range(0, tree.data.shape[0], block_size):
        block_end = min(block_start + block_size, tree.data.shape[0])
        block_size_actual = block_end - block_start
        
        # Reset only the portion we'll use (more cache-friendly)
        max_block_component_bounds[:block_size_actual] = np.inf
        
        # Parallel processing within the block
        for i in numba.prange(block_start, block_end):
            distance_lower_bound = point_to_node_lower_bound_rdist(tree.node_bounds[0, 0], tree.node_bounds[1, 0],
                                                                   tree.data[i])
            heap_p, heap_i = candidate_distances[i:i + 1], candidate_indices[i:i + 1]
            
            # Use current global bounds for this component
            current_component = point_components[i]
            local_component_bound = component_nearest_neighbor_dist[current_component:current_component + 1]
            
            component_aware_query_recursion(
                tree,
                0,
                data[i],
                heap_p,
                heap_i,
                core_distances[i],
                core_distances,
                point_components[i],
                node_components,
                point_components,
                distance_lower_bound,
                local_component_bound
            )
            
            # Store the potentially updated bound for this point
            max_block_component_bounds[i - block_start] = local_component_bound[0]
        
        # Sequential update of global component bounds after the block
        update_component_bounds_from_block(component_nearest_neighbor_dist, max_block_component_bounds, 
                                          point_components, block_start, block_end)
    
    return candidate_distances, candidate_indices

@numba.njit(
    locals={
        "i": numba.types.int32,
        "j": numba.types.int32,
        "k": numba.types.int32,
        "result_idx": numba.types.int32,
        "from_component": numba.types.int32,
        "to_component": numba.types.int32
    }, 
    parallel=True, 
    cache=True
)
def initialize_boruvka_from_knn(knn_indices, knn_distances, core_distances, disjoint_set):
    # component_edges = {0:(np.int32(0), np.int32(1), np.float32(0.0)) for i in range(0)}
    component_edges = np.full((knn_indices.shape[0], 3), -1, dtype=np.float64)

    for i in numba.prange(knn_indices.shape[0]):
        for j in range(1, knn_indices.shape[1]):
            k = np.int32(knn_indices[i, j])
            if core_distances[i] >= core_distances[k]:
                component_edges[i] = (np.float64(i), np.float64(k), np.float64(core_distances[i]))
                break

    result = np.empty((len(component_edges), 3), dtype=np.float64)
    result_idx = 0

    # Add the best edges to the edge set and merge the relevant components
    for edge in component_edges:
        if edge[0] < 0:
            continue
        from_component = ds_find(disjoint_set, np.int32(edge[0]))
        to_component = ds_find(disjoint_set, np.int32(edge[1]))
        if from_component != to_component:
            result[result_idx] = (np.float64(edge[0]), np.float64(edge[1]), np.float64(edge[2]))
            result_idx += 1

            ds_union_by_rank(disjoint_set, from_component, to_component)

    return result[:result_idx]


def parallel_boruvka(tree, min_samples=10, reproducible=False):
    components_disjoint_set = ds_rank_create(tree.data.shape[0])
    point_components = np.arange(tree.data.shape[0])
    node_components = np.full(tree.node_data.shape[0], -1)
    n_components = point_components.shape[0]

    if min_samples > 1:
        distances, neighbors = parallel_tree_query(tree, tree.data, k=min_samples + 1, output_rdist=True)
        core_distances = distances.T[-1]
        initial_edges = initialize_boruvka_from_knn(neighbors, distances, core_distances, components_disjoint_set)
        update_component_vectors(tree, components_disjoint_set, node_components, point_components)
    else:
        core_distances = np.zeros(tree.data.shape[0], dtype=np.float32)
        distances, neighbors = parallel_tree_query(tree, tree.data, k=2)
        initial_edges = initialize_boruvka_from_knn(neighbors, distances, core_distances, components_disjoint_set)
        update_component_vectors(tree, components_disjoint_set, node_components, point_components)

    # Get number of threads for block size calculation
    num_threads = numba.get_num_threads()
    
    # Count initial components after initialization
    n_components = len(np.unique(point_components))
    
    # Use list to accumulate edges, then convert at end (more efficient than vstack)
    all_edges = [initial_edges]

    while n_components > 1:
        if reproducible:
            # Calculate adaptive block size based on current component sizes
            block_size = calculate_block_size(n_components, tree.data.shape[0], num_threads)
            candidate_distances, candidate_indices = boruvka_tree_query_reproducible(
                tree, node_components, point_components, core_distances, block_size)
        else:
            candidate_distances, candidate_indices = boruvka_tree_query(
                tree, node_components, point_components, core_distances)
        
        new_edges = merge_components(components_disjoint_set, candidate_indices, candidate_distances, point_components)
        
        # Update component count more efficiently - subtract merged components
        n_components -= len(new_edges)
        
        update_component_vectors(tree, components_disjoint_set, node_components, point_components)

        if len(new_edges) > 0:
            all_edges.append(new_edges)

    # Combine all edges at once (more efficient than repeated vstack)
    edges = np.vstack(all_edges)
    edges[:, 2] = np.sqrt(edges.T[2])
    return edges

