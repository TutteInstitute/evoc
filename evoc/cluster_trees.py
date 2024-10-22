import numba
import numpy as np

from collections import namedtuple

from .disjoint_set import ds_rank_create, ds_find, ds_union_by_rank

LinkageMergeData = namedtuple("LinkageMergeData", ["parent", "size", "next"])


@numba.njit(cache=True)
def create_linkage_merge_data(base_size):
    parent = np.full(2 * base_size - 1, -1, dtype=np.intp)
    size = np.concatenate(
        (np.ones(base_size, dtype=np.intp), np.zeros(base_size - 1, dtype=np.intp))
    )
    next_parent = np.array([base_size], dtype=np.intp)

    return LinkageMergeData(parent, size, next_parent)


@numba.njit(cache=True)
def linkage_merge_find(linkage_merge, node):
    relabel = node
    while linkage_merge.parent[node] != -1 and linkage_merge.parent[node] != node:
        node = linkage_merge.parent[node]

    linkage_merge.parent[node] = node

    # label up to the root
    while linkage_merge.parent[relabel] != node:
        next_relabel = linkage_merge.parent[relabel]
        linkage_merge.parent[relabel] = node
        relabel = next_relabel

    return node


@numba.njit(cache=True)
def linkage_merge_join(linkage_merge, left, right):
    linkage_merge.size[linkage_merge.next[0]] = (
        linkage_merge.size[left] + linkage_merge.size[right]
    )
    linkage_merge.parent[left] = linkage_merge.next[0]
    linkage_merge.parent[right] = linkage_merge.next[0]
    linkage_merge.next[0] += 1


@numba.njit(cache=True)
def mst_to_linkage_tree(sorted_mst):
    result = np.empty((sorted_mst.shape[0], sorted_mst.shape[1] + 1))

    n_samples = sorted_mst.shape[0] + 1
    linkage_merge = create_linkage_merge_data(n_samples)

    for index in range(sorted_mst.shape[0]):

        left = np.intp(sorted_mst[index, 0])
        right = np.intp(sorted_mst[index, 1])
        delta = sorted_mst[index, 2]

        left_component = linkage_merge_find(linkage_merge, left)
        right_component = linkage_merge_find(linkage_merge, right)

        if left_component > right_component:
            result[index][0] = left_component
            result[index][1] = right_component
        else:
            result[index][1] = left_component
            result[index][0] = right_component

        result[index][2] = delta
        result[index][3] = (
            linkage_merge.size[left_component] + linkage_merge.size[right_component]
        )

        linkage_merge_join(linkage_merge, left_component, right_component)

    return result


@numba.njit(cache=True)
def bfs_from_hierarchy(hierarchy, bfs_root, num_points):
    to_process = [bfs_root]
    result = []

    while to_process:
        result.extend(to_process)
        next_to_process = []
        for n in to_process:
            if n >= num_points:
                i = n - num_points
                next_to_process.append(int(hierarchy[i, 0]))
                next_to_process.append(int(hierarchy[i, 1]))
        to_process = next_to_process

    return result


@numba.njit(cache=True)
def eliminate_branch(
    branch_node,
    parent_node,
    lambda_value,
    parents,
    children,
    lambdas,
    sizes,
    idx,
    ignore,
    hierarchy,
    num_points,
):
    if branch_node < num_points:
        parents[idx] = parent_node
        children[idx] = branch_node
        lambdas[idx] = lambda_value
        idx += 1
    else:
        for sub_node in bfs_from_hierarchy(hierarchy, branch_node, num_points):
            if sub_node < num_points:
                children[idx] = sub_node
                parents[idx] = parent_node
                lambdas[idx] = lambda_value
                idx += 1
            else:
                ignore[sub_node] = True

    return idx


CondensedTree = namedtuple(
    "CondensedTree", ["parent", "child", "lambda_val", "child_size"]
)


@numba.njit(fastmath=True, cache=True)
def condense_tree(hierarchy, min_cluster_size=10):
    root = 2 * hierarchy.shape[0]
    num_points = hierarchy.shape[0] + 1
    next_label = num_points + 1

    node_list = bfs_from_hierarchy(hierarchy, root, num_points)

    relabel = np.zeros(root + 1, dtype=np.int64)
    relabel[root] = num_points

    parents = np.ones(root, dtype=np.int64)
    children = np.empty(root, dtype=np.int64)
    lambdas = np.empty(root, dtype=np.float32)
    sizes = np.ones(root, dtype=np.int64)

    ignore = np.zeros(root + 1, dtype=np.bool)

    idx = 0

    for node in node_list:
        if ignore[node] or node < num_points:
            continue

        parent_node = relabel[node]
        l, r, d, _ = hierarchy[node - num_points]
        left = np.int64(l)
        right = np.int64(r)
        if d > 0.0:
            lambda_value = 1.0 / d
        else:
            lambda_value = np.inf

        left_count = (
            np.int64(hierarchy[left - num_points, 3]) if left >= num_points else 1
        )
        right_count = (
            np.int64(hierarchy[right - num_points, 3]) if right >= num_points else 1
        )

        # The logic here is in a strange order, but it has non-trivial performance gains ...
        # The most common case by far is a singleton on the left; and cluster on the right take care of this separately
        if left < num_points and right_count >= min_cluster_size:
            relabel[right] = parent_node
            parents[idx] = parent_node
            children[idx] = left
            lambdas[idx] = lambda_value
            idx += 1
        # Next most common is a small left cluster and a large right cluster: relabel the right node; eliminate the left branch
        elif left_count < min_cluster_size and right_count >= min_cluster_size:
            relabel[right] = parent_node
            idx = eliminate_branch(
                left,
                parent_node,
                lambda_value,
                parents,
                children,
                lambdas,
                sizes,
                idx,
                ignore,
                hierarchy,
                num_points,
            )
        # Then we have a large left cluster and a small right cluster: relabel the left node; elimiate the right branch
        elif left_count >= min_cluster_size and right_count < min_cluster_size:
            relabel[left] = parent_node
            idx = eliminate_branch(
                right,
                parent_node,
                lambda_value,
                parents,
                children,
                lambdas,
                sizes,
                idx,
                ignore,
                hierarchy,
                num_points,
            )
        # If both clusters are small then eliminate all branches
        elif left_count < min_cluster_size and right_count < min_cluster_size:
            idx = eliminate_branch(
                left,
                parent_node,
                lambda_value,
                parents,
                children,
                lambdas,
                sizes,
                idx,
                ignore,
                hierarchy,
                num_points,
            )
            idx = eliminate_branch(
                right,
                parent_node,
                lambda_value,
                parents,
                children,
                lambdas,
                sizes,
                idx,
                ignore,
                hierarchy,
                num_points,
            )
        # and finally if we actually have a legitimate cluster split, handle that correctly
        else:
            relabel[left] = next_label

            parents[idx] = parent_node
            children[idx] = next_label
            lambdas[idx] = lambda_value
            sizes[idx] = left_count
            next_label += 1
            idx += 1

            relabel[right] = next_label

            parents[idx] = parent_node
            children[idx] = next_label
            lambdas[idx] = lambda_value
            sizes[idx] = right_count
            next_label += 1
            idx += 1

    return CondensedTree(parents[:idx], children[:idx], lambdas[:idx], sizes[:idx])


@numba.njit(cache=True)
def extract_leaves(condensed_tree, allow_single_cluster=True):
    n_nodes = condensed_tree.parent.max() + 1
    n_points = condensed_tree.parent.min()
    leaf_indicator = np.ones(n_nodes, dtype=np.bool_)
    leaf_indicator[:n_points] = False

    for parent, child_size in zip(condensed_tree.parent, condensed_tree.child_size):
        if child_size > 1:
            leaf_indicator[parent] = False

    return np.nonzero(leaf_indicator)[0]


@numba.njit(cache=True, fastmath=True)
def score_condensed_tree_nodes(condensed_tree):
    result = {0: 0.0 for i in range(0)}

    for i in range(condensed_tree.parent.shape[0]):
        parent = condensed_tree.parent[i]
        if parent in result:
            result[parent] += (
                condensed_tree.lambda_val[i] * condensed_tree.child_size[i]
            )
        else:
            result[parent] = condensed_tree.lambda_val[i] * condensed_tree.child_size[i]

        if condensed_tree.child_size[i] > 1:
            child = condensed_tree.child[i]
            if child in result:
                result[child] -= (
                    condensed_tree.lambda_val[i] * condensed_tree.child_size[i]
                )
            else:
                result[child] = (
                    -condensed_tree.lambda_val[i] * condensed_tree.child_size[i]
                )

    return result


@numba.njit(cache=True)
def cluster_tree_from_condensed_tree(condensed_tree):
    mask = condensed_tree.child_size > 1
    return CondensedTree(
        condensed_tree.parent[mask],
        condensed_tree.child[mask],
        condensed_tree.lambda_val[mask],
        condensed_tree.child_size[mask],
    )


@numba.njit(cache=True)
def unselect_below_node(node, cluster_tree, selected_clusters):
    for child in cluster_tree.child[cluster_tree.parent == node]:
        unselect_below_node(child, cluster_tree, selected_clusters)
        selected_clusters[child] = False


@numba.njit(fastmath=True, cache=True)
def eom_recursion(node, cluster_tree, node_scores, selected_clusters):
    current_score = node_scores[node]

    children = cluster_tree.child[cluster_tree.parent == node]
    child_score_total = 0.0

    for child_node in children:
        child_score_total += eom_recursion(
            child_node, cluster_tree, node_scores, selected_clusters
        )

    if child_score_total > current_score:
        return child_score_total
    else:
        selected_clusters[node] = True
        unselect_below_node(node, cluster_tree, selected_clusters)
        return current_score


@numba.njit(cache=True)
def extract_eom_clusters(condensed_tree, cluster_tree, allow_single_cluster=False):
    node_scores = score_condensed_tree_nodes(condensed_tree)
    selected_clusters = {node: False for node in node_scores}

    if len(cluster_tree.parent) == 0:
        return np.zeros(0, dtype=np.int64)

    cluster_tree_root = cluster_tree.parent.min()

    if allow_single_cluster:
        eom_recursion(cluster_tree_root, cluster_tree, node_scores, selected_clusters)
    elif len(node_scores) > 1:
        root_children = cluster_tree.child[cluster_tree.parent == cluster_tree_root]
        for child_node in root_children:
            eom_recursion(child_node, cluster_tree, node_scores, selected_clusters)

    return np.asarray(
        [node for node, selected in selected_clusters.items() if selected]
    )


@numba.njit(cache=True)
def cluster_epsilon_search(clusters, cluster_tree, min_persistence=0.0):
    selected = list()
    # only way to create a typed empty set
    processed = {np.int64(0)}
    processed.clear()

    root = cluster_tree.parent.min()
    for cluster in clusters:
        eps = 1 / cluster_tree.lambda_val[cluster_tree.child == cluster][0]
        if eps < min_persistence:
            if cluster not in processed:
                parent = traverse_upwards(cluster_tree, min_persistence, root, cluster)
                selected.append(parent)
                processed |= segments_in_branch(cluster_tree, parent)
        else:
            selected.append(cluster)
    return np.asarray(selected)


@numba.njit(cache=True)
def traverse_upwards(cluster_tree, min_persistence, root, segment):
    parent = cluster_tree.parent[cluster_tree.child == segment][0]
    if parent == root:
        return root
    parent_eps = 1 / cluster_tree.lambda_val[cluster_tree.child == parent][0]
    if parent_eps >= min_persistence:
        return parent
    else:
        return traverse_upwards(cluster_tree, min_persistence, root, parent)


@numba.njit(cache=True)
def segments_in_branch(cluster_tree, segment):
    # only way to create a typed empty set
    result = {np.intp(0)}
    result.clear()
    to_process = {segment}

    while len(to_process) > 0:
        result |= to_process
        to_process = set(
            cluster_tree.child[in_set_parallel(cluster_tree.parent, to_process)]
        )

    return result


@numba.njit(parallel=True, cache=True)
def in_set_parallel(values, targets):
    mask = np.empty(values.shape[0], dtype=numba.boolean)
    for i in numba.prange(values.shape[0]):
        mask[i] = values[i] in targets
    return mask


@numba.njit(parallel=True, cache=True)
def get_cluster_labelling_at_cut(linkage_tree, cut, min_cluster_size):

    root = 2 * linkage_tree.shape[0]
    num_points = linkage_tree.shape[0] + 1
    result = np.empty(num_points, dtype=np.intp)
    disjoint_set = ds_rank_create(root + 1)

    cluster = num_points
    for i in range(linkage_tree.shape[0]):
        if linkage_tree[i, 2] < cut:
            ds_union_by_rank(disjoint_set, np.intp(linkage_tree[i, 0]), cluster)
            ds_union_by_rank(disjoint_set, np.intp(linkage_tree[i, 1]), cluster)
        cluster += 1

    cluster_size = np.zeros(cluster, dtype=np.intp)
    for n in range(num_points):
        cluster = ds_find(disjoint_set, n)
        cluster_size[cluster] += 1
        result[n] = cluster

    cluster_label_map = {-1: -1}
    cluster_label = 0
    unique_labels = np.unique(result)

    for cluster in unique_labels:
        if cluster_size[cluster] < min_cluster_size:
            cluster_label_map[cluster] = -1
        else:
            cluster_label_map[cluster] = cluster_label
            cluster_label += 1

    for n in numba.prange(num_points):
        result[n] = cluster_label_map[result[n]]

    return result


@numba.njit(cache=True)
def get_single_cluster_label_vector(
    tree,
    cluster,
    cluster_selection_epsilon,
    n_samples,
):
    if len(tree.parent) == 0:
        return np.full(n_samples, -1, dtype=np.intp)

    result = np.full(n_samples, -1, dtype=np.intp)
    max_lambda = tree.lambda_val[tree.parent == cluster].max()

    for i in range(tree.child.shape[0]):
        n = tree.child[i]
        cur_lambda = tree.lambda_val[i]
        if cluster_selection_epsilon > 0.0:
            if cur_lambda >= 1 / cluster_selection_epsilon:
                result[n] = 0
            else:
                result[n] = -1
        elif cur_lambda >= max_lambda:
            result[n] = 0

    return result


@numba.njit(cache=True)
def get_cluster_label_vector(
    tree,
    clusters,
    cluster_selection_epsilon,
    n_samples,
):
    if len(clusters) == 1:
        return get_single_cluster_label_vector(
            tree, clusters[0], cluster_selection_epsilon, n_samples
        )

    if len(tree.parent) == 0:
        return np.full(n_samples, -1, dtype=np.intp)
    root_cluster = tree.parent.min()
    result = np.full(n_samples, -1, dtype=np.intp)
    cluster_label_map = {c: n for n, c in enumerate(np.sort(clusters))}

    disjoint_set = ds_rank_create(max(tree.parent.max() + 1, tree.child.max() + 1))
    clusters = set(clusters)

    for n in range(tree.parent.shape[0]):
        child = tree.child[n]
        parent = tree.parent[n]
        if child not in clusters:
            ds_union_by_rank(disjoint_set, parent, child)

    for n in range(n_samples):
        cluster = ds_find(disjoint_set, n)
        if cluster <= root_cluster:
            result[n] = -1
        else:
            result[n] = cluster_label_map[cluster]

    return result


@numba.njit(cache=True)
def max_lambdas(tree, clusters):
    result = {c: 0.0 for c in clusters}

    for n in range(tree.parent.shape[0]):
        cluster = tree.parent[n]
        if cluster in clusters and tree.child_size[n] == 1:
            result[cluster] = max(result[cluster], tree.lambda_val[n])

    return result


@numba.njit(cache=True)
def get_point_membership_strength_vector(tree, clusters, labels):
    result = np.zeros(labels.shape[0], dtype=np.float32)
    deaths = max_lambdas(tree, set(clusters))
    root_cluster = tree.parent.min()
    cluster_index_map = {n: c for n, c in enumerate(np.sort(clusters))}

    for n in range(tree.child.shape[0]):
        point = tree.child[n]
        if point >= root_cluster or labels[point] < 0:
            continue

        cluster = cluster_index_map[labels[point]]
        max_lambda = deaths[cluster]
        if max_lambda == 0.0 or not np.isfinite(tree.lambda_val[n]):
            result[point] = 1.0
        else:
            lambda_val = min(tree.lambda_val[n], max_lambda)
            result[point] = lambda_val / max_lambda

    return result
