"""Adding samples to a fitted EVoC model."""
from warnings import warn

import numba
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from .common_nndescent import flagged_heap_push, make_heap
from .knn_graph import knn_graph, nn_descent
from .search.kernels import output_to_internal

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


@numba.njit(cache=True)
def _reverse_insert(heap_indices, heap_distances, heap_flags, new_ids, inds, dists):
    """Offer each new point to the heaps of the neighbours it found."""
    for i in range(new_ids.shape[0]):
        x = new_ids[i]
        for j in range(inds.shape[1]):
            y = inds[i, j]
            if y < 0:
                continue
            flagged_heap_push(heap_distances[y], heap_indices[y], heap_flags[y], dists[i, j], x)


@numba.njit(cache=True)
def _offer_edges(heap_indices, heap_distances, heap_flags, sources, inds, dists):
    """Offer edges sources[i] -> inds[i, j] to the sources' heaps."""
    for i in range(sources.shape[0]):
        x = sources[i]
        for j in range(inds.shape[1]):
            if inds[i, j] < 0:
                continue
            flagged_heap_push(
                heap_distances[x], heap_indices[x], heap_flags[x], dists[i, j], inds[i, j]
            )


def _rows_to_heap(indices, internal, flag):
    """Sorted (nearest first) rows as max-heaps: reversed rows are valid heaps."""
    heap_indices = np.ascontiguousarray(indices[:, ::-1], dtype=np.int32)
    heap_distances = np.ascontiguousarray(internal[:, ::-1], dtype=np.float32)
    heap_distances[heap_indices < 0] = np.inf
    heap_flags = np.full(heap_indices.shape, flag, dtype=np.uint8)
    return heap_indices, heap_distances, heap_flags


def update_knn_graph(
    data,
    new_data,
    nn_inds,
    nn_dists,
    new_inds,
    new_dists,
    input_dtype,
    random_state,
    n_iters=None,
    delta=0.001,
    delta_improv=0.001,
    max_candidates=None,
    seed_new_edges=True,
):
    """Extend a k-NN graph (as returned by ``knn_graph``) with new samples.

    Parameters
    ----------
    data, new_data : arrays
        Old and new samples, prepared as for search (float data normalised).

    nn_inds, nn_dists : arrays of shape (n_old, n_neighbors)
        The current k-NN graph, self in column 0, output distance units.

    new_inds, new_dists : arrays of shape (n_new, n_neighbors - 1)
        Neighbours of the new samples among the old ones (from a search),
        output distance units.

    Returns
    -------
    data_all : array of shape (n_old + n_new, n_features)

    nn_inds, nn_dists : arrays of shape (n_old + n_new, n_neighbors)
        The updated graph, self in column 0, output distance units.
    """
    n_old = data.shape[0]
    n_new = new_data.shape[0]
    n_all = n_old + n_new
    n_neighbors = nn_inds.shape[1]
    k = n_neighbors - 1
    data_all = np.ascontiguousarray(np.vstack([data, new_data]))

    old_rows = np.ascontiguousarray(nn_inds[:, 1:], dtype=np.int32)
    old_internal = output_to_internal(nn_dists[:, 1:], input_dtype)
    new_rows = np.full((n_new, k), -1, dtype=np.int32)
    new_internal = np.full((n_new, k), np.inf, dtype=np.float32)
    n_found = min(k, new_inds.shape[1])
    new_rows[:, :n_found] = new_inds[:, :n_found]
    new_internal[:, :n_found] = output_to_internal(new_dists[:, :n_found], input_dtype)

    old_heap = _rows_to_heap(old_rows, old_internal, 0)
    new_heap = _rows_to_heap(new_rows, new_internal, 1)
    heap = make_heap(n_all, k)
    for part in range(3):
        heap[part][:n_old] = old_heap[part]
        heap[part][n_old:] = new_heap[part]

    # Offer each new point to the old points it found; accepted edges are
    # flagged new, so NN-descent only joins around them.
    new_ids = np.arange(n_old, n_all, dtype=np.int32)
    _reverse_insert(
        heap[0], heap[1], heap[2], new_ids, new_rows, np.ascontiguousarray(new_internal)
    )
    if seed_new_edges and n_new > n_neighbors:
        # Neighbours among the new points themselves, which the search of the
        # old points cannot provide
        new_new_inds, new_new_dists = knn_graph(
            new_data, n_neighbors=n_neighbors, random_state=random_state
        )
        new_new_internal = output_to_internal(new_new_dists[:, 1:], input_dtype)
        _offer_edges(
            heap[0],
            heap[1],
            heap[2],
            new_ids,
            np.ascontiguousarray(new_new_inds[:, 1:] + n_old, dtype=np.int32),
            np.ascontiguousarray(new_new_internal),
        )

    random_state = random_state if random_state is not None else np.random
    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    if n_iters is None:
        n_iters = max(5, int(round(np.log2(n_all))))
    if max_candidates is None:
        max_candidates = min(60, int(n_neighbors * 1.5))
    # Convergence is judged relative to the number of new samples, since only
    # the region around them changes
    scale = max(n_new, 1) / n_all
    indices, distances = nn_descent(
        data_all,
        k,
        rng_state,
        max_candidates,
        n_iters,
        delta * scale,
        input_dtype,
        use_sorted_updates=True,
        delta_improv=None if delta_improv is None else delta_improv * scale,
        init_graph=heap,
    )
    if np.any(indices < 0):
        warn(
            "Failed to find n_neighbors for some samples while updating the "
            "k-NN graph. Results may be less than ideal."
        )
    out_inds = np.empty((n_all, n_neighbors), dtype=indices.dtype)
    out_dists = np.empty((n_all, n_neighbors), dtype=distances.dtype)
    out_inds[:, 0] = np.arange(n_all)
    out_dists[:, 0] = 0.0
    out_inds[:, 1:] = indices
    out_dists[:, 1:] = distances
    return data_all, out_inds, out_dists


def warm_start_positions(old_embedding, new_inds, new_weights, n_old, random_state):
    """Initial embedding positions for new samples.

    Each new sample starts at the weighted mean of its already placed
    neighbours' positions (old samples first, then new samples placed in
    earlier passes); samples with no placed neighbour start near the mean of
    the embedding.
    """
    from .prediction import weighted_positions

    n_new = new_inds.shape[0]
    dim = old_embedding.shape[1]
    positions = np.full((n_old + n_new, dim), np.nan, dtype=np.float32)
    positions[:n_old] = old_embedding
    pending = np.arange(n_new)
    for _ in range(3):
        if pending.shape[0] == 0:
            break
        inds = new_inds[pending]
        placed = np.zeros(inds.shape, dtype=bool)
        valid = inds >= 0
        placed[valid] = ~np.isnan(positions[inds[valid], 0])
        weights = np.where(placed, new_weights[pending], 0.0).astype(np.float32)
        new_positions = weighted_positions(
            np.where(placed, inds, -1).astype(np.int32), weights, positions
        )
        positions[n_old + pending] = new_positions
        pending = pending[np.isnan(new_positions[:, 0])]
    if pending.shape[0] > 0:
        scale = 0.1 * old_embedding.std(axis=0)
        positions[n_old + pending] = old_embedding.mean(axis=0) + scale * (
            random_state.normal(size=(pending.shape[0], dim))
        )
    return positions


def match_layers(old_layers, new_layers, n_old):
    """For each new layer, the old layer it corresponds to.

    The old layer with the most similar number of clusters (on a log scale) is
    chosen, with ties broken by the adjusted Rand index on the old samples.
    """
    old_counts = np.array([len(np.unique(l[l >= 0])) for l in old_layers])
    matches = []
    for new_labels in new_layers:
        count = len(np.unique(new_labels[new_labels >= 0]))
        gaps = np.abs(np.log1p(old_counts) - np.log1p(count))
        candidates = np.flatnonzero(np.isclose(gaps, gaps.min()))
        if candidates.shape[0] > 1:
            scores = [
                adjusted_rand_score(old_layers[c], new_labels[:n_old])
                for c in candidates
            ]
            matches.append(int(candidates[int(np.argmax(scores))]))
        else:
            matches.append(int(candidates[0]))
    return matches


def stable_relabel(old_labels, new_labels):
    """Relabel ``new_labels`` to agree with ``old_labels`` on the old samples.

    New clusters are matched to old clusters by maximum overlap (Hungarian
    assignment on the contingency table of the old samples, noise excluded) and
    take the matched cluster's label; unmatched new clusters get fresh labels
    above every old label. Returns the relabelled labels and the label for each
    original new label.
    """
    n_old = old_labels.shape[0]
    n_new_clusters = int(new_labels.max()) + 1 if new_labels.shape[0] else 0
    mapping = np.full(max(n_new_clusters, 0), -1, dtype=np.int64)
    if n_new_clusters == 0:
        return new_labels.copy(), mapping
    old_part = new_labels[:n_old]
    both = (old_labels >= 0) & (old_part >= 0)
    old_ids = np.unique(old_labels[old_labels >= 0])
    if old_ids.shape[0] > 0 and np.any(both):
        old_pos = np.searchsorted(old_ids, old_labels[both])
        table = np.zeros((n_new_clusters, old_ids.shape[0]), dtype=np.int64)
        np.add.at(table, (old_part[both], old_pos), 1)
        rows, cols = linear_sum_assignment(-table)
        for r, c in zip(rows, cols):
            if table[r, c] > 0:
                mapping[r] = old_ids[c]
    next_label = int(old_labels.max()) + 1 if old_labels.shape[0] else 0
    next_label = max(next_label, 0)
    for r in range(n_new_clusters):
        if mapping[r] < 0:
            mapping[r] = next_label
            next_label += 1
    relabelled = np.where(new_labels >= 0, mapping[np.maximum(new_labels, 0)], -1)
    return relabelled.astype(new_labels.dtype), mapping
