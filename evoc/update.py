"""Adding samples to a fitted EVoC model."""
from warnings import warn

import numba
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from .common_nndescent import flagged_heap_push, make_heap
from .knn_graph import knn_graph, nn_descent
from .search.kernels import (
    four_internal_distances_to,
    internal_distance_to,
    output_to_internal,
)

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


def _remap_rows(indices, distances, old_to_new):
    """Renumber neighbour indices after removing rows.

    Neighbours that were removed become holes (-1, inf), moved to the end of
    each row so rows stay sorted by distance.
    """
    remapped = np.where(indices >= 0, old_to_new[np.maximum(indices, 0)], -1)
    distances = np.where(remapped >= 0, distances, np.inf).astype(np.float32)
    order = np.argsort(remapped < 0, axis=1, kind="stable")
    return (
        np.ascontiguousarray(np.take_along_axis(remapped, order, axis=1), dtype=np.int32),
        np.ascontiguousarray(np.take_along_axis(distances, order, axis=1)),
    )


# How rows that lose neighbours are repaired, per input dtype, as
# (n_first, n_second, n_reflag): each such row is offered the first n_second
# neighbours of its first n_first surviving neighbours, and its n_reflag
# nearest edges are flagged as new so that NN-descent re-explores around it
# (otherwise NN-descent never revisits a row's surviving edges). For float and
# int8 data a small repair plus re-flagging measured faster, and more
# accurate, than offering all neighbours of neighbours; bit-packed distances
# are cheap enough that the full repair without re-flagging is fastest.
REPAIR_SETTINGS = {
    np.dtype(np.float32): (5, 5, 5),
    np.dtype(np.int8): (5, 5, 5),
    np.dtype(np.uint8): (14, 14, 0),
}


@numba.njit(parallel=True, fastmath=True, cache=True)
def _repair_rows(
    heap_indices, heap_distances, heap_flags, rows, affected, data, inv_norms,
    n_first, n_second, n_reflag,
):
    """Offer each affected row a few of its neighbours' neighbours, then flag
    its ``n_reflag`` nearest edges as new.

    Candidates are the first ``n_second`` neighbours of the row's first
    ``n_first`` surviving neighbours (``rows`` are the renumbered neighbour
    rows), skipping the row itself, current heap entries and repeats.
    Distances are computed in internal units, four at a time where possible,
    and go into the affected row's own heap only, so rows can be processed in
    parallel. Candidates with no similarity at all are skipped.
    """
    k = rows.shape[1]
    heap_size = heap_indices.shape[1]
    n_second = min(n_second, k)
    for r in numba.prange(affected.shape[0]):
        a = affected[r]
        if inv_norms.shape[0] > 0:
            inv_a = inv_norms[a]
        else:
            inv_a = np.float32(1.0)
        candidates = np.empty(n_first * n_second, dtype=np.int32)
        n_candidates = 0
        n_used = 0
        for j in range(k):
            b = rows[a, j]
            if b < 0:
                continue
            n_used += 1
            if n_used > n_first:
                break
            for t in range(n_second):
                c = rows[b, t]
                if c < 0 or c == a:
                    continue
                seen = False
                for m in range(heap_size):
                    if heap_indices[a, m] == c:
                        seen = True
                        break
                if not seen:
                    for m in range(n_candidates):
                        if candidates[m] == c:
                            seen = True
                            break
                if not seen:
                    candidates[n_candidates] = c
                    n_candidates += 1

        query = data[a]
        dists = np.empty(4, dtype=np.float32)
        q = 0
        while q + 4 <= n_candidates:
            four_internal_distances_to(
                query, inv_a, data, inv_norms, candidates[q : q + 4], dists
            )
            for m in range(4):
                if dists[m] < 0.0:
                    flagged_heap_push(
                        heap_distances[a],
                        heap_indices[a],
                        heap_flags[a],
                        dists[m],
                        candidates[q + m],
                    )
            q += 4
        while q < n_candidates:
            d = internal_distance_to(query, inv_a, data, inv_norms, candidates[q])
            if d < 0.0:
                flagged_heap_push(
                    heap_distances[a], heap_indices[a], heap_flags[a], d, candidates[q]
                )
            q += 1

        remaining = heap_distances[a].copy()
        for _ in range(min(n_reflag, heap_size)):
            nearest = np.argmin(remaining)
            if not np.isfinite(remaining[nearest]):
                break
            heap_flags[a, nearest] = 1
            remaining[nearest] = np.inf


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
    keep=None,
    inv_norms=None,
):
    """Update a k-NN graph (as returned by ``knn_graph``): remove samples and
    add new ones.

    Parameters
    ----------
    data, new_data : arrays
        Current and new samples, prepared as for search (float data
        normalised).

    nn_inds, nn_dists : arrays of shape (n_current, n_neighbors)
        The current k-NN graph, self in column 0, output distance units.

    new_inds, new_dists : arrays of shape (n_new, m)
        Neighbours of the new samples among the current ones (from a search),
        nearest first, output distance units. Neighbours that are removed are
        dropped, and the first ``n_neighbors - 1`` remaining ones are used.

    keep : boolean array of shape (n_current,) or None
        Which current samples to keep; None keeps all. Rows that lose
        neighbours are offered some of their neighbours' neighbours, and their
        nearest edges are flagged as new, before NN-descent.

    inv_norms : array or None
        Inverse row norms of int8 data (current then new); computed if None.

    Returns
    -------
    data_all : array of shape (n_kept + n_new, n_features)

    nn_inds, nn_dists : arrays of shape (n_kept + n_new, n_neighbors)
        The updated graph, self in column 0, output distance units.
    """
    n_current = data.shape[0]
    n_neighbors = nn_inds.shape[1]
    k = n_neighbors - 1
    if keep is None:
        keep = np.ones(n_current, dtype=bool)
    old_to_new = np.full(n_current, -1, dtype=np.int64)
    old_to_new[keep] = np.arange(int(keep.sum()))
    n_old = int(keep.sum())
    n_new = new_data.shape[0]
    n_all = n_old + n_new
    data_all = np.ascontiguousarray(np.vstack([data[keep], new_data]))

    old_rows, old_dists = _remap_rows(nn_inds[keep, 1:], nn_dists[keep, 1:], old_to_new)
    affected = np.flatnonzero(np.any(old_rows < 0, axis=1)).astype(np.int32)
    old_internal = output_to_internal(old_dists, input_dtype)
    old_internal[old_rows < 0] = np.inf

    new_rows = np.full((n_new, k), -1, dtype=np.int32)
    new_internal = np.full((n_new, k), np.inf, dtype=np.float32)
    if n_new > 0:
        found_rows, found_dists = _remap_rows(
            np.asarray(new_inds), np.asarray(new_dists, dtype=np.float32), old_to_new
        )
        n_found = min(k, found_rows.shape[1])
        new_rows[:, :n_found] = found_rows[:, :n_found]
        new_internal[:, :n_found] = output_to_internal(
            found_dists[:, :n_found], input_dtype
        )
        new_internal[new_rows < 0] = np.inf

    old_heap = _rows_to_heap(old_rows, old_internal, 0)
    new_heap = _rows_to_heap(new_rows, new_internal, 1)
    heap = make_heap(n_all, k)
    for part in range(3):
        heap[part][:n_old] = old_heap[part]
        heap[part][n_old:] = new_heap[part]

    if affected.shape[0] > 0:
        # Rows that lost neighbours: offer some neighbours' neighbours, and
        # flag their nearest edges as new for NN-descent
        if input_dtype == np.int8 and inv_norms is None:
            from .int8_nndescent import int8_inverse_norms

            inv_norms = int8_inverse_norms(data_all)
        elif inv_norms is None:
            inv_norms = np.zeros(0, dtype=np.float32)
        _repair_rows(
            heap[0],
            heap[1],
            heap[2],
            old_rows,
            affected,
            data_all,
            inv_norms,
            *REPAIR_SETTINGS[np.dtype(data_all.dtype)],
        )

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
    # Convergence is judged relative to the number of new and affected
    # samples, since only the region around them changes
    scale = max(n_new + affected.shape[0], 1) / n_all
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


def stable_relabel(old_labels, new_labels, min_new_label=None):
    """Relabel ``new_labels`` to agree with ``old_labels`` on the old samples.

    New clusters are matched to old clusters by maximum overlap (Hungarian
    assignment on the contingency table of the old samples, noise excluded) and
    take the matched cluster's label; unmatched new clusters get fresh labels
    starting at ``min_new_label`` (default: above every old label). Returns the
    relabelled labels and the label for each original new label.
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
    if min_new_label is not None:
        next_label = max(next_label, int(min_new_label))
    for r in range(n_new_clusters):
        if mapping[r] < 0:
            mapping[r] = next_label
            next_label += 1
    relabelled = np.where(new_labels >= 0, mapping[np.maximum(new_labels, 0)], -1)
    return relabelled.astype(new_labels.dtype), mapping


def resolve_expired(X_expired, data, search_index, n_candidates=16):
    """Which current rows to keep, given the rows to expire.

    ``X_expired`` is either 1-D (integer row positions, or a boolean mask over
    the rows) or 2-D (data rows). Each data row removes one matching current
    row, the oldest one not already removed. Rows match when they are equal
    in the stored representation: exactly for int8 and uint8 data, and to
    within float32 rounding after L2 normalisation for float data (EVoC only
    uses the direction of float vectors).

    Returns a boolean keep mask of shape (n_rows,).
    """
    n_rows = data.shape[0]
    keep = np.ones(n_rows, dtype=bool)
    expired = np.asarray(X_expired)
    if expired.size == 0:
        return keep
    if expired.ndim == 1:
        if expired.dtype == bool:
            if expired.shape[0] != n_rows:
                raise ValueError(
                    f"A boolean X_expired mask must have one entry per row "
                    f"({n_rows}), not {expired.shape[0]}"
                )
            keep[expired] = False
            return keep
        if not np.issubdtype(expired.dtype, np.integer):
            raise ValueError(
                "A 1-D X_expired must hold integer row positions or be a boolean "
                "mask; pass a 2-D array to expire data rows"
            )
        if expired.min() < 0 or expired.max() >= n_rows:
            raise ValueError(
                f"X_expired row positions must be in [0, {n_rows}), got values "
                f"from {expired.min()} to {expired.max()}"
            )
        if np.unique(expired).shape[0] != expired.shape[0]:
            raise ValueError("X_expired row positions must not repeat")
        keep[expired] = False
        return keep
    if expired.ndim != 2:
        raise ValueError(
            f"X_expired must be 1-D or 2-D, got an array of shape {expired.shape}"
        )

    # Validates dtype and number of features
    queries, _, _, _ = search_index._prepare_queries(expired)
    k = min(n_candidates, n_rows)
    candidates, _ = search_index.query(expired, k, epsilon=0.2)
    if data.dtype == np.float32:
        tolerance = 1e-5

        def matches(row, positions):
            return np.all(np.abs(data[positions] - row) <= tolerance, axis=1)

    else:

        def matches(row, positions):
            return np.all(data[positions] == row, axis=1)

    for i in range(queries.shape[0]):
        row = queries[i]
        found = np.sort(candidates[i][candidates[i] >= 0])
        found = found[keep[found]]
        hits = found[matches(row, found)] if found.shape[0] else found
        if hits.shape[0] == 0:
            # Fall back to scanning every remaining row
            remaining = np.flatnonzero(keep)
            hits = remaining[matches(row, remaining)]
            if hits.shape[0] == 0:
                raise ValueError(
                    f"Row {i} of X_expired does not match any current sample"
                )
        keep[hits[0]] = False
    return keep
