"""Distance kernels for searching a fitted EVoC model's training data.

All distances here are in EVoC's *output* units: the units of the distances
returned by :func:`evoc.knn_graph.knn_graph` (and stored as ``nn_dists_``).

* float data (L2 normalised): ``max(-log2(cos), 0)``, where a non-positive
  cosine is replaced by the smallest positive float32 (so the distance is about
  126), exactly as ``knn_graph`` reports it;
* int8 data: the same, with ``cos = dot * inv_norm_x * inv_norm_y``;
* uint8 (bit-packed) data: ``max(-log2(jaccard), 0)``, which is ``inf`` when the
  two vectors share no set bits (or are both all zero).

During NN-descent construction EVoC works with *internal* distances (negated
similarities, see ``fast_cosine``, ``fast_int_cosine_dissimilarity`` and
``fast_bit_jaccard``). :func:`internal_to_output` is the conversion
``knn_graph`` applies, and :func:`internal_distances` recomputes internal
distances for given pairs; keep all unit conversions in this module.

The search code calls :func:`distance_to` and :func:`four_distances_to`, which
dispatch on the data dtype at compile time, so one search function serves
every input dtype without runtime branching on a kernel id.
"""
import numba
import numpy as np
from numba.extending import overload

from ..uint8_nndescent import load_u64_unaligned, popcnt_u64, popcnt_u8

FLOAT32_MAX = np.float32(np.finfo(np.float32).max)
EXP_NEG_INF = np.float32(np.finfo(np.float32).tiny)
FLOAT32_EPS = np.float32(np.finfo(np.float32).eps)
INF = np.float32(np.inf)


# --------------------------------------------------------------------------- scalar transforms
@numba.njit(fastmath=True, inline="always", cache=True)
def similarity_to_output(s):
    """Output-unit distance for a cosine (or cosine-like) similarity ``s``."""
    if s <= 0.0:
        s = EXP_NEG_INF
    d = -np.log2(np.float32(s))
    if d < 0.0:
        return np.float32(0.0)
    return np.float32(d)


@numba.njit(fastmath=True, inline="always", cache=True)
def jaccard_to_output(j):
    """Output-unit distance for a Jaccard similarity ``j`` (inf if ``j <= 0``)."""
    if j <= 0.0:
        return INF
    d = -np.log2(np.float32(j))
    if d < 0.0:
        return np.float32(0.0)
    return np.float32(d)


# --------------------------------------------------------------------------- pairwise kernels
@numba.njit(fastmath=True, inline="always", boundscheck=False, cache=True)
def float_distance(x, y):
    s = np.float32(0.0)
    for t in range(x.shape[0]):
        s += x[t] * y[t]
    return similarity_to_output(s)


@numba.njit(fastmath=True, inline="always", boundscheck=False, cache=True)
def int8_distance(x, y, inv_x, inv_y):
    s = np.int32(0)
    for t in range(x.shape[0]):
        s += np.int32(x[t]) * np.int32(y[t])
    return similarity_to_output(np.float32(s) * inv_x * inv_y)


@numba.njit(fastmath=True, inline="always", boundscheck=False, cache=True)
def bit_distance(x, y):
    # Same counting as fast_bit_jaccard: 64-bit words, then the byte tail
    dim = x.shape[0]
    n_words = dim // 8
    both = np.uint64(0)
    either = np.uint64(0)
    for w in range(n_words):
        offset = w * 8
        x_val = load_u64_unaligned(x, offset)
        y_val = load_u64_unaligned(y, offset)
        both += popcnt_u64(x_val & y_val)
        either += popcnt_u64(x_val | y_val)
    for d in range(n_words * 8, dim):
        both += popcnt_u8(x[d] & y[d])
        either += popcnt_u8(x[d] | y[d])
    if either == 0:
        return INF
    return jaccard_to_output(np.float32(both) / np.float32(either))


# --------------------------------------------------------------------------- dtype dispatch
def distance_to(query, query_inv_norm, data, inv_norms, candidate):
    """Output-unit distance from ``query`` to ``data[candidate]``.

    ``query_inv_norm`` and ``inv_norms`` are only used for int8 data.
    """
    raise NotImplementedError("distance_to is only available in numba code")


@overload(distance_to, jit_options={"fastmath": True})
def _ov_distance_to(query, query_inv_norm, data, inv_norms, candidate):
    if data.dtype == numba.types.uint8:

        def impl(query, query_inv_norm, data, inv_norms, candidate):
            return bit_distance(query, data[candidate])

    elif data.dtype == numba.types.int8:

        def impl(query, query_inv_norm, data, inv_norms, candidate):
            return int8_distance(
                query, data[candidate], query_inv_norm, inv_norms[candidate]
            )

    else:

        def impl(query, query_inv_norm, data, inv_norms, candidate):
            return float_distance(query, data[candidate])

    return impl


def four_distances_to(query, query_inv_norm, data, inv_norms, candidates, out):
    """Distances from ``query`` to the four rows ``data[candidates]`` into ``out``."""
    raise NotImplementedError("four_distances_to is only available in numba code")


@numba.njit(fastmath=True, inline="always", boundscheck=False, cache=True)
def _four_dots_float(x, y0, y1, y2, y3):
    s0 = s1 = s2 = s3 = np.float32(0.0)
    for t in range(x.shape[0]):
        xv = x[t]
        s0 += xv * y0[t]
        s1 += xv * y1[t]
        s2 += xv * y2[t]
        s3 += xv * y3[t]
    return s0, s1, s2, s3


@numba.njit(fastmath=True, inline="always", boundscheck=False, cache=True)
def _four_dots_int8(x, y0, y1, y2, y3):
    s0 = s1 = s2 = s3 = np.int32(0)
    for t in range(x.shape[0]):
        xv = np.int32(x[t])
        s0 += xv * np.int32(y0[t])
        s1 += xv * np.int32(y1[t])
        s2 += xv * np.int32(y2[t])
        s3 += xv * np.int32(y3[t])
    return s0, s1, s2, s3


@overload(four_distances_to, jit_options={"fastmath": True})
def _ov_four_distances_to(query, query_inv_norm, data, inv_norms, candidates, out):
    if data.dtype == numba.types.uint8:
        # Batching popcounts was measured slower than four single comparisons

        def impl(query, query_inv_norm, data, inv_norms, candidates, out):
            out[0] = bit_distance(query, data[candidates[0]])
            out[1] = bit_distance(query, data[candidates[1]])
            out[2] = bit_distance(query, data[candidates[2]])
            out[3] = bit_distance(query, data[candidates[3]])

    elif data.dtype == numba.types.int8:

        def impl(query, query_inv_norm, data, inv_norms, candidates, out):
            s0, s1, s2, s3 = _four_dots_int8(
                query,
                data[candidates[0]],
                data[candidates[1]],
                data[candidates[2]],
                data[candidates[3]],
            )
            out[0] = similarity_to_output(
                np.float32(s0) * query_inv_norm * inv_norms[candidates[0]]
            )
            out[1] = similarity_to_output(
                np.float32(s1) * query_inv_norm * inv_norms[candidates[1]]
            )
            out[2] = similarity_to_output(
                np.float32(s2) * query_inv_norm * inv_norms[candidates[2]]
            )
            out[3] = similarity_to_output(
                np.float32(s3) * query_inv_norm * inv_norms[candidates[3]]
            )

    else:

        def impl(query, query_inv_norm, data, inv_norms, candidates, out):
            s0, s1, s2, s3 = _four_dots_float(
                query,
                data[candidates[0]],
                data[candidates[1]],
                data[candidates[2]],
                data[candidates[3]],
            )
            out[0] = similarity_to_output(s0)
            out[1] = similarity_to_output(s1)
            out[2] = similarity_to_output(s2)
            out[3] = similarity_to_output(s3)

    return impl


# --------------------------------------------------------------------------- bulk helpers
@numba.njit(parallel=True, fastmath=True, cache=True)
def pair_distances(data, inv_norms, rows, cols):
    """Output-unit distances between ``data[rows[i]]`` and ``data[cols[i]]``.

    Negative indices give ``inf``.
    """
    result = np.empty(rows.shape[0], dtype=np.float32)
    for i in numba.prange(rows.shape[0]):
        r = rows[i]
        c = cols[i]
        if r < 0 or c < 0:
            result[i] = INF
        else:
            if inv_norms.shape[0] > 0:
                inv_r = inv_norms[r]
            else:
                inv_r = np.float32(1.0)
            result[i] = distance_to(data[r], inv_r, data, inv_norms, c)
    return result


@numba.njit(parallel=True, fastmath=True, cache=True)
def query_distances(queries, query_inv_norms, data, inv_norms, indices):
    """Output-unit distances from each query row to ``data[indices[i, j]]``."""
    n, k = indices.shape
    result = np.full((n, k), INF, dtype=np.float32)
    for i in numba.prange(n):
        if query_inv_norms.shape[0] > 0:
            inv_q = query_inv_norms[i]
        else:
            inv_q = np.float32(1.0)
        for j in range(k):
            c = indices[i, j]
            if c >= 0:
                result[i, j] = distance_to(queries[i], inv_q, data, inv_norms, c)
    return result


def internal_to_output(distances):
    """Convert internal NN-descent distances to output units, as knn_graph does."""
    with np.errstate(divide="ignore"):
        return np.maximum(-np.log2(-np.asarray(distances, dtype=np.float32)), 0.0)


def output_to_internal(distances, input_dtype):
    """Convert output-unit distances back to internal NN-descent distances.

    Exact up to rounding except where the output transform clips: output 0
    (similarity >= 1) maps to -1, and the float/int8 "no similarity" value maps
    to about ``-EXP_NEG_INF``. Prefer :func:`internal_distances` where the data
    is at hand.
    """
    distances = np.asarray(distances, dtype=np.float32)
    with np.errstate(over="ignore"):
        result = -np.exp2(-distances.astype(np.float64))
    if input_dtype == np.uint8:
        # knn_graph reports Jaccard 0 as inf, whose internal value is 0
        result[np.isinf(distances)] = 0.0
    return result.astype(np.float32)


def internal_distances(data, inv_norms, rows, cols):
    """Internal (negated similarity) distances between rows of ``data``."""
    return output_to_internal(
        pair_distances(data, inv_norms, rows, cols), data.dtype.type
    )


def prepare_data(data):
    """Return (data, inv_norms) in the form the search code expects.

    Float data is converted to L2-normalised float32 (zero rows left as zero),
    exactly as :func:`evoc.knn_graph.knn_graph` does; int8 data gets its inverse
    row norms; uint8 data is used as is. ``inv_norms`` is empty unless the data
    is int8.
    """
    from ..int8_nndescent import int8_inverse_norms

    if data.dtype == np.uint8:
        return np.ascontiguousarray(data), np.zeros(0, dtype=np.float32)
    if data.dtype == np.int8:
        data = np.ascontiguousarray(data)
        return data, int8_inverse_norms(data)
    norms = np.einsum("ij,ij->i", data, data)
    np.sqrt(norms, norms)
    norms[norms == 0.0] = 1.0
    if np.allclose(norms, 1.0):
        data = np.ascontiguousarray(data, dtype=np.float32)
    else:
        data = np.array(data, dtype=np.float32, order="C", copy=True)
        data /= norms[:, np.newaxis]
    return data, np.zeros(0, dtype=np.float32)


# ------------------------------------------------------- internal-unit kernels
# Distances in the internal (NN-descent) units of each dtype, for code that
# pushes into NN-descent heaps: negated cosine similarity for float and int8
# data (-EXP_NEG_INF when not positive, as fast_cosine), negated Jaccard
# similarity for bit-packed data (0 when there is no overlap).
def internal_distance_to(query, query_inv_norm, data, inv_norms, candidate):
    """Internal-unit distance from ``query`` to ``data[candidate]``."""
    raise NotImplementedError("internal_distance_to is only available in numba code")


@overload(internal_distance_to, jit_options={"fastmath": True})
def _ov_internal_distance_to(query, query_inv_norm, data, inv_norms, candidate):
    if data.dtype == numba.types.uint8:
        from ..uint8_nndescent import fast_bit_jaccard

        def impl(query, query_inv_norm, data, inv_norms, candidate):
            return fast_bit_jaccard(query, data[candidate])

    elif data.dtype == numba.types.int8:
        from ..int8_nndescent import fast_int_cosine_dissimilarity

        def impl(query, query_inv_norm, data, inv_norms, candidate):
            return fast_int_cosine_dissimilarity(
                query, data[candidate], query_inv_norm, inv_norms[candidate]
            )

    else:
        from ..float_nndescent import fast_cosine

        def impl(query, query_inv_norm, data, inv_norms, candidate):
            return fast_cosine(query, data[candidate])

    return impl


def four_internal_distances_to(query, query_inv_norm, data, inv_norms, candidates, out):
    """Internal-unit distances from ``query`` to the four rows ``data[candidates]``."""
    raise NotImplementedError(
        "four_internal_distances_to is only available in numba code"
    )


@numba.njit(fastmath=True, inline="always", cache=True)
def _negated_similarity(s):
    if s > 0.0:
        return np.float32(-s)
    return -EXP_NEG_INF


@overload(four_internal_distances_to, jit_options={"fastmath": True})
def _ov_four_internal_distances_to(query, query_inv_norm, data, inv_norms, candidates, out):
    if data.dtype == numba.types.uint8:
        from ..uint8_nndescent import fast_bit_jaccard

        # Batching popcounts was measured slower than four single comparisons
        def impl(query, query_inv_norm, data, inv_norms, candidates, out):
            out[0] = fast_bit_jaccard(query, data[candidates[0]])
            out[1] = fast_bit_jaccard(query, data[candidates[1]])
            out[2] = fast_bit_jaccard(query, data[candidates[2]])
            out[3] = fast_bit_jaccard(query, data[candidates[3]])

    elif data.dtype == numba.types.int8:

        def impl(query, query_inv_norm, data, inv_norms, candidates, out):
            s0, s1, s2, s3 = _four_dots_int8(
                query,
                data[candidates[0]],
                data[candidates[1]],
                data[candidates[2]],
                data[candidates[3]],
            )
            out[0] = _negated_similarity(
                np.float32(s0) * query_inv_norm * inv_norms[candidates[0]]
            )
            out[1] = _negated_similarity(
                np.float32(s1) * query_inv_norm * inv_norms[candidates[1]]
            )
            out[2] = _negated_similarity(
                np.float32(s2) * query_inv_norm * inv_norms[candidates[2]]
            )
            out[3] = _negated_similarity(
                np.float32(s3) * query_inv_norm * inv_norms[candidates[3]]
            )

    else:

        def impl(query, query_inv_norm, data, inv_norms, candidates, out):
            s0, s1, s2, s3 = _four_dots_float(
                query,
                data[candidates[0]],
                data[candidates[1]],
                data[candidates[2]],
                data[candidates[3]],
            )
            out[0] = _negated_similarity(s0)
            out[1] = _negated_similarity(s1)
            out[2] = _negated_similarity(s2)
            out[3] = _negated_similarity(s3)

    return impl
