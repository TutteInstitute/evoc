"""Relative-neighbourhood pruning (diversification) of a k-NN graph.

Mirrors ``pynndescent.pynndescent_.diversify`` (pynndescent 1f460b2) with the
EVoC distance kernels from :mod:`evoc.search.kernels`, in output units.
"""
import numba
import numpy as np

from ._pynnd_utils import hash_uniform
from .kernels import FLOAT32_EPS, distance_to


@numba.njit(inline="always")
def _prune_edge(prune_probability, seed, i, j, m):
    if prune_probability >= 1.0:
        return True
    return hash_uniform(seed, i, j, m) < prune_probability


@numba.njit(parallel=True, fastmath=True, cache=True)
def diversify(indices, distances, data, inv_norms, seed, prune_probability=1.0):
    """Relative-neighbourhood pruning of each row of a k-NN graph, in place.

    Each row (without self, sorted by increasing distance) is scanned in order;
    a candidate u is dropped if some already retained neighbour c (other than
    an exact duplicate of the row's point) is closer to u than the row's point
    is, with probability ``prune_probability`` per such c. Retained neighbours
    keep their order and the row is padded with (-1, inf).
    """
    n, k = indices.shape
    for i in numba.prange(n):
        n_retained = 1
        for j in range(1, k):
            u = indices[i, j]
            if u < 0:
                break
            d_iu = distances[i, j]
            if inv_norms.shape[0] > 0:
                inv_u = inv_norms[u]
            else:
                inv_u = np.float32(1.0)
            keep = True
            for m in range(n_retained):
                if distances[i, m] > FLOAT32_EPS:
                    if distance_to(data[u], inv_u, data, inv_norms, indices[i, m]) < d_iu:
                        if _prune_edge(prune_probability, seed, i, j, m):
                            keep = False
                            break
            if keep:
                indices[i, n_retained] = u
                distances[i, n_retained] = d_iu
                n_retained += 1
        for j in range(n_retained, k):
            indices[i, j] = -1
            distances[i, j] = np.inf

    return indices, distances
