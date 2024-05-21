import numpy as np
import numba

from scipy.sparse import coo_matrix

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1
SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf


@numba.njit(
    locals={
        "psum": numba.types.float32,
        "lo": numba.types.float32,
        "mid": numba.types.float32,
        "hi": numba.types.float32,
    },
    fastmath=True,
    parallel=True,
    cache=True,
)
def smooth_knn_dist(distances, k, n_iter=64, bandwidth=1.0):
    target = np.log2(k) * bandwidth
    rho = np.zeros(distances.shape[0], dtype=np.float32)
    sigma = np.zeros(distances.shape[0], dtype=np.float32)

    mean_distances = np.mean(distances)

    for i in numba.prange(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= 1:
            rho[i] = non_zero_dists[0]

        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1.0

            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0

        sigma[i] = mid

        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if sigma[i] < MIN_K_DIST_SCALE * mean_ith_distances:
                sigma[i] = MIN_K_DIST_SCALE * mean_ith_distances
        else:
            if sigma[i] < MIN_K_DIST_SCALE * mean_distances:
                sigma[i] = MIN_K_DIST_SCALE * mean_distances

    return sigma, rho


@numba.njit(
    locals={
        "knn_dists": numba.types.float32[:, ::1],
        "sigmas": numba.types.float32[::1],
        "rhos": numba.types.float32[::1],
        "sigma": numba.types.float32,
        "rho": numba.types.float32,
        "val": numba.types.float32,
    },
    parallel=True,
    fastmath=True,
    cache=True,
)
def compute_membership_strengths(
    knn_indices,
    knn_dists,
    sigmas,
    rhos,
):
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float32)

    for i in range(n_samples):
        rho = rhos[i]
        sigma = sigmas[i]
        for j in range(n_neighbors):
            idx = knn_indices[i, j]
            if idx == -1:
                continue  # We didn't get the full knn for i
            elif idx == i:
                val = 0.0
            elif (knn_dists[i, j] - rho) <= 0.0 or sigma == 0.0:
                val = 1.0
            else:
                val = np.exp(-((knn_dists[i, j] - rhos[i]) / (sigma)))

            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = idx
            vals[i * n_neighbors + j] = val

    return rows, cols, vals


def neighbor_graph_matrix(
    n_neighbors,
    knn_indices,
    knn_dists,
    symmetrize=True,
):
    knn_dists = knn_dists.astype(np.float32)

    sigmas, rhos = smooth_knn_dist(
        knn_dists,
        float(n_neighbors),
    )

    rows, cols, vals = compute_membership_strengths(
        knn_indices, knn_dists, sigmas, rhos
    )

    result = coo_matrix(
        (vals, (rows, cols)), shape=(knn_indices.shape[0], knn_indices.shape[0]), dtype=np.float32,
    )
    result.eliminate_zeros()

    if symmetrize:
        transpose = result.transpose()

        prod_matrix = result.multiply(transpose)
        result = result + transpose - prod_matrix
    else:
        result = result.tocsr()

    result.eliminate_zeros()

    return result