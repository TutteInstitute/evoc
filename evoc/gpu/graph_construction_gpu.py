"""GPU-accelerated graph construction from KNN data.

Ports the smooth_knn_dist binary search and membership strength computation
to fully vectorized GPU operations via PyTorch.
"""

import numpy as np
import torch

from scipy.sparse import coo_array

from . import get_device

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3


def smooth_knn_dist_gpu(distances, k, n_iter=64, bandwidth=1.0):
    """GPU-accelerated computation of smooth KNN distances.

    Vectorised binary search across all samples simultaneously on GPU.

    Parameters
    ----------
    distances : ndarray of shape (n_samples, n_neighbors), float32
        KNN distance matrix.

    k : float
        Effective number of neighbors for kernel width estimation.

    n_iter : int, default=64
        Maximum binary search iterations.

    bandwidth : float, default=1.0
        Kernel bandwidth scaling factor.

    Returns
    -------
    sigma : ndarray of shape (n_samples,), float32
    rho : ndarray of shape (n_samples,), float32
    """
    device = get_device()
    target = np.log2(k) * bandwidth
    n_samples, n_neighbors = distances.shape

    dists = torch.from_numpy(distances.astype(np.float32)).to(device)
    mean_distances = dists.mean().item()

    # Compute rho: distance to nearest non-zero neighbor
    dists_masked = dists.clone()
    dists_masked[dists_masked <= 0] = float("inf")
    rho = dists_masked.min(dim=1).values
    rho[rho == float("inf")] = 0.0

    # Binary search for sigma — fully vectorized across all samples
    lo = torch.zeros(n_samples, device=device, dtype=torch.float32)
    hi = torch.full((n_samples,), float("inf"), device=device, dtype=torch.float32)
    mid = torch.ones(n_samples, device=device, dtype=torch.float32)

    for _ in range(n_iter):
        # psum = sum_{j=1..k} exp(-(d[i,j] - rho[i]) / mid[i]) if d>rho else 1
        d = dists[:, 1:] - rho.unsqueeze(1)  # (n_samples, n_neighbors-1)
        psum = torch.where(
            d > 0,
            torch.exp(-d / mid.unsqueeze(1).clamp(min=1e-10)),
            torch.ones_like(d),
        ).sum(dim=1)

        converged = torch.abs(psum - target) < SMOOTH_K_TOLERANCE
        too_high = psum > target
        too_low = ~too_high & ~converged

        # Update hi/lo/mid
        hi = torch.where(too_high, mid, hi)
        lo = torch.where(too_low, mid, lo)

        inf_hi = hi == float("inf")
        new_mid = torch.where(
            too_high,
            (lo + hi) / 2.0,
            torch.where(
                too_low & inf_hi,
                mid * 2.0,
                torch.where(too_low & ~inf_hi, (lo + hi) / 2.0, mid),
            ),
        )
        mid = torch.where(converged, mid, new_mid)

    sigma = mid

    # Minimum distance scaling
    mean_per_sample = dists.mean(dim=1)
    rho_pos = rho > 0
    sigma = torch.where(
        rho_pos & (sigma < MIN_K_DIST_SCALE * mean_per_sample),
        MIN_K_DIST_SCALE * mean_per_sample,
        sigma,
    )
    sigma = torch.where(
        ~rho_pos & (sigma < MIN_K_DIST_SCALE * mean_distances),
        torch.full_like(sigma, MIN_K_DIST_SCALE * mean_distances),
        sigma,
    )

    return sigma.cpu().numpy(), rho.cpu().numpy()


def compute_membership_strengths_gpu(knn_indices, knn_dists, sigmas, rhos):
    """GPU-accelerated membership strength computation.

    Computes the Gaussian kernel weights for each KNN edge in parallel on GPU.

    Parameters
    ----------
    knn_indices : ndarray of shape (n_samples, n_neighbors), int32
    knn_dists : ndarray of shape (n_samples, n_neighbors), float32
    sigmas : ndarray of shape (n_samples,), float32
    rhos : ndarray of shape (n_samples,), float32

    Returns
    -------
    rows, cols, vals : ndarrays
        COO format sparse matrix components.
    """
    device = get_device()
    n_samples, n_neighbors = knn_indices.shape

    inds = torch.from_numpy(knn_indices.astype(np.int64)).to(device)
    dists = torch.from_numpy(knn_dists.astype(np.float32)).to(device)
    sig = torch.from_numpy(sigmas.astype(np.float32)).to(device)
    rho = torch.from_numpy(rhos.astype(np.float32)).to(device)

    # Row indices (sample ids)
    sample_idx = (
        torch.arange(n_samples, device=device).unsqueeze(1).expand_as(inds)
    )

    # Compute kernel weights: exp(-(d - rho) / sigma)
    d = dists - rho.unsqueeze(1)
    sigma_exp = sig.unsqueeze(1).clamp(min=1e-10)
    vals = torch.exp(-d / sigma_exp)

    # val = 0 for self-edges and missing neighbors (-1)
    vals = torch.where(inds == sample_idx, torch.zeros_like(vals), vals)
    vals = torch.where(inds == -1, torch.zeros_like(vals), vals)
    # val = 1 where d <= 0 or sigma == 0 (unless self/missing)
    close_mask = (d <= 0) | (sig.unsqueeze(1) == 0)
    vals = torch.where(
        close_mask & (inds != sample_idx) & (inds != -1),
        torch.ones_like(vals),
        vals,
    )

    rows = sample_idx.reshape(-1).cpu().numpy().astype(np.int32)
    cols = inds.reshape(-1).cpu().numpy().astype(np.int32)
    values = vals.reshape(-1).cpu().numpy().astype(np.float32)

    return rows, cols, values


def neighbor_graph_matrix_gpu(n_neighbors, knn_indices, knn_dists, symmetrize=True):
    """GPU-accelerated neighbor graph matrix construction.

    Parameters
    ----------
    n_neighbors : float
        Effective number of neighbors for kernel width estimation.

    knn_indices : ndarray of shape (n_samples, k), int32
    knn_dists : ndarray of shape (n_samples, k), float32
    symmetrize : bool, default=True

    Returns
    -------
    graph : scipy sparse matrix
        Weighted KNN graph.
    """
    knn_dists = knn_dists.astype(np.float32)

    sigmas, rhos = smooth_knn_dist_gpu(knn_dists, float(n_neighbors))
    rows, cols, vals = compute_membership_strengths_gpu(
        knn_indices, knn_dists, sigmas, rhos
    )

    result = coo_array(
        (vals, (rows, cols)),
        shape=(knn_indices.shape[0], knn_indices.shape[0]),
        dtype=np.float32,
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
