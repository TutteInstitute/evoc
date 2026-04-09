"""GPU-accelerated brute-force k-nearest neighbor graph construction.

Replaces the CPU NN-Descent algorithm with GPU matrix multiplication for
exact KNN computation. For normalized vectors, cosine similarity is computed
as X @ X.T, which maps perfectly to GPU GEMM operations.

Supports float32 (cosine), int8 (quantized cosine), and uint8 (bit Jaccard).
"""

import numpy as np
import torch

from . import get_device, _auto_batch_size


def knn_graph_gpu(data, n_neighbors=30, batch_size=None, verbose=False):
    """Construct a k-nearest neighbor graph using GPU brute-force computation.

    For float32 data, cosine similarity is computed via normalized matrix multiply.
    For int8 data, inner product similarity is used.
    For uint8 binary data, bit Jaccard similarity via unpacked matrix multiply.

    Parameters
    ----------
    data : ndarray of shape (n_samples, n_features)
        Input data. dtype determines the distance metric used.

    n_neighbors : int, default=30
        Number of nearest neighbors to compute for each sample.

    batch_size : int or None, default=None
        Rows to process per GPU batch. Auto-selected if None.

    verbose : bool, default=False
        If True, print progress information.

    Returns
    -------
    indices : ndarray of shape (n_samples, n_neighbors), dtype int32
        Indices of nearest neighbors for each sample.

    distances : ndarray of shape (n_samples, n_neighbors), dtype float32
        Transformed distances matching the CPU EVoC distance space.
    """
    if data.dtype == np.uint8:
        return _knn_uint8(data, n_neighbors, batch_size, verbose)
    elif data.dtype == np.int8:
        return _knn_int8(data, n_neighbors, batch_size, verbose)
    else:
        return _knn_float(data, n_neighbors, batch_size, verbose)


def _knn_float(data, k, batch_size, verbose):
    """GPU KNN for float32 data using cosine similarity."""
    device = get_device()
    data = np.ascontiguousarray(data, dtype=np.float32)

    # Normalize for cosine similarity
    norms = np.einsum("ij,ij->i", data, data)
    np.sqrt(norms, norms)
    norms[norms == 0.0] = 1.0
    if not np.allclose(norms, 1.0):
        data = data.copy()
        data /= norms[:, np.newaxis]

    n, d = data.shape
    k = min(k, n - 1)
    if batch_size is None:
        batch_size = _auto_batch_size(n, d)

    data_gpu = torch.from_numpy(data).to(device)
    indices = np.empty((n, k), dtype=np.int32)
    distances = np.empty((n, k), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bsz = end - start

        # Cosine similarity via matmul of normalized vectors
        sims = torch.mm(data_gpu[start:end], data_gpu.T)

        # Exclude self-similarity
        sims[torch.arange(bsz, device=device), torch.arange(start, end, device=device)] = -2.0

        topk_vals, topk_inds = torch.topk(sims, k, dim=1, sorted=True)

        inds_np = topk_inds.cpu().numpy().astype(np.int32)
        sims_np = topk_vals.cpu().numpy().astype(np.float32)

        # Transform to EVoC distance space: max(-log2(cos_sim), 0)
        # Matches CPU: heap stores -cos_sim, then max(-log2(-val), 0)
        sims_np = np.clip(sims_np, 1e-10, 1.0)
        distances[start:end] = np.maximum(-np.log2(sims_np), 0.0).astype(np.float32)
        indices[start:end] = inds_np

        if verbose and start % (batch_size * 10) == 0:
            print(f"  GPU KNN: {end}/{n} samples processed")

    del data_gpu
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return indices, distances


def _knn_int8(data, k, batch_size, verbose):
    """GPU KNN for int8 quantized embeddings via inner product."""
    device = get_device()
    n, d = data.shape
    k = min(k, n - 1)
    if batch_size is None:
        batch_size = _auto_batch_size(n, d)

    # Cast to float32 for GPU matmul
    data_gpu = torch.from_numpy(data.astype(np.float32)).to(device)
    indices = np.empty((n, k), dtype=np.int32)
    distances = np.empty((n, k), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bsz = end - start

        sims = torch.mm(data_gpu[start:end], data_gpu.T)
        sims[torch.arange(bsz, device=device), torch.arange(start, end, device=device)] = -1e30

        topk_vals, topk_inds = torch.topk(sims, k, dim=1, sorted=True)

        inds_np = topk_inds.cpu().numpy().astype(np.int32)
        sims_np = topk_vals.cpu().numpy().astype(np.float32)

        # Transform: CPU does 1.0 / (-heap_val) where heap_val = -dot_product
        sims_np = np.maximum(sims_np, 1e-10)
        distances[start:end] = (1.0 / sims_np).astype(np.float32)
        indices[start:end] = inds_np

    del data_gpu
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return indices, distances


def _knn_uint8(data, k, batch_size, verbose):
    """GPU KNN for uint8 binary embeddings using bit Jaccard similarity.

    Unpacks bytes to individual bits (float32) and computes Jaccard similarity
    via matrix multiply: intersection = batch_bits @ all_bits.T
    """
    device = get_device()
    n, n_bytes = data.shape
    k = min(k, n - 1)

    if batch_size is None:
        # Smaller batches for uint8 because unpacked bits use 8x more memory
        batch_size = _auto_batch_size(n, n_bytes * 8)
        batch_size = min(batch_size, 2048)

    # Unpack bytes to individual bits as float32 for matmul
    bit_shifts = torch.arange(8, device=device, dtype=torch.uint8)
    data_gpu = torch.from_numpy(data).to(device)
    bits = ((data_gpu.unsqueeze(-1) >> bit_shifts) & 1).reshape(n, -1).float()
    del data_gpu

    # Pre-compute popcount per sample for Jaccard denominator
    pop_all = bits.sum(dim=1, keepdim=True)  # (n, 1)

    indices = np.empty((n, k), dtype=np.int32)
    distances = np.empty((n, k), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bsz = end - start

        batch_bits = bits[start:end]
        pop_batch = pop_all[start:end]  # (bsz, 1)

        # intersection(a, b) = dot(a_bits, b_bits) for 0/1 vectors
        intersection = torch.mm(batch_bits, bits.T)
        union = pop_batch + pop_all.T - intersection

        # Jaccard similarity = intersection / union
        jaccard = intersection / union.clamp(min=1.0)

        # Exclude self
        jaccard[torch.arange(bsz, device=device), torch.arange(start, end, device=device)] = -1.0

        topk_vals, topk_inds = torch.topk(jaccard, k, dim=1, sorted=True)

        inds_np = topk_inds.cpu().numpy().astype(np.int32)
        sims_np = topk_vals.cpu().numpy().astype(np.float32)

        # Transform: CPU does -log2(-heap_val) where heap_val = -jaccard_sim
        sims_np = np.clip(sims_np, 1e-10, 1.0)
        distances[start:end] = (-np.log2(sims_np)).astype(np.float32)
        indices[start:end] = inds_np

    del bits, pop_all
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return indices, distances
