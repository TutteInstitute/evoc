"""GPU-accelerated node embedding via edge-parallel SGD.

Ports the UMAP-like graph embedding optimization to GPU using PyTorch.
All active edges within an epoch are processed in parallel via vectorized
gradient computation and scatter_add_ for embedding updates (Hogwild!-style).
"""

import numpy as np
import torch

from tqdm import tqdm

from . import get_device


def _make_epochs_per_sample(weights, n_epochs):
    """Compute how frequently each edge should be sampled."""
    n_samples = np.maximum(n_epochs * (weights / weights.max()), 1.0)
    return (float(n_epochs) / np.float32(n_samples)).astype(np.float32)


def node_embedding_gpu(
    graph,
    n_components,
    n_epochs,
    initial_embedding=None,
    initial_alpha=0.5,
    negative_sample_rate=1.0,
    noise_level=0.5,
    random_state=None,
    reproducible_flag=True,
    verbose=False,
    tqdm_kwds=None,
):
    """Learn a low-dimensional embedding of a graph on GPU.

    Uses edge-parallel stochastic gradient descent with attractive forces
    on connected edges and repulsive forces on random negative pairs.
    All active edges in an epoch are processed simultaneously.

    Parameters
    ----------
    graph : scipy sparse matrix
        Weighted adjacency matrix of the graph.

    n_components : int
        Number of embedding dimensions.

    n_epochs : int
        Number of training epochs.

    initial_embedding : ndarray or None
        Starting embedding. Random if None.

    initial_alpha : float, default=0.5
        Initial learning rate (decays linearly).

    negative_sample_rate : float, default=1.0
        Ratio of negative to positive samples.

    noise_level : float, default=0.5
        Controls attractive gradient strength.

    random_state : RandomState or None
        For initial embedding generation.

    reproducible_flag : bool, default=True
        Ignored on GPU (GPU operations are inherently non-deterministic).
        Kept for API compatibility.

    verbose : bool, default=False
        Show progress bar.

    tqdm_kwds : dict or None
        Additional tqdm keyword arguments.

    Returns
    -------
    embedding : ndarray of shape (n_vertices, n_components), float32
    """
    device = get_device()

    if random_state is None:
        random_state = np.random.RandomState()

    n_vertices = graph.shape[0]

    # Initialize embedding
    if initial_embedding is not None:
        embedding = torch.from_numpy(
            np.ascontiguousarray(initial_embedding, dtype=np.float32)
        ).to(device)
    else:
        embedding = (
            torch.from_numpy(
                random_state.normal(
                    scale=0.25, size=(n_vertices, n_components)
                ).astype(np.float32)
            ).to(device)
        )

    # Prepare graph edges in COO format
    coo = graph.tocoo()
    head = torch.from_numpy(coo.row.astype(np.int64)).to(device)
    tail = torch.from_numpy(coo.col.astype(np.int64)).to(device)
    weights = coo.data.astype(np.float32)

    n_edges = head.shape[0]
    dim = n_components

    # Epoch scheduling: higher-weight edges are processed more frequently
    eps = _make_epochs_per_sample(weights, n_epochs)
    eps_neg = (eps / negative_sample_rate).astype(np.float32)

    eps_gpu = torch.from_numpy(eps).to(device)
    eps_neg_gpu = torch.from_numpy(eps_neg).to(device)
    epoch_of_next_sample = eps_gpu.clone()
    epoch_of_next_neg = eps_neg_gpu.clone()

    if tqdm_kwds is None:
        tqdm_kwds = {}
    if "disable" not in tqdm_kwds:
        tqdm_kwds["disable"] = not verbose

    for epoch in tqdm(range(n_epochs), **tqdm_kwds):
        alpha = np.float32(initial_alpha * (1.0 - float(epoch) / float(n_epochs)))

        # ---- Positive (attractive) updates ----
        pos_mask = epoch_of_next_sample <= epoch
        if pos_mask.any():
            active_idx = pos_mask.nonzero(as_tuple=True)[0]
            ah = head[active_idx]
            at = tail[active_idx]

            current = embedding[ah]  # (n_active, dim)
            other = embedding[at]

            diff = current - other
            dist_sq = (diff * diff).sum(dim=1, keepdim=True)  # (n_active, 1)
            dist = torch.sqrt(dist_sq.clamp(min=1e-10))

            # Attractive gradient coefficient (matches CPU formula)
            grad_coeff = (-2.0 * noise_level * dist - 2.0) / (
                2.0 * dist_sq - 0.5 * dist + 1.0
            )
            grad = grad_coeff * diff * alpha

            # Apply to both endpoints via atomic scatter_add_
            embedding.scatter_add_(0, ah.unsqueeze(1).expand(-1, dim), grad)
            embedding.scatter_add_(0, at.unsqueeze(1).expand(-1, dim), -grad)

            # Advance schedule for active edges
            epoch_of_next_sample[active_idx] += eps_gpu[active_idx]

        # ---- Negative (repulsive) updates ----
        neg_mask = epoch_of_next_neg <= epoch
        if neg_mask.any():
            neg_idx = neg_mask.nonzero(as_tuple=True)[0]
            nh = head[neg_idx]
            nt = torch.randint(0, n_vertices, (neg_idx.shape[0],), device=device)

            curr_neg = embedding[nh]
            other_neg = embedding[nt]

            diff_neg = curr_neg - other_neg
            dist_sq_neg = (diff_neg * diff_neg).sum(dim=1, keepdim=True)

            # Only apply when points are far enough apart
            valid = dist_sq_neg.squeeze(1) > 1e-2
            grad_coeff_neg = 4.0 / (
                (1.0 + 0.25 * dist_sq_neg) * dist_sq_neg.clamp(min=1e-10)
            )
            grad_neg = torch.clamp(grad_coeff_neg * diff_neg * alpha, -4.0, 4.0)
            grad_neg[~valid] = 0.0

            embedding.scatter_add_(0, nh.unsqueeze(1).expand(-1, dim), grad_neg)

            epoch_of_next_neg[neg_idx] += eps_neg_gpu[neg_idx]

    result = embedding.cpu().numpy()

    del embedding, head, tail, eps_gpu, eps_neg_gpu
    del epoch_of_next_sample, epoch_of_next_neg
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result
