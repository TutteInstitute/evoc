"""GPU acceleration backend for EVoC clustering.

Provides GPU-accelerated implementations of the most compute-intensive
operations in the EVoC pipeline using PyTorch as the GPU backend.

Supports CUDA (NVIDIA), ROCm (AMD), and MPS (Apple Silicon) devices.
"""

_GPU_AVAILABLE = False
_GPU_BACKEND = None


def _detect_gpu():
    """Detect available GPU backend on import."""
    global _GPU_AVAILABLE, _GPU_BACKEND
    try:
        import torch

        if torch.cuda.is_available():
            _GPU_AVAILABLE = True
            _GPU_BACKEND = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _GPU_AVAILABLE = True
            _GPU_BACKEND = "mps"
    except ImportError:
        pass


def gpu_available():
    """Return True if a supported GPU backend is available."""
    return _GPU_AVAILABLE


def get_device():
    """Get the PyTorch device for GPU operations.

    Returns
    -------
    torch.device
        The best available GPU device.

    Raises
    ------
    RuntimeError
        If no GPU backend is available.
    """
    import torch

    if _GPU_BACKEND == "cuda":
        return torch.device("cuda")
    elif _GPU_BACKEND == "mps":
        return torch.device("mps")
    raise RuntimeError(
        "No GPU available. Install PyTorch with CUDA or MPS support, "
        "or set use_gpu=False."
    )


def _auto_batch_size(n_samples, n_features, element_bytes=4):
    """Estimate batch size for GPU KNN that fits in GPU memory.

    Uses ~40% of GPU memory for the similarity matrix computation,
    accounting for the full dataset already being on-device.
    """
    import torch

    if _GPU_BACKEND == "cuda":
        total_mem = torch.cuda.get_device_properties(0).total_memory
    else:
        # MPS does not easily expose memory; assume 8GB
        total_mem = 8 * 1024**3

    data_bytes = n_samples * n_features * element_bytes
    usable = max(int(total_mem * 0.4) - data_bytes, int(total_mem * 0.1))
    bs = max(128, usable // max(n_samples * element_bytes, 1))
    return min(bs, n_samples)


_detect_gpu()
