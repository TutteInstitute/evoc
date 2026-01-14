import os
import sys
import numba


def supports_safe_nesting():
    # Check if user explicitly set a layer
    layer = os.environ.get("NUMBA_THREADING_LAYER", "")

    if layer in ("tbb", "omp"):
        return True

    # Check loaded libraries (if numba has already initialized)
    try:
        if "tbb" in numba.threading_layer():
            return True
    except (ValueError, numba.errors.NumbaError):
        # Numba hasn't selected a layer yet, or multiple are available.
        pass

    # Heuristic: If on Mac and TBB is not strictly enforced/present, assume unsafe.
    if sys.platform == "darwin":
        # You could try importing tbb to be sure
        try:
            import tbb

            return True
        except ImportError:
            return False

    return True


ENABLE_NESTED_PARALLELISM = supports_safe_nesting()
