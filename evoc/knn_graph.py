import numpy as np
import numba
import time

from sklearn.utils import check_array, check_random_state

from warnings import warn
from .float_nndescent import (
    make_float_forest,
    nn_descent_float,
    nn_descent_float_sorted,
)
from .uint8_nndescent import (
    make_uint8_forest,
    nn_descent_uint8,
    nn_descent_uint8_sorted,
)
from .int8_nndescent import make_int8_forest, nn_descent_int8, nn_descent_int8_sorted

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1


# Generates a timestamp for use in logging messages when verbose=True
def ts():
    return time.ctime(time.time())


def make_forest(
    data,
    n_neighbors,
    n_trees,
    leaf_size,
    random_state,
    input_dtype,
    max_depth=200,
):
    """Build a random projection forest with ``n_trees``.

    Parameters
    ----------
    data
    n_neighbors
    n_trees
    leaf_size
    rng_state
    angular

    Returns
    -------
    forest: list
        A list of random projection trees.
    """

    if leaf_size is None:
        leaf_size = max(10, np.int32(n_neighbors))

    rng_states = random_state.randint(INT32_MIN, INT32_MAX, size=(n_trees, 3)).astype(
        np.int64
    )
    try:
        if input_dtype == np.uint8:
            result = make_uint8_forest(data, rng_states, leaf_size, max_depth)
        elif input_dtype == np.int8:
            result = make_int8_forest(data, rng_states, leaf_size, max_depth)
        else:
            result = make_float_forest(data, rng_states, leaf_size, max_depth)
    except (RuntimeError, RecursionError, SystemError):
        warn(
            "Random Projection forest initialisation failed due to recursion"
            "limit being reached. Something is a little strange with your "
            "graph_data, and this may take longer than normal to compute."
        )
        return np.empty((0, 0), dtype=np.int32)

    # different trees can end up with different max leaf_sizes if the tree depth is insufficient
    max_leaf_size = np.max([leaf_array.shape[1] for leaf_array in result])

    # pad each leaf_array from each tree out to the max_leaf_size from any tree
    # so that vstack can succeed. Check np.pad docs for the specific semantics
    return np.vstack(
        [
            np.pad(
                leaf_array,
                ((0, 0), (0, max_leaf_size - leaf_array.shape[1])),
                constant_values=-1,
            )
            for leaf_array in result
        ]
    )


def nn_descent(
    data,
    n_neighbors,
    rng_state,
    effective_max_candidates,
    n_iters,
    delta,
    input_dtype,
    leaf_array=None,
    verbose=False,
    use_sorted_updates=True,
    delta_improv=None,
):
    if input_dtype == np.uint8:
        if use_sorted_updates:
            neighbor_graph = nn_descent_uint8_sorted(
                data,
                n_neighbors,
                rng_state,
                effective_max_candidates,
                n_iters,
                delta,
                delta_improv=delta_improv,
                leaf_array=leaf_array,
                verbose=verbose,
            )
        else:
            neighbor_graph = nn_descent_uint8(
                data,
                n_neighbors,
                rng_state,
                effective_max_candidates,
                n_iters,
                delta,
                delta_improv=delta_improv,
                leaf_array=leaf_array,
                verbose=verbose,
            )
        neighbor_graph[1][:] = -np.log2(-neighbor_graph[1])
    elif input_dtype == np.int8:
        if use_sorted_updates:
            neighbor_graph = nn_descent_int8_sorted(
                data,
                n_neighbors,
                rng_state,
                effective_max_candidates,
                n_iters,
                delta,
                delta_improv=delta_improv,
                leaf_array=leaf_array,
                verbose=verbose,
            )
        else:
            neighbor_graph = nn_descent_int8(
                data,
                n_neighbors,
                rng_state,
                effective_max_candidates,
                n_iters,
                delta,
                delta_improv=delta_improv,
                leaf_array=leaf_array,
                verbose=verbose,
            )
        neighbor_graph[1][:] = 1.0 / (-neighbor_graph[1])
    else:
        if use_sorted_updates:
            neighbor_graph = nn_descent_float_sorted(
                data,
                n_neighbors,
                rng_state,
                effective_max_candidates,
                n_iters,
                delta,
                delta_improv=delta_improv,
                leaf_array=leaf_array,
                verbose=verbose,
            )
        else:
            neighbor_graph = nn_descent_float(
                data,
                n_neighbors,
                rng_state,
                effective_max_candidates,
                n_iters,
                delta,
                delta_improv=delta_improv,
                leaf_array=leaf_array,
                verbose=verbose,
            )
        neighbor_graph[1][:] = np.maximum(-np.log2(-neighbor_graph[1]), 0.0)

    return neighbor_graph


def knn_graph(
    data,
    n_neighbors=30,
    n_trees=None,
    leaf_size=None,
    random_state=None,
    max_candidates=None,
    max_rptree_depth=200,
    n_iters=None,
    delta=0.001,
    delta_improv=0.001,
    n_jobs=None,
    verbose=False,
    use_sorted_updates=True,
):
    """Construct a k-nearest neighbor graph using the NN-Descent algorithm.

    This function builds a k-nearest neighbor graph using random projection trees
    for initialization followed by the NN-Descent algorithm for refinement. It
    supports multiple data types (float32 for normalized embeddings, int8 for
    quantized embeddings, uint8 for binary embeddings) with appropriate distance
    metrics for each.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        The data for which to compute nearest neighbors. If float32, cosine distance
        is used. If int8, quantized cosine distance is used. If uint8, Jaccard
        distance (based on Hamming distance for binary embeddings) is used.

    n_neighbors : int, default=30
        The number of nearest neighbors to compute for each sample.

    n_trees : int or None, default=None
        The number of random projection trees to build. If None, defaults to
        between 4 and 8 depending on the number of available threads.

    leaf_size : int or None, default=None
        The maximum number of points per leaf in the random projection trees.
        If None, defaults to max(10, n_neighbors).

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the algorithm. Pass an int for reproducible
        output across multiple function calls.

    max_candidates : int or None, default=None
        The maximum number of candidate neighbors to evaluate during NN-Descent.
        If None, defaults to min(60, int(n_neighbors * 1.5)).

    max_rptree_depth : int, default=200
        Maximum depth of the random projection trees.

    n_iters : int or None, default=None
        Number of iterations for the NN-Descent algorithm. If None, defaults to
        max(5, int(round(log2(n_samples)))).

    delta : float, default=0.001
        Convergence threshold for the NN-Descent algorithm.

    delta_improv : float, default=0.001
        Improvement threshold for early stopping in NN-Descent.

    n_jobs : int or None, default=None
        The number of threads to use. If -1, uses all available threads.
        If None, preserves the current numba thread setting.

    verbose : bool, default=False
        If True, print progress messages during computation.

    use_sorted_updates : bool, default=True
        If True, uses a more efficient sorted update strategy in NN-Descent.

    Returns
    -------
    neighbor_graph : tuple of (array, array)
        A tuple containing:
        - indices : array-like of shape (n_samples, n_neighbors)
            The indices of the k-nearest neighbors for each sample.
        - distances : array-like of shape (n_samples, n_neighbors)
            The distances from each sample to its k-nearest neighbors.
            Distances are transformed to a uniform scale based on the input dtype.
    """
    if data.dtype == np.uint8:
        data = check_array(data, dtype=np.uint8, order="C")
        _input_dtype = np.uint8
        _bit_trees = True
    elif data.dtype == np.int8:
        data = check_array(data, dtype=np.int8, order="C")
        _input_dtype = np.int8
        _bit_trees = False
    else:
        norms = np.einsum("ij,ij->i", data, data)
        np.sqrt(norms, norms)
        norms[norms == 0.0] = 1.0
        if np.allclose(norms, 1.0):
            # Data is already normalized, just ensure C-contiguity and float32
            data = np.ascontiguousarray(data, dtype=np.float32)
        else:
            # Efficiently create a modifiable float32 C-contiguous copy
            data = np.array(data, dtype=np.float32, order="C", copy=True)
            data /= norms[:, np.newaxis]
        _input_dtype = np.float32
        _bit_trees = False

    current_random_state = check_random_state(random_state)
    rng_state = current_random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)

    # Set threading constraints
    _original_num_threads = numba.get_num_threads()
    if n_jobs != -1 and n_jobs is not None:
        numba.set_num_threads(n_jobs)

    if n_trees is None:
        n_trees = numba.get_num_threads()
        n_trees = max(4, min(8, n_trees))  # Only so many trees are useful
    if n_iters is None:
        n_iters = max(5, int(round(np.log2(data.shape[0]))))

    if verbose:
        print(ts(), "Building RP forest with", str(n_trees), "trees")

    leaf_array = make_forest(
        data,
        n_neighbors,
        n_trees,
        leaf_size,
        current_random_state,
        _input_dtype,
        max_depth=max_rptree_depth,
    )

    if max_candidates is None:
        effective_max_candidates = min(60, int(n_neighbors * 1.5))
    else:
        effective_max_candidates = max_candidates

    if verbose:
        print(ts(), "NN descent for", str(n_iters), "iterations")

    neighbor_graph = nn_descent(
        data,
        n_neighbors,
        rng_state,
        effective_max_candidates,
        n_iters,
        delta,
        _input_dtype,
        leaf_array=leaf_array,
        verbose=verbose,
        use_sorted_updates=use_sorted_updates,
        delta_improv=delta_improv,
    )

    if np.any(neighbor_graph[0] < 0):
        warn(
            "Failed to correctly find n_neighbors for some samples."
            " Results may be less than ideal. Try re-running with"
            " different parameters."
        )

    if n_jobs != -1 and n_jobs is not None:
        numba.set_num_threads(_original_num_threads)
    return neighbor_graph
