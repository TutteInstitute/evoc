import numpy as np
import numba
import time

from sklearn.utils import check_array, check_random_state

from warnings import warn
from .float_nndescent import make_float_forest, nn_descent_float
from .uint8_nndescent import make_uint8_forest, nn_descent_uint8
from .int8_nndescent import make_int8_forest, nn_descent_int8

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
):

    if input_dtype == np.uint8:
        neighbor_graph = nn_descent_uint8(
            data,
            n_neighbors,
            rng_state,
            effective_max_candidates,
            n_iters,
            delta,
            leaf_array=leaf_array,
            verbose=verbose,
        )
        neighbor_graph[1][:] = -np.log2(-neighbor_graph[1])
    elif input_dtype == np.int8:
        neighbor_graph = nn_descent_int8(
            data,
            n_neighbors,
            rng_state,
            effective_max_candidates,
            n_iters,
            delta,
            leaf_array=leaf_array,
            verbose=verbose,
        )
        neighbor_graph[1][:] = 1.0 / (-neighbor_graph[1])
    else:
        neighbor_graph = nn_descent_float(
            data,
            n_neighbors,
            rng_state,
            effective_max_candidates,
            n_iters,
            delta,
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
    n_jobs=None,
    verbose=False,
):
    if data.dtype == np.uint8:
        data = check_array(data, dtype=np.uint8, order="C")
        _input_dtype = np.uint8
        _bit_trees = True
    elif data.dtype == np.int8:
        data = check_array(data, dtype=np.int8, order="C")
        _input_dtype = np.int8
        _bit_trees = False
    else:
        data = check_array(data, dtype=np.float32, order="C", copy=True)
        norms = np.einsum("ij,ij->i", data, data)
        np.sqrt(norms, norms)
        norms[norms == 0.0] = 1.0
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
        effective_max_candidates = min(60, n_neighbors)
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
    )

    if np.any(neighbor_graph[0] < 0):
        warn(
            "Failed to correctly find n_neighbors for some samples."
            " Results may be less than ideal. Try re-running with"
            " different parameters."
        )

    numba.set_num_threads(_original_num_threads)
    return neighbor_graph
