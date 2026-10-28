"""Nearest-neighbour search over a fitted EVoC model's training data."""
import numba
import numpy as np
from sklearn.utils import check_random_state

from ..int8_nndescent import int8_inverse_norms
from ._pynnd_utils import FlatTree, make_heap, resort_tree_indices
from .cached_search import (
    deheap_sort_parallel,
    deheap_sort_serial,
    make_cached_search_function,
    make_search_struct,
)
from .diversify import diversify
from .hub_tree import make_hub_search_tree
from .kernels import FLOAT32_EPS, prepare_data
from .search_graph import build_search_graph, permute_graph, permute_rows

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1
PARALLEL_MIN_QUERIES = 256


def _input_kind(dtype):
    dtype = np.dtype(dtype)
    if dtype == np.uint8:
        return np.uint8
    if dtype == np.int8:
        return np.int8
    if np.issubdtype(dtype, np.floating):
        return np.float32
    raise ValueError(
        f"Unsupported data dtype {dtype}; expected a float dtype, int8 or uint8"
    )


class SearchIndex:
    """Nearest-neighbour search over a fitted EVoC model's training data.

    Builds a search graph (the diversified, symmetrised and degree-pruned k-NN
    graph) and a hub-based search tree from the training data and its k-NN
    graph, and answers k-nearest-neighbour queries for new samples. Distances
    are in EVoC's output units, the units of ``knn_graph`` (and of an ``EVoC``
    model's ``nn_dists_``): ``-log2(cosine similarity)`` for float and int8
    data, ``-log2(Jaccard similarity)`` for bit-packed uint8 data.

    The search functions are compiled once and cached on disk, so building an
    index compiles nothing after the first use.

    Parameters
    ----------
    data : array of shape (n_samples, n_features)
        The training data, as passed to ``knn_graph`` (float, int8 or
        bit-packed uint8). Float data is L2 normalised internally.

    nn_inds, nn_dists : arrays of shape (n_samples, n_neighbors)
        The k-NN graph of ``data`` from ``knn_graph``. If column 0 is each
        point itself (as ``knn_graph`` returns it) it is ignored.

    prune_degree_multiplier : float, default=1.5
        The search graph keeps at most ``prune_degree_multiplier *
        n_neighbors`` edges per vertex.

    diversify_prob : float, default=1.0
        The probability of pruning an occluded edge during diversification.

    leaf_size : int, default=30
        Leaf size of the search tree.

    random_state : int, RandomState instance or None, default=None
        Seeds the search tree and the search's random initialisation.

    parallel_batch_queries : bool or None, default=None
        Search batches of queries in parallel. None searches batches of at
        least ``PARALLEL_MIN_QUERIES`` queries in parallel (when numba has more
        than one thread) and smaller batches serially.

    Attributes
    ----------
    order : array of shape (n_samples,)
        The stored data are permuted into search-tree order; ``order[i]`` is
        the original training row of stored row ``i``.

    n_graph_edges : int
        Number of edges in the search graph.
    """

    def __init__(
        self,
        data,
        nn_inds,
        nn_dists,
        *,
        prune_degree_multiplier=1.5,
        diversify_prob=1.0,
        leaf_size=30,
        max_depth=200,
        random_state=None,
        parallel_batch_queries=None,
    ):
        self.input_dtype = _input_kind(data.dtype)
        nn_inds = np.asarray(nn_inds)
        nn_dists = np.asarray(nn_dists)
        if data.ndim != 2 or nn_inds.shape[0] != data.shape[0]:
            raise ValueError("nn_inds must have one row per row of data")
        self.n_features = data.shape[1]
        self.n_samples = data.shape[0]
        self.prune_degree_multiplier = prune_degree_multiplier
        self.diversify_prob = diversify_prob
        self.leaf_size = leaf_size
        self.max_depth = max_depth
        self.parallel_batch_queries = parallel_batch_queries

        random_state = check_random_state(random_state)
        self._rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(
            np.int64
        )
        tree_seed = int(random_state.randint(0, INT32_MAX))
        diversify_seed = np.int64(random_state.randint(0, INT32_MAX))

        data, inv_norms = prepare_data(data)

        if nn_inds.shape[1] > 1 and np.all(
            nn_inds[:, 0] == np.arange(self.n_samples)
        ):
            rows = nn_inds[:, 1:]
            dists = nn_dists[:, 1:]
        else:
            rows = nn_inds
            dists = nn_dists
        rows = np.ascontiguousarray(rows, dtype=np.int32)
        dists = np.ascontiguousarray(dists, dtype=np.float32)
        self.n_neighbors = rows.shape[1] + 1

        # Search tree (built from the undiversified graph, as pynndescent does)
        if self.input_dtype == np.int8:
            tree_data = data.astype(np.float32) * inv_norms[:, None]
        else:
            tree_data = data
        tree = make_hub_search_tree(
            tree_data,
            rows,
            leaf_size=leaf_size,
            max_depth=max_depth,
            angular=self.input_dtype != np.uint8,
            seed=tree_seed,
        )

        # Search graph
        rows, dists = diversify(
            rows.copy(), dists.copy(), data, inv_norms, diversify_seed,
            np.float64(diversify_prob),
        )
        # Preserve any distance 0 points
        dists[dists == 0.0] = FLOAT32_EPS
        max_degree = int(np.round(prune_degree_multiplier * self.n_neighbors))
        indptr, indices, min_distance = build_search_graph(rows, dists, max_degree)

        # Reorder everything according to the search tree leaf order
        order = np.ascontiguousarray(tree.indices, dtype=np.int32)
        indptr, indices = permute_graph(indptr, indices, order)
        data = permute_rows(data, order)
        if inv_norms.shape[0] > 0:
            inv_norms = np.ascontiguousarray(inv_norms[order])
        tree = resort_tree_indices(tree, np.argsort(order))

        self.order = order
        self._data = data
        self._inv_norms = inv_norms
        self._indptr = indptr
        self._indices = indices
        self._tree = FlatTree(
            np.ascontiguousarray(tree.hyperplanes),
            np.ascontiguousarray(tree.offsets, dtype=np.float32),
            np.ascontiguousarray(tree.children, dtype=np.int32),
            np.ascontiguousarray(tree.indices, dtype=np.int32),
            tree.leaf_size,
        )
        self.min_distance = np.float32(min_distance)
        self.n_graph_edges = int(indices.shape[0])
        self._init_search()

    def _init_search(self):
        self._visited = np.zeros(self.n_samples // 8 + 1, dtype=np.uint8)
        self._struct = make_search_struct(
            self._data,
            self._inv_norms,
            self._indptr,
            self._indices,
            self._tree,
            self.n_neighbors,
            self.min_distance,
        )
        self._searches = {}
        # Make sure everything is compiled (or loaded from cache) up front
        self.query(self._original_rows(1), 1, parallel=False)
        if self.parallel_batch_queries is not False:
            self.query(self._original_rows(1), 1, parallel=True)

    def _get_search(self, parallel):
        if parallel not in self._searches:
            query_dtype = self._data.dtype
            tree_query_dtype = (
                np.uint8 if self.input_dtype == np.uint8 else np.float32
            )
            self._searches[parallel] = (
                make_cached_search_function(
                    self._struct, query_dtype, tree_query_dtype, parallel
                ),
                deheap_sort_parallel if parallel else deheap_sort_serial,
            )
        return self._searches[parallel]

    def _original_rows(self, n):
        rows = self._data[:n]
        if self.input_dtype == np.float32:
            return rows.copy()
        return np.ascontiguousarray(rows)

    def _prepare_queries(self, X):
        """Validate queries; return (queries, tree_queries, inv_norms, zero_rows)."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(
                f"Expected a 2D array of queries, got an array of shape {X.shape}"
            )
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Queries have {X.shape[1]} features, but the index was built "
                f"with {self.n_features} features"
            )
        try:
            kind = _input_kind(X.dtype)
        except ValueError:
            kind = None
        if kind != self.input_dtype:
            raise ValueError(
                f"Queries have dtype {X.dtype}, but the index was built on "
                f"{np.dtype(self.input_dtype).name} data"
            )

        empty = np.zeros(0, dtype=np.float32)
        if self.input_dtype == np.uint8:
            X = np.ascontiguousarray(X)
            zero_rows = None
            return X, X, empty, zero_rows
        if self.input_dtype == np.int8:
            X = np.ascontiguousarray(X)
            inv_norms = int8_inverse_norms(X)
            zero_rows = inv_norms == 0
            tree_X = X.astype(np.float32) * inv_norms[:, None]
            return X, tree_X, inv_norms, zero_rows
        X = np.array(X, dtype=np.float32, order="C", copy=True)
        norms = np.sqrt(np.einsum("ij,ij->i", X, X))
        zero_rows = norms == 0
        norms[zero_rows] = 1.0
        X /= norms[:, None]
        return X, X, empty, zero_rows

    def query(self, X, k, epsilon=0.1, parallel=None):
        """Find the k nearest training samples to each query.

        Parameters
        ----------
        X : array of shape (n_queries, n_features)
            Queries, of the same kind (float, int8 or uint8) as the training
            data.

        k : int
            The number of neighbours to return.

        epsilon : float, default=0.1
            Controls the accuracy/speed trade-off of the graph search; larger
            values search more of the graph.

        parallel : bool or None, default=None
            Search the queries in parallel; None follows
            ``parallel_batch_queries``.

        Returns
        -------
        indices : array of shape (n_queries, k), int32
            Indices into the original training data, nearest first; -1 where
            fewer than ``k`` neighbours were found.

        distances : array of shape (n_queries, k), float32
            Distances in EVoC's output units (``inf`` where no neighbour).
        """
        k = int(k)
        if k < 1:
            raise ValueError("k must be at least 1")
        if k > self.n_samples:
            raise ValueError(
                f"k={k} is larger than the number of indexed samples "
                f"({self.n_samples})"
            )
        queries, tree_queries, inv_norms, zero_rows = self._prepare_queries(X)
        if parallel is None:
            parallel = self.parallel_batch_queries
            if parallel is None:
                parallel = (
                    queries.shape[0] >= PARALLEL_MIN_QUERIES
                    and numba.get_num_threads() > 1
                )
        search, deheap = self._get_search(bool(parallel))
        result = make_heap(queries.shape[0], k)
        search(
            queries,
            tree_queries,
            inv_norms,
            result,
            np.int64(k),
            np.float32(epsilon),
            self._visited,
            self._rng_state,
        )
        indices, distances = deheap(result[0], result[1])
        found = indices >= 0
        indices[found] = self.order[indices[found]]
        if zero_rows is not None and np.any(zero_rows):
            indices[zero_rows] = -1
            distances[zero_rows] = np.inf
        return indices, distances

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ("_struct", "_searches", "_visited"):
            state.pop(key, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._init_search()


def prepare_search(data, nn_inds, nn_dists, **kwargs):
    """Build a :class:`SearchIndex` over ``data`` and its k-NN graph."""
    return SearchIndex(data, nn_inds, nn_dists, **kwargs)
