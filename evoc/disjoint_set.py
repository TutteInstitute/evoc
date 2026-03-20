import numba
import numpy as np

from collections import namedtuple

RankDisjointSet = namedtuple("RankDisjointSet", ["parent", "rank"])
SizeDisjointSet = namedtuple("SizeDisjointSet", ["parent", "size"])

_sentinel_rank_ds = RankDisjointSet(
    parent=np.empty(1, dtype=np.int32),
    rank=np.empty(1, dtype=np.int32),
)
_sentinel_size_ds = SizeDisjointSet(
    parent=np.empty(1, dtype=np.int32),
    size=np.empty(1, dtype=np.int32),
)
RankDisjointSetType = numba.typeof(_sentinel_rank_ds)
SizeDisjointSetType = numba.typeof(_sentinel_size_ds)


@numba.njit(cache=True)
def ds_rank_create(n_elements):
    return RankDisjointSet(
        np.arange(n_elements, dtype=np.int32), np.zeros(n_elements, dtype=np.int32)
    )


@numba.njit(cache=True)
def ds_size_create(n_elements):
    return SizeDisjointSet(
        np.arange(n_elements, dtype=np.int32), np.ones(n_elements, dtype=np.int32)
    )


@numba.njit(cache=True)
def ds_find(disjoint_set, x):
    while disjoint_set.parent[x] != x:
        x, disjoint_set.parent[x] = (
            disjoint_set.parent[x],
            disjoint_set.parent[disjoint_set.parent[x]],
        )

    return x


@numba.njit(
    numba.void(
        RankDisjointSetType,
        numba.int32,
        numba.int32,
    ),
    cache=True,
)
def ds_union_by_rank(disjoint_set, x, y):
    x = ds_find(disjoint_set, x)
    y = ds_find(disjoint_set, y)

    if x == y:
        return

    if disjoint_set.rank[x] < disjoint_set.rank[y]:
        x, y = y, x

    disjoint_set.parent[y] = x
    if disjoint_set.rank[x] == disjoint_set.rank[y]:
        disjoint_set.rank[x] += 1


@numba.njit(
    numba.void(
        SizeDisjointSetType,
        numba.int32,
        numba.int32,
    ),
    cache=True,
)
def ds_union_by_size(disjoint_set, x, y):
    x = ds_find(disjoint_set, x)
    y = ds_find(disjoint_set, y)

    if x == y:
        return

    if disjoint_set.size[x] < disjoint_set.size[y]:
        x, y = y, x

    disjoint_set.parent[y] = x
    disjoint_set.size[x] += disjoint_set.size[y]
