import numpy as np
from evoc.cluster_trees import (
    create_linkage_merge_data,
    eliminate_branch,
    linkage_merge_find,
    linkage_merge_join,
    mst_to_linkage_tree,
)

from sklearn.datasets import make_blobs
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

import scipy.sparse


n_clusters = 3
# X = generate_clustered_data(n_clusters=n_clusters, n_samples_per_cluster=50)
X, y = make_blobs(n_samples=200, random_state=10)
X, y = shuffle(X, y, random_state=7)
X = StandardScaler().fit_transform(X)

# Remove this when there are actual tests
def test_noop():
    pass

