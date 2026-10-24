import evoc
from sklearn.datasets import make_blobs

data, _ = make_blobs(n_samples=1000, n_features=64, centers=10)

clusterer = evoc.EVoC()
cluster_labels = clusterer.fit_predict(data)
