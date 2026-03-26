Quick Start Guide
================

This guide provides a quick introduction to using EVōC for clustering high-dimensional embedding vectors. EVōC 
specifically targets modern embedding vectors such as those produced by CLIP, sentence-transformers, and other 
dense vector representations. It seeks to provide fast and effective results with as little parameter tuning as possible.

Basic Usage
-----------

The simplest way to use EVōC is to import the EVoC class, create an instance with default parameters, and call fit_predict on your data:

.. code-block:: python

   from evoc import EVoC
   from sklearn.datasets import make_blobs
   import numpy as np

   # Generate sample embedding data
   blob_data, blob_labels = make_blobs(n_samples=10_000, n_features=512, centers=256)

   # Create and fit the clusterer
   clusterer = EVoC()
   labels = clusterer.fit_predict(blob_data)

   # Analyze results
   n_clusters = len(np.unique(labels[labels >= 0]))
   n_noise = np.sum(labels == -1)

   print(f"Found {n_clusters} clusters")
   print(f"Noise points: {n_noise}")

EVōC uses the sklearn API, so you can drop it in to any existing clustering workflow that expects a fit_predict method. 
The default parameters are designed to work well for typical embedding data, but you can adjust them as needed
(see the Parameter Selection section below).

Understanding the Output
------------------------

EVōC uses standard sklearn conventions for its output. After fitting, the clusterer will have the following attributes:

* **labels_**: Cluster labels for each point (-1 for noise)
* **membership_strengths_**: Confidence scores for cluster membership
* **cluster_layers_**: Multiple clustering granularities 
* **cluster_tree_**: Hierarchical structure of clusters

The ``labels_`` attribute is the expected vector of cluster assignments you would get from any sklearn clustering algorithm. 
The ``membership_strengths_`` attribute provides additional information about how strongly each point belongs to its assigned 
cluster, which can be useful for filtering or analyzing borderline cases; the is equivalent to the ``probabilities_`` attribute 
in HDBSCAN.

The ``cluster_layers_`` and ``cluster_tree_`` attributes are more novel. EVōC is not a hierarchical clustering algorithm in the 
traditional sense,  instead it produces multiple layers of clustering resolution, that can be results that can be cast into a 
hierarchy.

.. code-block:: python

   # Access different clustering layers
   print(f"Available layers: {len(clusterer.cluster_layers_)}")

   # Get membership strengths
   strengths = clusterer.membership_strengths_
   print(f"Average membership strength: {np.mean(strengths):.3f}")

   # Access the cluster hierarchy
   tree = clusterer.cluster_tree_
   print(f"Hierarchical structure: {tree}")

Layers are sorted from most fine-grained (many small clusters) at index 0 to most coarse-grained (fewer large clusters).
Each layer is a label vector, just like ``labels_``, but with a different clustering resolution. The ``labels_`` attribute 
corresponds to the layer that has clusters persisting across the widest range of cluster resolution scales, and is usually 
the most stable and meaningful clustering result. However, depending on your needs, other cluster layers may be more appropriate.

The ``cluster_tree_`` attribute provides a hierarchical structure of the clusters across layers. 
It shows how clusters in finer layers relate to clusters in coarser layers, effectively creating a tree of cluster relationships. 
This can be useful for understanding the multi-scale structure of your data and for selecting clusters at 
different levels of granularity.

The tree is structured as a dictionary. Each cluster is identified as a tuple of (layer_index, cluster_id), 
and the value is a list of child clusters in the more fine-grained layers.

Parameter Selection
-------------------

Key parameters to adjust:

**n_neighbors** (default=15)
   Number of neighbors for graph construction. Increase for more global connectivity.

**base_min_cluster_size** (default=5)
   Minimum cluster size at the base layer.

**approx_n_clusters** (default=None)
   Target number of clusters (returns single layer if specified).

.. code-block:: python

   # Example with custom parameters
   clusterer = EVoC(
       n_neighbors=25,          # More neighbors for denser graphs
       base_min_cluster_size=10, # Larger minimum clusters
       max_layers=5             # Limit hierarchy depth
   )

   labels = clusterer.fit_predict(blob_data)

Working with Different Data Types
---------------------------------

EVoC automatically detects data types and uses appropriate distance metrics:

* **float32/float64**: Cosine distance (default for embeddings)
* **int8**: Quantized cosine distance  
* **uint8**: Bitwise Jaccard distance (for binary embeddings)

We can take out blob data and convert it to different formats to see how EVoC handles them.
In practice, you would typically be working with actual embedding data that comes
pre-quantized or binarized depending on the model and/or storage format you are using.

   embeddings = normalize(blob_data)  # Example embedding data

   # For standard embeddings (float)
   X_float = embeddings.astype(np.float32)
   labels_cosine = EVoC().fit_predict(X_float)

   # For quantized embeddings (int8)
   X_quantized = (StandardScaler().fit_transform(embeddings) * 127).astype(np.int8)  
   labels_quantized = EVoC().fit_predict(X_quantized)

   # For binary embeddings (packed uint8)
   X_binary = np.packbits(embeddings > 0.0, axis=1)
   labels_binary = EVoC().fit_predict(X_binary)

Next Steps
----------

* See the :doc:`user_guide` for detailed parameter explanations
* Refer to :doc:`api/index` for complete API documentation
