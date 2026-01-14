Quick Start Guide
================

This guide provides a quick introduction to using EVōC for clustering high-dimensional embedding vectors.

Basic Usage
-----------

The simplest way to use EVōC:

.. code-block:: python

   from evoc import EVoC
   import numpy as np

   # Generate sample embedding data
   X = np.random.rand(1000, 512)  # 1000 samples, 512 dimensions

   # Create and fit the clusterer
   clusterer = EVoC()
   labels = clusterer.fit_predict(X)

   # Analyze results
   n_clusters = len(np.unique(labels[labels >= 0]))
   n_noise = np.sum(labels == -1)

   print(f"Found {n_clusters} clusters")
   print(f"Noise points: {n_noise}")

Understanding the Output
------------------------

EVōC produces:

* **labels_**: Cluster labels for each point (-1 for noise)
* **membership_strengths_**: Confidence scores for cluster membership
* **cluster_layers_**: Multiple clustering granularities 
* **cluster_tree_**: Hierarchical structure of clusters

.. code-block:: python

   # Access different clustering layers
   print(f"Available layers: {len(clusterer.cluster_layers_)}")

   # Get membership strengths
   strengths = clusterer.membership_strengths_
   print(f"Average membership strength: {np.mean(strengths):.3f}")

   # Access the cluster hierarchy
   tree = clusterer.cluster_tree_
   print(f"Hierarchical structure: {tree}")

Parameter Selection
-------------------

Key parameters to adjust:

**n_neighbors** (default=15)
   Number of neighbors for graph construction. Increase for more global connectivity.

**noise_level** (default=0.5)
   Expected noise level. Higher values = stricter clustering.

**base_min_cluster_size** (default=5)
   Minimum cluster size at the base layer.

**approx_n_clusters** (default=None)
   Target number of clusters (returns single layer if specified).

.. code-block:: python

   # Example with custom parameters
   clusterer = EVoC(
       n_neighbors=25,          # More neighbors for denser graphs
       noise_level=0.3,         # Lower noise threshold
       base_min_cluster_size=10, # Larger minimum clusters
       max_layers=5             # Limit hierarchy depth
   )

   labels = clusterer.fit_predict(X)

Working with Different Data Types
---------------------------------

EVoC automatically detects data types and uses appropriate distance metrics:

* **float32/float64**: Cosine distance (default for embeddings)
* **int8**: Quantized cosine distance  
* **uint8**: Bitwise Jaccard distance (for binary embeddings)

.. code-block:: python

   # For standard embeddings (float)
   X_float = embeddings.astype(np.float32)
   labels_cosine = EVoC().fit_predict(X_float)

   # For quantized embeddings (int8)
   X_quantized = (embeddings * 127).astype(np.int8)  
   labels_quantized = EVoC().fit_predict(X_quantized)

   # For binary embeddings (uint8)
   X_binary = (embeddings > 0.5).astype(np.uint8)
   labels_binary = EVoC().fit_predict(X_binary)

Next Steps
----------

* See the :doc:`user_guide` for detailed parameter explanations
* Check out :doc:`tutorials/index` for step-by-step examples
* Browse :doc:`notebooks/index` for interactive examples  
* Refer to :doc:`api/index` for complete API documentation
