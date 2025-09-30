User Guide
==========

This comprehensive guide covers EVoC's features, parameters, and best practices for different use cases.

Algorithm Overview
------------------

EVoC (Embedding Vector Oriented Clustering) combines two key techniques:

1. **Graph Embedding**: Constructs a k-nearest neighbor graph and learns a lower-dimensional embedding (similar to UMAP)
2. **Density Clustering**: Applies hierarchical density-based clustering to the embedding (similar to HDBSCAN)

This combination provides several advantages for high-dimensional embedding vectors:

* Efficient processing of dense, high-dimensional data
* Multiple clustering granularities through hierarchical layers
* Robust handling of noise and outliers
* Optimized distance metrics for different embedding types

Parameter Reference
-------------------

Core Parameters
~~~~~~~~~~~~~~~

**noise_level** : float, default=0.5
   Controls the trade-off between cluster purity and data coverage. Lower values (0.0-0.3) include more 
   points in clusters but may reduce cluster quality. Higher values (0.7-1.0) produce purer clusters 
   but classify more points as noise.

**base_min_cluster_size** : int, default=5  
   Minimum number of points required to form a cluster at the base (finest) granularity level. 
   Larger values produce fewer, more stable clusters.

**n_neighbors** : int, default=15
   Number of neighbors used in k-NN graph construction. More neighbors capture more global structure 
   but increase computational cost.

**min_samples** : int, default=5
   Minimum samples for density estimation in the final clustering step. Should typically match 
   or be smaller than base_min_cluster_size.

Clustering Control
~~~~~~~~~~~~~~~~~~

**base_n_clusters** : int, optional
   Target number of clusters for the base layer. When specified, EVoC will search for the clustering 
   granularity that produces approximately this many clusters, then build additional layers on top.

**approx_n_clusters** : int, optional  
   Target number of clusters for the final output. When specified, EVoC returns only a single 
   clustering layer (no hierarchy) with approximately this many clusters.

**max_layers** : int, default=10
   Maximum number of hierarchical clustering layers to generate. More layers provide finer control 
   over clustering granularity but increase computation time.

**min_similarity_threshold** : float, default=0.2
   Minimum Jaccard similarity threshold for layer selection. Prevents nearly identical clustering 
   layers in the hierarchy.

Advanced Parameters  
~~~~~~~~~~~~~~~~~~~

**node_embedding_dim** : int, optional
   Dimensionality of the intermediate node embedding. If None, defaults to min(max(n_neighbors // 4, 4), 15).
   Higher dimensions can capture more complex structure but increase computation.

**neighbor_scale** : float, default=1.0
   Scales the effective number of neighbors (neighbor_scale × n_neighbors). Values > 1.0 create 
   denser graphs, values < 1.0 create sparser graphs focused on local structure.

**n_epochs** : int, default=50
   Number of optimization epochs for the node embedding. More epochs improve embedding quality 
   but increase computation time.

**node_embedding_init** : {'label_prop', None}, default='label_prop'
   Initialization method for the node embedding. 'label_prop' uses label propagation for initialization, 
   None uses random initialization.

**n_label_prop_iter** : int, default=20
   Number of label propagation iterations when using 'label_prop' initialization.

**symmetrize_graph** : bool, default=True
   Whether to make the k-NN graph symmetric. Recommended for most use cases.

**random_state** : int, optional
   Random seed for reproducible results. When specified, enables deterministic mode.

Best Practices
--------------

Choosing Parameters for Different Use Cases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Small Datasets (< 1,000 samples)**:

.. code-block:: python

   clusterer = EVoC(
       n_neighbors=10,
       base_min_cluster_size=3,
       min_samples=3,
       noise_level=0.3
   )

**Medium Datasets (1,000 - 100,000 samples)**:

.. code-block:: python

   clusterer = EVoC(
       n_neighbors=15,
       base_min_cluster_size=5,
       min_samples=5,
       noise_level=0.5
   )

**Large Datasets (> 100,000 samples)**:

.. code-block:: python

   clusterer = EVoC(
       n_neighbors=20,
       base_min_cluster_size=10,
       min_samples=10,
       noise_level=0.7,
       max_layers=5
   )

**High-Dimensional Embeddings (> 1,000 dimensions)**:

.. code-block:: python

   clusterer = EVoC(
       n_neighbors=25,
       node_embedding_dim=20,
       neighbor_scale=0.8,
       n_epochs=75
   )

Working with Hierarchical Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

EVoC provides multiple clustering layers with different granularities:

.. code-block:: python

   clusterer = EVoC(max_layers=5)
   clusterer.fit(X)

   # Explore different granularities
   for i, layer in enumerate(clusterer.cluster_layers_):
       n_clusters = len(np.unique(layer[layer >= 0]))
       n_noise = np.sum(layer == -1)
       persistence = clusterer.persistence_scores_[i]

       print(f"Layer {i}: {n_clusters} clusters, {n_noise} noise points, "
             f"persistence: {persistence:.3f}")

   # Use cluster tree for hierarchical analysis
   tree = clusterer.cluster_tree_
   # ... analyze hierarchical structure ...

Handling Different Embedding Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Standard Dense Embeddings (CLIP, sentence-transformers)**:

.. code-block:: python

   # Ensure float32 for optimal performance
   X = embeddings.astype(np.float32)
   clusterer = EVoC(neighbor_scale=1.2)  # Slightly denser graph
   labels = clusterer.fit_predict(X)

**Quantized Embeddings (int8)**:

.. code-block:: python

   # Quantize to int8 range
   X_quantized = (embeddings * 127).clip(-127, 127).astype(np.int8)
   clusterer = EVoC(n_neighbors=20)  # More neighbors for quantized data
   labels = clusterer.fit_predict(X_quantized)

**Binary Embeddings (uint8)**:

.. code-block:: python

   # Binarize embeddings  
   X_binary = (embeddings > threshold).astype(np.uint8)
   clusterer = EVoC(
       n_neighbors=30,         # More neighbors for binary data
       neighbor_scale=1.5,     # Denser graph
       noise_level=0.4         # Lower noise threshold
   )
   labels = clusterer.fit_predict(X_binary)

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

For large datasets or when performance is critical:

.. code-block:: python

   clusterer = EVoC(
       n_neighbors=15,         # Balance between quality and speed
       n_epochs=30,           # Fewer epochs for faster embedding  
       max_layers=3,          # Limit hierarchy depth
       node_embedding_dim=10,  # Lower embedding dimension
       random_state=42        # Enable optimizations
   )

Memory Management
~~~~~~~~~~~~~~~~~

For memory-constrained environments:

.. code-block:: python

   # Process in smaller batches or reduce parameters
   clusterer = EVoC(
       n_neighbors=10,         # Smaller graphs
       node_embedding_dim=8,   # Lower embedding dimension
       max_layers=2           # Fewer layers to store
   )

Troubleshooting
---------------

**Problem**: Too many small clusters
   **Solution**: Increase base_min_cluster_size or noise_level

**Problem**: Most points classified as noise  
   **Solution**: Decrease noise_level or reduce min_samples

**Problem**: Clustering too slow
   **Solution**: Reduce n_neighbors, n_epochs, or max_layers

**Problem**: Poor cluster quality
   **Solution**: Increase n_neighbors, n_epochs, or try different node_embedding_init

**Problem**: Inconsistent results
   **Solution**: Set random_state for reproducible results

Evaluation and Validation
-------------------------

EVoC provides several ways to evaluate clustering quality:

.. code-block:: python

   # Basic cluster statistics
   labels = clusterer.labels_
   n_clusters = len(np.unique(labels[labels >= 0]))
   n_noise = np.sum(labels == -1)

   # Membership strength analysis
   strengths = clusterer.membership_strengths_
   avg_strength = np.mean(strengths[labels >= 0])  # Exclude noise points

   # Layer-wise analysis
   for i, (layer, persistence) in enumerate(zip(
       clusterer.cluster_layers_, clusterer.persistence_scores_)):

       layer_n_clusters = len(np.unique(layer[layer >= 0]))  
       print(f"Layer {i}: {layer_n_clusters} clusters, persistence: {persistence:.3f}")

   # External validation (if ground truth available)
   from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

   ari = adjusted_rand_score(true_labels, labels)
   nmi = normalized_mutual_info_score(true_labels, labels)
   print(f"ARI: {ari:.3f}, NMI: {nmi:.3f}")
