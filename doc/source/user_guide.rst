User Guide
==========

This comprehensive guide covers EVoC's features, parameters, and best practices. For a high-level overview of how EVōC works, start with :doc:`how_evoc_works`.

Why Parameters Matter
---------------------

EVoC is designed to work well out-of-the-box with default parameters. However, understanding the parameters helps you tune the algorithm for your specific data and use case.

Think of parameters in three groups:

1. **Resolution Control** — How many clusters do you get?
2. **Data Understanding** — How does the algorithm perceive your embeddings?
3. **Noise & Outliers** — What gets included vs. excluded?

We'll go through each group, then discuss how to combine them for different scenarios.

Understanding Parameters by Group
----------------------------------

**Group 1: Resolution Control** — How many clusters do you get?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most important parameters for controlling clustering granularity:

**noise_level** (default=0.5)
   This is the main "dial" for cluster count. It controls the trade-off between including points in clusters vs. marking them as noise.
   
   - **Low values (0.1-0.3)**: More clusters, more points included, looser clustering
   - **Default (0.5)**: Balanced — usually gives natural cluster structure
   - **High values (0.7-1.0)**: Fewer clusters, stricter quality requirement, more noise points
   
   **When to adjust:** If EVoC finds too many clusters, increase this. If it finds too few, decrease it.

**base_min_cluster_size** (default=5)
   Minimum number of points needed to form a cluster. Larger values naturally produce fewer, more stable clusters.
   
   - **Small values (2-5)**: Allows small, tight clusters
   - **Default (5)**: Good for most datasets
   - **Large values (20-50)**: Forces larger, more robust clusters
   
   **When to adjust:** If you care about small structural details, use smaller values. For coarser groupings, increase it.

**approx_n_clusters** (optional, default=None)
   If you know roughly how many clusters you want, specify it here. EVoC will find clustering at that granularity.
   
   .. code-block:: python
   
      # Force approximately 5 clusters
      clusterer = EVoC(approx_n_clusters=5)
      labels = clusterer.fit_predict(X)
   
   **When to use:** When you have domain knowledge about expected cluster count. Overrides automatic selection.

**max_layers** (default=10)
   How many different cluster granularities to compute. More layers = more exploration, slightly slower.
   
   - **Small values (2-3)**: Fast, coarse granularity exploration
   - **Default (10)**: Full exploration
   - **Large values (15+)**: Very detailed hierarchy
   
   **When to adjust:** For quick clustering, reduce this. For detailed analysis, keep it high.

**Group 2: Data Understanding** — How does the algorithm perceive your embeddings?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These parameters control how the algorithm understands the geometry of your data:

**n_neighbors** (default=15)
   Number of nearest neighbors considered when building the graph. This controls what "neighborhood" means.
   
   - **Small values (5-10)**: Focus on very local structure
   - **Default (15)**: Balanced local-to-global trade-off
   - **Large values (30-50)**: More global structure understanding
   
   **Intuition:** Think of this as "how wide is my search radius?" More neighbors = broader awareness.
   
   **When to adjust:**
   - Increase for sparse datasets or when you want to capture global structure
   - Decrease for datasets where you care most about local neighborhoods
   - For very high-dimensional data (> 1000 dims), you might increase this slightly

**neighbor_scale** (default=1.0)
   Multiplies the effective number of neighbors. A quick way to make the graph denser or sparser without changing n_neighbors.
   
   - Values < 1.0: Sparser graph, more local focus
   - 1.0: Use as-is
   - Values > 1.0: Denser graph, more global awareness
   
   **Example:**
   
   .. code-block:: python
   
      # n_neighbors=15, but effectively use 22 neighbors
      clusterer = EVoC(n_neighbors=15, neighbor_scale=1.5)

**Group 3: Noise & Outliers** — What gets included vs. excluded?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

How the algorithm handles points that don't fit cleanly into clusters:

**min_samples** (default=5)
   Used in density estimation. Points in regions with fewer than min_samples neighbors may be marked as noise.
   
   - **Small values (2-3)**: Include more points, fewer marked as noise
   - **Default (5)**: Balanced
   - **Large values (10-20)**: Strict — more noise points
   
   **Typically:** Set this to match or be slightly smaller than base_min_cluster_size.

**Advanced: Fine-Tuning the Graph Embedding**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to control the intermediate representation (rarely needed):

**node_embedding_dim** (optional, default=auto-selected)
   Dimensionality of the compressed representation learned in Stage 1. Defaults to a value between 4-15 based on n_neighbors.
   
   - Usually you don't need to change this
   - Larger values can capture more complex structure but slow things down
   - Smaller values are faster but might lose detail

**n_epochs** (default=50)
   How many optimization iterations for learning the embedding. More iterations = better quality, slower.
   
   - **Low values (20-30)**: Fast, decent quality
   - **Default (50)**: Good balance
   - **High values (100+)**: Best quality, slower
   
   **When to adjust:** For large datasets, reduce this for speed. For maximum quality, increase it.

**node_embedding_init** (default='label_prop')
   How to initialize the embedding. 'label_prop' is usually better. Leave as-is unless troubleshooting.

**Other Useful Parameters**

**random_state** (optional)
   Set this to any integer for reproducible results. Without it, results vary slightly between runs due to random initialization.
   
   .. code-block:: python
   
      clusterer = EVoC(random_state=42)  # Always get same results

**symmetrize_graph** (default=True)
   Whether to make the k-NN graph symmetric (A→B and B→A both exist). Keep True for embeddings.

**base_n_clusters** (optional)
   Like approx_n_clusters but for the base (finest) layer specifically. Advanced, rarely needed.

Best Practices & Recipes
------------------------

Choosing the Right Parameters for Your Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**For most use cases, defaults work fine.** Only adjust if you see unexpected behavior.

**Dataset Size Strategy**

Small Datasets (< 1,000 samples):

.. code-block:: python

   clusterer = EVoC(
       n_neighbors=10,              # Smaller graph for small data
       base_min_cluster_size=3,     # Allow small clusters
       noise_level=0.4              # Slightly more inclusive
   )

These settings let you discover small structural details. Noise level is slightly lower because small datasets can support more granular clustering.

Medium Datasets (1,000 - 100,000 samples):

.. code-block:: python

   clusterer = EVoC()  # Defaults usually work perfectly

Defaults are calibrated for this range. Try them first.

Large Datasets (> 100,000 samples):

.. code-block:: python

   clusterer = EVoC(
       n_neighbors=20,              # Broader context for large data
       base_min_cluster_size=10,    # Larger clusters
       noise_level=0.6,             # More selective clustering
       n_epochs=30,                 # Reduce for speed
       max_layers=5                 # Limit hierarchy for speed
   )

Focus on speed and stable large clusters. Reduce n_epochs for faster computation.

**Structure Types**

If your data has **very clear, obvious clusters**:

.. code-block:: python

   clusterer = EVoC(noise_level=0.7)  # Stricter clustering

If your data has **subtle, overlapping structure**:

.. code-block:: python

   clusterer = EVoC(noise_level=0.3)  # More inclusive, find nuance

If you're **exploring without prior knowledge**:

.. code-block:: python

   clusterer = EVoC()  # Use defaults, then examine cluster_layers_

**Dimensionality**

Very High-Dimensional Data (> 1000 dimensions):

.. code-block:: python

   clusterer = EVoC(
       n_neighbors=25,          # More neighbors in high dims
       node_embedding_dim=20,   # Larger intermediate representation
       neighbor_scale=0.8       # Slightly sparser effective graph
   )

The curse of dimensionality is real. More neighbors help maintain meaningful distance relationships.

Exploring Your Data with Cluster Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One of EVoC's strengths is providing multiple clustering granularities. Here's how to use them:

.. code-block:: python

   clusterer = EVoC()
   clusterer.fit(X)
   
   # See all available granularities
   print(f"Found {len(clusterer.cluster_layers_)} cluster layers:")
   
   for i, layer in enumerate(clusterer.cluster_layers_):
       n_clusters = len(np.unique(layer[layer >= 0]))
       n_noise = np.sum(layer == -1)
       persistence = clusterer.persistence_scores_[i]
       
       print(f"  Layer {i}: {n_clusters} clusters, "
             f"{n_noise} noise, persistence={persistence:.3f}")
   
   # Use the automatically selected layer (usually best)
   default_labels = clusterer.labels_
   
   # Or pick a different layer for your analysis
   coarse_labels = clusterer.cluster_layers_[0]  # Fewest clusters
   fine_labels = clusterer.cluster_layers_[-1]   # Most clusters

The default (clusterer.labels_) is selected to be persistent and natural. But explore others to see different views of your data.

Using Membership Strengths
~~~~~~~~~~~~~~~~~~~~~~~~~~

Membership strengths tell you how confidently a point is assigned to its cluster:

.. code-block:: python

   clusterer = EVoC()
   clusterer.fit(X)
   
   labels = clusterer.labels_
   strengths = clusterer.membership_strengths_
   
   # Points at cluster cores (high confidence)
   core_mask = strengths > 0.8
   core_labels = labels[core_mask]
   
   # Points at boundaries (lower confidence)
   boundary_mask = (strengths > 0.5) & (strengths <= 0.8)
   boundary_labels = labels[boundary_mask]
   
   # Noise points (not in any cluster)
   noise_mask = labels == -1

This is useful for:

- **Identifying cluster cores vs. boundaries** — for hierarchical analysis
- **Weighting downstream analysis** — trust core assignments more
- **Filtering uncertainty** — keep only high-confidence assignments
- **Finding transition zones** — study points with moderate membership

Working with Different Embedding Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

EVoC automatically chooses the right distance metric based on dtype:

**Float Embeddings (CLIP, sentence-transformers, etc.)**

.. code-block:: python

   X = embeddings.astype(np.float32)
   clusterer = EVoC()
   labels = clusterer.fit_predict(X)

Use as-is. No special handling needed. Cosine distance is used automatically.

**Quantized Embeddings (int8 for memory efficiency)**

.. code-block:: python

   # Quantize from float [-1, 1] to int8 [-127, 127]
   X_quantized = (embeddings * 127).clip(-127, 127).astype(np.int8)
   
   clusterer = EVoC(n_neighbors=20)  # Slightly more neighbors for quantized
   labels = clusterer.fit_predict(X_quantized)

Quantization maintains most clustering quality while using 4x less memory. Great for large-scale clustering.

**Binary Embeddings (uint8 for very compact representations)**

.. code-block:: python

   # Binarize: each dimension becomes 0 or 1
   X_binary = (embeddings > threshold).astype(np.uint8)
   
   clusterer = EVoC(
       n_neighbors=30,      # More neighbors needed for binary
       neighbor_scale=1.5,  # Denser graph
       noise_level=0.4      # May need adjustment
   )
   labels = clusterer.fit_predict(X_binary)

Binary embeddings are extremely compact but lose some precision. Adjust parameters to compensate.

Getting Reproducible Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For debugging or publication, you need consistent results:

.. code-block:: python

   clusterer = EVoC(random_state=42)
   labels = clusterer.fit_predict(X)
   
   # Run again, get identical results
   clusterer2 = EVoC(random_state=42)
   labels2 = clusterer2.fit_predict(X)
   
   assert np.array_equal(labels, labels2)  # True!

Always set random_state when:

- Comparing algorithm variants
- Publishing results
- Debugging issues
- Creating examples

Performance Tips
~~~~~~~~~~~~~~~~

**For speed:**

.. code-block:: python

   clusterer = EVoC(
       n_neighbors=15,      # Default is fine
       n_epochs=20,         # Reduce from default 50
       max_layers=3,        # Reduce from default 10
       node_embedding_dim=8 # Reduce from default
   )

**For memory efficiency:**

.. code-block:: python

   clusterer = EVoC(
       node_embedding_dim=8,  # Smaller intermediate representation
       max_layers=2,          # Fewer layers to store
       n_neighbors=15         # Reasonable graph size
   )

**For quality (at cost of speed/memory):**

.. code-block:: python

   clusterer = EVoC(
       n_neighbors=25,        # Broader context
       n_epochs=100,          # More optimization
       node_embedding_dim=25  # Richer representation
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
