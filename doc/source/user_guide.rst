User Guide
==========

This end-user oriented guide covers EVoC's features, parameters, and best practices for different use cases. To better 
understand the parameters that are available, it help help to bgin with an overview of the algorithm and its key
components.

Algorithm Overview
------------------

EVoC (Embedding Vector Oriented Clustering) combines two key techniques:

1. **Graph Embedding**: Constructs a k-nearest neighbor graph and learns a lower-dimensional embedding (similar to UMAP)
2. **Density Clustering**: Applies hierarchical density-based clustering to the embedding (similar to HDBSCAN and PLSCAN)

The advantage of EVoC is that it can optimize every part of these tasks for the specific task of clustering high-dimensional 
embedding vectors, providing both improved **performance** and **quality** compared to general-purpose clustering algorithms.
That is to say, EVoC not only runs much faster than a combination of UMAP and HDBSCAN, but also produces better clusters as
a result.

The combination of dimension reduction/manifold learning and density clustering tailored to embedding vectors provides several 
advantages for clustering embedding vectors:

* Efficient processing of dense, high-dimensional data
* Multiple clustering granularities through hierarchical layers
* Robust handling of noise and outliers
* Optimized distance metrics for different embedding types

Parameter Reference
-------------------

With that core idea -- a two part algorithm -- in mind, let's explore the key parameters that control EVoC's behavior. 
The parameters can be broadly categorized into three groups:

Core Parameters
~~~~~~~~~~~~~~~

These are the main parameters that most users will want to adjust based on their specific dataset and clustering goals:

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

These parameters control the clustering behavior and granularity:

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

These parameters provide more fine-grained control over the algorithm and are typically only adjusted by advanced users:

**noise_level** : float, default=0.5
   Controls the noise threshold for cluster membership. Higher values produce more noise points 
   and fewer clusters, while lower values produce more clusters and fewer noise points. In practice
   this only provides fine-tuning over the amount of noise, and is not as important as 
   base_min_cluster_size and min_samples.

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

As a general rule EVoC is desgined to largely be as parameter-free as possible. The default parameters 
should work well for a wide range of datasets and use cases, and most users will not need to adjust them.
So the best place to start is just running with default parameters and then adjusting based on the results. 
However, here are some best practices for different scenarios:

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

The layer 0 is always the most fine-grained layer as determined by ``base_min_cluster_size`` or ``base_n_clusters``. 
Each subsequent layer provides a coarser clustering, with fewer clusters. In general the most fine-grained layers
will have the most noise points, and the coarser layers will have fewer noise points. The persistence score
provides a measure of how stable each layer is across different parameter settings, with higher scores indicating more robust clusters.

If you are interested in getting very fine-grained clusters it is worth setting ``base_min_cluster_size`` or ``base_n_clusters`` 
explicitly to ensure you get clustering at that granularity. You can then inspect the other layers to see if the other natural
granularities align with your use case. If you are only interested in a single clustering, you can set ``approx_n_clusters`` 
to get the layer that is closest to that number of clusters.

You can also make use of the tree structure to analyze how clusters evolve across layers, and to identify stable clusters 
that persist across multiple layers. Alternatively you can use the tree structure to create a "mixed" resolution layer by selecting
clusters at a given layer, and then also selecting any clusters in lower layers that are no children of any of your selected clusters. 
This allows you to get a more fine-grained clustering in some parts of the data, while keeping a coarser clustering i
n other parts of the data.

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

Depending on your needs you may be willing to trade off some accuracy for speed, or vice versa. 
The default EVoC parameters are designed primarily for exploratory clustering, and thus produce clusters very quickly.
If you are looking for a more robust higher quality clustering, it can be worth tweaking the parameters to spend
more time to produce a better clustering result. For example, for a medium sized dataset (e.g. 10k-100k points) 
you can increase the number of epochs and neighbors to get a better embedding, which will lead to better clusters.
In such cases you will also likely want to fix a random seed to ensure reproducibility, as the optimization process is stochastic.

.. code-block:: python

   clusterer = EVoC(
       n_epochs=150,           # More epochs for better embedding  
       random_state=42        # Enable optimizations
   )

For larger datasets, you may want to reduce the number of neighbors and epochs to get a faster result, at the cost of some cluster quality.
In that case not setting a random seed can actually improve performance, as it allows the algorithm to skip some of the overhead 
of ensuring reproducibility.

.. code-block:: python

   clusterer = EVoC(
       n_neighbors=10,         # Balance between quality and speed
       n_epochs=30,           # Fewer epochs for faster embedding  
       max_layers=3,          # Limit hierarchy depth
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

