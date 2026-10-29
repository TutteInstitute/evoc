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

**prediction_data** : bool, default=False
   Whether to keep the extra information needed to predict clusters for new data and to update the model 
   (see `Predicting and Updating`_). This uses roughly as much extra memory as the training data.

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


Predicting and Updating
-----------------------

Once you have a clustering you are happy with, new data will often keep arriving. Rather than refitting from scratch 
every time, EVoC can assign new samples to the existing clusters, or fold new samples into the model (and drop old 
ones) while keeping cluster labels stable. To make this possible EVoC needs to keep some extra information from 
fitting, so you need to ask for it with ``prediction_data=True``:

.. code-block:: python

   clusterer = EVoC(prediction_data=True, random_state=42)
   clusterer.fit(X)

This keeps a copy of the training data, the node embedding, and the cluster trees, so expect it to use roughly as much 
extra memory as the training data itself. The clustering results are exactly the same whether or not you set it.

Predicting Clusters for New Samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``predict`` method assigns new samples to the clusters of a fitted model, just as you would expect from any 
sklearn style clusterer. By default you get labels for the same layer as ``labels_``, but you can ask for any layer, 
or for all of them at once with ``predict_layers``:

.. code-block:: python

   labels = clusterer.predict(X_new)

   # Labels and membership strengths for a particular layer
   labels, strengths = clusterer.predict(X_new, layer=0, return_strengths=True)

   # Labels and membership strengths for every layer
   layer_labels, layer_strengths = clusterer.predict_layers(X_new)

The new data needs to be the same kind of data as the training data (float, int8 or binary uint8) with the same number 
of features. Both methods start by finding the nearest training samples to each new sample, and then there are two 
ways to decide on a cluster:

* **method="vote"**: The default. Each new sample goes to the cluster that has the most support among its nearest 
  training samples, weighting each neighbor by how strongly it is connected to the new sample and how strongly it 
  belongs to its own cluster. Samples without enough support are labeled as noise, with the bar set so that roughly 
  the same fraction of samples are noise as in the training data (you can change it with ``noise_threshold``).
* **method="embedding"**: Places each new sample in the node embedding, near its neighbors, and then finds where it 
  would fall in the cluster tree. This follows the clustering itself more closely, which can matter for deciding 
  what is noise.

In practice the two methods agree on the vast majority of samples, and both are fast enough to use one sample at a 
time. If you want the positions of new samples in the node embedding you can get them with ``embed``.

Adding and Expiring Data
~~~~~~~~~~~~~~~~~~~~~~~~

If you want the new data to become part of the model, rather than just being labeled, use ``update``. This adds the 
new samples, updates the nearest neighbor graph and the node embedding, and re-clusters. EVoC then matches each new 
cluster layer to the corresponding old layer and relabels the clusters so that, as far as possible, every cluster keeps 
its label:

.. code-block:: python

   new_labels = clusterer.update(X_new)

   # labels_ now covers the original samples followed by the new ones
   clusterer.labels_

You can also remove data with the ``X_expired`` keyword argument. This makes it easy to keep a clustering of a moving 
window of data up to date as time goes by: each time new data arrives, expire the oldest data and add the new.

.. code-block:: python

   clusterer = EVoC(prediction_data=True).fit(first_window)

   for batch in new_batches:
       # Expire the oldest len(batch) samples and add the new batch
       clusterer.update(batch, X_expired=np.arange(len(batch)))

Expired samples are removed first, and new samples are added at the end, so the remaining samples keep their order. 
You can say which samples to expire in a few ways:

* **Row positions**: An array of integer positions, such as ``np.arange(1000)`` for the oldest thousand samples.
* **A boolean mask**: ``True`` for each sample to expire.
* **The data itself**: A 2D array of samples; each one removes the oldest matching sample in the model.

Since positions change as data comes and goes, every sample also gets an id in ``sample_ids_`` that never changes. 
Fitting numbers the samples from zero, and each update gives new samples the next ids. To expire samples by id you can 
use a mask such as ``np.isin(clusterer.sample_ids_, ids_to_expire)``. You can also call ``update`` with only 
``X_expired`` to drop data without adding any.

Clusters that have no counterpart in the old clustering get brand new labels, and the labels of clusters that 
disappear are never reused, so after an update the labels need not run from zero without gaps. A few options control 
how the update is done:

* **recluster**: Set to ``False`` to keep all existing labels and just assign the new samples to existing clusters (this 
  can't be combined with ``X_expired``).
* **relabel**: Set to ``False`` to label the clusters as a fresh fit would, rather than matching the old labels.
* **optimize**: By default only new samples are placed in the node embedding and existing samples stay put, which 
  keeps the clustering of existing data as stable as possible. Use ``optimize="all"`` to let the whole embedding adjust 
  to the new data instead.

Nearest Neighbor Search
~~~~~~~~~~~~~~~~~~~~~~~

The nearest neighbor search that powers ``predict`` and ``update`` can also be used on its own. Build a 
``SearchIndex`` from your data and its nearest neighbor graph, and then query it with new data:

.. code-block:: python

   from evoc.knn_graph import knn_graph
   from evoc.search import SearchIndex

   nn_inds, nn_dists = knn_graph(X, n_neighbors=15)
   index = SearchIndex(X, nn_inds, nn_dists)
   indices, distances = index.query(X_new, k=10)

Distances are reported the same way EVoC reports them for ``nn_dists_``, which is based on cosine similarity for float 
and int8 data, and Jaccard similarity for binary data.

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

