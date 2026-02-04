Quick Start Guide
================

Let's get started with EVōC! This guide walks you through the basics with concrete examples.

Your First Clustering
---------------------

The simplest way to use EVōC:

.. code-block:: python

   from evoc import EVoC
   import numpy as np

   # Generate sample embedding data
   X = np.random.rand(1000, 512)  # 1000 samples, 512 dimensions

   # Create and fit the clusterer
   clusterer = EVoC()
   labels = clusterer.fit_predict(X)

   # Check results
   n_clusters = len(np.unique(labels[labels >= 0]))
   n_noise = np.sum(labels == -1)

   print(f"Found {n_clusters} clusters")
   print(f"Noise points: {n_noise}")

That's it! EVōC figures out the structure automatically.

Understanding Your Results
---------------------------

EVōC gives you several pieces of information. Let's explore them:

.. code-block:: python

   # The main output: cluster assignment for each point
   labels = clusterer.labels_
   print(f"First 10 assignments: {labels[:10]}")
   # Note: -1 indicates noise points (isolated, not in any cluster)

   # How confident is each assignment? (0 = uncertain, 1 = very confident)
   strengths = clusterer.membership_strengths_
   print(f"Membership strength for first point: {strengths[0]:.3f}")

   # Different cluster granularities discovered
   print(f"Available cluster layers: {len(clusterer.cluster_layers_)}")
   
   for i, layer in enumerate(clusterer.cluster_layers_):
       n_clusters_in_layer = len(np.unique(layer[layer >= 0]))
       print(f"  Layer {i}: {n_clusters_in_layer} clusters")

**What does this mean?**

- ``labels`` tells you which cluster each point belongs to (-1 for noise)
- ``membership_strengths_`` tells you how confident that assignment is
- ``cluster_layers_`` gives you clustering at multiple resolutions

You can use the default layer (``labels_``) or explore different granularities. More on this in :doc:`how_evoc_works`.

Real Data: Clustering MNIST
----------------------------

Let's try with real data. Here's how to load MNIST embeddings and cluster them:

.. code-block:: python

   from sklearn.datasets import fetch_openml
   import numpy as np
   from evoc import EVoC

   # Load MNIST (this downloads ~20MB of embedding data)
   print("Loading MNIST embeddings...")
   X = fetch_openml('mnist_784', version=1, return_X_y=False).data
   X = X.astype(np.float32) / 255.0  # Normalize

   # Cluster
   clusterer = EVoC()
   labels = clusterer.fit_predict(X)

   # Analyze
   n_clusters = len(np.unique(labels[labels >= 0]))
   n_noise = np.sum(labels == -1)
   print(f"\nClustered {len(X)} images into {n_clusters} groups")
   print(f"Noise points: {n_noise}")

   # Look at different granularities
   print(f"\nExploring cluster layers:")
   for i, layer in enumerate(clusterer.cluster_layers_):
       n = len(np.unique(layer[layer >= 0]))
       print(f"  Layer {i}: {n} clusters")

This shows how EVōC discovers meaningful structure in real embedding data.

Controlling the Results
-----------------------

Sometimes you want to adjust what EVōC discovers. Here are the main parameters:

**Want more clusters?** Lower the ``noise_level``:

.. code-block:: python

   clusterer = EVoC(noise_level=0.3)  # Stricter = more, smaller clusters
   labels = clusterer.fit_predict(X)

**Want fewer clusters?** Raise the ``noise_level``:

.. code-block:: python

   clusterer = EVoC(noise_level=0.7)  # Looser = fewer, larger clusters
   labels = clusterer.fit_predict(X)

**Want a specific number of clusters?** Use ``approx_n_clusters``:

.. code-block:: python

   clusterer = EVoC(approx_n_clusters=10)  # Find ~10 clusters
   labels = clusterer.fit_predict(X)

**Fine-tune the graph construction** with ``n_neighbors``:

.. code-block:: python

   clusterer = EVoC(
       n_neighbors=20,  # More neighbors = broader context
       noise_level=0.5
   )
   labels = clusterer.fit_predict(X)

For a complete explanation of all parameters and when to use them, see the :doc:`user_guide`.

Working with Different Embedding Types
---------------------------------------

EVōC automatically handles different data types:

.. code-block:: python

   # Standard float embeddings (CLIP, sentence transformers, etc.)
   X_float = embeddings.astype(np.float32)
   labels = EVoC().fit_predict(X_float)

   # Quantized embeddings (int8) - for memory efficiency
   X_quantized = (embeddings * 127).clip(-127, 127).astype(np.int8)
   labels = EVoC().fit_predict(X_quantized)

   # Binary embeddings (uint8)
   X_binary = (embeddings > 0.5).astype(np.uint8)
   labels = EVoC().fit_predict(X_binary)

EVōC uses appropriate distance metrics for each type automatically.

Common Patterns
---------------

**Pattern 1: Automatic discovery (most common)**

.. code-block:: python

   clusterer = EVoC()
   labels = clusterer.fit_predict(embeddings)
   # Use the result as-is

**Pattern 2: Explore multiple granularities**

.. code-block:: python

   clusterer = EVoC()
   clusterer.fit(embeddings)
   
   for i, layer in enumerate(clusterer.cluster_layers_):
       print(f"Layer {i}: {len(np.unique(layer[layer >= 0]))} clusters")
       # Do something with this granularity...

**Pattern 3: Get confidence scores**

.. code-block:: python

   clusterer = EVoC()
   clusterer.fit(embeddings)
   labels = clusterer.labels_
   confidence = clusterer.membership_strengths_
   
   # Filter to high-confidence assignments
   high_conf = labels[confidence > 0.7]

**Pattern 4: Tune for your data**

.. code-block:: python

   clusterer = EVoC(
       n_neighbors=15,
       noise_level=0.5,
       base_min_cluster_size=5
   )
   labels = clusterer.fit_predict(embeddings)

Next Steps
----------

**Learn More:**

- **Algorithm intuition:** See :doc:`how_evoc_works` for a clear explanation
- **Parameter guidance:** Check the :doc:`user_guide` for detailed reference
- **Complete API:** See :doc:`api/index` for all available methods and options

**Try Interactive Examples:**

We provide seven notebooks covering different aspects and use cases:

1. :doc:`notebooks/01_getting_started` — Start here! Hands-on intro with synthetic and MNIST data
2. :doc:`notebooks/02_text_embeddings` — Cluster documents: 20 Newsgroups + arXiv ML
3. :doc:`notebooks/03_image_embeddings` — Real-world images: CIFAR-100 with CLIP embeddings
4. :doc:`notebooks/04_biological_data` — Domain-specific: Bird species from audio embeddings
5. :doc:`notebooks/05_quantized_embeddings` — Advanced: Memory-efficient int8 quantization
6. :doc:`notebooks/06_performance_benchmarks` — Compare EVōC vs. K-Means vs. UMAP+HDBSCAN
7. :doc:`notebooks/07_understanding_layers` — Deep dive: Hierarchical clustering structures
