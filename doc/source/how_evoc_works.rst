How EVōC Works
==============

Understanding EVōC at a high level helps you know what to expect from the algorithm and how to use it effectively. This guide explains the intuition behind the two-stage approach without diving into mathematical details.

The Problem We're Solving
--------------------------

Modern embedding vectors from models like CLIP, sentence-transformers, or other foundation models encode meaningful information in high-dimensional spaces. The challenge: clustering these embeddings is tricky. Standard algorithms like K-means assume Euclidean geometry that doesn't apply well in high dimensions, while approaches like DBSCAN struggle with the curse of dimensionality.

What we need is an algorithm that:

- Understands the structure of high-dimensional embeddings
- Doesn't require us to guess the number of clusters upfront
- Works efficiently even with hundreds of thousands of embeddings
- Provides interpretable results with confidence scores
- Discovers meaningful clusters across multiple granularities

EVōC addresses all of these concerns through a two-stage approach.

Stage 1: Graph Embedding
------------------------

The first stage focuses on **understanding the local structure** of your embeddings.

Imagine your embedding vectors as a cloud of points in high-dimensional space. Points that are similar (close together in embedding space) should probably belong to the same cluster. The first thing EVōC does is build a **k-nearest neighbor graph**: for each point, find its k most similar neighbors and connect them.

Why this matters: This graph captures the local neighborhood structure of your embeddings. But this graph is still in high dimensions, which makes the next step inefficient. So EVōC learns a **lower-dimensional embedding** of this graph structure—think of it as a compressed representation that preserves the neighborhood relationships but is easier to work with.

This stage is inspired by UMAP's approach, but optimized for the clustering task. Instead of trying to preserve all the structure, we focus on what matters for clustering: local density and neighborhood relationships.

**In simple terms:** "Compress your embeddings into a lower-dimensional space that preserves neighborhood structure."

Stage 2: Density Clustering
---------------------------

The second stage applies **intelligent cluster detection** to the learned representation.

Now that we have a cleaner, lower-dimensional representation, the next question is: where are the clusters? EVōC uses density-based clustering—the idea that clusters are regions of higher density separated by regions of lower density.

The algorithm applies **hierarchical clustering** to build a tree structure:

- Start with fine-grained clusters (many small, tight clusters)
- Gradually merge similar clusters together
- At each step, create a new clustering layer with coarser granularity

This gives you **multiple valid clusterings** at different resolutions in a single pass. You might want tight, detailed clusters for one analysis and broader, looser clusters for another—EVōC gives you both.

The algorithm also identifies **noise points**: embeddings that are isolated and don't belong to any high-density cluster. These are marked with a special label (-1 in scikit-learn convention) rather than forced into clusters.

**In simple terms:** "Find clusters by detecting dense regions, then show me clusters at different levels of detail."

Why This Two-Stage Approach?
----------------------------

You might wonder: why not just apply density clustering directly to the embeddings?

The answer has to do with the curse of dimensionality. In very high dimensions (hundreds or thousands), the notion of "distance" becomes less meaningful, and density-based clustering struggles. By first compressing the embeddings into a lower-dimensional space that preserves structure, we make the density clustering step more robust and efficient.

Additionally, the two-stage approach is **fast**: the graph embedding stage is computationally efficient, and density clustering on the reduced representation is quick even for large datasets.

Key Features Explained
----------------------

Automatic Cluster Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The biggest advantage of EVōC is that it **figures out the number of clusters automatically**. You don't need to specify k (as in K-means) or tune difficult DBSCAN parameters.

Here's why this works: during hierarchical clustering, the algorithm searches for a clustering layer where the clusters are most "persistent"—meaning they appear across multiple levels of the hierarchy and aren't just artifacts of noise. This is the natural cluster structure your data reveals.

Of course, you can override this if you want a specific number of clusters, but most of the time automatic selection works well.

Multi-Layer Clustering
~~~~~~~~~~~~~~~~~~~~~~

EVōC doesn't just give you one clustering—it gives you several, each at a different granularity. Access them via ``clusterer.cluster_layers_`` to explore your data at multiple resolutions:

- Coarse layers (fewer clusters, larger groups)
- Fine layers (more clusters, tighter groups)

This is particularly useful for hierarchical data or when you want to understand structure at different scales. For instance, in document clustering, a coarse layer might group documents by topic, while a finer layer might separate documents within a topic by subtopic.

Membership Strengths
~~~~~~~~~~~~~~~~~~~~

Rather than hard assignments (a point definitely is or isn't in a cluster), EVōC provides **membership strengths**: a confidence score for each point-cluster assignment. Points at the core of a cluster have high membership strength; points at the boundary have lower strength.

This gives you nuance: a point might be assigned to cluster A, but if its membership strength is only 0.6, you know it's somewhat ambiguous. This is more realistic than hard clustering and useful for downstream analysis.

Noise Handling
~~~~~~~~~~~~~~

EVōC gracefully handles **noise and outliers**. Instead of forcing every point into a cluster, isolated points that don't fit well into any high-density region are marked as noise (label -1). The ``noise_level`` parameter controls this trade-off: higher values = stricter clustering (more noise), lower values = include more points (fewer noise).

Practical Implications
----------------------

What does all this mean when you use EVōC?

**Simple usage:**

.. code-block:: python

   from evoc import EVoC
   import numpy as np

   # Your embeddings
   embeddings = np.load("embeddings.npy")

   # Cluster them
   clusterer = EVoC()
   clusterer.fit(embeddings)

   # Get labels
   labels = clusterer.labels_

   # Explore different cluster granularities
   for i, layer in enumerate(clusterer.cluster_layers_):
       n_clusters = len(np.unique(layer[layer >= 0]))
       print(f"Layer {i}: {n_clusters} clusters")

**With some parameter tuning:**

.. code-block:: python

   # If you want more/fewer clusters, adjust noise_level
   # Lower = more clusters, higher = fewer clusters
   clusterer = EVoC(noise_level=0.3)  # More clusters
   labels = clusterer.fit_predict(embeddings)

**Key takeaways:**

1. EVōC works well out-of-the-box with default parameters
2. You get multiple cluster granularities, not just one
3. Noise points are handled explicitly (not forced into clusters)
4. Results include confidence scores (membership strengths)
5. The algorithm is fast and scales to large embedding collections

When to Use EVōC
----------------

EVōC excels when you have:

- **High-dimensional embeddings** (hundreds to thousands of dimensions) from neural models
- **No predetermined number of clusters** (or you want to explore the data)
- **Real-world data** with potential noise and outliers
- **Large datasets** where computational efficiency matters
- **Need for multiple granularities** (exploring structure at different scales)

EVōC is less ideal if:

- You must have a fixed number of clusters and can't explore alternatives
- Your data is low-dimensional (< 10 dimensions) where standard DBSCAN works fine
- You need to cluster using custom distance metrics not supported by EVōC

Comparison with Other Approaches
---------------------------------

**vs. K-means:** K-means requires you to guess k upfront. EVōC figures out the natural number of clusters. However, K-means is simpler and faster for very large datasets where you know the number of clusters.

**vs. UMAP + HDBSCAN:** This was the standard "best practice" pipeline. EVōC combines and optimizes the two steps for embeddings, resulting in faster clustering. For non-embedding data, UMAP + HDBSCAN remains a good choice.

**vs. Spectral Clustering:** Spectral clustering finds a different kind of structure (global connectivity). EVōC finds local density-based clusters, which often matches human intuition better.

**vs. Agglomerative Clustering:** Both are hierarchical, but EVōC's density-aware approach handles high dimensions much better.

Next Steps
----------

Ready to get started? Here's the recommended learning path:

1. **Quick Start** — Jump into :doc:`quickstart` for a 5-minute introduction
2. **Interactive Learning** — Try :doc:`notebooks/01_getting_started` for hands-on examples
3. **Parameter Tuning** — See :doc:`user_guide` for guidance on adjusting parameters
4. **Real Applications** — Explore notebooks for your use case:
   
   - Text embeddings: :doc:`notebooks/02_text_embeddings`
   - Image embeddings: :doc:`notebooks/03_image_embeddings`  
   - Domain-specific data: :doc:`notebooks/04_biological_data`
   - Advanced techniques: :doc:`notebooks/05_quantized_embeddings`

5. **Deep Dives** — Understand hierarchies in :doc:`notebooks/07_understanding_layers` and performance in :doc:`notebooks/06_performance_benchmarks`
6. **Complete Reference** — See :doc:`api/index` for all available methods and options

For deeper mathematical details, see our research papers (coming soon).
