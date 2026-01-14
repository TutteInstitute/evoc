Examples
========

Collection of practical examples demonstrating EVoC usage in different scenarios.

Basic Examples
--------------

Simple Clustering
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from evoc import EVoC
   import numpy as np

   # Simple example with random data
   X = np.random.rand(500, 128)
   clusterer = EVoC()
   labels = clusterer.fit_predict(X)

   print(f"Found {len(np.unique(labels[labels >= 0]))} clusters")

Specify Number of Clusters
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # When you know the desired number of clusters
   clusterer = EVoC(approx_n_clusters=5)
   labels = clusterer.fit_predict(X)

Working with Real Embeddings
-----------------------------

CLIP Embeddings
~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import clip
   from evoc import EVoC

   # Load CLIP model
   device = "cuda" if torch.cuda.is_available() else "cpu"
   model, preprocess = clip.load("ViT-B/32", device=device)

   # Generate embeddings for images
   # (assuming you have a list of PIL images)
   embeddings = []
   with torch.no_grad():
       for image in images:
           image_input = preprocess(image).unsqueeze(0).to(device)
           embedding = model.encode_image(image_input)
           embeddings.append(embedding.cpu().numpy())

   X = np.vstack(embeddings)

   # Cluster the embeddings
   clusterer = EVoC(
       n_neighbors=20,
       noise_level=0.6,
       base_min_cluster_size=3
   )
   labels = clusterer.fit_predict(X)

Sentence Embeddings
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sentence_transformers import SentenceTransformer
   from evoc import EVoC

   # Load sentence transformer model
   model = SentenceTransformer('all-MiniLM-L6-v2')

   # Your text data
   texts = [
       "The cat sat on the mat",
       "Dogs are great pets", 
       "Machine learning is fascinating",
       # ... more texts
   ]

   # Generate embeddings
   embeddings = model.encode(texts)

   # Cluster similar texts
   clusterer = EVoC(
       n_neighbors=15,
       noise_level=0.4,
       base_min_cluster_size=2
   )
   labels = clusterer.fit_predict(embeddings)

   # Group texts by cluster
   clusters = {}
   for i, label in enumerate(labels):
       if label >= 0:  # Ignore noise points
           if label not in clusters:
               clusters[label] = []
           clusters[label].append(texts[i])

Advanced Usage
--------------

Hierarchical Analysis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get multiple clustering granularities
   clusterer = EVoC(max_layers=5)
   clusterer.fit(X)

   # Analyze each layer
   for i, layer in enumerate(clusterer.cluster_layers_):
       n_clusters = len(np.unique(layer[layer >= 0]))
       persistence = clusterer.persistence_scores_[i]

       print(f"Layer {i}: {n_clusters} clusters, "
             f"persistence: {persistence:.3f}")

   # Access the hierarchical structure
   tree = clusterer.cluster_tree_
   print(f"Hierarchical structure: {tree}")

Parameter Optimization
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.metrics import silhouette_score

   # Grid search over parameters
   best_score = -1
   best_params = None

   for n_neighbors in [10, 15, 20]:
       for noise_level in [0.3, 0.5, 0.7]:
           clusterer = EVoC(
               n_neighbors=n_neighbors,
               noise_level=noise_level,
               random_state=42
           )
           labels = clusterer.fit_predict(X)

           if len(np.unique(labels[labels >= 0])) > 1:
               score = silhouette_score(X, labels)
               if score > best_score:
                   best_score = score
                   best_params = {
                       'n_neighbors': n_neighbors,
                       'noise_level': noise_level
                   }

   print(f"Best parameters: {best_params}")
   print(f"Best silhouette score: {best_score:.3f}")

Memory-Efficient Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For large datasets, use smaller parameters
   clusterer = EVoC(
       n_neighbors=10,        # Reduce graph density
       node_embedding_dim=8,  # Lower embedding dimension  
       n_epochs=30,          # Fewer training epochs
       max_layers=3          # Limit hierarchy depth
   )

   # Process in chunks if needed
   chunk_size = 10000
   all_labels = []

   for i in range(0, len(X), chunk_size):
       chunk = X[i:i+chunk_size]
       chunk_labels = clusterer.fit_predict(chunk)
       all_labels.extend(chunk_labels)

Specialized Data Types
----------------------

Binary Embeddings
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For binary/hash embeddings
   binary_embeddings = (embeddings > 0.5).astype(np.uint8)

   clusterer = EVoC(
       n_neighbors=25,     # More neighbors for binary data
       neighbor_scale=1.5, # Denser graph
       noise_level=0.4     # Lower noise threshold
   )
   labels = clusterer.fit_predict(binary_embeddings)

Quantized Embeddings
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For int8 quantized embeddings
   quantized_embeddings = (embeddings * 127).clip(-127, 127).astype(np.int8)

   clusterer = EVoC(
       n_neighbors=20,
       base_min_cluster_size=8,
       noise_level=0.6
   )
   labels = clusterer.fit_predict(quantized_embeddings)

Evaluation and Validation
--------------------------

Cluster Quality Assessment
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.metrics import (
       silhouette_score, 
       calinski_harabasz_score,
       davies_bouldin_score
   )

   # Fit the clusterer
   labels = clusterer.fit_predict(X)

   # Calculate quality metrics
   if len(np.unique(labels[labels >= 0])) > 1:
       silhouette = silhouette_score(X, labels)
       calinski_harabasz = calinski_harabasz_score(X, labels)  
       davies_bouldin = davies_bouldin_score(X, labels)

       print(f"Silhouette Score: {silhouette:.3f}")
       print(f"Calinski-Harabasz Score: {calinski_harabasz:.3f}")
       print(f"Davies-Bouldin Score: {davies_bouldin:.3f}")

   # Analyze membership strengths
   strengths = clusterer.membership_strengths_
   print(f"Average membership strength: {np.mean(strengths):.3f}")
   print(f"Std of membership strengths: {np.std(strengths):.3f}")

Stability Analysis
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Test clustering stability across random seeds
   stability_scores = []

   for seed in range(10):
       clusterer = EVoC(random_state=seed)
       labels = clusterer.fit_predict(X)

       if len(np.unique(labels[labels >= 0])) > 1:
           score = silhouette_score(X, labels)
           stability_scores.append(score)

   print(f"Mean stability: {np.mean(stability_scores):.3f}")
   print(f"Std stability: {np.std(stability_scores):.3f}")
