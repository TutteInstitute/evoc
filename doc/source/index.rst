.. image:: evoc_logo_horizontal.png
  :width: 600
  :align: center
  :alt: EVōC Logo

EVōC: Embedding Vector Oriented Clustering
==========================================

.. image:: https://img.shields.io/badge/python-3.8%2B-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-BSD-green.svg
   :target: https://opensource.org/licenses/BSD-3-Clause
   :alt: License

EVōC (pronounced as "evoke") provides Embedding Vector Oriented Clustering.

EVōC (Embedding Vector Oriented Clustering) is a powerful clustering algorithm designed specifically for high-dimensional 
embedding vectors such as CLIP-vectors, sentence-transformers output, and other dense vector representations. 

The algorithm combines a node embedding approach (related to UMAP) with density-based clustering (related to HDBSCAN), 
providing improved efficiency and quality for clustering high-dimensional embedding vectors.

Key Features
------------

* **Optimized for High-Dimensional Embeddings**: Specifically designed for modern embedding vectors
* **Multi-Layer Clustering**: Provides hierarchical clustering with multiple granularity levels
* **Performance Optimized**: Uses Numba for high-performance computation
* **Flexible Parameters**: Extensive parameter set for fine-tuning clustering behavior
* **Scikit-learn Compatible**: Follows scikit-learn API conventions

Quick Start
-----------

.. code-block:: python

   from evoc import EVoC
   import numpy as np

   # Generate sample data
   X = np.random.rand(1000, 512)  # 1000 samples, 512-dimensional embeddings

   # Initialize and fit the clusterer
   clusterer = EVoC()
   labels = clusterer.fit_predict(X)

   # Access cluster layers and membership strengths
   print(f"Number of clusters: {len(np.unique(labels[labels >= 0]))}")
   print(f"Number of cluster layers: {len(clusterer.cluster_layers_)}")

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide
   api/index
   examples
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
