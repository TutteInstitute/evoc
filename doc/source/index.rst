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

**Cluster embedding vectors with automatic structure discovery**

If you're working with modern embedding representations—CLIP vectors, sentence embeddings, or other dense vector representations from foundation models—you face a clustering challenge: standard algorithms like K-means or DBSCAN don't perform well on high-dimensional embeddings. That's where EVōC comes in.

EVōC (pronounced "evoke") is a clustering algorithm specifically designed for embedding vectors. It automatically discovers meaningful cluster structures in your embeddings without requiring you to guess the number of clusters or tune many parameters. Under the hood, EVōC combines a fast graph-based embedding approach with intelligent density-based clustering, optimized to work efficiently on modern embeddings.

What's Different About EVōC?
----------------------------

**For Practitioners:** Get results fast with minimal parameter tuning. EVōC figures out the right number of clusters automatically. Just load your embeddings and run it.

**For Researchers:** Explore hierarchical cluster structures at multiple granularities in a single pass. Access membership strengths and detailed clustering metadata for analysis and experimentation.

**Performance:** Using optimized Numba kernels and an efficient two-stage algorithm, EVōC clusters large embedding collections faster than UMAP+HDBSCAN while maintaining or improving quality.

**Easy to Install:** Pure Python with just Numba as a core dependency. No complicated build steps or system-level dependencies.

The EVōC Approach: Two Stages
-----------------------------

EVōC works in two stages:

1. **Graph Embedding** — Build a k-nearest neighbor graph of your embeddings and learn an efficient intermediate representation (inspired by UMAP)
2. **Density Clustering** — Apply hierarchical density-based clustering to this representation (inspired by HDBSCAN)

This combination gives you the best of both worlds: fast, efficient processing of high-dimensional vectors, plus robust cluster discovery that handles noise gracefully.

Learn More
----------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   quickstart
   installation

.. toctree::
   :maxdepth: 2
   :caption: Understanding EVōC:

   how_evoc_works
   user_guide

.. toctree::
   :maxdepth: 2
   :caption: Learning Notebooks:

   notebooks/01_getting_started
   notebooks/02_text_embeddings
   notebooks/03_image_embeddings
   notebooks/04_biological_data
   notebooks/05_quantized_embeddings
   notebooks/06_performance_benchmarks
   notebooks/07_understanding_layers

.. toctree::
   :maxdepth: 2
   :caption: In Practice:

   examples
   api/index

.. toctree::
   :maxdepth: 1
   :caption: Reference:

   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
