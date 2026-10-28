Changelog
=========

This document records all notable changes to EVoC.

Unreleased
----------

**Features:**

* ``EVoC(prediction_data=True)`` keeps the state needed to work with new data
  (``prediction_data_``).
* ``EVoC.predict`` and ``EVoC.predict_layers`` assign new samples to the
  clusters of a fitted model, by a weighted vote of their nearest training
  samples (``method="vote"``) or by placing them in the node embedding and
  condensed cluster tree (``method="embedding"``); ``EVoC.embed`` returns the
  embedding positions of new samples.
* ``EVoC.update`` adds samples to a fitted model: the k-NN graph is extended
  incrementally, new samples are placed in the node embedding, and the layers
  are re-clustered and relabelled so that cluster ids stay stable.
* ``evoc.search.SearchIndex``: fast, disk-cached nearest-neighbour search
  over float, int8 and bit-packed uint8 data, adapted from pynndescent.
* The NN-descent drivers (and ``knn_graph.nn_descent``) accept an initial
  graph (``init_graph``).
* ``benchmarks/predict_update_bench.py`` benchmarks search, predict and update
  on MNIST and Fashion-MNIST.

**Changes:**

* int8 data now uses cosine similarity for both neighbour ranking and the
  reported neighbour distances (``-log2(cosine)``, as for float data).
  Previously int8 neighbours were ranked by raw inner product and distances
  were reported as ``1 / inner product``. This changes int8 clustering results.

**Fixes:**

* ``cluster_tree_`` handles cluster labels that are not contiguous.

Version 0.1.0 (TBD)
-------------------

Initial release of EVoC.

**Features:**

* Core clustering algorithm with hierarchical multi-layer support
* Scikit-learn compatible API
* Support for multiple embedding types (float, int8, uint8)  
* Optimized distance metrics (cosine, quantized cosine, bitwise Jaccard)
* Numba-accelerated performance
* Comprehensive parameter set for fine-tuning
* Built-in duplicate detection
* Extensive documentation and examples

**API Reference:**

* ``EVoC`` - Main clustering class
* ``evoc_clusters`` - Functional interface  
* ``build_cluster_layers`` - Multi-layer clustering construction

**Performance:**

* Efficient processing of high-dimensional embeddings
* Memory-optimized algorithms
* Multi-threaded computation support

**Documentation:**

* Complete API documentation with numpydoc formatting
* Interactive Jupyter notebook examples
* Comprehensive user guide and tutorials
* ReadTheDocs integration
