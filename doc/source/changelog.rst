Changelog
=========

This document records all notable changes to EVoC.

Unreleased
----------

**Features:**

* Prediction of clusters for new data with ``predict`` and ``predict_layers`` (requires ``prediction_data=True``)
* Incremental updates with ``update``, including expiring old data via ``X_expired`` for moving windows
* Stable cluster labels across updates, and stable sample ids in ``sample_ids_``
* Fast nearest neighbor search over fitted data with ``evoc.search.SearchIndex``
* Benchmark script for search, prediction and updates (``benchmarks/predict_update_bench.py``)

**Changes:**

* int8 data now uses cosine similarity for both finding neighbors and reporting neighbor distances, matching float 
  data; this changes clustering results for int8 data

**Fixes:**

* ``cluster_tree_`` now handles cluster labels with gaps

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
