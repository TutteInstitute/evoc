Changelog
=========

This document records all notable changes to EVoC.

Unreleased
----------

**Changes:**

* int8 data now uses cosine similarity for both neighbour ranking and the
  reported neighbour distances (``-log2(cosine)``, as for float data).
  Previously int8 neighbours were ranked by raw inner product and distances
  were reported as ``1 / inner product``. This changes int8 clustering results.

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
