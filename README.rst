.. image:: doc/evoc_logo.png
  :width: 600
  :align: center
  :alt: EVōC Logo

====
EVōC
====

EVōC (pronounced as "evoke") is Embedding Vector Oriented Clustering.
EVōC is a library for fast and flexible clustering of large datasets of high dimensional embedding vectors. 
If you have CLIP-vectors, outputs from sentence-transformers, or openAI, or Cohere embed, and you want
to quickly get good clusters out this is the library for you. EVōC takes all the good parts of the 
combination of UMAP + HDBSCAN for embedding clustering, improves upon them, and removes all 
the time-consuming parts. By specializing directly to embedding vectors we can get good
quality clustering with fewer hyper-parameters to tune and in a fraction of the time.

EVōC is the library to use if you want:

 * Fast clustering of embedding vectors on CPU
 * Multi-granularity clustering, and automatic selection of the number of clusters
 * Clustering of int8 or binary quantized embedding vectors that works out-of-the-box

 As of now this is very much an early beta version of the library. Things can and will break right now.
 We would welcome feedback, use cases and feature suggestions however.

-----------
Basic Usage
-----------

EVōC follows the scikit-learn API, so it should be familiar to most users. You can use EVōC wherever
you might have previously been using other sklearn clustering algorithms. Here is a simple example

.. code-block:: python

    import evoc
    from sklearn.datasets import make_blobs

    data, _ = make_blobs(n_samples=100_000, n_features=1024, centers=100)

    clusterer = evoc.EVoC()
    cluster_labels = clusterer.fit_predict(data)

Some more unique features include the generation of multiple layers of cluster granularity,
the ability to extract a hierarchy of clusters across those layers, and automatic duplicate 
(or very near duplicate) detection.

.. code-block:: python

    import evoc
    from sklearn.datasets import make_blobs

    data, _ = make_blobs(n_samples=100_000, n_features=1024, centers=100)

    clusterer = evoc.EVoC()
    cluster_labels = clusterer.fit_predict(data)
    cluster_layers = clusterer.cluster_layers_
    hierarchy = clusterer.cluster_tree_
    potential_duplicates = clusterer.duplicates_

The cluster layers are a list of cluster label vectors with the first being the finest grained
and later layers being coarser grained. This is ideal for layered topic modelling and use with
`DataMapPlot <https://github.com/TutteInstitute/datamapplot>`_. See 
`this data map <https://lmcinnes.github.io/datamapplot_examples/ArXiv_data_map_example.html>`_
for an example of using these layered clusters in topic modelling (zoom in to access finer 
grained topics).

------------
Installation
------------

EVōC has a small set of dependencies:

 * numpy
 * scikit-learn
 * numba
 * tqdm
 * tbb

At some point in the near future ... you can install EVōC from PyPI using pip:

.. code-block:: bash

    pip install evoc

For now install the latest version of EVōC from source you can do so by cloning the repository and running:

.. code-block:: bash

    git clone https://github.com/TutteInstitute/evoc
    cd evoc
    pip install .

-------
License
-------

EVōC is BSD (2-clause) licensed. See the LICENSE file for details.

------------
Contributing
------------

Contributions are more than welcome! If you have ideas for features of projects please get in touch. Everything from
code to notebooks to examples and documentation are all *equally valuable* so please don't feel you can't contribute.
To contribute please `fork the project <https://github.com/TutteInstitute/evoc/issues#fork-destination-box>`_ make your
changes and submit a pull request. We will do our best to work through any issues with you and get your code merged in.
