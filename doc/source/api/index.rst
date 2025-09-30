API Reference
=============

This section contains the complete API reference for EVoC.

Main Classes and Functions
--------------------------

.. currentmodule:: evoc

.. autosummary::
   :toctree: generated/
   :nosignatures:

   EVoC
   evoc_clusters
   build_cluster_layers

Core Clustering
---------------

.. autoclass:: EVoC
   :members:
   :inherited-members:
   :show-inheritance:

.. autofunction:: evoc_clusters

.. autofunction:: build_cluster_layers

Utility Functions
-----------------

.. currentmodule:: evoc.clustering_utilities

.. autosummary::
   :toctree: generated/
   :nosignatures:

   find_peaks
   binary_search_for_n_clusters
   select_diverse_peaks
   build_cluster_tree
   find_duplicates

.. autofunction:: find_peaks
.. autofunction:: binary_search_for_n_clusters  
.. autofunction:: select_diverse_peaks
.. autofunction:: build_cluster_tree
.. autofunction:: find_duplicates

Tree Operations
---------------

.. currentmodule:: evoc.cluster_trees

.. autosummary::
   :toctree: generated/
   :nosignatures:

   mst_to_linkage_tree
   condense_tree
   extract_leaves
   get_cluster_label_vector
   get_point_membership_strength_vector

.. autofunction:: mst_to_linkage_tree
.. autofunction:: condense_tree
.. autofunction:: extract_leaves
.. autofunction:: get_cluster_label_vector
.. autofunction:: get_point_membership_strength_vector

Graph Construction
------------------

.. currentmodule:: evoc.knn_graph

.. autosummary::
   :toctree: generated/
   :nosignatures:

   knn_graph

.. autofunction:: knn_graph

.. currentmodule:: evoc.graph_construction

.. autosummary::
   :toctree: generated/
   :nosignatures:

   neighbor_graph_matrix

.. autofunction:: neighbor_graph_matrix

Node Embedding
--------------

.. currentmodule:: evoc.node_embedding

.. autosummary::
   :toctree: generated/
   :nosignatures:

   node_embedding

.. autofunction:: node_embedding

.. currentmodule:: evoc.label_propagation

.. autosummary::
   :toctree: generated/
   :nosignatures:

   label_propagation_init

.. autofunction:: label_propagation_init

Algorithm Components
--------------------

.. currentmodule:: evoc.boruvka

.. autosummary::
   :toctree: generated/
   :nosignatures:

   parallel_boruvka

.. autofunction:: parallel_boruvka

.. currentmodule:: evoc.numba_kdtree

.. autosummary::
   :toctree: generated/
   :nosignatures:

   build_kdtree

.. autofunction:: build_kdtree
