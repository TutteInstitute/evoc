Installation
============

Requirements
------------

EVoC requires Python 3.8 or later and the following dependencies:

* numpy >= 1.21.0
* scipy >= 1.7.0
* scikit-learn >= 1.0.0
* numba >= 0.56.0

Install from PyPI
-----------------

.. code-block:: bash

   pip install evoc

Install from Source
-------------------

To install the latest development version:

.. code-block:: bash

   git clone https://github.com/TutteInstitute/evoc.git
   cd evoc
   pip install -e .

Development Installation
------------------------

For development, install with additional dependencies:

.. code-block:: bash

   git clone https://github.com/TutteInstitute/evoc.git
   cd evoc
   pip install -e ".[dev,docs,test]"

Verify Installation
-------------------

To verify that EVoC is installed correctly:

.. code-block:: python

   import evoc
   print(evoc.__version__)

   # Run a quick test
   from evoc import EVoC
   import numpy as np

   X = np.random.rand(100, 10)
   clusterer = EVoC()
   labels = clusterer.fit_predict(X)
   print(f"Clustering completed successfully! Found {len(np.unique(labels[labels >= 0]))} clusters.")
