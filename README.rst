uproot-methods
==============

.. inclusion-marker-1-do-not-remove

Pythonic mix-ins for ROOT classes.

.. inclusion-marker-1-5-do-not-remove

This package is typically used as a dependency for `uproot <https://github.com/scikit-hep/uproot>`__, to define methods on the classes that are automatically generated from ROOT files. This includes histograms (TH*) and physics objects like TLorentzVectors. The reason it's a separate library is so that we can add physics-specific functionality on a shorter timescale than we can update uproot itself, which is purely an I/O package.

Occasionally, this library is used without uproot, as a way to make arrays of TLorentzVectors.

.. inclusion-marker-2-do-not-remove

Installation
============

Install uproot-methods like any other Python package:

.. code-block:: bash

    pip install uproot-methods                # maybe with sudo or --user, or in virtualenv

or install with `conda <https://conda.io/en/latest/miniconda.html>`__:

.. code-block:: bash

    conda config --add channels conda-forge   # if you haven't added conda-forge already
    conda install uproot-methods

Both installers automatically install the dependencies.

Dependencies:
-------------

- `numpy <https://scipy.org/install.html>`__ (1.13.1+)
- `awkward-array <https://github.com/scikit-hep/awkward-array>`__ (0.11.0+)

.. inclusion-marker-3-do-not-remove

Reference documentation
=======================

TBD.

Acknowledgements
================

Support for this work was provided by NSF cooperative agreement OAC-1836650 (IRIS-HEP), grant OAC-1450377 (DIANA/HEP) and PHY-1520942 (US-CMS LHC Ops).

Thanks especially to the gracious help of `uproot-methods contributors <https://github.com/scikit-hep/uproot-methods/graphs/contributors>`__!
