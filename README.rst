uproot3-methods
===============

.. inclusion-marker-1-do-not-remove

Pythonic mix-ins for ROOT classes.

.. inclusion-marker-1-5-do-not-remove

This package is typically used as a dependency for `uproot 3.x <https://github.com/scikit-hep/uproot3>`__, to define methods on the classes that are automatically generated from ROOT files. This includes histograms (TH*) and physics objects like TLorentzVectors. The reason it's a separate library is so that we can add physics-specific functionality on a shorter timescale than we can update Uproot 3 itself, which is purely an I/O package.

Occasionally, this library is used without Uproot 3, as a way to make arrays of TLorentzVectors.

**Note:** this package is incompatible with ``awkward>=1.0`` and ``uproot>=4.0``! For Lorentz vectors, use `vector <https://github.com/scikit-hep/vector>`__. Since the versions of Awkward Array and Uproot that this is compatible with are deprecated, **this library is deprecated** as well.

.. inclusion-marker-2-do-not-remove

Installation
============

Install uproot3-methods like any other Python package:

.. code-block:: bash

    pip install uproot3-methods               # maybe with sudo or --user, or in virtualenv

Dependencies:
-------------

- `numpy <https://scipy.org/install.html>`__ (1.13.1+)
- `Awkward Array 0.x <https://github.com/scikit-hep/awkward-0.x>`__

.. inclusion-marker-3-do-not-remove

Reference documentation
=======================

TBD.

Acknowledgements
================

Support for this work was provided by NSF cooperative agreement OAC-1836650 (IRIS-HEP), grant OAC-1450377 (DIANA/HEP) and PHY-1520942 (US-CMS LHC Ops).

Thanks especially to the gracious help of `uproot3-methods contributors <https://github.com/scikit-hep/uproot3-methods/graphs/contributors>`__!
