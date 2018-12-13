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

    pip install uproot-methods

or similar (use ``sudo``, ``--user``, ``virtualenv``, or pip-in-conda if you wish).

Strict dependencies:
====================

- `Python <http://docs.python-guide.org/en/latest/starting/installation/>`__ (2.7+, 3.4+)

The following are installed automatically when you install uproot with pip:

- `Numpy <https://scipy.org/install.html>`__ (1.13.1+)
- `awkward-array <https://pypi.org/project/awkward>`__ (0.7.0+) to manipulate data from non-flat TTrees, such as jagged arrays (`part of Scikit-HEP <https://github.com/scikit-hep/awkward-array>`__)

.. inclusion-marker-3-do-not-remove

Reference documentation
=======================

TBD.
