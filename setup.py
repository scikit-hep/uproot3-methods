#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot-methods/blob/master/LICENSE

import os.path

from setuptools import find_packages
from setuptools import setup

def get_version():
    g = {}
    exec(open(os.path.join("uproot_methods", "version.py")).read(), g)
    return g["__version__"]

def get_description():
    description = open("README.rst", "rb").read().decode("utf8", "ignore")
    start = description.index(".. inclusion-marker-1-5-do-not-remove")
    stop = description.index(".. inclusion-marker-3-do-not-remove")

    before = ""
    after = """

Reference documentation
=======================

"""

    return description[start:stop].strip() # before + + after

setup(name = "uproot-methods",
      version = get_version(),
      packages = find_packages(exclude = ["tests"]),
      scripts = [],
      description = "Pythonic mix-ins for ROOT classes.",
      long_description = get_description(),
      author = "Jim Pivarski (IRIS-HEP)",
      author_email = "pivarski@princeton.edu",
      maintainer = "Jim Pivarski (IRIS-HEP)",
      maintainer_email = "pivarski@princeton.edu",
      url = "https://github.com/scikit-hep/uproot-methods",
      download_url = "https://github.com/scikit-hep/uproot-methods/releases",
      license = "BSD 3-clause",
      test_suite = "tests",
      install_requires = ["numpy>=1.13.1", "awkward>=0.11.0"],
      tests_require = [],
      classifiers = [
          "Development Status :: 5 - Production/Stable",
          "Intended Audience :: Developers",
          "Intended Audience :: Information Technology",
          "Intended Audience :: Science/Research",
          "License :: OSI Approved :: BSD License",
          "Operating System :: MacOS",
          "Operating System :: POSIX",
          "Operating System :: Unix",
          "Programming Language :: Python",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.4",
          "Programming Language :: Python :: 3.5",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
          "Topic :: Scientific/Engineering",
          "Topic :: Scientific/Engineering :: Information Analysis",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Scientific/Engineering :: Physics",
          "Topic :: Software Development",
          "Topic :: Utilities",
          ],
      platforms = "Any",
      )
