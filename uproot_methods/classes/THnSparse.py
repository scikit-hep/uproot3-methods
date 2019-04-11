#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot-methods/blob/master/LICENSE

import numpy

import uproot_methods.base

class Methods(uproot_methods.base.ROOTMethods):
    def hello(self):
        return "world", len(dir(self))
