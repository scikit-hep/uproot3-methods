#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot3-methods/blob/master/LICENSE

import numpy

import uproot3_methods.base

class Methods(uproot3_methods.base.ROOTMethods):
    def hello(self):
        return "world", len(dir(self))
