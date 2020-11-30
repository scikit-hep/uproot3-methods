#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot3-methods/blob/master/LICENSE

import awkward0
import awkward0.util

class ROOTMethods(awkward0.Methods):
    _arraymethods = None

    awkward = awkward0
    awkward0 = awkward0

    def __ne__(self, other):
        return not self.__eq__(other)

    def _trymemo(self, name, function):
        memoname = "_memo_" + name
        wrap, (array,) = awkward0.util.unwrap_jagged(type(self), self.JaggedArray, (self,))
        if not hasattr(array, memoname):
            setattr(array, memoname, function(array))
        return wrap(getattr(array, memoname))
