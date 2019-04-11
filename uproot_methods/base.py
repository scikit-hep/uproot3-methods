#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot-methods/blob/master/LICENSE


import awkward

from awkward.util import unwrap_jagged


class ROOTMethods(awkward.Methods):
    _arraymethods = None

    awkward = awkward

    def __ne__(self, other):
        return not self.__eq__(other)

    def _trymemo(self, name, function):
        memoname = "_memo_" + name
        wrap, (array,) = unwrap_jagged(type(self), self.JaggedArray, (self,))
        if not hasattr(array, memoname):
            setattr(array, memoname, function(array))
        return wrap(getattr(array, memoname))
