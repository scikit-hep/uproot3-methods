#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot-methods/blob/master/LICENSE


import awkward

from uproot_methods.wrapjagged import normalize_arrays, unwrap_jagged


class ROOTMethods(awkward.Methods):
    _arraymethods = None

    awkward = awkward

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def _normalize_arrays(cls, arrays):
        return normalize_arrays(cls, arrays)

    @classmethod
    def _unwrap_jagged(cls, awkcls, arrays):
        return unwrap_jagged(cls, awkcls, arrays)

    def _trymemo(self, name, function):
        memoname = "_memo_" + name
        wrap, (array,) = type(self)._unwrap_jagged(self.JaggedArray, (self,))
        if not hasattr(array, memoname):
            setattr(array, memoname, function(array))
        return wrap(getattr(array, memoname))
