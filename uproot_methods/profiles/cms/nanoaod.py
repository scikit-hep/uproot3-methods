#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot-methods/blob/master/LICENSE

import tokenize
import keyword
import re

import awkward.array.table

def _isidentifier(x):
    return isinstance(x, str) and _isidentifier.regex.match(x) and not keyword.iskeyword(x)
_isidentifier.regex = re.compile("^" + tokenize.Name + "$")

class AttributeTable(awkward.array.table.Table):
    def __setitem__(self, where, what):
        super(AttributeTable, self).__setitem__(where, what)
        if _isidentifier(where) and not where in dir(awkward.array.table.Table):
            self.__dict__[where] = self._contents[where]

    def __delitem__(self, where):
        if _isidentifier(where) and where in self._contents and where in self.__dict__:
            del self.__dict__[where]
        super(AttributeTable, self).__delitem__(where)

def transform(array):
    array._valid()
    array.check_whole_valid = False

    out = AttributeTable.named("Event")
    HLT = AttributeTable.named("HLT")

    for n in array.columns:
        if n.startswith("HLT_"):
            HLT[n[len("HLT_"):]] = array[n]

    if len(HLT.columns) > 0:
        out["HLT"] = HLT

    return out
