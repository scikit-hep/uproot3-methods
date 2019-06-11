#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot-methods/blob/master/LICENSE

import tokenize
import keyword
import re
from collections import OrderedDict

import awkward.array.chunked
import awkward.array.objects
import awkward.array.jagged
import awkward.array.table
import awkward.array.virtual

def _isidentifier(x):
    return isinstance(x, str) and _isidentifier.regex.match(x) and not keyword.iskeyword(x)
_isidentifier.regex = re.compile("^" + tokenize.Name + "$")

class Attribute(awkward.array.objects.Methods):
    def __setitem__(self, where, what):
        super(Attribute, self).__setitem__(where, what)
        if _isidentifier(where) and not where in dir(awkward.array.table.Table) and not where in dir(awkward.array.jagged.JaggedArray):
            self.__dict__[where] = self._contents[where]

    def __delitem__(self, where):
        if _isidentifier(where) and where in self._contents and where in self.__dict__:
            del self.__dict__[where]
        super(Attribute, self).__delitem__(where)

def unwrap(array, flatten):
    if isinstance(array, awkward.array.chunked.ChunkedArray):
        return array.copy(chunks=[unwrap(chunk, flatten) for chunk in array.chunks])
    elif isinstance(array, awkward.array.virtual.VirtualArray):
        return unwrap(array.array, flatten)
    elif isinstance(array, awkward.array.jagged.JaggedArray):
        if flatten:
            return array.content
        else:
            return array
    else:
        raise NotImplementedError(type(array))

class GenerateJaggedTable(object):
    def __init__(self, JaggedArray, Table, rowname, counts):
        self.JaggedArray = JaggedArray
        self.Table = Table
        self.rowname = rowname
        self.counts = counts
        self.fields = OrderedDict()

    def __call__(self):
        table = self.Table.named(self.rowname)
        for n, x in self.fields.items():
            table[n] = awkward.array.virtual.VirtualArray(unwrap, (x, True))
        return self.JaggedArray.fromcounts(unwrap(self.counts, False), table)
        
    # def __call__(self, rowgroup, column):
    #     return fromarrow(self.parquetfile.read_row_group(rowgroup, columns=[column]))[column]

    # def __getstate__(self):
    #     return {"file": self.file, "metadata": self.metadata, "common_metadata": self.common_metadata}

    # def __setstate__(self, state):
    #     self.file = state["file"]
    #     self.metadata = state["metadata"]
    #     self.common_metadata = state["common_metadata"]
    #     self._init()

    # def tojson(self):
    #     json.dumps([self.file, self.metadata, self.common_metadata])
    #     return {"file": self.file, "metadata": self.metadata, "common_metadata": self.common_metadata}

    # @classmethod
    # def fromjson(cls, state):
    #     return cls(state["file"], metadata=state["metadata"], common_metadata=state["common_metadata"])


def transform(array):
    array._valid()
    array.check_whole_valid = False

    AttributeJaggedArray = awkward.array.objects.Methods.mixin(Attribute, array.JaggedArray)
    AttributeTable = awkward.array.objects.Methods.mixin(Attribute, array.Table)

    if "nMuon" in array.columns:
        muons = GenerateJaggedTable(AttributeJaggedArray, AttributeTable, "Muon", array["nMuon"])
    else:
        muons = None
    flag = AttributeTable.named("Flags")
    HLT = AttributeTable.named("HLT")

    for n in array.columns:
        if n.startswith("Muon_") and muons is not None:
            muons.fields[n[len("Muon_"):]] = array[n]
        elif n.startswith("Flag_"):
            flag[n[len("Flag_"):]] = array[n]
        elif n.startswith("HLT_"):
            HLT[n[len("HLT_"):]] = array[n]

    out = AttributeTable.named("Event")
    out["raw"] = array
    if muons is not None and len(muons.fields) > 0:
        out["muons"] = awkward.array.virtual.VirtualArray(muons)
    if len(flag.columns) > 0:
        out["flag"] = flag
    if len(HLT.columns) > 0:
        out["HLT"] = HLT

    return out
