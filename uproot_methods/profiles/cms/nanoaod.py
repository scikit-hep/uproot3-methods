#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot-methods/blob/master/LICENSE

import awkward.type

# def unwrap(array, flatten):
#     if isinstance(array, awkward.array.chunked.ChunkedArray):
#         return array.copy(chunks=[unwrap(chunk, flatten) for chunk in array.chunks])
#     elif isinstance(array, awkward.array.virtual.VirtualArray):
#         return unwrap(array.array, flatten)
#     elif isinstance(array, awkward.array.jagged.JaggedArray):
#         if flatten:
#             return array.content
#         else:
#             return array
#     else:
#         raise NotImplementedError(type(array))

# class GenerateJaggedTable(object):
#     def __init__(self, JaggedArray, Table, rowname, counts):
#         self.JaggedArray = JaggedArray
#         self.Table = Table
#         self.rowname = rowname
#         self.counts = counts
#         self.fields = OrderedDict()

#     def __call__(self):
#         # table = self.Table.named(self.rowname)
#         # for n, x in self.fields.items():
#         #     table[n] = awkward.array.virtual.VirtualArray(unwrap, x)

#         VirtualArray = self.counts.VirtualArray
#         ChunkedArray = self.counts.ChunkedArray
#         return ChunkedArray([VirtualArray(makejagged, chunk) for chunk in self.counts.chunks], self.counts.counts)

#     # def __call__(self, rowgroup, column):
#     #     return fromarrow(self.parquetfile.read_row_group(rowgroup, columns=[column]))[column]

#     # def __getstate__(self):
#     #     return {"file": self.file, "metadata": self.metadata, "common_metadata": self.common_metadata}

#     # def __setstate__(self, state):
#     #     self.file = state["file"]
#     #     self.metadata = state["metadata"]
#     #     self.common_metadata = state["common_metadata"]
#     #     self._init()

#     # def tojson(self):
#     #     json.dumps([self.file, self.metadata, self.common_metadata])
#     #     return {"file": self.file, "metadata": self.metadata, "common_metadata": self.common_metadata}

#     # @classmethod
#     # def fromjson(cls, state):
#     #     return cls(state["file"], metadata=state["metadata"], common_metadata=state["common_metadata"])

def getcontent(virtual):
    return virtual.array.content

class GenerateJaggedTable(object):
    def __init__(self, rowname, counts, fields):
        self.rowname = rowname
        self.counts = counts
        self.fields = fields

    def __call__(self):
        Table = self.counts.Table
        JaggedArray = self.counts.JaggedArray
        VirtualArray = self.counts.VirtualArray

        offsets = JaggedArray.counts2offsets(self.counts.array)
        table = Table.named(self.rowname)
        for n, x in self.fields:
            table[n] = VirtualArray(getcontent, x, type=awkward.type.ArrayType(offsets[-1], x.type.to.to), cache=self.counts.cache, persistvirtual=self.counts.persistvirtual)
        return JaggedArray.fromoffsets(offsets, table)

def lazyjagged(countsarray, rowname, fields):
    VirtualArray = countsarray.VirtualArray
    ChunkedArray = countsarray.ChunkedArray
    chunks = []
    for i, countschunk in enumerate(countsarray.chunks):
        fieldschunks = []
        tabletype = awkward.type.TableType()
        for fieldname, field in fields:
            assert field.counts[i] == countsarray.counts[i]
            fieldschunks.append((fieldname, field.chunks[i]))
            tabletype[fieldname] = field.type.to.to
        generator = GenerateJaggedTable(rowname, countschunk, fieldschunks)
        chunks.append(VirtualArray(generator, type=awkward.type.ArrayType(len(countschunk), float("inf"), tabletype), cache=countschunk.cache, persistvirtual=countschunk.persistvirtual))
    return ChunkedArray(chunks, countsarray.counts)

def transform(array):
    array._valid()
    array.check_whole_valid = False

    muons = []
    flag = array.Table.named("Flags")
    HLT = array.Table.named("HLT")

    for n in array.columns:
        if n.startswith("Muon_"):
            muons.append((n[len("Muon_"):], array[n]))
        if n.startswith("Flag_"):
            flag[n[len("Flag_"):]] = array[n]
        elif n.startswith("HLT_"):
            HLT[n[len("HLT_"):]] = array[n]

    out = array.Table.named("Event")
    out["raw"] = array
    if len(muons) > 0 and "nMuon" in array.columns:
        out["muons"] = lazyjagged(array["nMuon"], "Muon", muons)
    if len(flag.columns) > 0:
        out["flag"] = flag
    if len(HLT.columns) > 0:
        out["HLT"] = HLT

    return out
