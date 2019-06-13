#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot-methods/blob/master/LICENSE

import awkward.type
import awkward.array.chunked

import uproot_methods.classes.TLorentzVector

def getcontent(virtual):
    return virtual.array.content

def jaggedtable(rowname, counts, fields):
    Table = counts.Table
    JaggedArray = counts.JaggedArray
    VirtualArray = counts.VirtualArray
    ChunkedArray = counts.ChunkedArray

    countsarray = counts.array
    if isinstance(countsarray, awkward.array.chunked.ChunkedArray):
        return lazyjagged(countsarray, rowname, [(n, x.array) for n, x in fields])
    else:
        offsets = JaggedArray.counts2offsets(countsarray)
        table = Table.named(rowname)
        for n, x in fields:
            table[n] = VirtualArray(getcontent, x, type=awkward.type.ArrayType(offsets[-1], x.type.to.to), cache=counts.cache, persistvirtual=counts.persistvirtual)
        columns = table.columns
        if "pt" in columns and "eta" in columns and "phi" in columns and "mass" in columns and "p4" not in columns:
            table["p4"] = VirtualArray(uproot_methods.classes.TLorentzVector.TLorentzVectorArray.from_ptetaphim, (table["pt"], table["eta"], table["phi"], table["mass"]), type=awkward.type.ArrayType(offsets[-1], uproot_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray), cache=counts.cache, persistvirtual=counts.persistvirtual)
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
        columns = tabletype.columns
        if "pt" in columns and "eta" in columns and "phi" in columns and "mass" in columns and "p4" not in columns:
            tabletype["p4"] = uproot_methods.classes.TLorentzVector.TLorentzVectorArray.from_ptetaphim
        chunks.append(VirtualArray(jaggedtable, (rowname, countschunk, fieldschunks), type=awkward.type.ArrayType(len(countschunk), float("inf"), tabletype), cache=countschunk.cache, persistvirtual=countschunk.persistvirtual))
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
        elif n.startswith("Flag_"):
            flag[n[len("Flag_"):]] = array[n]
        elif n.startswith("HLT_"):
            HLT[n[len("HLT_"):]] = array[n]
        else:
            print(n)

    out = array.Table.named("Event")
    out["raw"] = array
    if len(muons) > 0 and "nMuon" in array.columns:
        out["muons"] = lazyjagged(array["nMuon"], "Muon", muons)
    if len(flag.columns) > 0:
        out["flag"] = flag
    if len(HLT.columns) > 0:
        out["HLT"] = HLT

    return out
