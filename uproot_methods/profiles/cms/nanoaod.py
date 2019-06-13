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

    stuff = [("Muon_", "muons", []),
             ("Flag_", "flag", array.Table.named("flags")),
             ("HLT_", "HLT", array.Table.named("HLT"))]

    for n in array.columns:
        for prefix, collection, data in stuff:
            if n.startswith(prefix):
                if isinstance(data, list):
                    data.append((n[len(prefix):], array[n]))
                else:
                    data[n[len(prefix):]] = array[n]
                
    out = array.Table.named("Event")
    out["raw"] = array

    for prefix, collection, data in stuff:
        if isinstance(data, list):
            rowname = prefix[:-1]
            countname = "n" + rowname
            if len(data) > 0 and countname in array.columns:
                out[collection] = lazyjagged(array[countname], rowname, data)
        else:
            if len(data.columns) > 0:
                out[collection] = data

    return out
