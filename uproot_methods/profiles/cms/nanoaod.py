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
            assert field.chunksizes[i] == countsarray.chunksizes[i]
            fieldschunks.append((fieldname, field.chunks[i]))
            tabletype[fieldname] = field.type.to.to
        columns = tabletype.columns
        if "pt" in columns and "eta" in columns and "phi" in columns and "mass" in columns and "p4" not in columns:
            tabletype["p4"] = uproot_methods.classes.TLorentzVector.TLorentzVectorArray.from_ptetaphim
        chunks.append(VirtualArray(jaggedtable, (rowname, countschunk, fieldschunks), type=awkward.type.ArrayType(len(countschunk), float("inf"), tabletype), cache=countschunk.cache, persistvirtual=countschunk.persistvirtual))
    return ChunkedArray(chunks, countsarray.chunksizes)

def transform(array):
    array._valid()
    array.check_whole_valid = False

    ChunkedArray = array.ChunkedArray
    VirtualArray = array.VirtualArray
    Table = array.Table

    stuff = [("run",                               "run",                                None),
             ("luminosityBlock",                   "lumi",                               None),
             ("event",                             "event",                              None),
             ("Electron_",                         "electrons",                          []),
             ("Muon_",                             "muons",                              []),
             ("Tau_",                              "taus",                               []),
             ("Photon_",                           "photons",                            []),
             ("Jet_",                              "jets",                               []),
             ("FatJet_",                           "fatjets",                            []),
             ("SubJet_",                           "subjets",                            []),
             ("IsoTrack_",                         "isotracks",                          []),
             ("SoftActivityJet_",                  "softjets",                           []),
             ("SoftActivityJetHT",                 "softactivity.HT",                    None),
             ("SoftActivityJetHT2",                "softactivity.HT2",                   None),
             ("SoftActivityJetHT5",                "softactivity.HT5",                   None),
             ("SoftActivityJetHT10",               "softactivity.HT10",                  None),
             ("SoftActivityJetNjets2",             "softactivity.njets2",                None),
             ("SoftActivityJetNjets5",             "softactivity.njets5",                None),
             ("SoftActivityJetNjets10",            "softactivity.njets10",               None),
             ("fixedGridRhoFastjetAll",            "fixedGridRhoFastjet.everything",     None),
             ("fixedGridRhoFastjetCentralCalo",    "fixedGridRhoFastjet.centralcalo",    None),
             ("fixedGridRhoFastjetCentralNeutral", "fixedGridRhoFastjet.centralneutral", None),
             ("MET_",                              "MET",                                Table.named("MET")),
             ("RawMET_",                           "rawMET",                             Table.named("RawMET")),
             ("CaloMET_",                          "caloMET",                            Table.named("CaloMET")),
             ("PuppiMET_",                         "puppiMET",                           Table.named("PuppiMET")),
             ("TkMET_",                            "tkMET",                              Table.named("TkMET")),
             ("PV_",                               "PV",                                 Table.named("PV")),
             ("SV_",                               "SVs",                                []),
             ("OtherPV_",                          "otherPVs",                           []),
             ("Pileup_",                           "pileup",                             Table.named("Pileup")),
             ("Flag_",                             "flags",                              Table.named("Flags")),
             ("TrigObj_",                          "trigobjs",                           []),
             ("HLT_",                              "HLT",                                Table.named("HLT")),
             ("HLTriggerFirstPath",                "HLT.firstpath",                      None),
             ("HLTriggerFinalPath",                "HLT.finalpath",                      None),
             ("Generator_",                        "gen",                                Table.named("Generator")),
             ("GenDressedLepton_",                 "gen.dressedleptons",                 []),
             ("GenPart_",                          "gen.partons",                        []),
             ("GenJet_",                           "gen.jets",                           []),
             ("GenJetAK8_",                        "gen.jetsAK8",                        []),
             ("SubGenJetAK8_",                     "gen.subjetsAK8",                     []),
             ("GenVisTau_",                        "gen.vistaus",                        []),
             ("GenMET_",                           "gen.MET",                            Table.named("GenMET")),
             ("LHE_",                              "gen.LHE",                            Table.named("LHE")),
             ("LHEPart_",                          "gen.LHEpartons",                     []),
             ("genWeight",                         "gen.genweight",                      None),
             ("LHEPdfWeight",                      "gen.LHEpdfweight",                   None),
             ("LHEScaleWeight",                    "gen.LHEscaleweight",                 None),
             ("LHEWeight_originalXWGTUP",          "gen.LHEweight_originalXWGTUP",       None),
             ]

    others = []
    for n in array.columns:
        for prefix, rename, data in stuff:
            if n.startswith(prefix):
                if data is None:
                    pass
                elif isinstance(data, list):
                    data.append((n[len(prefix):], array[n]))
                else:
                    data[n[len(prefix):]] = array[n]
                break
            elif n == "n" + prefix.rstrip("_"):
                break
        else:
            others.append(n)

    events = Table.named("Event")

    def makecollection(rename):
        if "." in rename:
            outer, inner = rename.split(".")
            if outer not in events.columns:
                events[outer] = Table.named(outer.capitalize())
            return events[outer], inner
        else:
            return events, rename

    for prefix, rename, data in stuff:
        if data is None:
            if prefix in array.columns:
                collection, rename = makecollection(rename)
                collection[rename] = array[prefix]
        elif isinstance(data, list):
            rowname = prefix[:-1]
            countname = "n" + rowname
            if len(data) > 0 and countname in array.columns:
                collection, rename = makecollection(rename)
                collection[rename] = lazyjagged(array[countname], rowname, data)
        else:
            if len(data.columns) > 0:
                collection, rename = makecollection(rename)
                collection[rename] = data

    eventtype = events.type
    eventtype.takes = array.type.takes

    eventtype.to["electrons"].to["jet"] = awkward.type.OptionType(eventtype.to["jets"].to)
    eventtype.to["electrons"].to["photon"] = awkward.type.OptionType(eventtype.to["photons"].to)
    for i, chunk in enumerate(events["electrons"].chunks):
        assert events["electrons"].chunksizes[i] == events["jets"].chunksizes[i] == events["photons"].chunksizes[i]
        events["electrons"].chunks[i] = VirtualArray(crossref, (chunk, [
            (events["jets"].chunks[i], array["Electron_jetIdx"].chunks[i], "jet", eventtype.to["electrons"].to["jet"]),
            (events["photons"].chunks[i], array["Electron_photonIdx"].chunks[i], "photon", eventtype.to["electrons"].to["photon"]),
            ]), type=awkward.type.ArrayType(eventtype.takes, eventtype.to["electrons"]), cache=chunk.cache, persistvirtual=chunk.persistvirtual)

    eventtype.to["muons"].to["jet"] = awkward.type.OptionType(eventtype.to["jets"].to)
    for i, chunk in enumerate(events["muons"].chunks):
        assert events["muons"].chunksizes[i] == events["jets"].chunksizes[i]
        events["muons"].chunks[i] = VirtualArray(crossref, (chunk, [
            (events["jets"].chunks[i], array["Muon_jetIdx"].chunks[i], "jet", eventtype.to["muons"].to["jet"]),
            ]), type=awkward.type.ArrayType(eventtype.takes, eventtype.to["muons"]), cache=chunk.cache, persistvirtual=chunk.persistvirtual)



# Muon_jetIdx
# Tau_jetIdx
# Photon_electronIdx
# Photon_jetIdx
# Jet_electronIdx1
# Jet_electronIdx2
# Jet_muonIdx1
# Jet_muonIdx2
# FatJet_subJetIdx1
# FatJet_subJetIdx2



    if len(others) > 0:
        etc = events["etc"] = Table.named("OtherFields")
        for n in others:
            etc[n] = array[n]
    events["raw"] = array

    return events

def crossref(fromarray, links):
    out = fromarray.array
    for toarray, localindex, name, totype in links:
        out.content[name] = out.VirtualArray(indexedmask, (toarray, localindex), type=awkward.type.ArrayType(out.offsets[-1], totype), cache=fromarray.cache, persistvirtual=fromarray.persistvirtual)
    return out

def indexedmask(toarray, localindex):
    jagged = toarray.array
    localindex = localindex.array
    globalindex = localindex + jagged.starts
    globalindex.content[localindex.content < 0] = -1
    return toarray.IndexedMaskedArray(globalindex.content, jagged.content)
