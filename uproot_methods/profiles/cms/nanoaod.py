#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot-methods/blob/master/LICENSE

import awkward.type
import awkward.array.chunked
import awkward.array.objects

import uproot_methods.classes.TLorentzVector

def getcontent(virtual):
    return virtual.array.content

def jaggedtable(rowname, counts, fields):
    Table = counts.Table
    JaggedArray = counts.JaggedArray
    ChunkedArray = counts.ChunkedArray
    VirtualArray = counts.VirtualArray
    VirtualTLorentzVectorArray = awkward.array.objects.Methods.mixin(uproot_methods.classes.TLorentzVector.PtEtaPhiMassArrayMethods, VirtualArray)

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
            table["p4"] = VirtualTLorentzVectorArray(uproot_methods.classes.TLorentzVector.TLorentzVectorArray.from_ptetaphim, (table["pt"], table["eta"], table["phi"], table["mass"]), type=awkward.type.ArrayType(offsets[-1], uproot_methods.classes.TLorentzVector.PtEtaPhiMassLorentzVectorArray), cache=counts.cache, persistvirtual=counts.persistvirtual)
        return JaggedArray.fromoffsets(offsets, table)

def lazyjagged(countsarray, rowname, fields):
    ChunkedArray = countsarray.ChunkedArray
    VirtualArray = countsarray.VirtualArray

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

def crossref(fromarray, links, subj):
    out = fromarray.array
    ChunkedArray = out.ChunkedArray
    VirtualArray = out.VirtualArray

    if isinstance(out, awkward.array.chunked.ChunkedArray):
        chunks = []
        for j, chunk in enumerate(out.chunks):
            newtype = awkward.type.ArrayType(out.chunksizes[j], float("inf"), awkward.type.TableType())
            for n in chunk.type.to.to.columns:
                newtype.to.to[n] = chunk.type.to.to[n]
            for collection, subname, i, localindex, name, totype in links:
                newtype.to.to[name] = totype
            chunks.append(VirtualArray(crossref, (chunk, links, j), type=newtype, cache=fromarray.cache, persistvirtual=fromarray.persistvirtual))

        return ChunkedArray(chunks, out.chunksizes)

    else:
        for collection, subname, i, localindex, name, totype in links:
            toarray = collection[subname].chunks[i]
            out.content[name] = VirtualArray(indexedmask, (toarray, localindex, subj), type=awkward.type.ArrayType(out.offsets[-1], totype), cache=fromarray.cache, persistvirtual=fromarray.persistvirtual)
        return out

def indexedmask(toarray, localindex, subj):
    jagged = toarray.array
    localindex = localindex.array
    if subj is not None:
        jagged = jagged.chunks[subj].array
        localindex = localindex.chunks[subj].array

    globalindex = localindex + jagged.starts
    globalindex.content[localindex.content < 0] = -1
    return toarray.IndexedMaskedArray(globalindex.content, jagged.content)

def transform(array):
    array._valid()
    array.check_whole_valid = False

    Table = array.Table
    VirtualArray = array.VirtualArray

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

    eventtype.to["electrons"].to["photon"] = awkward.type.OptionType(eventtype.to["photons"].to)
    eventtype.to["electrons"].to["photon"].check = False
    eventtype.to["electrons"].to["jet"] = awkward.type.OptionType(eventtype.to["jets"].to)
    eventtype.to["electrons"].to["jet"].check = False
    for i, chunk in enumerate(events["electrons"].chunks):
        assert events["electrons"].chunksizes[i] == events["jets"].chunksizes[i] == events["photons"].chunksizes[i]
        events["electrons"].chunks[i] = VirtualArray(crossref, (chunk, [
            (events, "photons", i, array["Electron_photonIdx"].chunks[i], "photon", eventtype.to["electrons"].to["photon"]),
            (events, "jets", i, array["Electron_jetIdx"].chunks[i], "jet", eventtype.to["electrons"].to["jet"]),
            ], None), type=awkward.type.ArrayType(events["electrons"].chunksizes[i], eventtype.to["electrons"]), cache=chunk.cache, persistvirtual=chunk.persistvirtual)

    eventtype.to["muons"].to["jet"] = awkward.type.OptionType(eventtype.to["jets"].to)
    eventtype.to["muons"].to["jet"].check = False
    for i, chunk in enumerate(events["muons"].chunks):
        assert events["muons"].chunksizes[i] == events["jets"].chunksizes[i]
        events["muons"].chunks[i] = VirtualArray(crossref, (chunk, [
            (events, "jets", i, array["Muon_jetIdx"].chunks[i], "jet", eventtype.to["muons"].to["jet"]),
            ], None), type=awkward.type.ArrayType(events["muons"].chunksizes[i], eventtype.to["muons"]), cache=chunk.cache, persistvirtual=chunk.persistvirtual)

    eventtype.to["taus"].to["jet"] = awkward.type.OptionType(eventtype.to["jets"].to)
    eventtype.to["taus"].to["jet"].check = False
    for i, chunk in enumerate(events["taus"].chunks):
        assert events["taus"].chunksizes[i] == events["jets"].chunksizes[i]
        events["taus"].chunks[i] = VirtualArray(crossref, (chunk, [
            (events, "jets", i, array["Tau_jetIdx"].chunks[i], "jet", eventtype.to["taus"].to["jet"]),
            ], None), type=awkward.type.ArrayType(events["jets"].chunksizes[i], eventtype.to["taus"]), cache=chunk.cache, persistvirtual=chunk.persistvirtual)

    eventtype.to["taus"].to["jet"] = awkward.type.OptionType(eventtype.to["jets"].to)
    eventtype.to["taus"].to["jet"].check = False
    for i, chunk in enumerate(events["taus"].chunks):
        assert events["taus"].chunksizes[i] == events["jets"].chunksizes[i]
        events["taus"].chunks[i] = VirtualArray(crossref, (chunk, [
            (events, "jets", i, array["Tau_jetIdx"].chunks[i], "jet", eventtype.to["taus"].to["jet"]),
            ], None), type=awkward.type.ArrayType(events["taus"].chunksizes[i], eventtype.to["taus"]), cache=chunk.cache, persistvirtual=chunk.persistvirtual)

    eventtype.to["photons"].to["electron"] = awkward.type.OptionType(eventtype.to["electrons"].to)
    eventtype.to["photons"].to["electron"].check = False
    eventtype.to["photons"].to["jet"] = awkward.type.OptionType(eventtype.to["jets"].to)
    eventtype.to["photons"].to["jet"].check = False
    for i, chunk in enumerate(events["photons"].chunks):
        assert events["photons"].chunksizes[i] == events["jets"].chunksizes[i] == events["electrons"].chunksizes[i]
        events["photons"].chunks[i] = VirtualArray(crossref, (chunk, [
            (events, "electrons", i, array["Photon_electronIdx"].chunks[i], "electron", eventtype.to["photons"].to["electron"]),
            (events, "jets", i, array["Photon_jetIdx"].chunks[i], "jet", eventtype.to["photons"].to["jet"]),
            ], None), type=awkward.type.ArrayType(events["photons"].chunksizes[i], eventtype.to["photons"]), cache=chunk.cache, persistvirtual=chunk.persistvirtual)

    eventtype.to["jets"].to["electron1"] = awkward.type.OptionType(eventtype.to["electrons"].to)
    eventtype.to["jets"].to["electron1"].check = False
    eventtype.to["jets"].to["electron2"] = awkward.type.OptionType(eventtype.to["electrons"].to)
    eventtype.to["jets"].to["electron2"].check = False
    eventtype.to["jets"].to["muon1"] = awkward.type.OptionType(eventtype.to["muons"].to)
    eventtype.to["jets"].to["muon1"].check = False
    eventtype.to["jets"].to["muon2"] = awkward.type.OptionType(eventtype.to["muons"].to)
    eventtype.to["jets"].to["muon2"].check = False
    for i, chunk in enumerate(events["jets"].chunks):
        assert events["jets"].chunksizes[i] == events["electrons"].chunksizes[i] == events["muons"].chunksizes[i]
        events["jets"].chunks[i] = VirtualArray(crossref, (chunk, [
            (events, "electrons", i, array["Jet_electronIdx1"].chunks[i], "electron1", eventtype.to["jets"].to["electron1"]),
            (events, "electrons", i, array["Jet_electronIdx2"].chunks[i], "electron2", eventtype.to["jets"].to["electron2"]),
            (events, "muons", i, array["Jet_muonIdx1"].chunks[i], "muon1", eventtype.to["jets"].to["muon1"]),
            (events, "muons", i, array["Jet_muonIdx2"].chunks[i], "muon2", eventtype.to["jets"].to["muon2"]),
            ], None), type=awkward.type.ArrayType(events["jets"].chunksizes[i], eventtype.to["jets"]), cache=chunk.cache, persistvirtual=chunk.persistvirtual)

    eventtype.to["fatjets"].to["subjet1"] = awkward.type.OptionType(eventtype.to["jets"].to)
    eventtype.to["fatjets"].to["subjet1"].check = False
    eventtype.to["fatjets"].to["subjet2"] = awkward.type.OptionType(eventtype.to["jets"].to)
    eventtype.to["fatjets"].to["subjet2"].check = False
    for i, chunk in enumerate(events["fatjets"].chunks):
        assert events["fatjets"].chunksizes[i] == events["jets"].chunksizes[i]
        events["fatjets"].chunks[i] = VirtualArray(crossref, (chunk, [
            (events, "jets", i, array["FatJet_subJetIdx1"].chunks[i], "subjet1", eventtype.to["fatjets"].to["subjet1"]),
            (events, "jets", i, array["FatJet_subJetIdx2"].chunks[i], "subjet2", eventtype.to["fatjets"].to["subjet2"]),
            ], None), type=awkward.type.ArrayType(events["fatjets"].chunksizes[i], eventtype.to["fatjets"]), cache=chunk.cache, persistvirtual=chunk.persistvirtual)

    if len(others) > 0:
        etc = events["etc"] = Table.named("OtherFields")
        for n in others:
            etc[n] = array[n]
    events["raw"] = array

    return events
