#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot-methods/blob/master/LICENSE

import numbers
import math
import sys

import numpy

import uproot_methods.base

class Methods(uproot_methods.base.ROOTMethods):
    def __repr__(self):
        if self.name is None:
            return "<{0} at 0x{1:012x}>".format(self._classname, id(self))
        else:
            return "<{0} {1} 0x{2:012x}>".format(self._classname, repr(self._fName), id(self))

    @property
    def name(self):
        return getattr(self, "_fName", None)

    @property
    def title(self):
        return self._fTitle

    @property
    def numbins(self):
        return self._fXaxis._fNbins

    @property
    def low(self):
        return self._fXaxis._fXmin

    @property
    def high(self):
        return self._fXaxis._fXmax

    @property
    def underflows(self):
        return self[0]

    @property
    def overflows(self):
        return self[-1]

    @property
    def edges(self):
        axis = self._fXaxis
        if len(getattr(axis, "_fXbins", [])) > 0:
            return numpy.array(axis._fXbins)
        else:
            return numpy.linspace(axis._fXmin, axis._fXmax, axis._fNbins + 1)

    @property
    def alledges(self):
        axis = self._fXaxis
        v = numpy.empty(axis._fNbins + 3)
        v[0] = -numpy.inf
        v[-1] = numpy.inf
        v[1:-1] = self.edges
        return v

    @property
    def bins(self):
        edges = self.edges
        out = numpy.empty((len(edges) - 1, 2))
        out[:, 0] = edges[:-1]
        out[:, 1] = edges[1:]
        return out

    @property
    def allbins(self):
        edges = self.alledges
        out = numpy.empty((len(edges) - 1, 2))
        out[:, 0] = edges[:-1]
        out[:, 1] = edges[1:]
        return out

    @property
    def values(self):
        return self.allvalues[1:-1]

    @property
    def allvalues(self):
        return numpy.array(self, dtype=getattr(self, "_dtype", numpy.dtype(numpy.float64)).newbyteorder("="))

    @property
    def variances(self):
        return self.allvariances[1:-1]

    @property
    def allvariances(self):
        if len(getattr(self, "_fSumw2", [])) != len(self):
            return numpy.array(self, dtype=numpy.float64)
        else:
            return numpy.array(self._fSumw2, dtype=numpy.float64)

    def numpy(self):
        return self.values, self.edges

    def allnumpy(self):
        return self.allvalues, self.alledges

    def interval(self, index):
        if index < 0:
            index += len(self)

        low = self._fXaxis._fXmin
        high = self._fXaxis._fXmax
        if index == 0:
            return (float("-inf"), low)
        elif index == len(self) - 1:
            return (high, float("inf"))
        elif len(getattr(self._fXaxis, "_fXbins", [])) == self._fXaxis._fNbins + 1:
            return (self._fXaxis._fXbins[index - 1], self._fXaxis._fXbins[index])
        else:
            norm = (high - low) / self._fXaxis._fNbins
            return (index - 1)*norm + low, index*norm + low

    @property
    def xlabels(self):
        if getattr(self._fXaxis, "_fLabels", None) is None:
            return None
        else:
            return [str(x) for x in self._fXaxis._fLabels]

    def show(self, width=80, minimum=None, maximum=None, stream=sys.stdout):
        if minimum is None:
            minimum = min(self)
            if minimum < 0:
                minimum *= 1.05
            else:
                minimum = 0

        if maximum is None:
            maximum = max(self) * 1.05

        if maximum <= minimum:
            average = (minimum + maximum) / 2.0
            minimum = average - 0.5
            maximum = average + 0.5

        if self.xlabels is None:
            intervals = ["[{0:<.5g}, {1:<.5g})".format(l, h) for l, h in [self.interval(i) for i in range(len(self))]]
            intervals[-1] = intervals[-1][:-1] + "]"   # last interval is closed on top edge
        else:
            intervals = ["(underflow)"] + [self.xlabels[i] if i < len(self.xlabels) else self.interval(i+1) for i in range(self.numbins)] + ["(overflow)"]

        intervalswidth = max(len(x) for x in intervals)

        values = ["{0:<.5g}".format(float(x)) for x in self]
        valueswidth = max(len(x) for x in values)

        minimumtext = "{0:<.5g}".format(minimum)
        maximumtext = "{0:<.5g}".format(maximum)

        plotwidth = max(len(minimumtext) + len(maximumtext), width - (intervalswidth + 1 + valueswidth + 1 + 2))
        scale = minimumtext + " "*(plotwidth + 2 - len(minimumtext) - len(maximumtext)) + maximumtext

        norm = float(plotwidth) / float(maximum - minimum)
        zero = int(round((0.0 - minimum)*norm))
        line = numpy.empty(plotwidth, dtype=numpy.uint8)

        formatter = "{0:<%s} {1:<%s} |{2}|" % (intervalswidth, valueswidth)
        line[:] = ord("-")
        if minimum != 0 and 0 <= zero < plotwidth:
            line[zero] = ord("+")
        capstone = " " * (intervalswidth + 1 + valueswidth + 1) + "+" + str(line.tostring().decode("ascii")) + "+"

        out = [" "*(intervalswidth + valueswidth + 2) + scale]
        out.append(capstone)
        for interval, value, x in zip(intervals, values, self):
            line[:] = ord(" ")

            pos = int(round((x - minimum)*norm))
            if x < 0:
                line[pos:zero] = ord("*")
            else:
                line[zero:pos] = ord("*")

            if minimum != 0 and 0 <= zero < plotwidth:
                line[zero] = ord("|")

            out.append(formatter.format(interval, value, str(line.tostring().decode("ascii"))))

        out.append(capstone)
        out = "\n".join(out)
        if stream is None:
            return out
        else:
            stream.write(out)
            stream.write("\n")

    def pandas(self, underflow=True, overflow=True, variance=True):

        if not underflow and not overflow:
            s = slice(1,-1)
        elif not underflow:
            s = slice(1, None)
        elif not overflow:
            s = slice(None,-1)
        else:
            s = slice(None)

        def get_name(obj):
            if getattr(obj, "_fTitle", b"") == b"":
                return None
            else:
                return obj._fTitle.decode("utf-8", "ignore")

        allbins = self.allbins
        if type(allbins) != tuple:
            allbins = (allbins,)

        dim = len(allbins)

        if dim == 1:
            index_name_objects = (self,)
        elif dim == 2:
            index_name_objects = (self._fXaxis, self._fYaxis)
        elif dim == 3:
            index_name_objects = (self._fXaxis, self._fYaxis, self._fZaxis)
        else:
            raise NotImplementedError

        data = {"count" : self.allvalues[(s,)*dim].flatten()}
        columns = ["count"]

        if variance:
            data["variance"] = self.allvariances[(s,)*dim].flatten()
            columns += ["variance"]

        import pandas

        intervals = ((pandas.Interval(a,b, closed="left") for a, b in bins[s]) for bins in allbins)
        names = (get_name(obj) for obj in index_name_objects)
        index = pandas.MultiIndex.from_product(intervals, names=names)

        return pandas.DataFrame(index=index, data=data, columns=columns)

    def physt(self):
        import physt.binnings
        import physt.histogram1d
        freq = numpy.array(self.allvalues, dtype=getattr(self, "_dtype", numpy.dtype(numpy.float64)).newbyteorder("="))
        if getattr(self._fXaxis, "_fXbins", None):
            binning = physt.binnings.NumpyBinning(numpy.array(self._fXaxis._fXbins))
        else:
            low = self._fXaxis._fXmin
            high = self._fXaxis._fXmax
            binwidth = (high - low) / self._fXaxis._fNbins
            binning = physt.binnings.FixedWidthBinning(binwidth, bin_count=self._fXaxis._fNbins, min=low)
        return physt.histogram1d.Histogram1D(
            binning,
            frequencies=freq[1:-1],
            underflow=freq[0],
            overflow=freq[-1],
            name=getattr(self, "_fTitle", b"").decode("utf-8", "ignore"))

    def hepdata(self, independent={"name": None, "units": None}, dependent={"name": "counts", "units": None}, qualifiers=[], yamloptions={}):
        if independent["name"] is None and getattr(self, "_fTitle", b""):
            independent = dict(independent)
            if isinstance(self._fTitle, bytes):
                independent["name"] = self._fTitle.decode("utf-8", "ignore")
            else:
                independent["name"] = self._fTitle

        if getattr(self._fXaxis, "_fXbins", None):
            independent_values = [{"low": float(low), "high": float(high)} for low, high in zip(self._fXaxis._fXbins[:-1], self._fXaxis._fXbins[1:])]
        else:
            low = self._fXaxis._fXmin
            high = self._fXaxis._fXmax
            norm = (high - low) / self._fXaxis._fNbins
            independent_values = [{"low": float(i*norm + low), "high": float((i + 1)*norm + low)} for i in range(self.numbins)]

        if getattr(self, "_fSumw2", None):
            dependent_values = [{"value": float(value), "errors": [{"symerror": math.sqrt(variance), "label": "stat"}]} for value, variance in zip(self.values, self.variances)]
        else:
            dependent_values = [{"value": float(value), "errors": [{"symerror": math.sqrt(value), "label": "stat"}]} for value in self.values]

        out = {"independent_variables": [{"header": independent, "values": independent_values}], "dependent_variables": [{"header": dependent, "qualifiers": qualifiers, "values": dependent_values}]}

        if yamloptions is None:
            return out
        else:
            import yaml
            return yaml.dump(out, **yamloptions)

def _histtype(content):
    if issubclass(content.dtype.type, (numpy.bool_, numpy.bool)):
        return b"TH1C", content.astype(">i1")
    elif issubclass(content.dtype.type, numpy.int8):
        return b"TH1C", content.astype(">i1")
    elif issubclass(content.dtype.type, numpy.uint8) and content.max() <= numpy.iinfo(numpy.int8).max:
        return b"TH1C", content.astype(">i1")
    elif issubclass(content.dtype.type, numpy.uint8):
        return b"TH1S", content.astype(">i2")
    elif issubclass(content.dtype.type, numpy.int16):
        return b"TH1S", content.astype(">i2")
    elif issubclass(content.dtype.type, numpy.uint16) and content.max() <= numpy.iinfo(numpy.int16).max:
        return b"TH1S", content.astype(">i2")
    elif issubclass(content.dtype.type, numpy.uint16):
        return b"TH1I", content.astype(">i4")
    elif issubclass(content.dtype.type, numpy.int32):
        return b"TH1I", content.astype(">i4")
    elif issubclass(content.dtype.type, numpy.uint32) and content.max() <= numpy.iinfo(numpy.int32).max:
        return b"TH1I", content.astype(">i4")
    elif issubclass(content.dtype.type, numpy.integer) and numpy.iinfo(numpy.int32).min <= content.min() and content.max() <= numpy.iinfo(numpy.int32).max:
        return b"TH1I", content.astype(">i4")
    elif issubclass(content.dtype.type, numpy.float32):
        return b"TH1F", content.astype(">f4")
    else:
        return b"TH1D", content.astype(">f8")

def from_numpy(histogram):
    content, edges = histogram[:2]

    class TH1(Methods, list):
        pass

    class TAxis(object):
        def __init__(self, edges):
            self._fNbins = len(edges) - 1
            self._fXmin = edges[0]
            self._fXmax = edges[-1]
            if numpy.array_equal(edges, numpy.linspace(self._fXmin, self._fXmax, len(edges), dtype=edges.dtype)):
                self._fXbins = numpy.array([], dtype=">f8")
            else:
                self._fXbins = edges.astype(">f8")

    out = TH1.__new__(TH1)
    out._fXaxis = TAxis(edges)

    centers = (edges[:-1] + edges[1:]) / 2.0
    out._fEntries = out._fTsumw = out._fTsumw2 = content.sum()
    out._fTsumwx = (content * centers).sum()
    out._fTsumwx2 = (content * centers**2).sum()

    if len(histogram) >= 3:
        out._fTitle = histogram[2]
    else:
        out._fTitle = b""

    out._classname, content = _histtype(content)

    valuesarray = numpy.empty(len(content) + 2, dtype=content.dtype)
    valuesarray[1:-1] = content
    valuesarray[0] = 0
    valuesarray[-1] = 0

    out.extend(valuesarray)
    out._fSumw2 = valuesarray ** 2

    return out

def from_pandas(histogram):
    import pandas

    histogram = histogram.sort_index(ascending=True, inplace=False)
    if not histogram.index.is_non_overlapping_monotonic:
        raise ValueError("intervals overlap; cannot form a histogram")

    sparse = histogram.index[numpy.isfinite(histogram.index.left) & numpy.isfinite(histogram.index.right)]
    if (sparse.right[:-1] == sparse.left[1:]).all():
        dense = sparse
    else:
        pairs = numpy.empty(len(sparse) * 2, dtype=numpy.float64)
        pairs[::2] = sparse.left
        pairs[1::2] = sparse.right
        nonempty = numpy.empty(len(pairs), dtype=numpy.bool_)
        nonempty[:-1] = (pairs[1:] != pairs[:-1])
        nonempty[-1] = True
        dense = pandas.IntervalIndex.from_breaks(pairs[nonempty], closed="left")

    densehist = pandas.DataFrame(index=dense.left).join(histogram.reindex(histogram.index.left))
    densehist.fillna(0, inplace=True)

    underflowhist = histogram[numpy.isinf(histogram.index.left)]
    overflowhist = histogram[numpy.isinf(histogram.index.right)]

    content = numpy.array(densehist["count"])

    sumw2 = numpy.empty(len(content) + 2, dtype=numpy.float64)
    if "variance" in densehist.columns:
        sumw2source = "variance"
    else:
        sumw2source = "count"
    sumw2[1:-1] = densehist[sumw2source]
    if len(underflowhist) == 0:
        sumw2[0] = 0
    else:
        sumw2[0] = underflowhist[sumw2source]
    if len(overflowhist) == 0:
        sumw2[-1] = 0
    else:
        sumw2[-1] = overflowhist[sumw2source]

    edges = numpy.empty(len(densehist) + 1, dtype=numpy.float64)
    edges[:-1] = dense.left
    edges[-1] = dense.right[-1]

    class TH1(Methods, list):
        pass

    class TAxis(object):
        def __init__(self, fNbins, fXmin, fXmax):
            self._fNbins = fNbins
            self._fXmin = fXmin
            self._fXmax = fXmax

    out = TH1.__new__(TH1)
    out._fXaxis = TAxis(len(edges) - 1, edges[0], edges[-1])
    out._fXaxis._fXbins = edges

    centers = (edges[:-1] + edges[1:]) / 2.0
    out._fEntries = content.sum()
    out._fTsumw = content.sum()
    out._fTsumw2 = sumw2.sum()
    out._fTsumwx = (content * centers).sum()
    out._fTsumwx2 = (content * centers**2).sum()

    if histogram.index.name is None:
        out._fTitle = b""
    elif isinstance(histogram.index.name, bytes):
        out._fTitle = histogram.index.name
    else:
        out._fTitle = histogram.index.name.encode("utf-8", "ignore")

    out._classname, content = _histtype(content)

    valuesarray = numpy.empty(len(content) + 2, dtype=content.dtype)
    valuesarray[1:-1] = content
    if len(underflowhist) == 0:
        valuesarray[0] = 0
    else:
        valuesarray[0] = underflowhist["count"]
    if len(overflowhist) == 0:
        valuesarray[-1] = 0
    else:
        valuesarray[-1] = overflowhist["count"]

    out.extend(valuesarray)

    return out

def from_physt(histogram):
    import physt.binnings
    import physt.histogram1d

    class TH1(Methods, list):
        pass

    class TAxis(object):
        def __init__(self, fNbins, fXmin, fXmax):
            self._fNbins = fNbins
            self._fXmin = fXmin
            self._fXmax = fXmax

    out = TH1.__new__(TH1)

    if isinstance(histogram.binning, physt.binnings.FixedWidthBinning):
        out._fXaxis = TAxis(histogram.binning.bin_count,
                            histogram.binning.first_edge,
                            histogram.binning.last_edge)
    elif isinstance(histogram.binning, physt.binnings.NumpyBinning):
        out._fXaxis = TAxis(histogram.binning.bin_count,
                            histogram.binning.first_edge,
                            histogram.binning.last_edge)
        if not histogram.binning.is_regular():
            out._fXaxis._fXbins = histogram.binning.numpy_bins.astype(">f8")
    else:
        raise NotImplementedError(histogram.binning)

    centers = histogram.bin_centers
    content = histogram.frequencies

    out._fSumw2 = [0] + list(histogram.errors2) + [0]

    mean = histogram.mean()
    variance = histogram.variance()
    out._fEntries = content.sum()   # is there a #entries independent of weights?
    out._fTsumw = content.sum()
    out._fTsumw2 = histogram.errors2.sum()
    if mean is None:
        out._fTsumwx = (content * centers).sum()
    else:
        out._fTsumwx = mean * out._fTsumw
    if mean is None or variance is None:
        out._fTsumwx2 = (content * centers**2).sum()
    else:
        out._fTsumwx2 = (mean**2 + variance) * out._fTsumw2

    if histogram.name is not None:
        out._fTitle = histogram.name
    else:
        out._fTitle = b""

    out._classname, content = _histtype(content)

    valuesarray = numpy.empty(len(content) + 2, dtype=content.dtype)
    valuesarray[1:-1] = content
    valuesarray[0] = histogram.underflow
    valuesarray[-1] = histogram.overflow

    out.extend(valuesarray)

    return out
