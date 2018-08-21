#!/usr/bin/env python

# Copyright (c) 2018, DIANA-HEP
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# 
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys

import numpy

import uproot_methods.base

class Methods(uproot_methods.base.ROOTMethods):
    def __repr__(self):
        if self._fName is None:
            return "<{0} at 0x{1:012x}>".format(self._classname, id(self))
        else:
            return "<{0} {1} 0x{2:012x}>".format(self._classname, repr(self._fName), id(self))

    @property
    def name(self):
        return self._fName

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
    def values(self):
        return self[1:-1]

    @property
    def allvalues(self):
        return self[:]

    @property
    def numpy(self):
        low = self._fXaxis._fXmin
        high = self._fXaxis._fXmax
        norm = (high - low) / self._fXaxis._fNbins
        freq = numpy.array(self.values, dtype=self._dtype.newbyteorder("="))
        edges = numpy.array([i*norm + low for i in range(self.numbins + 1)])
        return freq, edges

    def interval(self, index):
        if index < 0:
            index += len(self)

        low = self._fXaxis._fXmin
        high = self._fXaxis._fXmax
        if index == 0:
            return (float("-inf"), low)
        elif index == len(self) - 1:
            return (high, float("inf"))
        else:
            norm = (high - low) / self._fXaxis._fNbins
            return (index - 1)*norm + low, index*norm + low

    def index(self, data):
        low = self._fXaxis._fXmin
        high = self._fXaxis._fXmax
        if data < low:
            return 0
        elif data >= high:
            return len(self) - 1
        elif not math.isnan(data):
            return int(math.floor(self._fXaxis._fNbins * (data - low) / (high - low))) + 1

    def fill(self, datum):
        numbins = self._fXaxis._fNbins
        low = self._fXaxis._fXmin
        high = self._fXaxis._fXmax
        if datum < low:
            self[0] += 1
        elif datum >= high:
            self[-1] += 1
        else:
            self[int(math.floor(numbins * (datum - low) / (high - low))) + 1] += 1

    def fillw(self, datum, weight):
        numbins = self._fXaxis._fNbins
        low = self._fXaxis._fXmin
        high = self._fXaxis._fXmax
        if datum < low:
            self[0] += weight
        elif datum >= high:
            self[-1] += weight
        else:
            self[int(math.floor(numbins * (datum - low) / (high - low))) + 1] += weight

    def fillall(self, data):
        numbins = self._fXaxis._fNbins
        low = self._fXaxis._fXmin
        high = self._fXaxis._fXmax

        if not isinstance(data, numpy.ndarray):
            data = numpy.array(data)

        freq, edges = numpy.histogram(data, bins=numbins, range=(low, high), density=False)
        for i, x in enumerate(freq):
            self[i + 1] += x

        self[0] += (data < low).sum()
        self[-1] += (data >= high).sum()

    def fillallw(self, data, weights):
        numbins = self._fXaxis._fNbins
        low = self._fXaxis._fXmin
        high = self._fXaxis._fXmax

        if not isinstance(data, numpy.ndarray):
            data = numpy.array(data)

        if isinstance(weights, numbers.Real):
            weights = numpy.empty_like(data)

        freq, edges = numpy.histogram(data, bins=numbins, range=(low, high), weights=weights, density=False)
        for i, x in enumerate(freq):
            self[i + 1] += x

        self[0] += weights[data < low].sum()
        self[-1] += weights[data >= high].sum()

    def __add__(self, other):
        if not isinstance(other, TH1Methods) or self.numbins != other.numbins or self.low != other.low or self.high != other.high:
            raise TypeError("TH1 histograms can only be combined with other TH1 histograms with the same binning")
        return hist(self.numbins, self.low, self.high, name=(self.name if self.name is not None else other.name), title=(self.title if self.title is not None else other.title), allvalues=(numpy.array(self) + numpy.array(other)))

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if not isinstance(other, TH1Methods) or self.numbins != other.numbins or self.low != other.low or self.high != other.high:
            raise TypeError("TH1 histograms can only be combined with other TH1 histograms with the same binning")
        for i in range(len(self)):
            self[i] = other[i]
        return self

    @property
    def xlabels(self):
        if self._fXaxis._fLabels is None:
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
