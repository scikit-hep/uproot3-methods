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

import numbers
import sys

import numpy

import uproot_methods.base

class Methods(uproot_methods.base.ROOTMethods):
    @property
    def numbins(self):
        return self.xnumbins * self.ynumbins

    @property
    def xnumbins(self):
        return self._fXaxis._fNbins

    @property
    def ynumbins(self):
        return self._fYaxis._fNbins

    @property
    def low(self):
        return self.xlow, self.ylow

    @property
    def high(self):
        return self.xhigh, self.yhigh

    @property
    def xlow(self):
        return self._fXaxis._fXmin

    @property
    def xhigh(self):
        return self._fXaxis._fXmax

    @property
    def ylow(self):
        return self._fYaxis._fXmin

    @property
    def yhigh(self):
        return self._fYaxis._fXmax

    @property
    def underflows(self):
        uf = numpy.array(self.allvalues)
        xuf = uf[:,0]
        yuf = uf[0]
        return xuf, yuf

    @property
    def xunderflows(self):
        return self.underflows[0]

    @property
    def yunderflows(self):
        return self.underflows[1]

    @property
    def overflows(self):
        of = numpy.array(self.allvalues)
        xof = of[:,-1]
        yof = of[-1]
        return xof, yof

    @property
    def xoverflows(self):
        return self.overflows[0]

    @property
    def yoverflows(self):
        return self.overflows[1]

    @property
    def values(self):
        va = numpy.array(self.allvalues)
        va = va[1:self.ynumbins+1, 1:self.xnumbins+1]
        return va.tolist()

    @property
    def allvalues(self):
        v = numpy.array(self[:])
        v = v.reshape(self.ynumbins + 2, self.xnumbins + 2)
        return v.tolist()

    def numpy(self):
        xlow  = self.xlow
        xhigh = self.xhigh
        xbins = self._fXaxis._fXbins
        if not xbins:
            norm   = float(xhigh - xlow) / self.xnumbins
            xedges = numpy.array([i*norm + xlow for i in range(self.xnumbins + 1)])
        else:
            xedges = numpy.array(xbins)

        ylow  = self.ylow
        yhigh = self.yhigh
        ybins = self._fYaxis._fXbins
        if not ybins:
            norm   = (yhigh - ylow) / self.ynumbins
            yedges = numpy.array([i*norm + ylow for i in range(self.ynumbins + 1)])
        else:
            yedges = numpy.array(ybins)

        freq = numpy.array(self.values, dtype=self._dtype.newbyteorder("="))

        return freq, (xedges, yedges)

    def interval(self, index, axis):
        if axis == "x":
            low   = self.xlow
            high  = self.xhigh
            nbins = self.xnumbins
            bins  = self._fXaxis._fXbins
        elif axis == "y":
            low   = self.ylow
            high  = self.yhigh
            nbins = self.ynumbins
            bins  = self._fYaxis._fXbins
        else:
            raise ValueError("axis must be 'x' or 'y'")

        if index < 0:
            index += nbins

        if index == 0:
            return float("-inf"), low
        elif index == nbins + 1:
            return high, float("inf")
        else:
            if not bins:
                norm   = float(high-low) / nbins
                xedges = (index-1)*norm + low, index*norm + low
            else:
                xedges = bins[index-1], bins[index]
            return xedges

    def xinterval(self, index):
        return self.interval(index, "x")

    def yinterval(self, index):
        return self.interval(index, "y")

    def index(self, data, axis):
        if axis == "x":
            ind = 0
            nbins = self.xnumbins
        elif axis == "y":
            ind = 1
            nbins = self.ynumbins
        else:
            raise TypeError("Axis must be either 'x' or 'y' to obtain an index.")

        low  = self.low[ind]
        high = self.high[ind]

        if data < low:
            return 0
        elif data >= high:
            return nbins+1
        elif not math.isnan(data):
            return int(math.floor(nbins*(data-low) / (high-low))) + 1

    def xindex(self, data):
        return self.index(data, "x")

    def yindex(self, data):
        return self.index(data, "y")

    @property
    def ylabels(self):
        if self._fYaxis._fLabels is None:
            return None
        else:
            return [str(x) for x in self._fYaxis._fLabels]
