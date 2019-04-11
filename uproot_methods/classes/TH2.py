#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot-methods/blob/master/LICENSE

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
        xuf = uf[0]
        yuf = uf[:, 0]
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
        xof = of[-1]
        yof = of[:, -1]
        return xof, yof

    @property
    def xoverflows(self):
        return self.overflows[0]

    @property
    def yoverflows(self):
        return self.overflows[1]

    @property
    def edges(self):
        xaxis = self._fXaxis
        yaxis = self._fYaxis
        if len(getattr(xaxis, "_fXbins", [])) > 0:
            xedges = numpy.array(xaxis._fXbins)
        else:
            xedges = numpy.linspace(xaxis._fXmin, xaxis._fXmax, xaxis._fNbins + 1)
        if len(getattr(yaxis, "_fXbins", [])) > 0:
            yedges = numpy.array(yaxis._fXbins)
        else:
            yedges = numpy.linspace(yaxis._fXmin, yaxis._fXmax, yaxis._fNbins + 1)
        return xedges, yedges

    @property
    def alledges(self):
        xedges, yedges = self.edges
        vx = numpy.empty(len(xedges) + 2)
        vy = numpy.empty(len(yedges) + 2)
        vx[0] = -numpy.inf
        vx[-1] = numpy.inf
        vx[1:-1] = xedges
        vy[0] = -numpy.inf
        vy[-1] = numpy.inf
        vy[1:-1] = yedges
        return vx, vy

    @property
    def bins(self):
        xedges, yedges = self.edges
        vx = numpy.empty((len(xedges) - 1, 2))
        vy = numpy.empty((len(yedges) - 1, 2))
        vx[:, 0] = xedges[:-1]
        vx[:, 1] = xedges[1:]
        vy[:, 0] = yedges[:-1]
        vy[:, 1] = yedges[1:]
        return vx, vy

    @property
    def allbins(self):
        xedges, yedges = self.alledges
        vx = numpy.empty((len(xedges) - 1, 2))
        vy = numpy.empty((len(yedges) - 1, 2))
        vx[:, 0] = xedges[:-1]
        vx[:, 1] = xedges[1:]
        vy[:, 0] = yedges[:-1]
        vy[:, 1] = yedges[1:]
        return vx, vy

    @property
    def values(self):
        va = self.allvalues
        return va[1:self.xnumbins+1, 1:self.ynumbins+1]

    @property
    def allvalues(self):
        dtype = getattr(self, "_dtype", numpy.dtype(numpy.float64))
        v = numpy.array(self[:], dtype=dtype.newbyteorder("="))
        v = v.reshape(self.ynumbins + 2, self.xnumbins + 2)
        return v.T

    @property
    def variances(self):
        va = self.allvariances
        return va[1:self.xnumbins+1, 1:self.ynumbins+1]

    @property
    def allvariances(self):
        if len(getattr(self, "_fSumw2", [])) != len(self):
            v = numpy.array(self, dtype=numpy.float64)
        else:
            v = numpy.array(self._fSumw2, dtype=numpy.float64)
        v = v.reshape(self.ynumbins + 2, self.xnumbins + 2)
        return v.T

    def numpy(self):
        return (self.values, [self.edges])

    def allnumpy(self):
        return (self.allvalues, [self.alledges])

    def interval(self, index, axis):
        if axis == "x":
            low = self.xlow
            high = self.xhigh
            nbins = self.xnumbins
            bins = self._fXaxis._fXbins
        elif axis == "y":
            low = self.ylow
            high = self.yhigh
            nbins = self.ynumbins
            bins = self._fYaxis._fXbins
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
                norm = float(high-low) / nbins
                xedges = (index-1)*norm + low, index*norm + low
            else:
                xedges = bins[index-1], bins[index]
            return xedges

    def xinterval(self, index):
        return self.interval(index, "x")

    def yinterval(self, index):
        return self.interval(index, "y")

    @property
    def ylabels(self):
        if self._fYaxis._fLabels is None:
            return None
        else:
            return [str(x) for x in self._fYaxis._fLabels]
