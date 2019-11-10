#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot-methods/blob/master/LICENSE

import numpy

import uproot_methods.base


class Methods(uproot_methods.base.ROOTMethods):
    @property
    def numbins(self):
        return self.xnumbins * self.ynumbins * self.znumbins

    @property
    def xnumbins(self):
        return self._fXaxis._fNbins

    @property
    def ynumbins(self):
        return self._fYaxis._fNbins

    @property
    def znumbins(self):
        return self._fZaxis._fNbins

    @property
    def low(self):
        return self.xlow, self.ylow, self.zlow

    @property
    def high(self):
        return self.xhigh, self.yhigh, self.zlow

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
    def zlow(self):
        return self._fZaxis._fXmin

    @property
    def zhigh(self):
        return self._fZaxis._fXmax

    @property
    def underflows(self):
        uf = numpy.array(self.allvalues)
        xuf = uf[0]
        yuf = uf[:, 0, :]
        zuf = uf[:, :, 0]
        return xuf, yuf, zuf

    @property
    def xunderflows(self):
        return self.underflows[0]

    @property
    def yunderflows(self):
        return self.underflows[1]

    @property
    def zunderflows(self):
        return self.underflows[2]

    @property
    def overflows(self):
        of = numpy.array(self.allvalues)
        xof = of[-1]
        yof = of[:, -1, :]
        zof = of[:, :, -1]
        return xof, yof, zof

    @property
    def xoverflows(self):
        return self.overflows[0]

    @property
    def yoverflows(self):
        return self.overflows[1]

    @property
    def zoverflows(self):
        return self.overflows[2]

    @property
    def edges(self):
        xaxis = self._fXaxis
        yaxis = self._fYaxis
        zaxis = self._fZaxis
        if len(getattr(xaxis, "_fXbins", [])) > 0:
            xedges = numpy.array(xaxis._fXbins)
        else:
            xedges = numpy.linspace(xaxis._fXmin, xaxis._fXmax,
                                    xaxis._fNbins + 1)

        if len(getattr(yaxis, "_fXbins", [])) > 0:
            yedges = numpy.array(yaxis._fXbins)
        else:
            yedges = numpy.linspace(yaxis._fXmin, yaxis._fXmax,
                                    yaxis._fNbins + 1)

        if len(getattr(zaxis, "_fXbins", [])) > 0:
            zedges = numpy.array(zaxis._fXbins)
        else:
            zedges = numpy.linspace(zaxis._fXmin, zaxis._fXmax,
                                    zaxis._fNbins + 1)

        return xedges, yedges, zedges

    @property
    def alledges(self):
        xedges, yedges, zedges = self.edges
        vx = numpy.empty(len(xedges) + 2)
        vy = numpy.empty(len(yedges) + 2)
        vz = numpy.empty(len(zedges) + 2)
        vx[0] = -numpy.inf
        vx[-1] = numpy.inf
        vx[1:-1] = xedges
        vy[0] = -numpy.inf
        vy[-1] = numpy.inf
        vy[1:-1] = yedges
        vz[0] = -numpy.inf
        vz[-1] = numpy.inf
        vz[1:-1] = zedges
        return vx, vy, vz

    @property
    def bins(self):
        xedges, yedges, zedges = self.edges
        vx = numpy.empty((len(xedges) - 1, 2))
        vy = numpy.empty((len(yedges) - 1, 2))
        vz = numpy.empty((len(zedges) - 1, 2))
        vx[:, 0] = xedges[:-1]
        vx[:, 1] = xedges[1:]
        vy[:, 0] = yedges[:-1]
        vy[:, 1] = yedges[1:]
        vz[:, 0] = zedges[:-1]
        vz[:, 1] = zedges[1:]
        return vx, vy, vz

    @property
    def allbins(self):
        xedges, yedges, zedges = self.alledges
        vx = numpy.empty((len(xedges) - 1, 2))
        vy = numpy.empty((len(yedges) - 1, 2))
        vz = numpy.empty((len(zedges) - 1, 2))
        vx[:, 0] = xedges[:-1]
        vx[:, 1] = xedges[1:]
        vy[:, 0] = yedges[:-1]
        vy[:, 1] = yedges[1:]
        vz[:, 0] = zedges[:-1]
        vz[:, 1] = zedges[1:]
        return vx, vy, vz

    @property
    def values(self):
        va = self.allvalues
        return va[1:self.xnumbins+1, 1:self.ynumbins+1, 1:self.znumbins+1]

    @property
    def allvalues(self):
        dtype = getattr(self, "_dtype", numpy.dtype(numpy.float64))
        v = numpy.array(self[:], dtype=dtype.newbyteorder("="))
        v = v.reshape(self.znumbins + 2, self.ynumbins + 2, self.xnumbins + 2)
        return v.T

    @property
    def variances(self):
        va = self.allvariances
        return va[1:self.xnumbins+1, 1:self.ynumbins+1, 1:self.znumbins+1]

    @property
    def allvariances(self):
        if len(getattr(self, "_fSumw2", [])) != len(self):
            v = numpy.array(self, dtype=numpy.float64)
        else:
            v = numpy.array(self._fSumw2, dtype=numpy.float64)
        v = v.reshape(self.znumbins + 2, self.ynumbins + 2, self.xnumbins + 2)
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
        elif axis == "z":
            low = self.zlow
            high = self.zhigh
            nbins = self.znumbins
            bins = self._fZaxis._fXbins
        else:
            raise ValueError("axis must be 'x','y' or 'z'")

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

    def zinterval(self, index):
        return self.interval(index, "z")

    @property
    def ylabels(self):
        if self._fYaxis._fLabels is None:
            return None
        else:
            return [str(x) for x in self._fYaxis._fLabels]

    @property
    def zlabels(self):
        if self._fZaxis._fLabels is None:
            return None
        else:
            return [str(x) for x in self._fZaxis._fLabels]
