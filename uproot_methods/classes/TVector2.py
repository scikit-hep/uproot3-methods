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

import math

import awkward
import awkward.util

import uproot_methods.common.TVector
import uproot_methods.base
    
class Common(object):
    def dot(self, other):
        out = self.x * other.x
        out += self.y * other.y
        return out

class ArrayMethods(uproot_methods.base.ROOTMethods, Common, uproot_methods.common.TVector.ArrayMethods, awkward.ObjectArray):
    @property
    def x(self):
        return self["fX"]

    @property
    def y(self):
        return self["fY"]

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented

        if ufunc is awkward.util.numpy.add or ufunc is awkward.util.numpy.subtract:
            if not all(isinstance(x, (ArrayMethods, Methods)) for x in inputs):
                raise TypeError("(arrays of) TVector2 can only be added to/subtracted from other (arrays of) TVector2")
            out = self.empty_like()
            out["fX"] = getattr(ufunc, method)(*[x.x for x in inputs], **kwargs)
            out["fY"] = getattr(ufunc, method)(*[x.y for x in inputs], **kwargs)
            return out

        elif ufunc is awkward.util.numpy.absolute:
            return self.mag()

        else:
            return awkward.ObjectArray.__array_ufunc__(self, ufunc, method, *inputs, **kwargs)

class Methods(uproot_methods.base.ROOTMethods, Common, uproot_methods.common.TVector.Methods):
    _arraymethods = ArrayMethods

    @property
    def x(self):
        return self._fX

    @property
    def y(self):
        return self._fY

    def __repr__(self):
        return "TVector2({0:.4g}, {1:.4g})".format(self.x, self.y)

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return isinstance(other, Methods) and self.x == other.x and self.y == other.y

    def _scalar(self, operator, scalar):
        return TVector2(operator(self.x, scalar), operator(self.y, scalar))

    def _vector(self, operator, vector):
        if not isinstance(vector, Methods):
            raise TypeError("cannot {0} a TVector2 with a {1}".format(operator.__name__, type(vector).__name__))
        return TVector2(operator(self.x, vector.x), operator(self.y, vector.y))

class TVector2Array(ArrayMethods):
    def __init__(self, x, y):
        super(TVector2Array, self).__init__(awkward.Table(), lambda row: TVector2(row["fX"], row["fY"]))
        self.content.rowname = "TVector2"
        self["fX"] = x
        self["fY"] = y

    @classmethod
    def origin(cls, shape, dtype=None):
        if dtype is None:
            dtype = awkward.util.numpy.float64
        return cls(awkward.util.numpy.zeros(shape, dtype=dtype), awkward.util.numpy.zeros(shape, dtype=dtype))

    @classmethod
    def origin_like(cls, array):
        return cls.origin(array.shape, array.dtype)

    @classmethod
    def from_circular(cls, rho, phi):
        return cls(rho * awkward.util.numpy.cos(phi),
                   rho * awkward.util.numpy.sin(phi))

    @property
    def x(self):
        return self["fX"]

    @x.setter
    def x(self, value):
        self["fX"] = value

    @property
    def y(self):
        return self["fY"]

    @y.setter
    def y(self, value):
        self["fY"] = value

class TVector2(Methods):
    def __init__(self, x, y):
        self._fX = x
        self._fY = y

    @classmethod
    def origin(cls):
        return cls(0.0, 0.0)

    @classmethod
    def from_circular(cls, rho, phi):
        return cls(rho * math.cos(phi),
                   rho * math.sin(phi))

    @property
    def x(self):
        return self._fX

    @x.setter
    def x(self, value):
        self._fX = value

    @property
    def y(self):
        return self._fY

    @y.setter
    def y(self, value):
        self._fY = value
