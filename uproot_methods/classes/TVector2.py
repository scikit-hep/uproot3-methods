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
import numbers

import awkward
import awkward.util

import uproot_methods.common.TVector
import uproot_methods.base
    
class Common(object):
    def dot(self, other):
        out = self.x*other.x
        out = out + self.y*other.y
        return out

    # TODO:
    # def _rotate(self, angle)

class ArrayMethods(Common, uproot_methods.common.TVector.ArrayMethods, uproot_methods.base.ROOTMethods):
    def _initObjectArray(self, table):
        awkward.ObjectArray.__init__(self, table, lambda row: TVector2(row["fX"], row["fY"]))
        self.content.rowname = "TVector2"

    @property
    def x(self):
        return self["fX"]

    @property
    def y(self):
        return self["fY"]

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented

        inputs = list(inputs)
        for i in range(len(inputs)):
            if isinstance(inputs[i], awkward.util.numpy.ndarray) and inputs[i].dtype == awkward.util.numpy.dtype(object) and len(inputs[i]) > 0:
                idarray = awkward.util.numpy.frombuffer(inputs[i], dtype=awkward.util.numpy.uintp)
                if (idarray == idarray[0]).all():
                    inputs[i] = inputs[i][0]

        if ufunc is awkward.util.numpy.add or ufunc is awkward.util.numpy.subtract:
            if not all(isinstance(x, (ArrayMethods, Methods)) for x in inputs):
                raise TypeError("(arrays of) TVector2 can only be added to/subtracted from other (arrays of) TVector2")
            out = self.empty_like()
            out["fX"] = getattr(ufunc, method)(*[x.x for x in inputs], **kwargs)
            out["fY"] = getattr(ufunc, method)(*[x.y for x in inputs], **kwargs)
            return out

        elif ufunc is awkward.util.numpy.power and len(inputs) >= 2 and isinstance(inputs[1], (numbers.Number, awkward.util.numpy.number)):
            if inputs[1] == 2:
                return self.mag2
            else:
                return self.mag2**(0.5*inputs[1])

        elif ufunc is awkward.util.numpy.absolute:
            return self.mag

        else:
            return awkward.ObjectArray.__array_ufunc__(self, ufunc, method, *inputs, **kwargs)

class Methods(Common, uproot_methods.common.TVector.Methods, uproot_methods.base.ROOTMethods):
    _arraymethods = ArrayMethods

    @property
    def x(self):
        return self._fX

    @property
    def y(self):
        return self._fY

    def __repr__(self):
        return "TVector2({0:.5g}, {1:.5g})".format(self.x, self.y)

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return isinstance(other, Methods) and self.x == other.x and self.y == other.y

    def _scalar(self, operator, scalar, reverse=False):
        if not isinstance(scalar, (numbers.Number, awkward.util.numpy.number)):
            raise TypeError("cannot {0} a TVector2 with a {1}".format(operator.__name__, type(scalar).__name__))
        if reverse:
            return TVector2(operator(scalar, self.x), operator(scalar, self.y))
        else:
            return TVector2(operator(self.x, scalar), operator(self.y, scalar))
        
    def _vector(self, operator, vector, reverse=False):
        if not isinstance(vector, Methods):
            raise TypeError("cannot {0} a TVector2 with a {1}".format(operator.__name__, type(vector).__name__))
        if reverse:
            return TVector2(operator(vector.x, self.x), operator(vector.y, self.y))
        else:
            return TVector2(operator(self.x, vector.x), operator(self.y, vector.y))

    def _unary(self, operator):
        return TVector2(operator(self.x), operator(self.y))

class TVector2Array(ArrayMethods, awkward.ObjectArray):
    def __init__(self, x, y):
        self._initObjectArray(awkward.Table())
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
