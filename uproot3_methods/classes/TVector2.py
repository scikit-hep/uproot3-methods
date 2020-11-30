#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot3-methods/blob/master/LICENSE

import math
import numbers

import awkward0.array.jagged
import awkward0.util

import uproot3_methods.common.TVector
import uproot3_methods.base

class Common(object):
    def dot(self, other):
        out = self.x*other.x
        out = out + self.y*other.y
        return out

    def _rotate(self, angle):
        c = self.awkward0.numpy.cos(angle)
        s = self.awkward0.numpy.sin(angle)
        x = self.x*c - self.y*s
        y = self.x*s + self.y*c
        return x, y

class ArrayMethods(Common, uproot3_methods.common.TVector.ArrayMethods, uproot3_methods.base.ROOTMethods):
    def _initObjectArray(self, table):
        self.awkward0.ObjectArray.__init__(self, table, lambda row: TVector2(row["fX"], row["fY"]))

    def __awkward_serialize__(self, serializer):
        self._valid()
        x, y = self.x, self.y
        return serializer.encode_call(
            ["uproot3_methods.classes.TVector2", "TVector2Array", "from_cartesian"],
            serializer(x, "TVector3Array.x"),
            serializer(y, "TVector3Array.y"))

    @property
    def x(self):
        return self["fX"]

    @property
    def y(self):
        return self["fY"]

    def rotate(self, angle):
        x, y = self._rotate(angle)
        out = self.empty_like()
        out["fX"] = x
        out["fY"] = y
        return out

    def sum(self):
        if isinstance(self, self.awkward0.JaggedArray):
            return TVector2Array.from_cartesian(self.x.sum(), self.y.sum())
        else:
            return TVector2(self.x.sum(), self.y.sum())

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if "out" in kwargs:
            raise NotImplementedError("in-place operations not supported")

        if method != "__call__":
            return NotImplemented

        inputs = list(inputs)
        for i in range(len(inputs)):
            if isinstance(inputs[i], self.awkward0.numpy.ndarray) and inputs[i].dtype == self.awkward0.numpy.dtype(object) and len(inputs[i]) > 0:
                idarray = self.awkward0.numpy.frombuffer(inputs[i], dtype=self.awkward0.numpy.uintp)
                if (idarray == idarray[0]).all():
                    inputs[i] = inputs[i][0]

        if ufunc is self.awkward0.numpy.add or ufunc is self.awkward0.numpy.subtract:
            if not all(isinstance(x, (ArrayMethods, Methods)) for x in inputs):
                raise TypeError("(arrays of) TVector2 can only be added to/subtracted from other (arrays of) TVector2")
            out = self.empty_like()
            out["fX"] = getattr(ufunc, method)(*[x.x for x in inputs], **kwargs)
            out["fY"] = getattr(ufunc, method)(*[x.y for x in inputs], **kwargs)
            return out

        elif ufunc is self.awkward0.numpy.power and len(inputs) >= 2 and isinstance(inputs[1], (numbers.Number, self.awkward0.numpy.number)):
            if inputs[1] == 2:
                return self.mag2
            else:
                return self.mag2**(0.5*inputs[1])

        elif ufunc is self.awkward0.numpy.absolute:
            return self.mag

        else:
            return super(ArrayMethods, self).__array_ufunc__(ufunc, method, *inputs, **kwargs)

JaggedArrayMethods = ArrayMethods.mixin(ArrayMethods, awkward0.JaggedArray)

class Methods(Common, uproot3_methods.common.TVector.Methods, uproot3_methods.base.ROOTMethods):
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
        if not isinstance(scalar, (numbers.Number, self.awkward0.numpy.number)):
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

    def rotate(self, angle):
        x, y = self._rotate(angle)
        return TVector2(x, y)

class TVector2Array(ArrayMethods, uproot3_methods.base.ROOTMethods.awkward0.ObjectArray):

    def __init__(self, x, y):
        if isinstance(x, awkward0.array.jagged.JaggedArray) or isinstance(y, awkward0.array.jagged.JaggedArray):
            raise TypeError("TVector2Array constructor arguments must not be jagged; use TVector2.from_cartesian for jaggedness-handling")
        self._initObjectArray(self.awkward0.Table())
        self["fX"] = x
        self["fY"] = y

    @classmethod
    def origin(cls, shape, dtype=None):
        if dtype is None:
            dtype = cls.awkward0.numpy.float64
        return cls(cls.awkward0.numpy.zeros(shape, dtype=dtype), cls.awkward0.numpy.zeros(shape, dtype=dtype))

    @classmethod
    def origin_like(cls, array):
        return array * 0.0

    @classmethod
    @awkward0.util.wrapjaggedmethod(JaggedArrayMethods)
    def from_cartesian(cls, x, y):
        return cls(x, y)

    @classmethod
    @awkward0.util.wrapjaggedmethod(JaggedArrayMethods)
    def from_polar(cls, rho, phi):
        return cls(rho * cls.awkward0.numpy.cos(phi),
                   rho * cls.awkward0.numpy.sin(phi))

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
        self._fX = float(x)
        self._fY = float(y)

    @classmethod
    def origin(cls):
        return cls(0.0, 0.0)

    @classmethod
    def from_polar(cls, rho, phi):
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
