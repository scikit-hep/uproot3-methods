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
import operator

import awkward
import awkward.util

import uproot_methods.base
import uproot_methods.common.TVector
import uproot_methods.classes.TVector3

class Common(object):
    @property
    def E(self):
        return self.t

    def dot(self, other):
        out = self.t*other.t
        out = out - self.x*other.x
        out = out - self.y*other.y
        out = out - self.z*other.z
        return out

    @property
    def energy(self):
        return self.t

    @property
    def p(self):
        return self.p3.mag

    @property
    def p2(self):
        return self.p3.mag2

    @property
    def perp2(self):
        return self.p3.rho2

    @property
    def perp(self):
        return self.p3.rho

    @property
    def pt2(self):
        return self.p3.rho2

    @property
    def pt(self):
        return self.p3.rho

    @property
    def Et(self):
        return self.energy * self.pt / self.p

    @property
    def mag2(self):
        return self.dot(self)

    @property
    def mass2(self):
        return self.mag2

    @property
    def mass(self):
        return self.mag

    @property
    def mt2(self):
        return self.energy**2 - self.z**2
        
    @property
    def phi(self):
        return self.p3.phi

    @property
    def theta(self):
        return self.p3.theta

    @property
    def cottheta(self):
        return self.p3.cottheta

    @property
    def beta(self):
        return self.p / self.energy

    def delta_phi(self, other):
        return (self.phi - other.phi + math.pi) % (2*math.pi) - math.pi

    def delta_r2(self, other):
        return (self.eta - other.eta)**2 + self.delta_phi(other)**2

    def _rotate_axis(self, axis, angle):
        if not isinstance(axis, uproot_methods.classes.TVector3.Common):
            raise TypeError("axis must be an (array of) TVector3")
        p3 = self.p3._rotate_axis(axis, angle)
        return p3, self.t

    def _rotate_euler(self, phi, theta, psi):
        return self.p3._rotate_euler(phi, theta, psi), self.t

    def rotatex(self, angle):
        return self.rotate_axis(TVector3(1.0, 0.0, 0.0), angle)

    def rotatey(self, angle):
        return self.rotate_axis(TVector3(0.0, 1.0, 0.0), angle)

    def rotatez(self, angle):
        return self.rotate_axis(TVector3(0.0, 0.0, 1.0), angle)

    def isspacelike(self, tolerance=1e-10):
        return self.mag2 < -tolerance

    def istimelike(self, tolerance=1e-10):
        return self.mag2 > tolerance

    def __lt__(self, other):
        raise TypeError("Lorentz vectors have no natural ordering")

    def __gt__(self, other):
        raise TypeError("Lorentz vectors have no natural ordering")

    def __le__(self, other):
        raise TypeError("Lorentz vectors have no natural ordering")

    def __ge__(self, other):
        raise TypeError("Lorentz vectors have no natural ordering")

class ArrayMethods(Common, uproot_methods.base.ROOTMethods):
    def _initObjectArray(self, table):
        awkward.ObjectArray.__init__(self, table, lambda row: TLorentzVector(row["fX"], row["fY"], row["fZ"], row["fE"]))
        self.content.rowname = "TLorentzVector"

    @property
    def p3(self):
        out = self.empty_like(generator=lambda row: uproot_methods.classes.TVector3.TVector3(row["fX"], row["fY"], row["fZ"]))
        if isinstance(self, awkward.JaggedArray):
            out.__class__ = type("JaggedArray", (awkward.JaggedArray, uproot_methods.classes.TVector3.ArrayMethods), {})
        else:
            out.__class__ = uproot_methods.classes.TVector3.ArrayMethods
        out["fX"] = self.x
        out["fY"] = self.y
        out["fZ"] = self.z
        return out

    @property
    def x(self):
        return self["fX"]

    @property
    def y(self):
        return self["fY"]

    @property
    def z(self):
        return self["fZ"]

    @property
    def t(self):
        return self["fE"]

    @property
    def mag(self):
        return awkward.util.numpy.sqrt(self.mag2)

    @property
    def mt(self):
        mt2 = self.mt2
        sign = awkward.util.numpy.sign(mt2)
        return awkward.util.numpy.sqrt(awkward.util.numpy.absolute(mt2)) * sign

    @property
    def eta(self):
        return -awkward.util.numpy.log((1.0 - awkward.util.numpy.cos(self.theta)) / (1.0 + awkward.util.numpy.cos(self.theta))) / 2.0

    @property
    def rapidity(self):
        return awkward.util.numpy.log((self.t + self.z) / (self.t - self.z)) / 2.0

    @property
    def unit(self):
        return self / awkward.util.numpy.sqrt(self.mag)

    @property
    def boostp3(self):
        out = self.empty_like(generator=lambda row: uproot_methods.classes.TVector3.TVector3(row["fX"], row["fY"], row["fZ"]))
        if isinstance(self, awkward.JaggedArray):
            out.__class__ = type("JaggedArray", (awkward.JaggedArray, uproot_methods.classes.TVector3.ArrayMethods), {})
        else:
            out.__class__ = uproot_methods.classes.TVector3.ArrayMethods
        out["fX"] = self.x / self.t
        out["fY"] = self.y / self.t
        out["fZ"] = self.z / self.t
        return out

    def boost(self, p3):
        if not isinstance(p3, (uproot_methods.classes.TVector3.ArrayMethods, uproot_methods.classes.TVector3.Methods)):
            raise TypeError("boost p3 must be an (array of) TVector3")

        b2 = p3.mag2
        gamma = (1 - b2)**(-0.5)
        gamma2 = awkward.util.numpy.zeros(b2.shape, dtype=awkward.util.numpy.float64)
        mask = (b2 != 0)
        gamma2[mask] = (gamma[mask] - 1) / b2[mask]
        del mask

        bp = self.p3.dot(p3)
        v = self.p3 + gamma2*bp*p3 + gamma*p3*self.t
        out = self.empty_like()
        out["fX"] = v.x
        out["fY"] = v.y
        out["fZ"] = v.z
        out["fE"] = gamma*(self.t + bp)
        return out

    @property
    def gamma(self):
        out = self.beta
        mask = (out < 1) & (out > -1)
        out[mask] = (1 - out[mask]**2)**(-0.5)
        out[~mask] = awkward.util.numpy.inf
        return out

    def delta_r(self, other):
        return awkward.util.numpy.sqrt(self.delta_r2(other))

    def rotate_axis(self, axis, angle):
        p3, t = self._rotate_axis(axis, angle)
        out = self.empty_like()
        out["fX"] = p3.x
        out["fY"] = p3.y
        out["fZ"] = p3.z
        out["fE"] = t
        return out

    def rotate_euler(self, phi=0, theta=0, psi=0):
        p3, t = self._rotate_euler(phi, theta, psi)
        out = self.empty_like()
        out["fX"] = p3.x
        out["fY"] = p3.y
        out["fZ"] = p3.z
        out["fE"] = t
        return out

    def islightlike(self, tolerance=1e-10):
        return awkward.util.numpy.absolute(self.mag2) < tolerance

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            raise NotImplemented

        inputs = list(inputs)
        for i in range(len(inputs)):
            if isinstance(inputs[i], awkward.util.numpy.ndarray) and inputs[i].dtype == awkward.util.numpy.dtype(object) and len(inputs[i]) > 0:
                idarray = awkward.util.numpy.frombuffer(inputs[i], dtype=awkward.util.numpy.uintp)
                if (idarray == idarray[0]).all():
                    inputs[i] = inputs[i][0]

        if ufunc is awkward.util.numpy.add or ufunc is awkward.util.numpy.subtract:
            if not all(isinstance(x, (ArrayMethods, Methods)) for x in inputs):
                raise TypeError("(arrays of) TLorentzVector can only be added to/subtracted from other (arrays of) TLorentzVector")
            out = self.empty_like()
            out["fX"] = getattr(ufunc, method)(*[x.x for x in inputs], **kwargs)
            out["fY"] = getattr(ufunc, method)(*[x.y for x in inputs], **kwargs)
            out["fZ"] = getattr(ufunc, method)(*[x.z for x in inputs], **kwargs)
            out["fE"] = getattr(ufunc, method)(*[x.t for x in inputs], **kwargs)
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

class Methods(Common, uproot_methods.base.ROOTMethods):
    _arraymethods = ArrayMethods

    @property
    def p3(self):
        return self._fP

    @property
    def x(self):
        return self._fP._fX

    @property
    def y(self):
        return self._fP._fY

    @property
    def z(self):
        return self._fP._fZ

    @property
    def t(self):
        return self._fE

    def __repr__(self):
        return "TLorentzVector({0:.5g}, {1:.5g}, {2:.5g}, {3:.5g})".format(self._fP._fX, self._fP._fY, self._fP._fZ, self._fE)

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return isinstance(other, Methods) and self.x == other.x and self.y == other.y and self.z == other.z and self.t == other.t

    def _scalar(self, operator, scalar, reverse=False):
        if not isinstance(scalar, (numbers.Number, awkward.util.numpy.number)):
            raise TypeError("cannot {0} a TLorentzVector with a {1}".format(operator.__name__, type(scalar).__name__))
        if reverse:
            return TLorentzVector(operator(scalar, self.x), operator(scalar, self.y), operator(scalar, self.z), operator(scalar, self.t))
        else:
            return TLorentzVector(operator(self.x, scalar), operator(self.y, scalar), operator(self.z, scalar), operator(self.t, scalar))

    def _vector(self, operator, vector, reverse=False):
        if not isinstance(vector, Methods):
            raise TypeError("cannot {0} a TLorentzVector with a {1}".format(operator.__name__, type(vector).__name__))
        if reverse:
            return TLorentzVector(operator(vector.x, self.x), operator(vector.y, self.y), operator(vector.z, self.z), operator(vector.t, self.t))
        else:
            return TLorentzVector(operator(self.x, vector.x), operator(self.y, vector.y), operator(self.z, vector.z), operator(self.t, vector.t))

    def _unary(self, operator):
        return TLorentzVector(operator(self.x), operator(self.y), operator(self.z), operator(self.t))

    @property
    def mag(self):
        return math.sqrt(self.mag2)

    @property
    def mt(self):
        out = self.mt2
        if out >= 0:
            return math.sqrt(out)
        else:
            return -math.sqrt(out)

    @property
    def eta(self):
        return -math.log((1.0 - math.cos(self.theta)) / (1.0 + math.cos(self.theta)))/2.0

    @property
    def rapidity(self):
        return math.log((self.t + self.z) / (self.t - self.z)) / 2.0

    @property
    def unit(self):
        return self / math.sqrt(self.mag)

    @property
    def boostp3(self):
        return uproot_methods.classes.TVector3.TVector3(self.x/self.t, self.y/self.t, self.z/self.t)

    def boost(self, p3):
        if not isinstance(p3, uproot_methods.classes.TVector3.Methods):
            raise TypeError("boost p3 must be a TVector3")

        b2 = p3.mag2
        gamma = (1.0 - b2)**(-0.5)
        if b2 != 0:
            gamma2 = (gamma - 1.0) / b2
        else:
            gamma2 = 0.0

        bp = self.p3.dot(p3)
        v = self.p3 + gamma2*bp*p3 + gamma*p3*self.t
        return self.__class__(v.x, v.y, v.z, gamma*(self.t + bp))

    @property
    def gamma(self):
        out = self.beta
        if -1 < out < 1:
            return (1 - out**2)**(-0.5)
        else:
            return float("inf")

    def delta_r(self, other):
        return math.sqrt(self.delta_r2(other))

    def rotate_axis(self, axis, angle):
        p3, t = self._rotate_axis(axis, angle)
        return self.__class__(p3.x, p3.y, p3.z, t)

    def rotate_euler(self, phi=0, theta=0, psi=0):
        p3, t = self._rotate_euler(phi, theta, psi)
        return self.__class__(p3.x, p3.y, p3.z, t)

    def islightlike(self, tolerance=1e-10):
        return abs(self.mag2) < tolerance

    def __add__(self, other):
        return self._vector(operator.add, other)

    def __radd__(self, other):
        return self._vector(operator.add, other, True)

    def __sub__(self, other):
        return self._vector(operator.sub, other)

    def __rsub__(self, other):
        return self._vector(operator.sub, other, True)

    def __mul__(self, other):
        return self._scalar(operator.mul, other)

    def __rmul__(self, other):
        return self._scalar(operator.mul, other, True)

    def __div__(self, other):
        return self._scalar(operator.div, other)

    def __rdiv__(self, other):
        return self._scalar(operator.div, other, True)

    def __truediv__(self, other):
        return self._scalar(operator.truediv, other)

    def __rtruediv__(self, other):
        return self._scalar(operator.truediv, other, True)

    def __floordiv__(self, other):
        return self._scalar(operator.floordiv, other)

    def __rfloordiv__(self, other):
        return self._scalar(operator.floordiv, other, True)

    def __mod__(self, other):
        return self._scalar(operator.mod, other)

    def __rmod__(self, other):
        return self._scalar(operator.mod, other, True)

    def __divmod__(self, other):
        return self._scalar(operator.divmod, other)

    def __rdivmod__(self, other):
        return self._scalar(operator.divmod, other, True)

    def __pow__(self, other):
        if isinstance(other, (numbers.Number, awkward.util.numpy.number)):
            if other == 2:
                return self.mag2
            else:
                return self.mag2**(0.5*other)
        else:
            self._scalar(operator.pow, other)

    # no __rpow__

    def __lshift__(self, other):
        return self._scalar(operator.lshift, other)

    def __rlshift__(self, other):
        return self._scalar(operator.lshift, other, True)

    def __rshift__(self, other):
        return self._scalar(operator.rshift, other)

    def __rrshift__(self, other):
        return self._scalar(operator.rshift, other, True)

    def __and__(self, other):
        return self._scalar(operator.and_, other)

    def __rand__(self, other):
        return self._scalar(operator.and_, other, True)

    def __or__(self, other):
        return self._scalar(operator.or_, other)

    def __ror__(self, other):
        return self._scalar(operator.or_, other, True)

    def __xor__(self, other):
        return self._scalar(operator.xor, other)

    def __rxor__(self, other):
        return self._scalar(operator.xor, other, True)

    def __neg__(self):
        return self._unary(operator.neg)

    def __pos__(self):
        return self._unary(operator.pos)

    def __abs__(self):
        return self.mag

    def __invert__(self):
        return self._unary(operator.invert)

class TLorentzVectorArray(ArrayMethods, awkward.ObjectArray):
    def __init__(self, x, y, z, t):
        self._initObjectArray(awkward.Table())
        self["fX"] = x
        self["fY"] = y
        self["fZ"] = z
        self["fE"] = t

    @classmethod
    def origin(cls, shape, dtype=None):
        if dtype is None:
            dtype = awkward.util.numpy.float64
        return cls(awkward.util.numpy.zeros(shape, dtype=dtype),
                   awkward.util.numpy.zeros(shape, dtype=dtype),
                   awkward.util.numpy.zeros(shape, dtype=dtype),
                   awkward.util.numpy.zeros(shape, dtype=dtype))

    @classmethod
    def origin_like(cls, array):
        return cls.origin(array.shape, array.dtype)

    @classmethod
    def from_p3(cls, p3, t):
        out = cls.__new__(cls)
        out["fX"] = p3.x
        out["fY"] = p3.y
        out["fZ"] = p3.z
        out["fE"] = t
        return out

    @classmethod
    def from_spherical(cls, r, theta, phi, t):
        return cls.from_p3(uproot_methods.classes.TVector3.TVector3Array.from_spherical(r, theta, phi), t)

    @classmethod
    def from_cylindrical(cls, rho, phi, z, t):
        return cls.from_p3(uproot_methods.classes.TVector3.TVector3Array.from_cylindrical(rho, phi, z), t)

    @classmethod
    def from_xyzm(cls, x, y, z, m):
        return cls(x, y, z, awkward.util.numpy.sqrt(x*x + y*y + z*z + m*m*awkward.util.numpy.sign(m)))

    @classmethod
    def from_ptetaphi(cls, pt, eta, phi, energy):
        return cls(pt * awkward.util.numpy.cos(phi),
                   pt * awkward.util.numpy.sin(phi),
                   pt * awkward.util.numpy.sinh(eta),
                   energy)

    @classmethod
    def from_ptetaphim(cls, pt, eta, phi, mass):
        x = pt * awkward.util.numpy.cos(phi),
        y = pt * awkward.util.numpy.sin(phi),
        z = pt * awkward.util.numpy.sinh(eta)
        p3 = uproot_methods.classes.TVector3.TVector3Array(x, y, z)
        return cls.from_p3(p3, awkward.util.numpy.sqrt(x*x + y*y + z*z + m*m*awkward.util.numpy.sign(m)))

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

    @property
    def z(self):
        return self["fZ"]

    @z.setter
    def z(self, value):
        self["fZ"] = value

    @property
    def t(self):
        return self["fE"]

    @t.setter
    def t(self, value):
        self["fE"] = value

    @property
    def E(self):
        return self["fE"]

    @E.setter
    def E(self, value):
        self["fE"] = value

class TLorentzVector(Methods):
    def __init__(self, x, y, z, t):
        self._fP = uproot_methods.classes.TVector3.TVector3(x, y, z)
        self._fE = t

    @classmethod
    def origin(cls):
        return cls(0.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_p3(cls, p3, t):
        out = cls.__new__(cls)
        out._fP = p3
        out._fE = t
        return out

    @classmethod
    def from_spherical(cls, r, theta, phi, t):
        return cls.from_p3(uproot_methods.classes.TVector3.Methods.from_spherical(r, theta, phi), t)

    @classmethod
    def from_cylindrical(cls, rho, phi, z, t):
        return cls.from_p3(uproot_methods.classes.TVector3.Methods.from_cylindrical(rho, phi, z), t)

    @classmethod
    def from_xyzm(cls, x, y, z, m):
        return cls(x, y, z, math.sqrt(x*x + y*y + z*z + m*m*(1 if m >= 0 else -1)))

    @classmethod
    def from_ptetaphi(cls, pt, eta, phi, energy):
        return cls(pt * math.cos(phi),
                   pt * math.sin(phi),
                   pt * math.sinh(eta),
                   energy)
    
    @classmethod
    def from_ptetaphim(cls, pt, eta, phi, mass):
        tmp = cls.from_ptetaphi(pt, eta, phi, 0)
        return cls.from_xyzm(tmp.x, tmp.y, tmp.z, mass)

    @property
    def x(self):
        return self._fP._fX

    @x.setter
    def x(self, value):
        self._fP._fX = value

    @property
    def y(self):
        return self._fP._fY

    @y.setter
    def y(self, value):
        self._fP._fY = value

    @property
    def z(self):
        return self._fP._fZ

    @z.setter
    def z(self, value):
        self._fP._fZ = value

    @property
    def t(self):
        return self._fE

    @t.setter
    def t(self, value):
        self._fE = value

    @property
    def E(self):
        return self._fE

    @E.setter
    def E(self, value):
        self._fE = value
