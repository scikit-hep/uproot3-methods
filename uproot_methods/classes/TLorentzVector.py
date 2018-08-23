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

import uproot_methods.base
import uproot_methods.classes.TVector3

class Common(object):
    @property
    def E(self):
        return self.t

    @energy.setter
    def E(self, value):
        self.t = value

    def dot(self, other):
        out = self.t * other.t
        out -= self.x * other.x
        out -= self.y * other.y
        out -= self.z * other.z
        return out

    def energy(self):
        return self.t

    def p(self):
        return self.vect.mag()

    def p2(self):
        return self.vect.mag2()

    def pt2(self):
        return self.vect.rho2()

    def pt(self):
        return self.vect.rho()

    def Et(self):
        return self.energy() * self.pt() / self.p()

    def mass2(self):
        return self.mag2()

    def mass(self):
        return self.mag()

    def mt2(self):
        return self.energy()**2 - self.z**2
        
    def phi(self):
        return self.vect.phi()

    def theta(self):
        return self.vect.theta()

    def cottheta(self):
        return self.vect.cottheta()

    def beta(self):
        return self.p() / self.energy()

    def delta_r2(self):
        return (self.eta() - other.eta())**2 + self.delta_phi(other)**2

    def _rotate_axis(self, axis, angle):
        if isinstance(axis, uproot_methods.classes.TVector3.Common):
            vect = self.vect._rotate_axis(axis, angle)
            return vect, self.t
        else:
            vect = self.vect._rotate_axis(axis.vect, angle)
            return vect, self.t

    def _rotate_euler(self, phi, theta, psi):
        return self.vect._rotate_euler(phi, theta, psi), self.t

    def rotatex(self, angle):
        return self.rotate_axis(TVector3(1.0, 0.0, 0.0), angle)

    def rotatey(self, angle):
        return self.rotate_axis(TVector3(0.0, 1.0, 0.0), angle)

    def rotatez(self, angle):
        return self.rotate_axis(TVector3(0.0, 0.0, 1.0), angle)

    def isspacelike(self, tolerance=1e-10):
        return self.mag2() < -tolerance

    def istimelike(self, tolerance=1e-10):
        return self.mag2() > tolerance

class ArrayMethods(uproot_methods.base.ROOTMethods, Common):
    @property
    def vect(self):
        out = self._vect.empty_like()
        out["fX"] = self.x
        out["fY"] = self.y
        out["fZ"] = self.z
        return out

    @vect.setter
    def vect(self, value):
        self._vect = value

    @property
    def x(self):
        return self._vect["fX"]

    @x.setter
    def x(self, value):
        self._vect["fX"] = value

    @property
    def y(self):
        return self._vect["fY"]

    @y.setter
    def y(self, value):
        self._vect["fY"] = value

    @property
    def z(self):
        return self._vect["fZ"]

    @z.setter
    def z(self, value):
        self._vect["fZ"] = value

    def __getitem__(self, where):
        if awkward.util.isstringslice(where):
            if where == "fX" or where == "fY" or where == "fZ":
                return self._vect[where]
            else:
                return self[where]
        else:
            return super(ArrayMethods, self).__getitem__(where)

    def __setitem__(self, where, what):
        if awkward.util.isstringslice(where):
            if where == "fX" or where == "fY" or where == "fZ":
                self._vect[where] = what
            elif isinstance(where, awkward.util.string):
                self[where] = what
            else:
                if len(where) != len(what):
                    raise ValueError("number of keys ({0}) does not match number of provided arrays ({1})".format(len(where), len(what)))
                for x, y in zip(where, what):
                    if x == "fX" or x == "fY" or x == "fZ":
                        self._vect[x] = y
                    else:
                        self[x] = y
        else:
            super(ArrayMethods, self).__setitem__(where, what)
        
    @property
    def t(self):
        return self["fE"]

    @e.setter
    def t(self, value):
        self["fE"] = value

    def mt(self):
        mt2 = self.mt2()
        sign = awkward.util.numpy.sign(mt2)
        return awkward.util.numpy.sqrt(awkward.util.numpy.absolute(mt2)) * sign

    def eta(self):
        return -awkward.util.numpy.log((1.0 - awkward.util.numpy.cos(self.theta())) / (1.0 + awkward.util.numpy.cos(self.theta()))) / 2.0

    def rapidity(self):
        return awkward.util.numpy.log((self.t + self.z) / (self.t - self.z)) / 2.0

    def boost_vector(self):
        out = self._vect.empty_like()
        out["fX"] = self.x / self.t
        out["fY"] = self.y / self.t
        out["fZ"] = self.z / self.t
        return out

    def boost(self, vect, inplace=False):
        b2 = vect.mag2()
        gamma = (1 - b2)**(-0.5)
        gamma2 = awkward.util.numpy.zeros(b2.shape, dtype=awkward.util.numpy.float64)
        mask = (b2 != 0)
        gamma2[mask] = (gamma[mask] - 1) / b2[mask]
        del mask

        bp = self.vect.dot(vect)
        if inplace:
            self.vect += gamma2*bp*vect + gamma*vect*self.t
            self.t += pt
            self.t *= gamma
        else:
            v = self.vect + gamma2*bp*vect + gamma*vect*self.t
            out = self.empty_like()
            out._vect = self._vect.empty_like()
            out._vect["fX"] = v.x
            out._vect["fY"] = v.y
            out._vect["fZ"] = v.z
            out["fE"] = gamma*(self.t + bp)
            return out

    def gamma(self):
        out = self.beta()
        mask = (out < 1) & (out > -1)
        out[mask] = (1 - out[mask]**2)**(-0.5)
        out[~mask] = awkward.util.numpy.inf
        return out

    def delta_r(self):
        return awkward.util.numpy.sqrt(self.delta_r2())

    def rotate_axis(self, axis, angle):
        vect, t = self._rotate_axis(axis, angle)
        out = self.empty_like()
        out._vect = vect
        out["fE"] = t
        return out

    def rotate_euler(self, phi=0, theta=0, psi=0):
        vect, t = self._rotate_euler(phi, theta, psi)
        out = self.empty_like()
        out._vect = vect
        out["fE"] = t
        return out

    def islightlike(self, tolerance=1e-10):
        return awkward.util.numpy.absolute(self.mag2()) < tolerance

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            raise NotImplemented

        inputsx = []
        inputsy = []
        inputsz = []
        inputst = []
        for obj in inputs:
            if isinstance(obj, ArrayMethods):
                inputsx.append(obj.x)
                inputsy.append(obj.y)
                inputsz.append(obj.z)
                inputst.append(obj.t)
            else:
                inputsx.append(obj)
                inputsy.append(obj)
                inputsz.append(obj)
                inputst.append(obj)

        resultx = getattr(ufunc, method)(*inputsx, **kwargs)
        resulty = getattr(ufunc, method)(*inputsy, **kwargs)
        resultz = getattr(ufunc, method)(*inputsz, **kwargs)
        resultt = getattr(ufunc, method)(*inputst, **kwargs)

        if isinstance(resultx, tuple) and isinstance(resulty, tuple) and isinstance(resultz, tuple) and isinstance(resultt, tuple):
            out = []
            for x, y, z, t in zip(resultx, resulty, resultz, resultt):
                out.append(self.empty_like())
                out[-1]._vect = self._vect.empty_like()
                out[-1]._vect["fX"] = x
                out[-1]._vect["fY"] = y
                out[-1]._vect["fZ"] = z
                out[-1]["fE"] = t
            return tuple(out)

        elif method == "at":
            return None

        else:
            out = self.empty_like()
            out._vect = self._vect.empty_like()
            out._vect["fX"] = resultx
            out._vect["fY"] = resulty
            out._vect["fZ"] = resultz
            out["fE"] = resultt
            return out

class Methods(uproot_methods.base.ROOTMethods, Common):
    _arraymethods = ArrayMethods

    @property
    def vect(self):
        return self._fP

    @vect.setter
    def vect(self, value):
        self._fP = value

    @property
    def x(self):
        return self.p.x

    @x.setter
    def x(self, value):
        self.p.x = value

    @property
    def y(self):
        return self.p.y

    @y.setter
    def y(self, value):
        self.p.y = value

    @property
    def z(self):
        return self.p.z

    @z.setter
    def z(self, value):
        self.p.z = value

    @property
    def t(self):
        return self._fE

    @e.setter
    def t(self, value):
        self._fE = value

    def mt(self):
        out = self.mt2()
        if out >= 0:
            return math.sqrt(out)
        else:
            return -math.sqrt(out)

    def eta(self):
        return -math.log((1.0 - math.cos(self.theta())) / (1.0 + math.cos(self.theta())))/2.0

    def rapidity(self):
        return math.log((self.t + self.z) / (self.t - self.z)) / 2.0

    def boost_vector(self):
        return uproot_methods.classes.TVector3.TVector3(self.x/self.t, self.y/self.t, self.z/self.t)

    def boost(self, vect, inplace=False):
        b2 = vect.mag2()
        gamma = (1.0 - b2)**(-0.5)
        if b2 != 0:
            gamma2 = (gamma - 1.0) / b2
        else:
            gamma2 = 0.0

        bp = self.vect.dot(vect)
        if inplace:
            self.vect += gamma2*bp*vect + gamma*vect*self.t
            self.t += pt
            self.t *= gamma
        else:
            v = self.vect + gamma2*bp*vect + gamma*vect*self.t
            return self.__class__(v.x, v.y, v.z, gamma*(self.t + bp))

    def gamma(self):
        out = self.beta()
        if -1 < out < 1:
            return (1 - out**2)**(-0.5)
        else:
            return float("inf")

    def delta_r(self):
        return math.sqrt(self.delta_r2())

    def rotate_axis(self, axis, angle):
        vect, t = self._rotate_axis(axis, angle)
        return self.__class__(vect.x, vect.y, vect.z, t)

    def rotate_euler(self, phi=0, theta=0, psi=0):
        vect, t = self._rotate_euler(phi, theta, psi)
        return self.__class__(vect.x, vect.y, vect.z, t)

    def islightlike(self, tolerance=1e-10):
        return abs(self.mag2()) < tolerance

    def __repr__(self):
        return "TLorentzVector({0:.4g}, {1:.4g}, {2:.4g}, {3:.4g})".format(self._fP._fX, self._fP._fY, self._fP._fZ, self._fE)

    def __str__(self):
        return repr(self)

class TLorentzVectorArray(ArrayMethods, awkward.ObjectArray):
    def __init__(self, x, y, z, t):
        super(TLorentzVectorArray, self).__init__(awkward.Table(min(len(x), len(y), len(z), len(t))), lambda row: TLorentzVector(row["fX"], row["fY"], row["fZ"], row["fE"]))
        self._vect = uproot_methods.classes.TVector3.TVector3Array(x, y, z)
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
    def from_vect(cls, vect, t):
        out = cls.__new__(cls)
        out._vect = vect
        out["fE"] = t
        return out

    @classmethod
    def from_spherical(cls, r, theta, phi, t):
        return cls.from_vect(uproot_methods.classes.TVector3.TVector3Array.from_spherical(r, theta, phi), t)

    @classmethod
    def from_cylindrical(cls, rho, phi, z, t):
        return cls.from_vect(uproot_methods.classes.TVector3.TVector3Array.from_cylindrical(rho, phi, z), t)

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
        vect = uproot_methods.classes.TVector3.TVector3Array(x, y, z)
        return cls.from_vect(vect, awkward.util.numpy.sqrt(x*x + y*y + z*z + m*m*awkward.util.numpy.sign(m)))

class TLorentzVector(Methods):
    def __init__(self, x, y, z, t):
        self._fP = uproot_methods.classes.TVector3.TVector3(x, y, z)
        self._fE = t

    @classmethod
    def origin(cls):
        return cls(0.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_vect(cls, vect, t):
        out = cls.__new__(cls)
        out._fP = vect
        out._fE = t
        return out

    @classmethod
    def from_spherical(cls, r, theta, phi, t):
        return cls.from_vect(uproot_methods.classes.TVector3.Methods.from_spherical(r, theta, phi), t)

    @classmethod
    def from_cylindrical(cls, rho, phi, z, t):
        return cls.from_vect(uproot_methods.classes.TVector3.Methods.from_cylindrical(rho, phi, z), t)

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
