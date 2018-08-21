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

import numpy

import uproot_methods.base
import uproot_methods.classes.TVector3

class ArrayMethods(uproot_methods.base.ROOTMethods):
    @property
    def x(self):
        return self["fX"]

class Methods(ArrayMethods):
    _arraymethods = ArrayMethods

    def __init__(self, data):
        self._fP = uproot_methods.classes.TVector3.Methods(data)
        self._fE = data["fE"]

    @classmethod
    def from4vector(cls, other):
        return cls(other._fP._fX, other._fP._fY, other._fP._fZ, other._fE)

    @classmethod
    def from3vector(cls, other, t):
        return cls(other._fX, other._fY, other._fZ, t)

    @classmethod
    def fromiterable(cls, values):
        x, y, z, t = values
        return cls(x, y, z, t)

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
    def vector(self):
        return self._fP.copy()

    @property
    def t(self):
        return self._fE

    @t.setter
    def t(self, value):
        self._fE = value

    def costheta(self):
        return self._fP.costheta()

    def theta(self, deg=False):
        return self._fP.theta(deg)

    def phi(self, deg=False):
        return self._fP.phi(deg)

    @property
    def px(self):
        return self._fP._fX

    @px.setter
    def px(self, value):
        self._fP._fX = value

    @property
    def py(self):
        return self._fP._fY

    @py.setter
    def py(self, value):
        self._fP._fY = value

    @property
    def pz(self):
        return self._fP._fZ

    @pz.setter
    def pz(self, value):
        self._fP._fZ = value

    @property
    def e(self):
        return self._fE

    @e.setter
    def e(self, value):
        self._fE = value

    def set(self, x, y, z, t):
        self._fP._fX = x
        self._fP._fY = y
        self._fP._fZ = z
        self._fE = t

    def setpxpypzm(self, px, py, pz, m):
        self._fP.set(px, py, pz)
        if m > 0:
            self._fE = numpy.sqrt(px**2 + py**2 + pz**2 + m**2)
        else:
            self._fE = numpy.sqrt(px**2 + py**2 + pz**2 - m**2)

    def setpxpypze(self, px, py, pz, e):
        self.set(px, py, pz, e)

    def setptetaphim(self, pt, eta, phi, m):
        px = pt*numpy.cos(phi)
        py = pt*numpy.sin(phi)
        pz = pt*numpy.sinh(eta)
        self.setpxpypzm(px, py, pz, m)

    def setptetaphie(self, pt, eta, phi, e):
        px = pt*numpy.cos(phi)
        py = pt*numpy.sin(phi)
        pz = pt*numpy.sinh(eta)
        self.setpxpypze(px, py, pz, e)

    def tolist(self):
        return [self._fP._fX, self._fP._fY, self._fP._fZ, self._fE]

    def __setitem__(self, i, value):
        if i < 0:
            i += 4
        if i == 0:
            self._fP._fX = value
        elif i == 1:
            self._fP._fY = value
        elif i == 2:
            self._fP._fZ = value
        elif i == 3:
            self._fE = value
        else:
            raise IndexError("TLorentzVector is of length 4 only!")

    def __getitem__(self, i):
        if i < 0:
            i += 4
        if i == 0:
            return self._fP._fX
        elif i == 1:
            return self._fP._fY
        elif i == 2:
            return self._fP._fZ
        elif i == 3:
            return self._fE
        else:
            raise IndexError("TLorentzVector is of length 3 only!")

    def __len__(self):
        return 4

    @property
    def p(self):
        return self._fP.mag

    @property
    def pt(self):
        return self._fP.rho

    @property
    def et(self):
        return float(self._fE) * self.pt / self.p

    @property
    def m(self):
        return self.mag

    @property
    def m2(self):
        return self.mag2

    @property
    def mass(self):
        return self.mag

    @property
    def mass2(self):
        return self.mag2

    @property
    def mt(self):
        return self.transversemass

    @property
    def mt2(self):
        return self.transversemass2

    @property
    def transversemass(self):
        mt2 = self.transversemass2
        return numpy.sqrt(mt2) if mt2 >= 0 else -numpy.sqrt(-mt2)

    @property
    def transversemass2(self):
        return self._fE**2 - self._fP._fZ**2

    @property
    def beta(self):
        return self._fP.mag / self._fE

    @property
    def gamma(self):
        beta = self.beta
        if beta < 1:
            return 1.0 / numpy.sqrt(1.0 - beta**2)
        else:
            return float("inf")

    @property
    def eta(self):
        return self.pseudorapidity

    @property
    def boostvector(self):
        return self._fP.__class__(self._fP._fX / self._fE, self._fP._fY / self._fE, self._fP._fZ / self._fE)

    @property
    def pseudorapidity(self):
        costheta = self.costheta()
        if abs(costheta) < 1:
            return -0.5 * numpy.log((1.0 - costheta) / (1.0 + costheta))
        else:
            return float("inf") if self.z > 0 else float("-inf")

    @property
    def rapidity(self):
        return 0.5 * numpy.log((self._fE + self._fP._fZ) / (self._fE - self._fP._fZ))

    @property
    def mag(self):
        mag2 = self.mag2
        return numpy.sqrt(mag2) if mag2 >= 0 else -numpy.sqrt(-mag2)

    @property
    def mag2(self):
        return self._fE**2 - self._fP.mag2

    @property
    def perp2(self):
        return self._fP._fX**2 + self._fP._fY**2

    @property
    def perp(self):
        return numpy.sqrt(self.perp2)

    def copy(self):
        return self.__class__(self._fP._fX, self._fP._fY, self._fP._fZ, self._fE)

    def __iadd__(self, other):
        self._fP += other._fP
        self._fE += other._fE
        return self

    def __isub__(self, other):
        self._fP -= other._fP
        self._fE -= other._fE
        return self

    def __add__(self, other):
        return self.__class__(self._fP._fX + other._fP._fX, self._fP._fY + other._fP._fY, self._fP._fZ + other._fP._fZ, self._fE + other._fE)

    def __sub__(self, other):
        return self.__class__(self._fP._fX - other._fP._fX, self._fP._fY - other._fP._fY, self._fP._fZ - other._fP._fZ, self._fE - other._fE)

    def __imul__(self, scalar):
        self._fP *= scalar
        self._fE *= scalar
        return self

    def __mul__(self, other):
        if isinstance(other, TLorentzVectorMethods):
            return self.dot(other)
        else:
            return self.__class__(self._fP._fX * other, self._fP._fY * other, self._fP._fZ * other, self._fE * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __itruediv__(self, scalar):
        self._fP._fX /= float(scalar)
        self._fP._fY /= float(scalar)
        self._fP._fZ /= float(scalar)
        self._fE /= float(scalar)
        return self

    __idiv__ = __itruediv__

    def __truediv__(self, scalar):
        return self.__class__(self._fP._fX / scalar, self._fP._fY / scalar, self._fP._fZ / scalar, self._fE / scalar)

    __div__ = __truediv__

    def __eq__(self, other):
        if other == 0:
            return self._fP._fX == 0 and self._fP._fY == 0 and self._fP._fZ == 0 and self._fE == 0
        else:
            return self._fP._fX == other._fP._fX and self._fP._fY == other._fP._fY and self._fP._fZ == other._fP._fZ and self._fE == other._fE

    def __ne__(self, other):
        return not self.__eq__(other)

    def __iter__(self):
        return (self._fP._fX, self._fP._fY, self._fP._fZ, self._fE).__iter__()

    def boost(self, *args):
        if len(args) == 1 and isinstance(args[0], TVector3Methods):
            bx, by, bz = args[0]._fX, args[0]._fY, args[0]._fZ
        elif len(args) == 1 and len(args[0]) == 3:
            bx, by, bz = args[0]
        elif len(args) == 3:
            bx, by, bz = args
        else:
            raise TypeError("Input object not a TVector3 nor an iterable with 3 elements.")

        b2 = bx**2 + by**2 + bz**2
        gamma = 1.0 / numpy.sqrt(1.0 - b2)
        bp = bx * self._fP._fX + by * self._fP._fY + bz * self._fP._fZ
        if b2 > 0:
            gamma2 = (gamma - 1.0) / b2
        else:
            gamma2 = 0.0

        xp = self._fP._fX + gamma2 * bp * bx - gamma * bx * self._fE
        yp = self._fP._fY + gamma2 * bp * by - gamma * by * self._fE
        zp = self._fP._fZ + gamma2 * bp * bz - gamma * bz * self._fE
        tp = gamma * (self._fE - bp)

        return self.__class__(xp, yp, zp, tp)

    def rotate(self, angle, *args):
        v3p = self._fP.rotate(angle, *args)
        return self.__class__(v3p._fX, v3p._fY, v3p._fZ, self._fE)

    def rotatex(self, angle):
        return self.rotate(angle, 1, 0, 0)

    def rotatey(self, angle):
        return self.rotate(angle, 0, 1, 0)

    def rotatez(self, angle):
        return self.rotate(angle, 0, 0, 1)

    def dot(self, other):
        return self._fE*other._fE - self._fP.dot(other._fP)

    def deltaeta(self, other):
        return self.eta - other.eta

    def deltaphi(self, other):
        dphi = self.phi() - other.phi()
        while dphi > numpy.pi:
            dphi -= 2.0*numpy.pi
        while dphi < -numpy.pi:
            dphi += 2.0*numpy.pi
        return dphi

    def deltar(self, other):
        return numpy.sqrt(self.deltaeta(other)**2 + self.deltaphi(other)**2)

    def isspacelike(self):
        return self.mag2 < 0

    def istimelike(self):
        return self.mag2 > 0

    def islightlike(self):
        return self.mag2 == 0

    def __repr__(self):
        return "TLorentzVector({0:.4g}, {1:.4g}, {2:.4g}, {3:.4g})".format(self._fP._fX, self._fP._fY, self._fP._fZ, self._fE)

    def __str__(self):
        return repr(self)
    #     return str((self._fP._fX, self._fP._fY, self._fP._fZ, self._fE))
