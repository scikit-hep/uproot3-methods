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

import awkward.util

import uproot_methods.base
import uproot_methods.classes.TVector3

class Common(object):
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

    @property
    def energy(self):
        return self.t

    @energy.setter
    def energy(self, value):
        self.t = value

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

    def p(self):
        return self.vect.mag()

    def p2(self):
        return self.vect.mag2()

    def pt2(self):
        return self.vect.rho2()

    def pt(self):
        return self.vect.rho()

    def Et(self):
        return self.energy * self.pt / self.p

    def mass2(self):
        return self.mag2()

    def mass(self):
        return self.mag()

    def mt2(self):
        return self.energy**2 - self.z**2
        
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
    def __init__(self, data):
        self._fP = uproot_methods.classes.TVector3.ArrayMethods(data)
        self._fE = data["fE"]

    def mt(self):
        out = self.mt2()
        sign = awkward.util.numpy.sign(out)
        awkward.util.numpy.absolute(out, out=out)
        awkward.util.numpy.sqrt(out, out=out)
        return awkward.util.numpy.multiply(out, sign, out=out)

    def eta(self):
        return -awkward.util.numpy.log((1.0 - awkward.util.numpy.cos(self.theta())) / (1.0 + awkward.util.numpy.cos(self.theta())))/2.0

    def rapidity(self):
        return awkward.util.numpy.log((self.t + self.z) / (self.t - self.z)) / 2.0

    def boost_vector(self):
        return uproot_methods.classes.TVector3.ArrayMethods({"fX": self.x/self.t,
                                                             "fY": self.y/self.t,
                                                             "fZ": self.z/self.t})

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
            return self.__class__({"fX": v.x, "fY": v.y, "fZ": v.z, "fE": gamma*(self.t + bp)})

    def gamma(self):
        out = self.beta()
        mask = (out < 1)
        awkward.util.numpy.bitwise_and(mask, out > -1, out=mask)
        out[mask] = (1 - out[mask]**2)**(-0.5)
        out[~mask] = awkward.util.numpy.inf
        return out

    def delta_r(self):
        return awkward.util.numpy.sqrt(self.delta_r2())

    def rotate_axis(self, axis, angle):
        vect, t = self._rotate_axis(axis, angle)
        return self.__class__({"fX": vect.x, "fY": vect.y, "fZ": vect.z, "fE": t})

    def rotate_euler(self, phi=0, theta=0, psi=0):
        vect, t = self._rotate_euler(phi, theta, psi)
        return self.__class__({"fX": vect.x, "fY": vect.y, "fZ": vect.z, "fE": t})

    def islightlike(self, tolerance=1e-10):
        return awkward.util.numpy.absolute(self.mag2()) < tolerance

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raise NotImplementedError

class Methods(uproot_methods.base.ROOTMethods, Common):
    _arraymethods = ArrayMethods

    def __init__(self, x, y, z, t):
        self._fP = uproot_methods.classes.TVector3.Methods(x, y, z)
        self._fE = t

    @classmethod
    def origin(cls):
        return cls(0.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_vect(cls, vect, t):
        return cls(vect.x, vect.y, vect.z, t)

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
                   t)
    
    @classmethod
    def from_ptetaphim(cls, pt, eta, phi, m):
        tmp = Methods.from_ptetaphi(pt, eta, phi, 0)
        return Methods.from_xyzm(tmp.x, tmp.y, tmp.z, m)

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
        return uproot_methods.classes.TVector3.Methods(self.x/self.t, self.y/self.t, self.z/self.t)

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
