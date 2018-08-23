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

import uproot_methods.common.TVector
import uproot_methods.base

class Common(object):
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

    @property
    def z(self):
        return self._fZ

    @z.setter
    def z(self, value):
        self._fZ = value

    def dot(self, other):
        out = self.x * other.x
        out += self.y * other.y
        out += self.z * other.z
        return out

    def _cross(self, other):
        return (self.y*other.z - self.z*other.y,
                self.z*other.x - self.x*other.z,
                self.x*other.y - self.y*other.x)

    def r2(self):
        return self.dot(self)

    def _rotate_axis(self, axis, angle):
        u = axis.unit()
        c = awkward.util.numpy.cos(angle)
        s = awkward.util.numpy.sin(angle)
        c1 = 1 - c

        x = (c + u.x**2 * c1) * self.x + (u.x * u.y * c1 - u.z * s) * self.y + (u.x * u.z * c1 + u.y * s) * self.z
        y = (u.x * u.y * c1 + u.z * s) * self.x + (c + u.y**2 * c1) * self.y + (u.y * u.z * c1 - u.x * s) * self.z
        z = (u.x * u.z * c1 - u.y * s) * self.x + (u.y * u.z * c1 + u.x * s) * self.y + (c + u.z**2 * c1) * self.z

        return x, y, z

    def _rotate_euler(self, phi, theta, psi):
        # Rotate Z (phi)
        c1 = awkward.util.numpy.cos(phi)
        s1 = awkward.util.numpy.sin(phi)
        c2 = awkward.util.numpy.cos(theta)
        s2 = awkward.util.numpy.sin(theta)
        c3 = awkward.util.numpy.cos(psi)
        s3 = awkward.util.numpy.sin(psi)

        # Rotate Y (theta)
        fzx2 = -s2*c1
        fzy2 =  s2*s1
        fzz2 =  c2

        # Rotate Z (psi)
        fxx3 =  c3*c2*c1 - s3*s1
        fxy3 = -c3*c2*s1 - s3*c1
        fxz3 =  c3*s2
        fyx3 =  s3*c2*c1 + c3*s1
        fyy3 = -s3*c2*s1 + c3*c1
        fyz3 =  s3*s2

        # Transform v
        x = fxx3*self.x + fxy3*self.y + fxz3*self.z
        y = fyx3*self.x + fyy3*self.y + fyz3*self.z
        z = fzx2*self.x + fzy2*self.y + fzz2*self.z

        return x, y, z

    def rotatex(self, angle):
        return self.rotate_axis(TVector3(1.0, 0.0, 0.0), angle)

    def rotatey(self, angle):
        return self.rotate_axis(TVector3(0.0, 1.0, 0.0), angle)

    def rotatez(self, angle):
        return self.rotate_axis(TVector3(0.0, 0.0, 1.0), angle)

class ArrayMethods(uproot_methods.base.ROOTMethods, Common, uproot_methods.common.TVector.ArrayMethods):
    def __init__(self, data):
        self._fX = data["fX"]
        self._fY = data["fY"]
        self._fZ = data["fZ"]

    @classmethod
    def origin(cls, shape, dtype=None):
        if dtype is None:
            dtype = awkward.util.numpy.float64
        return cls({"fX": awkward.util.numpy.zeros(shape, dtype=dtype),
                    "fY": awkward.util.numpy.zeros(shape, dtype=dtype),
                    "fZ": awkward.util.numpy.zeros(shape, dtype=dtype)})

    @classmethod
    def origin_like(cls, array):
        return cls.origin(array.shape, array.dtype)

    @classmethod
    def from_spherical(cls, r, theta, phi):
        return cls({"fX": r * awkward.util.numpy.sin(theta) * awkward.util.numpy.cos(phi),
                    "fY": r * awkward.util.numpy.sin(theta) * awkward.util.numpy.sin(phi),
                    "fZ": r * awkward.util.numpy.cos(theta)})

    @classmethod
    def from_cylindrical(cls, rho, phi, z):
        return cls({"fX": rho * awkward.util.numpy.cos(phi),
                    "fY": rho * awkward.util.numpy.sin(phi),
                    "fZ": z})

    def cross(self, other):
        x, y, z = self._cross(other)
        return self.__class__({"fX": x, "fY": y, "fZ": z})

    def theta(self):
        out = self.rho()
        return awkward.util.numpy.arctan2(out, self.z, out=out)

    def r(self):
        out = self.r2()
        return awkward.util.numpy.sqrt(out, out=out)

    def rotate_axis(self, axis, angle):
        x, y, z = self._rotate_axis(axis, angle)
        return self.__class__({"fX": x, "fY": y, "fZ": z})

    def rotate_euler(self, phi=0, theta=0, psi=0):
        x, y, z = self._rotate_euler(phi, theta, psi)
        return self.__class__({"fX": x, "fY": y, "fZ": z})

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            raise NotImplemented

        inputsx = []
        inputsy = []
        inputsz = []
        for obj in inputs:
            if isinstance(obj, ArrayMethods):
                inputsx.append(obj.x)
                inputsy.append(obj.y)
                inputsz.append(obj.z)
            else:
                inputsx.append(obj)
                inputsy.append(obj)
                inputsz.append(obj)

        resultx = getattr(ufunc, method)(*inputsx, **kwargs)
        resulty = getattr(ufunc, method)(*inputsy, **kwargs)
        resultz = getattr(ufunc, method)(*inputsz, **kwargs)

        if isinstance(resultx, tuple) and isinstance(resulty, tuple) and isinstance(resultz, tuple):
            return tuple(self.__class__({"fX": x, "fY": y, "fZ": z}) for x, y, z in zip(resultx, resulty, resultz))
        elif method == "at":
            return None
        else:
            return self.__class__({"fX": resultx, "fY": resulty, "fZ": resultz})

class Methods(uproot_methods.base.ROOTMethods, Common, uproot_methods.common.TVector.Methods):
    _arraymethods = ArrayMethods

    def __init__(self, x, y, z):
        self._fX = x
        self._fY = y
        self._fZ = z

    @classmethod
    def origin(cls):
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def from_spherical(cls, r, theta, phi):
        return cls(r * math.sin(theta) * math.cos(phi),
                   r * math.sin(theta) * math.sin(phi),
                   r * math.cos(theta))

    @classmethod
    def from_cylindrical(cls, rho, phi, z):
        return cls(rho * math.cos(phi),
                   rho * math.sin(phi),
                   z)

    def __repr__(self):
        return "TVector3({0:.4g}, {1:.4g}, {2:.4g})".format(self._fX, self._fY, self._fZ)

    def __str__(self):
        return repr(self)

    def cross(self, other):
        x, y, z = self._cross(other)
        return self.__class__(x, y, z)

    def theta(self):
        return math.atan2(self.rho(), self.z)

    def r(self):
        return math.sqrt(self.r2())

    def rotate_axis(self, axis, angle):
        x, y, z = self._rotate_axis(axis, angle)
        return self.__class__(x, y, z)

    def rotate_euler(self, phi=0, theta=0, psi=0):
        return self.__class__(x, y, z)
