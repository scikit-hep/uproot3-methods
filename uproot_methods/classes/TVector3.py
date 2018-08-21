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

class ArrayMethods(uproot_methods.base.ROOTMethods):
    pass

class Methods(ArrayMethods):
    _arraymethods = ArrayMethods

    def __init__(self, data):
        self._fX = data["fX"]
        self._fY = data["fX"]
        self._fZ = data["fX"]

    @classmethod
    def origin(cls):
        return cls(0.0, 0.0, 0.0)

    @classmethod
    def frompoint(cls, x, y, z):
        return cls(x, y, z)

    @classmethod
    def fromvector(cls, other):
        return cls(other.x, other.y, other.z)

    @classmethod
    def fromsphericalcoords(cls, r, theta, phi):
        x = r * numpy.sin(theta) * numpy.cos(phi)
        y = r * numpy.sin(theta) * numpy.sin(phi)
        z = r * numpy.cos(theta)
        return cls(x, y, z)

    @classmethod
    def fromcylindricalcoords(cls, rho, phi, z):
        x = rho * numpy.cos(phi)
        y = rho * numpy.sin(phi)
        z = z
        return cls(x, y, z)

    @classmethod
    def fromiterable(cls, values):
        x, y, z = values
        return cls(x, y, z)

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

    @property
    def rho(self):
        return numpy.sqrt(self._fX**2 + self._fY**2)

    @property
    def r(self):
        return numpy.sqrt(self._fX**2 + self._fY**2 + self._fZ**2)

    def costheta(self):
        if self._fX == 0 and self._fY == 0 and self._fZ == 0:
            return 1.0
        else:
            return self._fZ / numpy.sqrt(self._fX**2 + self._fY**2 + self._fZ**2)

    def theta(self, deg=False):
        theta = numpy.acos(self.costheta())
        return theta if not deg else numpy.degrees(theta)

    def phi(self, deg=False):
        phi = numpy.atan2(self._fY, self._fX)
        return phi if not deg else numpy.degrees(phi)

    def set(self, x, y, z):
        self._fX = x
        self._fY = y
        self._fZ = z

    def __setitem__(self, i, value):
        if i < 0:
            i += 3
        if i == 0:
            self._fX = value
        elif i == 1:
            self._fY = value
        elif i == 2:
            self._fZ = value
        else:
            raise IndexError("TVector3 is of length 3 only!")

    def __getitem__(self, i):
        if i < 0:
            i += 3
        if i == 0:
            return self._fX
        elif i == 1:
            return self._fY
        elif i == 2:
            return self._fZ
        else:
            raise IndexError("TVector3 is of length 3 only!")

    def tolist(self):
        return [self._fX, self._fY, self._fZ]

    def __len__(self):
        return 3

    @property
    def mag(self):
        return numpy.sqrt(self._fX**2 + self._fY**2 + self._fZ**2)

    @property
    def mag2(self):
        return self._fX**2 + self._fY**2 + self._fZ**2

    def __abs__(self):
        return numpy.sqrt(self._fX**2 + self._fY**2 + self._fZ**2)

    def copy(self):
        return self.__class__(self._fX, self._fY, self._fZ)

    def unit(self):
        mag = self.mag
        if mag != 0 and mag != 1:
            return self.__class__(self._fX / mag, self._fY / mag, self._fZ / mag)
        else:
            return self

    def __iadd__(self, other):
        self._fX += other._fX
        self._fY += other._fY
        self._fZ += other._fZ
        return self

    def __isub__(self, other):
        self._fX -= other._fX
        self._fY -= other._fY
        self._fZ -= other._fZ
        return self

    def __add__(self, other):
        return self.__class__(self._fX + other._fX, self._fY + other._fY, self._fZ + other._fZ)

    def __sub__(self, other):
        return self.__class__(self._fX - other._fX, self._fY - other._fY, self._fZ - other._fZ)

    def __imul__(self, other):
        self._fX *= scalar
        self._fY *= scalar
        self._fZ *= scalar
        return self

    def __mul__(self, other):
        if isinstance(other, TVector3Methods):
            return self.dot(other)
        else:
            return self.__class__(self._fX * other, self._fY * other, self._fZ * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __itruediv__(self, scalar):
        self._fX /= float(scalar)
        self._fY /= float(scalar)
        self._fZ /= float(scalar)
        return self

    __idiv__ = __itruediv__

    def __truediv__(self, scalar):
        return self.__class__(self._fX / scalar, self._fY / scalar, self._fZ / scalar)

    __div__ = __truediv__

    def __eq__(self, other):
        if other == 0:
            return self._fX == 0 and self._fY == 0 and self._fZ == 0
        else:
            return self._fX == other._fX and self._fY == other._fY and self._fZ == other._fZ

    def __ne__(self, other):
        return not self.__eq__(other)

    def __nonzero__(self):
        return self._fX == 0 and self._fY == 0 and self._fZ == 0

    def __iter__(self):
        return (self._fX, self._fY, self._fZ).__iter__()

    def dot(self, other):
        return self._fX*other._fX + self._fY*other._fY + self._fZ*other._fZ

    def cross(self, other):
        return self.__class__(self._fY * other._fZ - self._fZ * other._fY,
                              self._fZ * other._fX - self._fX * other._fZ,
                              self._fX * other._fY - self._fY * other._fX)

    def rotate(self, angle, *args):
        if len(args) == 1 and isinstance(args[0], TVector3Methods):
            ux, uy, uz = args[0]._fX, args[0]._fY, args[0]._fZ
        elif len(args) == 1 and len(args[0]) == 3:
            ux, uy, uz = args[0]
        elif len(args) == 3:
            ux, uy, uz = args
        else:
            raise TypeError("Input object not a TVector3 nor an iterable with 3 elements.")

        norm = numpy.sqrt(ux**2 + uy**2 + uz**2)
        if norm != 1:
            ux = ux / norm
            uy = uy / norm
            uz = uz / norm
        c = numpy.cos(angle)
        s = numpy.sin(angle)
        c1 = 1.0 - c

        xp = (c + ux**2 * c1) * self._fX + (ux * uy * c1 - uz * s) * self._fY + (ux * uz * c1 + uy * s) * self._fZ
        yp = (ux * uy * c1 + uz * s) * self._fX + (c + uy**2 * c1) * self._fY + (uy * uz * c1 - ux * s) * self._fZ
        zp = (ux * uz * c1 - uy * s) * self._fX + (uy * uz * c1 + ux * s) * self._fY + (c + uz**2 * c1) * self._fZ

        return self.__class__(xp, yp, zp)

    def rotatex(self, angle):
        return self.rotate(angle, 1, 0, 0)

    def rotatey(self, angle):
        return self.rotate(angle, 0, 1, 0)

    def rotatez(self, angle):
        return self.rotate(angle, 0, 0, 1)

    def cosdelta(self, other):
        m1 = self.mag2
        m2 = other.mag2
        if m1 == 0 or m2 == 0:
            return 1.0
        r = self.dot(other) / numpy.sqrt(m1 * m2)
        return max(-1.0, min(1.0, r))

    def angle(self, other, deg=False):
        cd = self.cosdelta(other)
        return numpy.acos(cd) if not deg else numpy.degrees(numpy.acos(cd))

    def isparallel(self, other):
        return self.cosdelta(other) == 1

    def isantiparallel(self, other):
        return self.cosdelta(other) == -1

    def iscollinear(self, other):
        return abs(self.cosdelta(other)) == 1

    def isopposite(self, other):
        return (self + other) == 0

    def isperpendicular(self, other):
        return self.dot(other) == 0

    def __repr__(self):
        return "TVector3({0:.4g}, {1:.4g}, {2:.4g})".format(self._fX, self._fY, self._fZ)

    def __str__(self):
        return repr(self)
        # return str((self._fX, self._fY, self._fZ))
