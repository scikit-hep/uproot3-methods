#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot-methods/blob/master/LICENSE

import math
import numbers
import operator

class Common(object):
    @property
    def mag2(self):
        return self.dot(self)

    @property
    def mag(self):
        return self.awkward.numpy.sqrt(self.mag2)

    @property
    def rho2(self):
        out = self.x*self.x
        out = out + self.y*self.y
        return out

    def delta_phi(self, other):
        return (self.phi - other.phi + math.pi) % (2*math.pi) - math.pi

    def isparallel(self, other, tolerance=1e-10):
        return 1 - self.cosdelta(other) < tolerance

    def isantiparallel(self, other, tolerance=1e-10):
        return self.cosdelta(other) - (-1) < tolerance

    def iscollinear(self, other, tolerance=1e-10):
        return 1 - self.awkward.numpy.absolute(self.cosdelta(other)) < tolerance

    def __lt__(self, other):
        raise TypeError("spatial vectors have no natural ordering")

    def __gt__(self, other):
        raise TypeError("spatial vectors have no natural ordering")

    def __le__(self, other):
        raise TypeError("spatial vectors have no natural ordering")

    def __ge__(self, other):
        raise TypeError("spatial vectors have no natural ordering")

class ArrayMethods(Common):
    @property
    def unit(self):
        return self / self.mag

    @property
    def rho(self):
        out = self.rho2
        return self.awkward.numpy.sqrt(out)

    @property
    def phi(self):
        return self.awkward.numpy.arctan2(self.y, self.x)

    def cosdelta(self, other):
        denom = self.mag2 * other.mag2
        mask = (denom > 0)
        denom = denom[mask]
        denom[:] = self.awkward.numpy.sqrt(denom)

        out = self.dot(other)
        out[mask] /= denom

        mask = self.awkward.numpy.logical_not(mask)
        out[mask] = 1

        return self.awkward.numpy.clip(out, -1, 1)

    def angle(self, other, normal=None, degrees=False):
        out = self.awkward.numpy.arccos(self.cosdelta(other))
        if normal is not None:
            a = self.unit
            b = other.unit
            out = out * self.awkward.numpy.sign(normal.dot(a.cross(b)))
        if degrees:
            out = self.awkward.numpy.multiply(out, 180.0/self.awkward.numpy.pi)
        return out

    def isopposite(self, other, tolerance=1e-10):
        tmp = self + other
        tmp.x = self.awkward.numpy.absolute(tmp.x)
        tmp.y = self.awkward.numpy.absolute(tmp.y)
        tmp.z = self.awkward.numpy.absolute(tmp.z)

        out = (tmp.x < tolerance)
        out = self.awkward.numpy.bitwise_and(out, tmp.y < tolerance)
        out = self.awkward.numpy.bitwise_and(out, tmp.z < tolerance)
        return out

    def isperpendicular(self, other, tolerance=1e-10):
        tmp = self.dot(other)
        tmp.x = self.awkward.numpy.absolute(tmp.x)
        tmp.y = self.awkward.numpy.absolute(tmp.y)
        tmp.z = self.awkward.numpy.absolute(tmp.z)

        out = (tmp.x < tolerance)
        out = self.awkward.numpy.bitwise_and(out, tmp.y < tolerance)
        out = self.awkward.numpy.bitwise_and(out, tmp.z < tolerance)
        return out

class Methods(Common):
    @property
    def unit(self):
        return self / self.mag

    @property
    def rho(self):
        return math.sqrt(self.rho2)

    @property
    def phi(self):
        return math.atan2(self.y, self.x)

    def cosdelta(self, other):
        m1 = self.mag2
        m2 = other.mag2
        if m1 == 0 or m2 == 0:
            return 1.0
        r = self.dot(other) / math.sqrt(m1 * m2)
        return max(-1.0, min(1.0, r))

    def angle(self, other, degrees=False):
        out = math.acos(self.cosdelta(other))
        if degrees:
            out = out * 180.0/math.pi
        return out

    def isopposite(self, other, tolerance=1e-10):
        tmp = self + other
        return abs(tmp.x) < tolerance and abs(tmp.y) < tolerance and abs(tmp.z) < tolerance

    def isperpendicular(self, other, tolerance=1e-10):
        tmp = self.dot(other)
        return abs(tmp.x) < tolerance and abs(tmp.y) < tolerance and abs(tmp.z) < tolerance

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
        if isinstance(other, (numbers.Number, self.awkward.numpy.number)):
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
