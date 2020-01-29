#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot-methods/blob/master/LICENSE

import math
import numbers
import operator

import awkward.array.chunked
import awkward.array.jagged
import awkward.array.objects
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
    def Et(self):
        return self.energy * self.pt / self.p

    @property
    def mag2(self):
        return self.dot(self)

    @property
    def mass2(self):
        return self.mag2

    @property
    def mt2(self):
        return self.energy**2 - self.z**2

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
        return self.rotate_axis(uproot_methods.classes.TVector3.TVector3(1.0, 0.0, 0.0), angle)

    def rotatey(self, angle):
        return self.rotate_axis(uproot_methods.classes.TVector3.TVector3(0.0, 1.0, 0.0), angle)

    def rotatez(self, angle):
        return self.rotate_axis(uproot_methods.classes.TVector3.TVector3(0.0, 0.0, 1.0), angle)

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
        self.awkward.ObjectArray.__init__(self, table, lambda row: TLorentzVector(row["fX"], row["fY"], row["fZ"], row["fE"]))

    def __awkward_serialize__(self, serializer):
        self._valid()
        x, y, z, t = self.x, self.y, self.z, self.t
        return serializer.encode_call(
            ["uproot_methods.classes.TLorentzVector", "TLorentzVectorArray", "from_cartesian"],
            serializer(x, "TLorentzVectorArray.x"),
            serializer(y, "TLorentzVectorArray.y"),
            serializer(z, "TLorentzVectorArray.z"),
            serializer(t, "TLorentzVectorArray.t"))

    @staticmethod
    def _wrapmethods(node, awkwardlib):
        if isinstance(node, awkward.array.chunked.ChunkedArray):
            node.__class__ = type("ChunkedArrayMethods", (awkwardlib.ChunkedArray, uproot_methods.classes.TVector3.ArrayMethods), {})
            for chunk in node.chunks:
                ArrayMethods._wrapmethods(chunk, awkwardlib)
        elif isinstance(node, awkward.array.jagged.JaggedArray):
            node.__class__ = type("JaggedArrayMethods", (awkwardlib.JaggedArray, uproot_methods.classes.TVector3.ArrayMethods), {})
            ArrayMethods._wrapmethods(node.content, awkwardlib)
        elif isinstance(node, awkward.array.objects.ObjectArray):
            node.__class__ = type("ObjectArrayMethods", (awkwardlib.ObjectArray, uproot_methods.classes.TVector3.ArrayMethods), {})
        
    @property
    def p3(self):
        out = self.empty_like(generator=lambda row: uproot_methods.classes.TVector3.TVector3(row["fX"], row["fY"], row["fZ"]))
        ArrayMethods._wrapmethods(out, self.awkward)
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
    def pt(self):
        return self._trymemo("pt", lambda self: self.awkward.numpy.sqrt(self.pt2))

    @property
    def eta(self):
        return self._trymemo("eta", lambda self: self.awkward.numpy.arcsinh(self.z / self.awkward.numpy.sqrt(self.x**2 + self.y**2)))

    @property
    def phi(self):
        return self._trymemo("phi", lambda self: self.p3.phi)

    @property
    def mass(self):
        return self._trymemo("mass", lambda self: self.awkward.numpy.sqrt(self.mag2))

    @property
    def mag(self):
        return self.awkward.numpy.sqrt(self.mag2)

    @property
    def mt(self):
        mt2 = self.mt2
        sign = self.awkward.numpy.sign(mt2)
        return self.awkward.numpy.sqrt(self.awkward.numpy.absolute(mt2)) * sign

    @property
    def rapidity(self):
        return 0.5 * self.awkward.numpy.log((self.t + self.z) / (self.t - self.z))

    @property
    def unit(self):
        return self / self.awkward.numpy.sqrt(self.mag)

    @property
    def boostp3(self):
        out = self.empty_like(generator=lambda row: uproot_methods.classes.TVector3.TVector3(row["fX"], row["fY"], row["fZ"]))
        if isinstance(self, self.awkward.JaggedArray):
            out.__class__ = type("JaggedArrayMethods", (self.awkward.JaggedArray, uproot_methods.classes.TVector3.ArrayMethods), {})
        else:
            out.__class__ = type("ObjectArrayMethods", (self.awkward.ObjectArray, uproot_methods.classes.TVector3.ArrayMethods), {})
        out["fX"] = self.x / self.t
        out["fY"] = self.y / self.t
        out["fZ"] = self.z / self.t
        return out

    def boost(self, p3):
        if not isinstance(p3, (uproot_methods.classes.TVector3.ArrayMethods, uproot_methods.classes.TVector3.Methods)):
            raise TypeError("boost p3 must be an (array of) TVector3")

        b2 = p3.mag2
        gamma = (1 - b2)**(-0.5)
        gamma2 = self.awkward.numpy.zeros(b2.shape, dtype=self.awkward.numpy.float64)
        mask = (b2 != 0)
        gamma2[mask] = (gamma[mask] - 1) / b2[mask]
        del mask

        bp = self.p3.dot(p3)

        v = self.p3 + gamma2*bp*p3 + self.t*gamma*p3
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
        out[~mask] = self.awkward.numpy.inf
        return out

    def delta_r(self, other):
        return self.awkward.numpy.sqrt(self.delta_r2(other))

    def rotate_axis(self, axis, angle):
        p3, t = self._rotate_axis(axis, angle)
        x, y, z = p3
        out = self.empty_like()
        out["fX"] = x
        out["fY"] = y
        out["fZ"] = z
        out["fE"] = t
        return out

    def rotate_euler(self, phi=0, theta=0, psi=0):
        p3, t = self._rotate_euler(phi, theta, psi)
        x, y, z = p3
        out = self.empty_like()
        out["fX"] = x
        out["fY"] = y
        out["fZ"] = z
        out["fE"] = t
        return out

    def islightlike(self, tolerance=1e-10):
        return self.awkward.numpy.absolute(self.mag2) < tolerance

    def sum(self):
        if isinstance(self, awkward.AwkwardArray) and self._util_hasjagged(self):
            return TLorentzVectorArray.from_cartesian(self.x.sum(), self.y.sum(), self.z.sum(), self.t.sum())
        else:
            return TLorentzVector(self.x.sum(), self.y.sum(), self.z.sum(), self.t.sum())

    def _to_cartesian(self):
        return TLorentzVectorArray.from_cartesian(self.x,self.y,self.z,self.t)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if "out" in kwargs:
            raise NotImplementedError("in-place operations not supported")

        if method != "__call__":
            return NotImplemented

        inputs = list(inputs)
        for i in range(len(inputs)):
            if isinstance(inputs[i], self.awkward.numpy.ndarray) and inputs[i].dtype == self.awkward.numpy.dtype(object) and len(inputs[i]) > 0:
                idarray = self.awkward.numpy.frombuffer(inputs[i], dtype=self.awkward.numpy.uintp)
                if (idarray == idarray[0]).all():
                    inputs[i] = inputs[i][0]

        if ufunc is self.awkward.numpy.add or ufunc is self.awkward.numpy.subtract:
            if not all(isinstance(x, (ArrayMethods, Methods)) for x in inputs):
                raise TypeError("(arrays of) TLorentzVector can only be added to/subtracted from other (arrays of) TLorentzVector")
            cart_inputs = [x._to_cartesian() for x in inputs]
            out = cart_inputs[0].empty_like()
            out["fX"] = getattr(ufunc, method)(*[x.x for x in cart_inputs], **kwargs)
            out["fY"] = getattr(ufunc, method)(*[x.y for x in cart_inputs], **kwargs)
            out["fZ"] = getattr(ufunc, method)(*[x.z for x in cart_inputs], **kwargs)
            out["fE"] = getattr(ufunc, method)(*[x.t for x in cart_inputs], **kwargs)
            return out

        elif ufunc is self.awkward.numpy.power and len(inputs) >= 2 and isinstance(inputs[1], (numbers.Number, self.awkward.numpy.number)):
            if inputs[1] == 2:
                return self.mag2
            else:
                return self.mag2**(0.5*inputs[1])

        elif ufunc is self.awkward.numpy.absolute:
            return self.mag

        else:
            return super(ArrayMethods, self).__array_ufunc__(ufunc, method, *inputs, **kwargs)

JaggedArrayMethods = ArrayMethods.mixin(ArrayMethods, awkward.JaggedArray)

class PtEtaPhiMassArrayMethods(ArrayMethods):
    def _initObjectArray(self, table):
        self.awkward.ObjectArray.__init__(self, table, lambda row: PtEtaPhiMassLorentzVector(row["fPt"], row["fEta"], row["fPhi"], row["fMass"]))

    def __awkward_serialize__(self, serializer):
        self._valid()
        pt, eta, phi, mass = self.pt, self.eta, self.phi, self.mass
        return serializer.encode_call(
            ["uproot_methods.classes.TLorentzVector", "TLorentzVectorArray", "from_ptetaphim"],
            serializer(pt, "TLorentzVectorArray.pt"),
            serializer(eta, "TLorentzVectorArray.eta"),
            serializer(phi, "TLorentzVectorArray.phi"),
            serializer(mass, "TLorentzVectorArray.mass"))
    
    @property
    def x(self):
        return self._trymemo("x",lambda self: self.pt * self.awkward.numpy.cos(self.phi))
    
    @property
    def y(self):
        return self._trymemo("y",lambda self: self.pt * self.awkward.numpy.sin(self.phi))
    
    @property
    def z(self):
        return self._trymemo("z",lambda self: self.pt * self.awkward.numpy.sinh(self.eta))
    
    @property
    def t(self):
        return self._trymemo("t",lambda self: self.awkward.numpy.hypot(self.mass, self.p))

    @property
    def pt(self):
        return self["fPt"]
    
    @property
    def pt2(self):
        return self["fPt"]**2
    
    @property
    def perp(self):
        return self["fPt"]
    
    @property
    def perp2(self):
        return self["fPt"]**2
    
    @property
    def eta(self):
        return self["fEta"]
    
    @property
    def phi(self):
        return self["fPhi"]
    
    @property
    def mass(self):
        return self["fMass"]

    @property
    def mass2(self):
        return self["fMass"]**2
    
    @property
    def mag(self):
        return self["fMass"]

    @property
    def mag2(self):
        return self["fMass"]**2

    @property
    def mt(self):
        return self.awkward.numpy.sqrt(self.mt2)

    @property
    def mt2(self):
        return self["fMass"]**2 + self["fPt"]**2

    @property
    def p(self):
        return self._trymemo("p",lambda self: self["fPt"]*self.awkward.numpy.cosh(self["fEta"]))
    
    @property
    def p2(self):
        return self.p**2

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if "out" in kwargs:
            raise NotImplementedError("in-place operations not supported")

        if method != "__call__":
            return NotImplemented

        inputs = list(inputs)
        for i in range(len(inputs)):
            if isinstance(inputs[i], self.awkward.numpy.ndarray) and inputs[i].dtype == self.awkward.numpy.dtype(object) and len(inputs[i]) > 0:
                idarray = self.awkward.numpy.frombuffer(inputs[i], dtype=self.awkward.numpy.uintp)
                if (idarray == idarray[0]).all():
                    inputs[i] = inputs[i][0]

        if ufunc is self.awkward.numpy.multiply or ufunc is self.awkward.numpy.divide:
            if sum(isinstance(x, PtEtaPhiMassArrayMethods) for x in inputs) > 1:
                raise ValueError("cannot multiply or divide two PtEtaPhiMassArrayMethods")
            this_input = None
            for i in range(len(inputs)):
                if isinstance(inputs[i], PtEtaPhiMassArrayMethods) and not isinstance(inputs[i], self.awkward.JaggedArray) and this_input is None:
                    this_input = inputs[i]
                    inputs[i] = self.awkward.Table(fPt=inputs[i]['fPt'], fMass=inputs[i]['fMass'])

            out = super(PtEtaPhiMassArrayMethods, self).__array_ufunc__(ufunc, method, *inputs, **kwargs)
            if this_input is not None:
                out['fEta'] = this_input['fEta']
                out['fPhi'] = this_input['fPhi']
                out.__class__ = this_input.__class__
            return out

        else:
            return super(PtEtaPhiMassArrayMethods, self).__array_ufunc__(ufunc, method, *inputs, **kwargs)

PtEtaPhiMassJaggedArrayMethods = PtEtaPhiMassArrayMethods.mixin(PtEtaPhiMassArrayMethods, awkward.JaggedArray)

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

    def _to_cartesian(self):
        return TLorentzVector(self.x,self.y,self.z,self.t)
    
    def __repr__(self):
        return "TLorentzVector(x={0:.5g}, y={1:.5g}, z={2:.5g}, t={3:.5g})".format(self._fP._fX, self._fP._fY, self._fP._fZ, self._fE)

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        return isinstance(other, Methods) and self.x == other.x and self.y == other.y and self.z == other.z and self.t == other.t

    def _scalar(self, operator, scalar, reverse=False):
        cart = self._to_cartesian()
        if not isinstance(scalar, (numbers.Number, self.awkward.numpy.number)):
            raise TypeError("cannot {0} a TLorentzVector with a {1}".format(operator.__name__, type(scalar).__name__))
        if reverse:
            return TLorentzVector(operator(scalar, cart.x), operator(scalar, cart.y), operator(scalar, cart.z), operator(scalar, cart.t))
        else:
            return TLorentzVector(operator(cart.x, scalar), operator(cart.y, scalar), operator(cart.z, scalar), operator(cart.t, scalar))

    def _vector(self, operator, vector, reverse=False):
        cart = self._to_cartesian()
        if not isinstance(vector, Methods):
            raise TypeError("cannot {0} a TLorentzVector with a {1}".format(operator.__name__, type(vector).__name__))
        if reverse:
            return TLorentzVector(operator(vector.x, cart.x), operator(vector.y, cart.y), operator(vector.z, cart.z), operator(vector.t, cart.t))
        else:
            return TLorentzVector(operator(cart.x, vector.x), operator(cart.y, vector.y), operator(cart.z, vector.z), operator(cart.t, vector.t))

    def _unary(self, operator):
        cart = self._to_cartesian()
        return TLorentzVector(operator(cart.x), operator(cart.y), operator(cart.z), operator(cart.t))

    @property
    def pt(self):
        return math.sqrt(self.pt2)

    @property
    def eta(self):
        return math.asinh(self.z / math.sqrt(self.x**2 + self.y**2))

    @property
    def phi(self):
        return self.p3.phi

    @property
    def mass(self):
        return math.sqrt(self.mag2)

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
        x, y, z = p3
        return self.__class__(x, y, z, t)

    def rotate_euler(self, phi=0, theta=0, psi=0):
        p3, t = self._rotate_euler(phi, theta, psi)
        x, y, z = p3
        return self.__class__(x, y, z, t)

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

class PtEtaPhiMassMethods(Methods):
    _arraymethods = PtEtaPhiMassArrayMethods
    
    @property
    def pt(self):
        return self._fPt
    
    @property
    def eta(self):
        return self._fEta
    
    @property
    def phi(self):
        return self._fPhi
    
    @property
    def mass(self):
        return self._fMass
    
    @property
    def mag(self):
        return self._fMass
    
    @property
    def mag2(self):
        return self._fMass**2
    
    @property
    def mt(self):
        out = self.mt2
        if out >= 0:
            return math.sqrt(out)
        else:
            return -math.sqrt(out)
    
    @property
    def mt2(self):
        return self._fMass**2 + self._fPt**2
    
    @property
    def p3(self):
        return uproot_methods.classes.TVector3.TVector3(self.x, self.y, self.z)
    
    @property
    def x(self):
        return self.pt * self.awkward.numpy.cos(self.phi)
    
    @property
    def y(self):
        return self.pt * self.awkward.numpy.sin(self.phi)
    
    @property
    def z(self):
        return self.pt * self.awkward.numpy.sinh(self.eta)
    
    @property
    def t(self):
        x = self.x
        y = self.y
        z = self.z
        mass = self.mass
        return self.awkward.numpy.sqrt(x*x + y*y + z*z + mass*mass*self.awkward.numpy.sign(mass))
    
    def __repr__(self):
        return "PtEtaPhiMassLorentzVector(pt={0:.5g}, eta={1:.5g}, phi={2:.5g}, mass={3:.5g})".format(self._fPt, self._fEta, self._fPhi, self._fMass)

class PtEtaPhiMassLorentzVectorArray(PtEtaPhiMassArrayMethods, uproot_methods.base.ROOTMethods.awkward.ObjectArray):
    def __init__(self, pt, eta, phi, mass):
        if isinstance(pt, awkward.array.jagged.JaggedArray) or isinstance(eta, awkward.array.jagged.JaggedArray) or isinstance(phi, awkward.array.jagged.JaggedArray) or isinstance(mass, awkward.array.jagged.JaggedArray):
            raise TypeError("PtEtaPhiMassLorentzVectorArray constructor arguments must not be jagged; use TLorentzVectorArray.from_ptetaphim for jaggedness-handling")
        self._initObjectArray(self.awkward.Table())
        self["fPt"]   = pt
        self["fEta"]  = eta
        self["fPhi"]  = phi
        self["fMass"] = mass

    @property
    def pt(self):
        return self["fPt"]
    
    @pt.setter
    def pt(self, value):
        self["fPt"] = value

    @property
    def eta(self):
        return self["fEta"]
    
    @eta.setter
    def eta(self, value):
        self["fEta"] = value

    @property
    def phi(self):
        return self["fPhi"]
    
    @phi.setter
    def phi(self, value):
        self["fPhi"] = value

    @property
    def mass(self):
        return self["fMass"]
    
    @mass.setter
    def mass(self, value):
        self["fMass"] = value

class TLorentzVectorArray(ArrayMethods, uproot_methods.base.ROOTMethods.awkward.ObjectArray):

    def __init__(self, x, y, z, t):
        if isinstance(x, awkward.array.jagged.JaggedArray) or isinstance(y, awkward.array.jagged.JaggedArray) or isinstance(z, awkward.array.jagged.JaggedArray) or isinstance(t, awkward.array.jagged.JaggedArray):
            raise TypeError("TLorentzVectorArray constructor arguments must not be jagged; use TLorentzVectorArray.from_cartesian for jaggedness-handling")
        self._initObjectArray(self.awkward.Table())
        self["fX"] = x
        self["fY"] = y
        self["fZ"] = z
        self["fE"] = t

    @classmethod
    def origin(cls, shape, dtype=None):
        if dtype is None:
            dtype = cls.awkward.numpy.float64
        return cls(cls.awkward.numpy.zeros(shape, dtype=dtype),
                   cls.awkward.numpy.zeros(shape, dtype=dtype),
                   cls.awkward.numpy.zeros(shape, dtype=dtype),
                   cls.awkward.numpy.zeros(shape, dtype=dtype))

    @classmethod
    def origin_like(cls, array):
        return array * 0.0

    @classmethod
    def from_p3(cls, p3, t):
        return cls.from_cartesian(p3.x, p3.y, p3.z, t)

    @classmethod
    @awkward.util.wrapjaggedmethod(JaggedArrayMethods)
    def from_cartesian(cls, x, y, z, t):
        return cls(x, y, z, t)

    @classmethod
    @awkward.util.wrapjaggedmethod(JaggedArrayMethods)
    def from_spherical(cls, r, theta, phi, t):
        return cls.from_p3(uproot_methods.classes.TVector3.TVector3Array.from_spherical(r, theta, phi), t)

    @classmethod
    @awkward.util.wrapjaggedmethod(JaggedArrayMethods)
    def from_cylindrical(cls, rho, phi, z, t):
        return cls.from_p3(uproot_methods.classes.TVector3.TVector3Array.from_cylindrical(rho, phi, z), t)

    @classmethod
    @awkward.util.wrapjaggedmethod(JaggedArrayMethods)
    def from_xyzm(cls, x, y, z, m):
        return cls(x, y, z, cls.awkward.numpy.sqrt(x*x + y*y + z*z + m*m*cls.awkward.numpy.sign(m)))

    @classmethod
    @awkward.util.wrapjaggedmethod(JaggedArrayMethods)
    def from_ptetaphi(cls, pt, eta, phi, energy):
        out = cls(pt * cls.awkward.numpy.cos(phi),
                  pt * cls.awkward.numpy.sin(phi),
                  pt * cls.awkward.numpy.sinh(eta),
                  energy)
        out._memo_pt = pt
        out._memo_eta = eta
        out._memo_phi = phi
        return out

    @classmethod
    def from_ptetaphie(cls, pt, eta, phi, energy):
        return cls.from_ptetaphi(pt, eta, phi, energy)

    @classmethod
    @awkward.util.wrapjaggedmethod(PtEtaPhiMassJaggedArrayMethods)
    def from_ptetaphim(cls, pt, eta, phi, mass):
        return PtEtaPhiMassLorentzVectorArray(pt,eta,phi,mass)

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

class PtEtaPhiMassLorentzVector(PtEtaPhiMassMethods):
    def __init__(self, pt, eta, phi, mass):
        self._fPt   = float(pt)
        self._fEta  = float(eta)
        self._fPhi  = float(phi)
        self._fMass = float(mass)
    
    @property
    def pt(self):
        return self._fPt
    
    @pt.setter
    def pt(self,value):
        self._fPt = value
                                           
    @property
    def eta(self):
        return self._fEta
                                       
    @eta.setter
    def eta(self, value):
        self._fEta = value
                                       
    @property
    def phi(self):
        return self._fPhi
                                       
    @phi.setter
    def phi(self, value):
        self._fPhi = value
                                       
    @property
    def mass(self):
        return self._fMass
                                       
    @mass.setter
    def mass(self, value):
        self._fMass = value
                                       
class TLorentzVector(Methods):
    def __init__(self, x, y, z, t):
        self._fP = uproot_methods.classes.TVector3.TVector3(float(x), float(y), float(z))
        self._fE = float(t)

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
    def from_ptetaphie(cls, pt, eta, phi, energy):
        return cls.from_ptetaphi(pt, eta, phi, energy)
    
    @classmethod
    def from_ptetaphim(cls, pt, eta, phi, mass):
        return PtEtaPhiMassLorentzVector(pt,eta,phi,mass)

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
