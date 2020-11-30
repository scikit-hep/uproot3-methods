#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot3-methods/blob/master/LICENSE

import unittest

import numpy

import awkward0
import uproot3_methods
from uproot3_methods import *

import inspect

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_issue10(self):
        p4 = TLorentzVectorArray.from_ptetaphim(awkward0.JaggedArray.fromiter([[1.0]]), awkward0.JaggedArray.fromiter([[1.0]]), awkward0.JaggedArray.fromiter([[1.0]]), awkward0.JaggedArray.fromiter([[1.0]]))
        assert p4.mass.tolist() == [[1.0]]
        assert p4[0].mass.tolist() == [1.0]
        assert p4[0][0].mass == 1.0
        assert p4[0][0]._to_cartesian().mass == 0.9999999999999999
        assert type(p4.mass) is awkward0.JaggedArray
        assert type(p4.x) is awkward0.JaggedArray

        p3 = TVector3Array.from_cylindrical(awkward0.JaggedArray.fromiter([[1.0]]), awkward0.JaggedArray.fromiter([[1.0]]), awkward0.JaggedArray.fromiter([[1.0]]))
        assert p3.rho.tolist() == [[1.0]]
        assert p3[0].rho.tolist() == [1.0]
        assert p3[0][0].rho == 1.0
        assert type(p3.rho) is awkward0.JaggedArray
        assert type(p3.x) is awkward0.JaggedArray

        p2 = TVector2Array.from_polar(awkward0.JaggedArray.fromiter([[1.0]]), awkward0.JaggedArray.fromiter([[1.0]]))
        assert p2.rho.tolist() == [[1.0]]
        assert p2[0].rho.tolist() == [1.0]
        assert p2[0][0].rho == 1.0
        assert type(p2.rho) is awkward0.JaggedArray
        assert type(p2.x) is awkward0.JaggedArray

    def test_issue39(self):
        counts = [2,2,2]
        mask = [True, False, True]

        pt = awkward0.JaggedArray.fromcounts(counts, [42.71, 31.46, 58.72, 30.19, 47.75, 10.83])
        eta = awkward0.JaggedArray.fromcounts(counts, [0.54, 1.57, -2.33, -1.22, -2.03, -0.37])
        phi = awkward0.JaggedArray.fromcounts(counts, [-2.13, 0.65, 2.74, 0.36, 2.87, -0.47])

        pt = pt[mask]
        eta = eta[mask]
        phi = phi[mask]

        electrons = uproot3_methods.TLorentzVectorArray.from_ptetaphim(pt, eta, phi, 0.000511)

    def test_issue61(self):
        assert TVector2(2, 0).rotate(numpy.pi/6).rotate(-numpy.pi/6) == TVector2(2, 0)

        _xs = numpy.array([2, 0, 1])
        _ys = numpy.array([0, 2, 1])
        arr = TVector2Array.from_cartesian(_xs, _ys).rotate(numpy.pi/4).rotate(-numpy.pi/4)

        _jxs = awkward0.JaggedArray.fromiter([[2,], [], [0, 1]])
        _jys = awkward0.JaggedArray.fromiter([[0,], [], [2, 1]])
        jarr = TVector2Array.from_cartesian(_jxs, _jys).rotate(numpy.pi/3).rotate(-numpy.pi/3)
