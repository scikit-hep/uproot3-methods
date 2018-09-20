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

import unittest

import numpy

import awkward
import uproot_methods
from uproot_methods import *

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_vector2(self):
        a = TVector2(4.4, 5.5)
        self.assertEqual(a.dot(a), 49.61)
        self.assertEqual(a + TVector2(1000, 2000), TVector2(1004.4, 2005.5))
        self.assertEqual(a - TVector2(1000, 2000), TVector2(-995.6, -1994.5))
        self.assertEqual(TVector2(1000, 2000) - a, TVector2(995.6, 1994.5))
        self.assertEqual(a * 1000, TVector2(4400, 5500))
        self.assertEqual(1000 * a, TVector2(4400, 5500))
        self.assertEqual(a / 1000, TVector2(0.0044, 0.0055))
        self.assertEqual(1000 / a, TVector2(227.27272727272725, 181.8181818181818))
        self.assertEqual(a**2, 49.61)
        self.assertEqual(a**1, 7.043436661176133)
        self.assertEqual(abs(a), 7.043436661176133)
        self.assertEqual(-a, TVector2(-4.4, -5.5))
        self.assertEqual(+a, TVector2(4.4, 5.5))

        a += TVector2(100, 200)
        self.assertEqual(a, TVector2(104.4, 205.5))
        a *= 10
        self.assertEqual(a, TVector2(1044, 2055))

    def test_vector2_array(self):
        a = TVector2Array(numpy.zeros(10), numpy.arange(10))
        self.assertEqual(a.tolist(), [TVector2(0, 0), TVector2(0, 1), TVector2(0, 2), TVector2(0, 3), TVector2(0, 4), TVector2(0, 5), TVector2(0, 6), TVector2(0, 7), TVector2(0, 8), TVector2(0, 9)])
        self.assertEqual(a[5], TVector2(0, 5))
        self.assertEqual(a.y.tolist(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(a.mag2.tolist(), [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0])
        self.assertEqual((a + TVector2(1000, 2000))[5], TVector2(1000, 2005))
        self.assertEqual((a + TVector2(1000, 2000) == TVector2Array(numpy.full(10, 1000), numpy.arange(2000, 2010))).tolist(), [True, True, True, True, True, True, True, True, True, True])
        self.assertEqual((a**2).tolist(), [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0])
        self.assertEqual((a**3).tolist(), [0.0, 1.0, 8.0, 27.0, 64.0, 125.0, 216.0, 343.0, 512.0, 729.0])

    def test_vector2_jagged(self):
        TVector2Jagged = type("TVector2Jagged", (awkward.JaggedArray, uproot_methods.classes.TVector2.ArrayMethods), {})
        a = TVector2Jagged.fromoffsets([0, 3, 3, 5, 10], TVector2Array(numpy.zeros(10), numpy.arange(10)))
        a._generator = uproot_methods.classes.TVector2.TVector2
        a._args = ()
        a._kwargs = {}
        self.assertEqual(a.tolist(), [[TVector2(0, 0), TVector2(0, 1), TVector2(0, 2)], [], [TVector2(0, 3), TVector2(0, 4)], [TVector2(0, 5), TVector2(0, 6), TVector2(0, 7), TVector2(0, 8), TVector2(0, 9)]])
        self.assertEqual(a.x.tolist(), [[0.0, 0.0, 0.0], [], [0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertEqual(a.y.tolist(), [[0, 1, 2], [], [3, 4], [5, 6, 7, 8, 9]])
        self.assertEqual((a + TVector2(1000, 2000)).tolist(), [[TVector2(1000, 2000), TVector2(1000, 2001), TVector2(1000, 2002)], [], [TVector2(1000, 2003), TVector2(1000, 2004)], [TVector2(1000, 2005), TVector2(1000, 2006), TVector2(1000, 2007), TVector2(1000, 2008), TVector2(1000, 2009)]])
        self.assertEqual((a + TVector2Array(numpy.full(4, 1000), numpy.arange(1000, 5000, 1000))).tolist(), [[TVector2(1000, 1000), TVector2(1000, 1001), TVector2(1000, 1002)], [], [TVector2(1000, 3003), TVector2(1000, 3004)], [TVector2(1000, 4005), TVector2(1000, 4006), TVector2(1000, 4007), TVector2(1000, 4008), TVector2(1000, 4009)]])

    def test_vector3(self):
        a = TVector3(4.4, 5.5, 0)
        self.assertEqual(a.dot(a), 49.61)
        self.assertEqual(a + TVector3(1000, 2000, 0), TVector3(1004.4, 2005.5, 0))
        self.assertEqual(a - TVector3(1000, 2000, 0), TVector3(-995.6, -1994.5, 0))
        self.assertEqual(TVector3(1000, 2000, 0) - a, TVector3(995.6, 1994.5, 0))
        self.assertEqual(a * 1000, TVector3(4400, 5500, 0))
        self.assertEqual(1000 * a, TVector3(4400, 5500, 0))
        self.assertEqual(a / 1000, TVector3(0.0044, 0.0055, 0))
        self.assertEqual(1000 / (a + TVector3(0, 0, 1)), TVector3(227.27272727272725, 181.8181818181818, 1000))
        self.assertEqual(a**2, 49.61)
        self.assertEqual(a**1, 7.043436661176133)
        self.assertEqual(abs(a), 7.043436661176133)
        self.assertEqual(-a, TVector3(-4.4, -5.5, 0))
        self.assertEqual(+a, TVector3(4.4, 5.5, 0))

        a += TVector3(100, 200, 0)
        self.assertEqual(a, TVector3(104.4, 205.5, 0))
        a *= 10
        self.assertEqual(a, TVector3(1044, 2055, 0))

    def test_vector3_array(self):
        a = TVector3Array(numpy.zeros(10), numpy.arange(10), numpy.zeros(10))
        self.assertEqual(a.tolist(), [TVector3(0, 0, 0), TVector3(0, 1, 0), TVector3(0, 2, 0), TVector3(0, 3, 0), TVector3(0, 4, 0), TVector3(0, 5, 0), TVector3(0, 6, 0), TVector3(0, 7, 0), TVector3(0, 8, 0), TVector3(0, 9, 0)])
        self.assertEqual(a[5], TVector3(0, 5, 0))
        self.assertEqual(a.y.tolist(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(a.mag2.tolist(), [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0])
        self.assertEqual((a + TVector3(1000, 2000, 0))[5], TVector3(1000, 2005, 0))
        self.assertEqual((a + TVector3(1000, 2000, 0) == TVector3Array(numpy.full(10, 1000), numpy.arange(2000, 2010), numpy.zeros(10))).tolist(), [True, True, True, True, True, True, True, True, True, True])
        self.assertEqual((a**2).tolist(), [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0])
        self.assertEqual((a**3).tolist(), [0.0, 1.0, 8.0, 27.0, 64.0, 125.0, 216.0, 343.0, 512.0, 729.0])

    def test_vector3_jagged(self):
        TVector3Jagged = type("TVector3Jagged", (awkward.JaggedArray, uproot_methods.classes.TVector3.ArrayMethods), {})
        a = TVector3Jagged.fromoffsets([0, 3, 3, 5, 10], TVector3Array(numpy.zeros(10), numpy.arange(10), numpy.zeros(10)))
        a._generator = uproot_methods.classes.TVector3.TVector3
        a._args = ()
        a._kwargs = {}
        self.assertEqual(a.tolist(), [[TVector3(0, 0, 0), TVector3(0, 1, 0), TVector3(0, 2, 0)], [], [TVector3(0, 3, 0), TVector3(0, 4, 0)], [TVector3(0, 5, 0), TVector3(0, 6, 0), TVector3(0, 7, 0), TVector3(0, 8, 0), TVector3(0, 9, 0)]])
        self.assertEqual(a.x.tolist(), [[0.0, 0.0, 0.0], [], [0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertEqual(a.y.tolist(), [[0, 1, 2], [], [3, 4], [5, 6, 7, 8, 9]])
        self.assertEqual((a + TVector3(1000, 2000, 0)).tolist(), [[TVector3(1000, 2000, 0), TVector3(1000, 2001, 0), TVector3(1000, 2002, 0)], [], [TVector3(1000, 2003, 0), TVector3(1000, 2004, 0)], [TVector3(1000, 2005, 0), TVector3(1000, 2006, 0), TVector3(1000, 2007, 0), TVector3(1000, 2008, 0), TVector3(1000, 2009, 0)]])
        self.assertEqual((a + TVector3Array(numpy.full(4, 1000), numpy.arange(1000, 5000, 1000), numpy.zeros(4))).tolist(), [[TVector3(1000, 1000, 0), TVector3(1000, 1001, 0), TVector3(1000, 1002, 0)], [], [TVector3(1000, 3003, 0), TVector3(1000, 3004, 0)], [TVector3(1000, 4005, 0), TVector3(1000, 4006, 0), TVector3(1000, 4007, 0), TVector3(1000, 4008, 0), TVector3(1000, 4009, 0)]])

    def test_lorentzvector(self):
        a = TLorentzVector(4.4, 5.5, 0, 0)
        self.assertEqual(a.dot(a), -49.61)
        self.assertEqual(a + TLorentzVector(1000, 2000, 0, 0), TLorentzVector(1004.4, 2005.5, 0, 0))
        self.assertEqual(a - TLorentzVector(1000, 2000, 0, 0), TLorentzVector(-995.6, -1994.5, 0, 0))
        self.assertEqual(TLorentzVector(1000, 2000, 0, 0) - a, TLorentzVector(995.6, 1994.5, 0, 0))
        self.assertEqual(a * 1000, TLorentzVector(4400, 5500, 0, 0))
        self.assertEqual(1000 * a, TLorentzVector(4400, 5500, 0, 0))
        self.assertEqual(a / 1000, TLorentzVector(0.0044, 0.0055, 0, 0))
        self.assertEqual(1000 / (a + TLorentzVector(0, 0, 1, 1)), TLorentzVector(227.27272727272725, 181.8181818181818, 1000, 1000))
        self.assertEqual(a**2, -49.61)
        self.assertEqual((a + TLorentzVector(0, 0, 0, 10))**1, 7.098591409568521)
        self.assertEqual(abs(a + TLorentzVector(0, 0, 0, 10)), 7.098591409568521)
        self.assertEqual(-a, TLorentzVector(-4.4, -5.5, 0, 0))
        self.assertEqual(+a, TLorentzVector(4.4, 5.5, 0, 0))

        a += TLorentzVector(100, 200, 0, 0)
        self.assertEqual(a, TLorentzVector(104.4, 205.5, 0, 0))
        a *= 10
        self.assertEqual(a, TLorentzVector(1044, 2055, 0, 0))

    def test_lorentzvector_array(self):
        a = TLorentzVectorArray(numpy.zeros(10), numpy.arange(10), numpy.zeros(10), numpy.zeros(10))
        self.assertEqual(a.tolist(), [TLorentzVector(0, 0, 0, 0), TLorentzVector(0, 1, 0, 0), TLorentzVector(0, 2, 0, 0), TLorentzVector(0, 3, 0, 0), TLorentzVector(0, 4, 0, 0), TLorentzVector(0, 5, 0, 0), TLorentzVector(0, 6, 0, 0), TLorentzVector(0, 7, 0, 0), TLorentzVector(0, 8, 0, 0), TLorentzVector(0, 9, 0, 0)])
        self.assertEqual(a[5], TLorentzVector(0, 5, 0, 0))
        self.assertEqual(a.y.tolist(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(a.mag2.tolist(), [0.0, -1.0, -4.0, -9.0, -16.0, -25.0, -36.0, -49.0, -64.0, -81.0])
        self.assertEqual((a + TLorentzVector(1000, 2000, 0, 0))[5], TLorentzVector(1000, 2005, 0, 0))
        self.assertEqual((a + TLorentzVector(1000, 2000, 0, 0) == TLorentzVectorArray(numpy.full(10, 1000), numpy.arange(2000, 2010), numpy.zeros(10), numpy.zeros(10))).tolist(), [True, True, True, True, True, True, True, True, True, True])
        self.assertEqual((a**2).tolist(), [0.0, -1.0, -4.0, -9.0, -16.0, -25.0, -36.0, -49.0, -64.0, -81.0])

    def test_lorentzvector_jagged(self):
        TLorentzVectorJagged = type("TLorentzVectorJagged", (awkward.JaggedArray, uproot_methods.classes.TLorentzVector.ArrayMethods), {})
        a = TLorentzVectorJagged.fromoffsets([0, 3, 3, 5, 10], TLorentzVectorArray(numpy.zeros(10), numpy.arange(10), numpy.zeros(10), numpy.zeros(10)))
        a._generator = uproot_methods.classes.TLorentzVector.TLorentzVector
        a._args = ()
        a._kwargs = {}
        self.assertEqual(a.tolist(), [[TLorentzVector(0, 0, 0, 0), TLorentzVector(0, 1, 0, 0), TLorentzVector(0, 2, 0, 0)], [], [TLorentzVector(0, 3, 0, 0), TLorentzVector(0, 4, 0, 0)], [TLorentzVector(0, 5, 0, 0), TLorentzVector(0, 6, 0, 0), TLorentzVector(0, 7, 0, 0), TLorentzVector(0, 8, 0, 0), TLorentzVector(0, 9, 0, 0)]])
        self.assertEqual(a.x.tolist(), [[0.0, 0.0, 0.0], [], [0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertEqual(a.y.tolist(), [[0, 1, 2], [], [3, 4], [5, 6, 7, 8, 9]])
        self.assertEqual((a + TLorentzVector(1000, 2000, 0, 0)).tolist(), [[TLorentzVector(1000, 2000, 0, 0), TLorentzVector(1000, 2001, 0, 0), TLorentzVector(1000, 2002, 0, 0)], [], [TLorentzVector(1000, 2003, 0, 0), TLorentzVector(1000, 2004, 0, 0)], [TLorentzVector(1000, 2005, 0, 0), TLorentzVector(1000, 2006, 0, 0), TLorentzVector(1000, 2007, 0, 0), TLorentzVector(1000, 2008, 0, 0), TLorentzVector(1000, 2009, 0, 0)]])
        self.assertEqual((a + TLorentzVectorArray(numpy.full(4, 1000), numpy.arange(1000, 5000, 1000), numpy.zeros(4), numpy.zeros(4))).tolist(), [[TLorentzVector(1000, 1000, 0, 0), TLorentzVector(1000, 1001, 0, 0), TLorentzVector(1000, 1002, 0, 0)], [], [TLorentzVector(1000, 3003, 0, 0), TLorentzVector(1000, 3004, 0, 0)], [TLorentzVector(1000, 4005, 0, 0), TLorentzVector(1000, 4006, 0, 0), TLorentzVector(1000, 4007, 0, 0), TLorentzVector(1000, 4008, 0, 0), TLorentzVector(1000, 4009, 0, 0)]])
