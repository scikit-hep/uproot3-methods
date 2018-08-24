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

from uproot_methods import *
from awkward import *

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_vector2(self):
        a = TVector2(4.4, 5.5)
        self.assertEqual(a.dot(a), 49.61)

        print(a + TVector2(1000, 2000))

    def test_vector2_array(self):
        a = TVector2Array(numpy.zeros(10), numpy.arange(10))
        self.assertEqual(a.tolist(), [TVector2(0, 0), TVector2(0, 1), TVector2(0, 2), TVector2(0, 3), TVector2(0, 4), TVector2(0, 5), TVector2(0, 6), TVector2(0, 7), TVector2(0, 8), TVector2(0, 9)])
        self.assertEqual(a[5], TVector2(0, 5))
        self.assertEqual(a.y.tolist(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(a.mag2().tolist(), [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0])
        self.assertEqual((a + TVector2(1000, 2000))[5], TVector2(1000, 2005))
        self.assertEqual((a + TVector2(1000, 2000) == TVector2Array(numpy.full(10, 1000), numpy.arange(2000, 2010))).tolist(), [True, True, True, True, True, True, True, True, True, True])
