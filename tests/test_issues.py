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

    def test_issue10(self):
        p4 = TLorentzVectorArray.from_ptetaphim(awkward.JaggedArray.fromiter([[1.0]]), awkward.JaggedArray.fromiter([[1.0]]), awkward.JaggedArray.fromiter([[1.0]]), awkward.JaggedArray.fromiter([[1.0]]))
        assert p4.mass.tolist() == [[1.0]]
        assert p4[0].mass.tolist() == [0.9999999999999999]
        assert p4[0][0].mass == 0.9999999999999999
        assert type(p4.mass) is awkward.JaggedArray
        assert type(p4.x) is awkward.JaggedArray

        p3 = TVector3Array.from_cylindrical(awkward.JaggedArray.fromiter([[1.0]]), awkward.JaggedArray.fromiter([[1.0]]), awkward.JaggedArray.fromiter([[1.0]]))
        assert p3.rho.tolist() == [[1.0]]
        assert p3[0].rho.tolist() == [1.0]
        assert p3[0][0].rho == 1.0
        assert type(p3.rho) is awkward.JaggedArray
        assert type(p3.x) is awkward.JaggedArray

        p2 = TVector2Array.from_polar(awkward.JaggedArray.fromiter([[1.0]]), awkward.JaggedArray.fromiter([[1.0]]))
        assert p2.rho.tolist() == [[1.0]]
        assert p2[0].rho.tolist() == [1.0]
        assert p2[0][0].rho == 1.0
        assert type(p2.rho) is awkward.JaggedArray
        assert type(p2.x) is awkward.JaggedArray
