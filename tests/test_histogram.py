#!/usr/bin/env python

# Copyright (c) 2019, IRIS-HEP
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
import numpy as np

class Test(unittest.TestCase):
    def runTest(self):
        pass

    def test_th1(self):
        from uproot_methods.classes.TH1 import Methods, _histtype, from_numpy

        edges = np.array((0., 1., 2.))
        values = np.array([2, 3])

        h = from_numpy((values, edges))

        assert h.name is None
        assert h.numbins == 2
        assert h.title == b""
        assert h.low == 0
        assert h.high == 2
        assert h.underflows == 0
        assert h.overflows == 0

        np.testing.assert_equal(h.edges, edges)
        np.testing.assert_equal(h.values, values)
        np.testing.assert_equal(h.variances, values ** 2)

        np.testing.assert_equal(h.alledges, [-np.inf] + list(edges) + [np.inf])
        np.testing.assert_equal(h.allvalues, [0] + list(values) + [0])
        np.testing.assert_equal(h.allvariances, [0] + list(values ** 2) + [0])

        np.testing.assert_equal(h.bins, ((0, 1), (1, 2)))
        np.testing.assert_equal(h.allbins, ((-np.inf, 0), (0, 1), (1, 2), (2, np.inf)))

        assert h.interval(0) == (-np.inf, 0)
        assert h.interval(1) == (0, 1)
        assert h.interval(2) == (1, 2)
        assert h.interval(3) == (2, np.inf)
        assert h.interval(-1) == h.interval(3)
