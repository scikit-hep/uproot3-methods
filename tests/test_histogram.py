#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot-methods/blob/master/LICENSE

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
