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

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import awkward

class ROOTMethods(awkward.Methods):
    _arraymethods = None

    awkward = awkward

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def _normalize_arrays(cls, arrays):
        length = None
        for i in range(len(arrays)):
            if isinstance(arrays[i], Iterable):
                if length is None:
                    length = len(arrays[i])
                    break
        if length is None:
            raise TypeError("cannot construct an array if all arguments are scalar")

        arrays = list(arrays)
        jaggedtype = [cls.awkward.JaggedArray] * len(arrays)
        starts, stops = None, None
        for i in range(len(arrays)):
            if starts is None and isinstance(arrays[i], cls.awkward.JaggedArray):
                starts, stops = arrays[i].starts, arrays[i].stops

            if isinstance(arrays[i], cls.awkward.JaggedArray):
                jaggedtype[i] = type(arrays[i])

            if not isinstance(arrays[i], Iterable):
                arrays[i] = cls.awkward.numpy.full(length, arrays[i])

            arrays[i] = cls.awkward.util.toarray(arrays[i], cls.awkward.numpy.float64)

        if starts is None:
            return arrays

        for i in range(len(arrays)):
            if not isinstance(arrays[i], cls.awkward.JaggedArray) or not (cls.awkward.numpy.array_equal(starts, arrays[i].starts) and cls.awkward.numpy.array_equal(stops, arrays[i].stops)):
                content = cls.awkward.numpy.zeros(stops.max(), dtype=cls.awkward.numpy.float64)
                arrays[i] = jaggedtype[i](starts, stops, content) + arrays[i]    # invoke jagged broadcasting to align arrays

        return arrays

    @classmethod
    def _unwrap_jagged(cls, ArrayMethods, arrays):
        if not isinstance(arrays[0], cls.awkward.JaggedArray):
            return lambda x: x, arrays
        else:
            if ArrayMethods is None:
                awkcls = arrays[0].JaggedArray
            else:
                awkcls = ArrayMethods.mixin(ArrayMethods, arrays[0].JaggedArray)
            counts = arrays[0].counts.reshape(-1)
            offsets = awkcls.counts2offsets(counts)
            starts, stops = offsets[:-1], offsets[1:]
            starts = starts.reshape(arrays[0].starts.shape[:-1] + (-1,))
            stops = stops.reshape(arrays[0].stops.shape[:-1] + (-1,))
            wrap, arrays = cls._unwrap_jagged(ArrayMethods, [x.flatten() for x in arrays])
            return lambda x: awkcls(starts, stops, wrap(x)), arrays

    def _trymemo(self, name, function):
        memoname = "_memo_" + name
        wrap, (array,) = type(self)._unwrap_jagged(None, (self,))
        if not hasattr(array, memoname):
            setattr(array, memoname, function(array))
        return wrap(getattr(array, memoname))
