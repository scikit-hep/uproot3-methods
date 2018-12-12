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

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import awkward
import awkward.util

def _normalize_arrays(arrays):
    length = None
    for i in range(len(arrays)):
        if isinstance(arrays[i], Iterable):
            if length is None:
                length = len(arrays[i])
                break
    if length is None:
        raise TypeError("cannot construct an array if all arguments are scalar")

    arrays = list(arrays)
    starts, stops = None, None
    for i in range(len(arrays)):
        if starts is None and isinstance(arrays[i], awkward.JaggedArray):
            starts, stops = arrays[i].starts, arrays[i].stops

        if not isinstance(arrays[i], Iterable):
            arrays[i] = awkward.util.numpy.full(length, arrays[i])

        arrays[i] = awkward.util.toarray(arrays[i], awkward.util.numpy.float64)

    if starts is None:
        return arrays

    for i in range(len(arrays)):
        if not isinstance(arrays[i], awkward.JaggedArray) or not (awkward.util.numpy.array_equal(starts, arrays[i].starts) and awkward.util.numpy.array_equal(stops, arrays[i].stops)):
            content = awkward.util.numpy.zeros(stops.max(), dtype=awkward.util.numpy.float64)
            arrays[i] = awkward.JaggedArray(starts, stops, content) + arrays[i]    # invoke jagged broadcasting to align arrays

    return arrays

def _unwrap_jagged(ArrayMethods, arrays):
    if not isinstance(arrays[0], awkward.JaggedArray):
        return lambda x: x, arrays
    else:
        if ArrayMethods is None:
            cls = awkward.JaggedArray
        else:
            cls = ArrayMethods.mixin(ArrayMethods, awkward.JaggedArray)
        starts, stops = arrays[0].starts, arrays[0].stops
        wrap, arrays = _unwrap_jagged(ArrayMethods, [x.content for x in arrays])
        return lambda x: cls(starts, stops, wrap(x)), arrays

def memo(function):
    memoname = "_memo_" + function.__name__
    def memofunction(array):
        wrap, (array,) = _unwrap_jagged(None, (array,))
        if not hasattr(array, memoname):
            setattr(array, memoname, function(array))
        return wrap(getattr(array, memoname))
    return memofunction

class ROOTMethods(awkward.Methods):
    _arraymethods = None

    def __ne__(self, other):
        return not self.__eq__(other)
