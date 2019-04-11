#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot-methods/blob/master/LICENSE

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from functools import wraps


def normalize_arrays(cls, arrays):
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

def unwrap_jagged(cls, awkcls, arrays):
    if not isinstance(arrays[0], cls.awkward.JaggedArray):
        return lambda x: x, arrays

    counts = arrays[0].counts.reshape(-1)
    offsets = awkcls.counts2offsets(counts)
    starts, stops = offsets[:-1], offsets[1:]
    starts = starts.reshape(arrays[0].starts.shape[:-1] + (-1,))
    stops = stops.reshape(arrays[0].stops.shape[:-1] + (-1,))
    wrap, arrays = unwrap_jagged(cls, awkcls, [x.flatten() for x in arrays])
    return lambda x: awkcls(starts, stops, wrap(x)), arrays

def wrapjaggedmethod(awkcls):
    def wrapjagged_decorator(func):
        @wraps(func)
        def func_wrapper(cls, *arrays):
            wrap, arrays = unwrap_jagged(cls, awkcls, normalize_arrays(cls, arrays))
            return wrap(func(cls, *arrays))
        return func_wrapper
    return wrapjagged_decorator
