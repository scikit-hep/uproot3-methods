#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot-methods/blob/master/LICENSE

import importlib

# uses numpy, not awkward.numpy, because it operates on TFile data, not TTree data
import numpy

def towriteable(obj):
    def identity(x):
        return x

    def resolve(obj):
        def types(cls, obj):
            if cls is numpy.ndarray:
                yield ("numpy", "ndarray", len(obj.shape), str(obj.dtype))
            elif cls.__module__ == "pandas.core.frame" and cls.__name__ == "DataFrame":
                yield ("pandas.core.frame", "DataFrame", obj.index.__class__.__name__, set(obj.columns))
            else:
                yield (cls.__module__, cls.__name__)
            for x in cls.__bases__:
                for y in types(x, obj):
                    yield y

        if any(x == ("builtins", "bytes") or x == ("builtins", "str") or x == ("__builtin__", "str") or x == ("__builtin__", "unicode") for x in types(obj.__class__, obj)):
            return (None, None, "uproot.write.objects.TObjString", "TObjString")

        # made with numpy.histogram
        elif isinstance(obj, tuple) and len(obj) == 2 and any(x[:2] == ("numpy", "ndarray") for x in types(obj[0].__class__, obj[0])) and any(x[:2] == ("numpy", "ndarray") for x in types(obj[1].__class__, obj[1])) and len(obj[0].shape) == 1 and len(obj[1].shape) == 1 and obj[0].shape[0] == obj[1].shape[0] - 1:
            return ("uproot_methods.classes.TH1", "from_numpy", "uproot.write.objects.TH", "TH")

        # made with numpy.histogram2d
        elif isinstance(obj, tuple) and len(obj) == 3 and any(x[:2] == ("numpy", "ndarray") for x in types(obj[0].__class__, obj[0])) and any(x[:2] == ("numpy", "ndarray") for x in types(obj[1].__class__, obj[1])) and any(x[:2] == ("numpy", "ndarray") for x in types(obj[2].__class__, obj[2])) and len(obj[0].shape) == 2 and len(obj[1].shape) == 1 and len(obj[2].shape) == 1 and obj[0].shape[0] == obj[1].shape[0] - 1 and obj[0].shape[1] == obj[2].shape[0] - 1:
            return ("uproot_methods.classes.TH2", "from_numpy", "uproot.write.objects.TH", "TH")

        # made with numpy.histogramdd (2-dimensional)
        elif isinstance(obj, tuple) and len(obj) == 2 and any(x[:2] == ("numpy", "ndarray") for x in types(obj[0].__class__, obj[0])) and isinstance(obj[1], list) and len(obj[1]) == 2 and any(x[:2] == ("numpy", "ndarray") for x in types(obj[1][0].__class__, obj[1][0])) and any(x[:2] == ("numpy", "ndarray") for x in types(obj[1][1].__class__, obj[1][1])) and len(obj[0].shape) == 2 and len(obj[1][0].shape) == 1 and len(obj[1][1].shape) == 1 and obj[0].shape[0] == obj[1][0].shape[0] - 1 and obj[0].shape[1] == obj[1][1].shape[0] - 1:
            return ("uproot_methods.classes.TH2", "from_numpy", "uproot.write.objects.TH", "TH")

        elif any(x[:3] == ("pandas.core.frame", "DataFrame", "IntervalIndex") and "count" in x[3] for x in types(obj.__class__, obj)):
            return ("uproot_methods.classes.TH1", "from_pandas", "uproot.write.objects.TH", "TH")

        elif any(x == ("physt.histogram1d", "Histogram1D") for x in types(obj.__class__, obj)):
            return ("uproot_methods.classes.TH1", "from_physt", "uproot.write.objects.TH", "TH")

        elif any(x == ("uproot_methods.classes.TH1", "Methods") or x == ("TH1", "Methods") for x in types(obj.__class__, obj)):
            return (None, None, "uproot.write.objects.TH", "TH")

        elif any(x == ("uproot_methods.classes.TH2", "Methods") or x == ("TH2", "Methods") for x in types(obj.__class__, obj)):
            return (None, None, "uproot.write.objects.TH", "TH")

        elif any(x == ("uproot_methods.classes.TH3", "Methods") or x == ("TH3", "Methods") for x in types(obj.__class__, obj)):
            return (None, None, "uproot.write.objects.TH", "TH")

        else:
            raise TypeError("type {0} from module {1} is not writeable by uproot".format(obj.__class__.__name__, obj.__class__.__module__))

    convertmod, convertfcn, mod, cls = resolve(obj)

    if convertfcn is not None:
        convert = getattr(importlib.import_module(convertmod), convertfcn)
        obj = convert(obj)

    cls = getattr(importlib.import_module(mod), cls)
    return cls(obj)
