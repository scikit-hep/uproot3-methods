#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot3-methods/blob/master/LICENSE

import importlib
import re

def transformer(name):
    m = re.match(r"^([a-zA-Z_][a-zA-Z_0-9]*)(\.[a-zA-Z_][a-zA-Z_0-9]*)*$", name)
    if m is None:
        raise ValueError("profile name must match \"identifier(.identifier)*\"")
    return getattr(importlib.import_module("uproot3_methods.profiles." + m.string), "transform")
