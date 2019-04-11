#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot-methods/blob/master/LICENSE

def hasmethods(name):
    if name not in globals() and name in hasmethods.loaders:
        globals()[name] = hasmethods.loaders[name].load_module(name)
    return name in globals() and isinstance(getattr(globals()[name], "Methods", None), type)

import pkgutil

hasmethods.loaders = dict([(module_name, loader.find_module(module_name)) for loader, module_name, is_pkg in pkgutil.walk_packages(__path__)])

del pkgutil
