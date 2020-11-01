#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot-methods/blob/master/LICENSE

import re

__version__ = "0.8.0"
version = __version__
version_info = tuple(re.split(r"[-\.]", __version__))

del re
