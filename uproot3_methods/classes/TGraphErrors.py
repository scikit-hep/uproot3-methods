#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot3-methods/blob/master/LICENSE

import uproot3_methods.base

class Methods(uproot3_methods.base.ROOTMethods):

	@property
	def xerrors(self):
		return self._fEX

	@property
	def yerrors(self):
		return self._fEY

	def matplotlib(self, showtitle=True, show=False, **kwargs):
		import matplotlib.pyplot as pyplot

		_xlabel = _decode(self.xlabel if self.xlabel is not None else "")
		_ylabel = _decode(self.ylabel if self.ylabel is not None else "")

		pyplot.errorbar(self.xvalues, self.yvalues, xerr=self.xerrors, yerr=self.yerrors, **kwargs)
		pyplot.xlabel(_xlabel)
		pyplot.ylabel(_ylabel)
		if showtitle:
			_title = _decode(self.title)
			pyplot.title(_title)

		if show:
			pyplot.show()

def _decode(sequence):
	return sequence.decode() if isinstance(sequence, bytes) else sequence
