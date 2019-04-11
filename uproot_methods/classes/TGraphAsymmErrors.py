#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot-methods/blob/master/LICENSE

import uproot_methods.base

class Methods(uproot_methods.base.ROOTMethods):
	
	@property
	def xerrorshigh(self):
		return self._fEXhigh
		
	@property
	def xerrorslow(self):
		return self._fEXlow
		
	@property
	def yerrorshigh(self):
		return self._fEYhigh
		
	@property
	def yerrorslow(self):
		return self._fEYlow
		
	def matplotlib(self, showtitle=True, show=False, **kwargs):
		import matplotlib.pyplot as pyplot
		
		_xerrs = [self.xerrorslow, self.xerrorshigh]
		_yerrs = [self.yerrorslow, self.yerrorshigh]

		_xlabel = _decode(self.xlabel if self.xlabel is not None else "")
		_ylabel = _decode(self.ylabel if self.ylabel is not None else "")
		
		pyplot.errorbar(self.xvalues, self.yvalues, xerr=_xerrs, yerr=_yerrs, **kwargs)
		pyplot.xlabel(_xlabel)
		pyplot.ylabel(_ylabel)
		if showtitle:
			_title = _decode(self.title)
			pyplot.title(_title)
			
		if show:
			pyplot.show()
			
def _decode(sequence):
	return sequence.decode() if isinstance(sequence, bytes) else sequence
