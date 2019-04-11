#!/usr/bin/env python

# BSD 3-Clause License; see https://github.com/scikit-hep/uproot-methods/blob/master/LICENSE

import uproot_methods.base

class Methods(uproot_methods.base.ROOTMethods):
	def __repr__(self):
		if self._fName is None:
			return "<{0} at 0x{1:012x}>".format(self._classname, id(self))
		else:
			return "<{0} {1} 0x{2:012x}>".format(self._classname, repr(self._fName), id(self))
			
	@property
	def name(self):
		return self._fName

	@property
	def title(self):
		return self._fTitle
		
	@property
	def maximum(self):
		return self._fMaximum
		
	@property
	def minimum(self):
		return self._fMinimum
		
	@property
	def npoints(self):
		return self._fNpoints
		
	@property
	def xvalues(self):
		return self._fX
		
	@property
	def yvalues(self):
		return self._fY
		
	@property
	def xlabel(self):
		if self._fHistogram is None:
			return None
		elif getattr(self._fHistogram, "_fXaxis", None) is None:
			return None
		else: 
			return getattr(self._fHistogram._fXaxis, "_fTitle", None)

	@property
	def ylabel(self):
		if self._fHistogram is None:
			return None
		elif getattr(self._fHistogram, "_fYaxis", None) is None:
			return None
		else: 
			return getattr(self._fHistogram._fYaxis, "_fTitle", None)

	def matplotlib(self, showtitle=True, show=False, fmt="", **kwargs):
		import matplotlib.pyplot as pyplot
		
		_xlabel = _decode(self.xlabel if self.xlabel is not None else "")
		_ylabel = _decode(self.ylabel if self.ylabel is not None else "")
		
		pyplot.plot(self.xvalues, self.yvalues, fmt, **kwargs)
		pyplot.xlabel(_xlabel)
		pyplot.ylabel(_ylabel)
		if showtitle:
			_title = _decode(self.title)
			pyplot.title(_title)
			
		if show:
			pyplot.show()
			
def _decode(sequence):
	return sequence.decode() if isinstance(sequence, bytes) else sequence
