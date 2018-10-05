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