import uproot3_methods.base


def _decode(seq):
    if isinstance(seq, bytes):
        return seq.decode("UTF-8")
    else:
        return seq


class Methods(uproot3_methods.base.ROOTMethods):
    def __repr__(self):
        if self._fName is None:
            return "<{0} at 0x{1:012x}>".format(_decode(self._classname), id(self))
        else:
            return "<{0} {1} 0x{2:012x}>".format(
                _decode(self._classname), _decode(self._fName), id(self)
            )

    def __str__(self):
        return str(self._fVal)

    @property
    def name(self):
        return _decode(self._fName)

    @property
    def value(self):
        return self._fVal
