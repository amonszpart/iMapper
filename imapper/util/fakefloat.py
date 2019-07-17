import numpy as np


class FakeFloat(float):
    """JSON encodable wrapper for float values"""

    def __init__(self, value):
        super(FakeFloat, self).__init__(value)
        self._value = value
        assert self._value == value, \
            "Something went wrong: %f %f" % (self._value, value)

    def __repr__(self):
        return str(self._value)


def default_encode(o):
    """JSON encoder for float values"""
    if isinstance(o, float) or isinstance(o, np.float32):
        return FakeFloat(o)
    else:
        print("type: %s" % type(o))
    raise TypeError(repr(o) + " is not JSON serializable")
