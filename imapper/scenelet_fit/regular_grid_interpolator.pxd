# cython: language_level=3
# import numpy as np
cimport numpy as np

cimport typedefs
from typedefs cimport DTYPE_t, DTYPEfloat32_t, DTYPEint32_t, bool

cdef class RegularGridInterpolator(object):
    # this class is based on code originally programmed by Johannes Buchner,
    # see https://github.com/JohannesBuchner/regulargrid

    cdef object method
    cdef bool bounds_error
    cdef DTYPEfloat32_t fill_value
    cdef object grid
    cdef np.ndarray grid_array
    cdef object values
