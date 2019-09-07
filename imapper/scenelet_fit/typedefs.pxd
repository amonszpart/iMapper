# cython: language_level=3
import numpy as np
cimport numpy as np

from cpython cimport bool


FTYPE = np.float32
ctypedef public np.float32_t FTYPE_t
ctypedef public np.float32_t DTYPEfloat32_t
DTYPE = np.float64
ctypedef public np.float64_t DTYPE_t
ITYPE = np.int32
ctypedef public np.int32_t ITYPE_t
ctypedef public np.int32_t DTYPEint32_t
