
import numpy as np
cimport numpy as np
from cython.parallel cimport prange
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def long_2d_array_row_copy(np.ndarray[np.int64_t, ndim=2] target_array, np.ndarray[np.int64_t, ndim=2] source_array):
    cdef long long [:, :] target_view = target_array
    cdef long long [:, :] source_view = source_array
    cdef Py_ssize_t total_count = source_array.shape[0]
    cdef long long length = source_array.shape[1]
    cdef Py_ssize_t index
    cdef long long row
    cdef Py_ssize_t col
    with nogil:
        for row in prange(total_count, schedule="static", chunksize=30):
            target_view[row, :length] = source_view[row, :]
    return target_array

@cython.boundscheck(False)
@cython.wraparound(False)
def long_2d_array_col_copy(np.ndarray[np.int64_t, ndim=2] target_array, np.ndarray[np.int64_t, ndim=2] source_array):
    cdef long long [:, :] target_view = target_array
    cdef long long [:, :] source_view = source_array
    cdef Py_ssize_t total_count = source_array.shape[0]
    cdef long long length = source_array.shape[1]
    cdef Py_ssize_t index
    cdef long long row
    cdef Py_ssize_t col
    with nogil:
        for row in prange(total_count, schedule="static", chunksize=30):
            target_view[row, :] = source_view[row, :]
    return target_array

