
import numpy as np
cimport numpy as np
cimport cython
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libc.stdlib cimport rand, RAND_MAX
from cython.parallel cimport prange

cdef extern from "<random>" namespace "std":
   cdef cppclass mt19937:
      mt19937() except +
      mt19937(unsigned int) except +

@cython.boundscheck(False)
@cython.wraparound(False)
def long_2d_array_slice(np.ndarray[np.int64_t, ndim=2] array, np.ndarray[np.int64_t, ndim=1] indices):
    """Building Edge Index
    """

    cdef np.ndarray[np.int64_t, ndim=2] res = np.zeros([indices.shape[0], array.shape[1]], dtype=np.int64)
    cdef long long [:, :] array_view = array
    cdef long long [:] indices_view = indices
    cdef long long [:, :] res_view = res
    cdef Py_ssize_t total_indices = indices.shape[0]
    cdef Py_ssize_t index
    cdef long long row
    cdef Py_ssize_t col
    with nogil:
        for index in prange(total_indices, schedule="static"):
            row = indices_view[index]
            res_view[index, :] = array_view[row, :]
    return res

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
        for row in prange(total_count, schedule="static"):
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
        for row in prange(total_count, schedule="static"):
            target_view[row, :] = source_view[row, :]
    return target_array


#TODO: COMMING SOON
'''
@cython.boundscheck(False)
@cython.wraparound(False)
def neighbor_sampling(np.ndarray[np.int64_t, ndim=1] csr_row, np.ndarray[np.int64_t, ndim=1] csr_col,
                      np.ndarray[np.int64_t, ndim=1] seeds, long long neighbors, bool replace=False):
    pass

@cython.boundscheck(False)
@cython.wraparound(False)
cdef select_index():
    pass
'''





