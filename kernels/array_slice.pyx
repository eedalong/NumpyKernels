
import numpy as np
cimport numpy as np
cimport cython
from libcpp.map cimport map
from libcpp.set cimport set
from libcpp.unordered_set cimport unordered_set
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libc.stdlib cimport rand, RAND_MAX

from cython.parallel cimport prange
import time

from cython.parallel cimport prange
cimport openmp
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
        for index in prange(total_indices, schedule="static", chunksize=30):
            row = indices_view[index]
            res_view[index, :] = array_view[row, :]
    print(openmp.omp_get_num_threads())
    return res





