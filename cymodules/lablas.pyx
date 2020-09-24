# cython: language_level=3
# cython: infer_types=True
# cython: boundscheck=False
# cython: wraparound=False
import numba
import numpy as np
cimport numpy as np
cimport scipy.linalg.cython_blas as blas
cimport scipy.linalg.cython_lapack as lapack
cimport cython


ctypedef float complex floatcomplex
ctypedef double complex doublecomplex


ctypedef fused realcomp:
    float
    double
    float complex
    double complex


def xgels(realcomp[:, ::1] a, realcomp[::1] b):
    """
    Find the minimum norm solution of an overdetermiend system
        minimize \| a*x - b \|
    with a m-by-n-array, b m-array, m > n.
    """
    cdef:
        int m, n, nrhs, lda, ldb, lwork, info
        realcomp get_lwork
        # since xgels overwrites a and b, make copies and convert to F-contiguous
        realcomp[::1, :] acpy = a.copy_fortran()
        realcomp[::1] bcpy = b.copy_fortran()
        realcomp[::1] workv
    m = a.shape[0]
    n = a.shape[1]
    nrhs = 1
    lda = a.shape[0]
    ldb = b.shape[0]

    # request optimal work array size, then do the actual computation
    lwork = -1
    if realcomp is float:
        lapack.sgels('N', &m, &n, &nrhs, &acpy[0, 0], &lda, &bcpy[0], &ldb, &get_lwork, &lwork, &info)
        lwork = <int> get_lwork
        workv = np.empty(lwork, dtype=np.float32)
        lapack.sgels('N', &m, &n, &nrhs, &acpy[0, 0], &lda, &bcpy[0], &ldb, &workv[0], &lwork, &info)
    elif realcomp is double:
        lapack.dgels('N', &m, &n, &nrhs, &acpy[0, 0], &lda, &bcpy[0], &ldb, &get_lwork, &lwork, &info)
        lwork = <int> get_lwork
        workv = np.empty(lwork, dtype=np.float64)
        lapack.dgels('N', &m, &n, &nrhs, &acpy[0, 0], &lda, &bcpy[0], &ldb, &workv[0], &lwork, &info)
    elif realcomp is floatcomplex:
        lapack.cgels('N', &m, &n, &nrhs, &acpy[0, 0], &lda, &bcpy[0], &ldb, &get_lwork, &lwork, &info)
        lwork = <int> get_lwork.real
        workv = np.empty(lwork, dtype=np.complex64)
        lapack.cgels('N', &m, &n, &nrhs, &acpy[0, 0], &lda, &bcpy[0], &ldb, &workv[0], &lwork, &info)
    elif realcomp is doublecomplex:
        lapack.zgels('N', &m, &n, &nrhs, &acpy[0, 0], &lda, &bcpy[0], &ldb, &get_lwork, &lwork, &info)
        lwork = <int> get_lwork.real
        workv = np.empty(lwork, dtype=np.complex128)
        lapack.zgels('N', &m, &n, &nrhs, &acpy[0, 0], &lda, &bcpy[0], &ldb, &workv[0], &lwork, &info)

    return np.asarray(bcpy[:n])
