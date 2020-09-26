# cython: language_level=3
# cython: infer_types=True
import numpy as np
cimport numpy as np
cimport scipy.linalg.cython_blas as blas
cimport scipy.linalg.cython_lapack as lapack
cimport cython
from libc.math cimport fabs


ctypedef fused sdcz:
    float
    double
    float complex
    double complex


cpdef dtype_cy2np(dtype):
    if dtype == 'float':
        return np.float32
    elif dtype == 'double':
        return np.float64
    elif dtype == 'float complex':
        return np.complex64
    elif dtype == 'double complex':
        return np.complex128
    else:
        return -1


@cython.boundscheck(False)
@cython.wraparound(False)
def omp(double[:, ::1] a, double[::1] b, int s):
    # initialize result vectors
    x = np.zeros(a.shape[1], dtype=dtype_cy2np(cython.typeof(a[0, 0])))
    S = np.zeros(a.shape[1], dtype=np.uint8)

    cdef:
        Py_ssize_t i
        int m, n, lda, ldb, incx, incy, argmaximum, nrhs, info, j, counter, lwork
        double alpha, beta, maximum, get_lwork
        double[::1] xv, workv, bcpy
        double[::1, :] atmp, acpy
        np.uint8_t[::1] Sv
    acpy = a.copy_fortran()
    bcpy = b.copy()
    atmp = a[:, :s].copy_fortran()
    xv = x
    Sv = S
    m = a.shape[0]
    n = a.shape[1]
    lda = a.shape[0]
    ldb = b.shape[0]
    incx = 1
    incy = 1
    nrhs = 1
    lwork = 1

    # request maximum needed size of work array
    lapack.dgels('N', &m, &s, &nrhs, &atmp[0, 0], &lda, &b[0], &ldb, &get_lwork, &lwork, &info)
    lwork = <int> get_lwork
    workv = np.empty(lwork, dtype=np.float64)

    # since bcpy is overwritten by dgemv and dgels, the original vector needs to be restored via copying repeatedly
    for j in range(1, s+1):
        if j > 1:
            bcpy[:] = b
        # bcpy <-- -acpy @ x + bcpy
        alpha = -1.0
        beta = 1.0
        blas.dgemv('N', &m, &n, &alpha, &acpy[0, 0], &lda, &xv[0], &incx, &beta, &bcpy[0], &incy)

        # x <-- acpy.T @ bcpy
        alpha = 1.0
        beta = 0.0
        blas.dgemv('T', &m, &n, &alpha, &acpy[0, 0], &lda, &bcpy[0], &incx, &beta, &xv[0], &incy)

        # argmax(abs(x))
        maximum = 0.0
        argmaximum = 0
        for i in range(xv.shape[0]):
            if fabs(xv[i]) > maximum:
                maximum = fabs(xv[i])
                argmaximum = i
        Sv[argmaximum] = 1

        # atmp consists of the columns i of acpy which correspond to S[i] == 1
        counter = 0
        for i in range(Sv.shape[0]):
            if Sv[i] > 0:
                atmp[:, counter] = acpy[:, i]
                counter += 1
        # lstsq
        bcpy[:] = b
        lapack.dgels('N', &m, &j, &nrhs, &atmp[0, 0], &lda, &bcpy[0], &ldb, &workv[0], &lwork, &info)

        # extract lstsq solution
        xv[:] = 0.0
        counter = 0
        for i in range(Sv.shape[0]):
            if Sv[i] > 0:
                xv[i] = bcpy[counter]
                counter += 1

    return x, S
