# cython: language_level=3
# cython: infer_types=True
import numpy as np
cimport numpy as np
cimport scipy.linalg.cython_blas as blas
cimport scipy.linalg.cython_lapack as lapack
cimport cython
from libc.math cimport fabs


cdef extern from 'complex.h' nogil:
    float cabsf 'cabs' (float complex z)
    double cabs 'cabs' (double complex z)


ctypedef fused sdcz:
    float
    double
    float complex
    double complex


ctypedef fused dz:
    double
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


@cython.boundscheck(False)
@cython.wraparound(False)
def omp(sdcz[:, ::1] a, sdcz[::1] b, int s):
    r"""
    Orthogonal Matching Pursuit, a greedy algorithm to solve
        \min_x \| a x - b \|
    for a s-sparse x.
    Args:
        a: An m-by-n matrix (typically m < n).
        b: An m-dimensional right-hand side vector.
        s: The sparsity (number of non-zero entires of x)

    Returns:
        x: A s-sparse n-dimensional vector which (is supposed to) solve(s) \min_x \| A x - y \|.
        S: A n-dimensional boolean vector indicating the support of x.

    Source:
        See "A mathematical introduction to compressvie sensing" by Foucart, Section 3.23.
    """
    # get numpy data type
    dtype_np = dtype_cy2np(cython.typeof(a[0, 0]))

    # initialize result vectors
    x = np.zeros(a.shape[1], dtype=dtype_np)
    S = np.zeros(a.shape[1], dtype=np.uint8)

    cdef:
        Py_ssize_t i
        int m, n, lda, ldb, incx, incy, argmaximum, nrhs, info, j, counter, lwork
        sdcz alpha, beta, get_lwork
        sdcz[::1] xv, workv, bcpy
        sdcz[::1, :] atmp, acpy
        double maximum
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
    if sdcz == cython.float:
        lapack.sgels('N', &m, &s, &nrhs, &atmp[0, 0], &lda, &b[0], &ldb, &get_lwork, &lwork, &info)
    elif sdcz == cython.double:
        lapack.dgels('N', &m, &s, &nrhs, &atmp[0, 0], &lda, &b[0], &ldb, &get_lwork, &lwork, &info)
    elif sdcz == cython.floatcomplex:
        lapack.cgels('N', &m, &s, &nrhs, &atmp[0, 0], &lda, &b[0], &ldb, &get_lwork, &lwork, &info)
    elif sdcz == cython.doublecomplex:
        lapack.zgels('N', &m, &s, &nrhs, &atmp[0, 0], &lda, &b[0], &ldb, &get_lwork, &lwork, &info)
    lwork = <int> get_lwork
    workv = np.empty(lwork, dtype=dtype_np)

    # since bcpy is overwritten by xgemv and xgels, the original vector needs to be restored via copying repeatedly
    for j in range(1, s+1):
        if j > 1:
            bcpy[:] = b

        # compute a.conj().T @ (b - a @ x)
        if sdcz == cython.float:
            alpha = -1.0
            beta = 1.0
            # bcpy <-- -acpy @ x + bcpy
            blas.sgemv('N', &m, &n, &alpha, &acpy[0, 0], &lda, &xv[0], &incx, &beta, &bcpy[0], &incy)
            # x <-- acpy.T @ bcpy
            alpha = 1.0
            beta = 0.0
            blas.sgemv('T', &m, &n, &alpha, &acpy[0, 0], &lda, &bcpy[0], &incx, &beta, &xv[0], &incy)
        elif sdcz == cython.double:
            alpha = -1.0
            beta = 1.0
            blas.dgemv('N', &m, &n, &alpha, &acpy[0, 0], &lda, &xv[0], &incx, &beta, &bcpy[0], &incy)
            alpha = 1.0
            beta = 0.0
            blas.dgemv('T', &m, &n, &alpha, &acpy[0, 0], &lda, &bcpy[0], &incx, &beta, &xv[0], &incy)
        elif sdcz == cython.floatcomplex:
            alpha = -1.0
            beta = 1.0
            blas.cgemv('N', &m, &n, &alpha, &acpy[0, 0], &lda, &xv[0], &incx, &beta, &bcpy[0], &incy)
            alpha = 1.0
            beta = 0.0
            blas.cgemv('C', &m, &n, &alpha, &acpy[0, 0], &lda, &bcpy[0], &incx, &beta, &xv[0], &incy)
        elif sdcz == cython.doublecomplex:
            alpha = -1.0
            beta = 1.0
            blas.zgemv('N', &m, &n, &alpha, &acpy[0, 0], &lda, &xv[0], &incx, &beta, &bcpy[0], &incy)
            alpha = 1.0
            beta = 0.0
            blas.zgemv('C', &m, &n, &alpha, &acpy[0, 0], &lda, &bcpy[0], &incx, &beta, &xv[0], &incy)

        # find argmax of abs(x)
        maximum = 0.0
        argmaximum = 0
        if sdcz == cython.float or sdcz == cython.double:
            for i in range(xv.shape[0]):
                if fabs(xv[i]) > maximum:
                    maximum = fabs(xv[i])
                    argmaximum = i
        elif sdcz == cython.floatcomplex or sdcz == cython.doublecomplex:
            for i in range(xv.shape[0]):
                if cabs(xv[i]) > maximum:
                    maximum = cabs(xv[i])
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
        if sdcz == cython.float:
            lapack.sgels('N', &m, &j, &nrhs, &atmp[0, 0], &lda, &bcpy[0], &ldb, &workv[0], &lwork, &info)
        elif sdcz == cython.double:
            lapack.dgels('N', &m, &j, &nrhs, &atmp[0, 0], &lda, &bcpy[0], &ldb, &workv[0], &lwork, &info)
        elif sdcz == cython.floatcomplex:
            lapack.cgels('N', &m, &j, &nrhs, &atmp[0, 0], &lda, &bcpy[0], &ldb, &workv[0], &lwork, &info)
        elif sdcz == cython.doublecomplex:
            lapack.zgels('N', &m, &j, &nrhs, &atmp[0, 0], &lda, &bcpy[0], &ldb, &workv[0], &lwork, &info)

        # extract lstsq solution
        xv[:] = 0.0
        counter = 0
        for i in range(Sv.shape[0]):
            if Sv[i] > 0:
                xv[i] = bcpy[counter]
                counter += 1

    return x, S


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def power_iteration(dz[:, ::1] a, dz[::1] v_init, double eps, int max_iterations):
    """
    Find the largest (in absolute value) eigenvalue and corresponding
    eigenvector of a square symmetric/self-adjoint matrix A using the
    initialization vector v_init. The algorithm converges if the maximum number
    of iterations, max_iterations, is reached or if the relative change of the
    estimated eigenvalue from one iteration to the next is smaller than or
    equal to eps.

    Returns:
        mu: The largest eigenvalue.
        v: The corresponding n-dimensional eigenvector.
    """
    # square matrix
    assert a.shape[0] == a.shape[1]
    assert a.shape[0] == v_init.shape[0]

    cdef:
        dz[::1] Avv
        dz[::1, :] acpy
        Py_ssize_t i, j
        int N, incx, incy
        double norm_Av
        dz alpha = 1.0
        dz beta = 0.0
        dz mu = 0.0
        dz mu_previous = 0.0
    N = a.shape[0]
    incx = 1
    incy = 1
    Av = np.empty(a.shape[0], dtype=dtype_cy2np(cython.typeof(a[0, 0])))
    Avv = Av
    acpy = a.copy_fortran()

    if dz == cython.double:
        # Avv <-- a @ v_init
        blas.dsymv('u', &N, &alpha, &acpy[0, 0], &N, &v_init[0], &incx, &beta, &Avv[0], &incy)
        for j in range(max_iterations):
            # v_init <-- Avv / norm_Av
            norm_Av = blas.dnrm2(&N, &Avv[0], &incx)
            for i in range(a.shape[0]):
                v_init[i] = Avv[i] / norm_Av
            # Avv <-- a @ v_init
            blas.dsymv('u', &N, &alpha, &acpy[0, 0], &N, &v_init[0], &incx, &beta, &Avv[0], &incy)
            # mu <-- v_init.T @ Avv
            mu = blas.ddot(&N, &v_init[0], &incx, &Avv[0], &incy)

            if fabs((mu - mu_previous) / mu) <= eps:
                break

            mu_previous = mu
        return mu, np.array(v_init)

    elif dz == cython.doublecomplex:
        # Avv <-- a @ v_init
        blas.zhemv('u', &N, &alpha, &acpy[0, 0], &N, &v_init[0], &incx, &beta, &Avv[0], &incy)
        for j in range(max_iterations):
            # v_init <-- Avv / norm_Av
            norm_Av = blas.dznrm2(&N, &Avv[0], &incx)
            for i in range(a.shape[0]):
                v_init[i] = Avv[i] / norm_Av
            # Avv <-- a @ v_init
            blas.zhemv('u', &N, &alpha, &acpy[0, 0], &N, &v_init[0], &incx, &beta, &Avv[0], &incy)
            # mu <-- v_init.conj().T @ Avv
            mu = blas.zdotc(&N, &v_init[0], &incx, &Avv[0], &incy)

            if cabs((mu - mu_previous) / mu) <= eps:
                break

            mu_previous = mu
        return mu.real, np.array(v_init)
