import numba
import numpy as np


def omp(A, y, s):
    r"""
    Orthogonal Matching Pursuit, a greedy algorithm to solve
        \min_x \| A x - y \|
    for a s-sparse x.
    Args:
        A: An m-by-n matrix (typically m < n).
        y: An m-dimensional right-hand side vector.
        s: The sparsity (number of non-zero entires of x)

    Returns:
        x: A s-sparse n-dimensional vector which (is supposed to) solve(s) \min_x \| A x - y \|.
        S: A n-dimensional boolean vector indicating the support of x.

    Source:
        See "A mathematical introduction to compressvie sensing" by Foucart, Section 3.23.
    """
    x = np.zeros(A.shape[1], dtype=np.complex)
    S = list()
    for _ in range(s):
        S.append(np.argmax(np.abs(A.conj().T @ (y - A @ x))))
        x.fill(0.0)
        x[S] = np.linalg.lstsq(A[:, S], y, rcond=None)[0]
    return x, S


@numba.njit
def omp_jit(A, y, s):
    r"""
    Orthogonal Matching Pursuit, a greedy algorithm to solve
        \min_x \| A x - y \|
    for a s-sparse x.
    Args:
        A: An m-by-n matrix (typically m < n).
        y: An m-dimensional right-hand side vector.
        s: The sparsity (number of non-zero entires of x)

    Returns:
        x: A s-sparse n-dimensional vector which (is supposed to) solve(s) \min_x \| A x - y \|.
        S: A n-dimensional boolean vector indicating the support of x.

    Source:
        See "A mathematical introduction to compressvie sensing" by Foucart, Section 3.23.
    """
    x = np.zeros(A.shape[1], dtype=A.dtype)
    S = np.zeros(A.shape[1], dtype=numba.boolean)
    for _ in range(s):
        S[np.argmax(np.abs(A.conj().T @ (y - A @ x)))] = True
        x.fill(0.0)
        x[S] = np.linalg.lstsq(A[:, S], y, rcond=-1)[0]
    return x, S


def power_iteration(A, v_init, eps=1e-12, max_iterations=100):
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

    Source:
        https://github.com/TheAlgorithms/Python/blob/master/linear_algebra/src/power_iteration.py
    """
    # square matrix
    assert A.shape[0] == A.shape[1]
    assert A.shape[0] == v_init.shape[0]

    # make sure A is symmetric/self-adjoint
    A = (A.conj().T + A) * 0.5

    mu = 0.0
    mu_previous = 0.0
    for _ in range(max_iterations):
        v = A @ v_init
        v_init = v / np.linalg.norm(v)

        # Rayleigh quotient
        mu = v_init.conj().T @ A @ v_init

        # check relative change in eigenvalue
        if np.abs((mu - mu_previous) / mu) <= eps:
            break

        mu_previous = mu
    return np.real(mu), v_init


@numba.njit
def power_iteration_jit(A, v_init, eps=1e-12, max_iterations=100):
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
    assert A.shape[0] == A.shape[1]
    assert A.shape[0] == v_init.shape[0]

    # make sure A is symmetric/self-adjoint
    A = (A.conj().T + A) * 0.5

    mu = 0.0
    mu_previous = 0.0
    # only compute A @ v_init once per loop
    Av = A @ v_init
    for _ in range(max_iterations):
        v_init = Av / np.linalg.norm(Av)

        # Rayleigh quotient
        Av = A @ v_init
        mu = v_init.conj().T @ Av

        # check relative change in eigenvalue
        if np.abs((mu - mu_previous) / mu) <= eps:
            break

        mu_previous = mu
    return np.real(mu), v_init
