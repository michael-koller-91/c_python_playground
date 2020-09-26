import numba
import numpy as np


def omp(A, y, s):
    r"""
    Orthogonal Matching Pursuit, a greedy algorithm to solve
        \min_x \| A x - y \|
    for a s-sparse x.
    Args:
        A: A m-by-n matrix (typically m < n).
        y: An m-dimensional right-hand side vector.
        s: The sparsity (number of non-zero entires of x)

    Returns:
        x: A s-sparse n-dimensional vector which (is supposed to) solve \min_x \| A x - y \|.

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
        A: A m-by-n matrix (typically m < n).
        y: An m-dimensional right-hand side vector.
        s: The sparsity (number of non-zero entires of x)

    Returns:
        x: A s-sparse n-dimensional vector which (is supposed to) solve \min_x \| A x - y \|.

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
