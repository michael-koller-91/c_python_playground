import sys
import unittest
import numpy as np

sys.path.append('..')

from modules import utils as ut     # noqa: E402
from modules import algs    # noqa: E402
from cymodules import algs as calgs     # noqa: E402


class TestOMP(unittest.TestCase):
    def test_omp(self):
        np.random.seed(123)
        for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
            for (m, n) in [(15, 20), (30, 100), (100, 200)]:
                a = ut.randn((m, n), dtype)
                b = ut.randn((m,), dtype)
                for s in [1, 3, 10]:
                    x = algs.omp(a, b, s)[0]
                    if dtype in [np.float32, np.float64]:
                        x = np.real(x)
                    x_jit = algs.omp_jit(a, b, s)[0]
                    x_c = calgs.omp(a, b, s)[0]
                    np.testing.assert_allclose(x_jit, x, atol=1e-6)
                    np.testing.assert_allclose(x_c, x, atol=1e-6)
