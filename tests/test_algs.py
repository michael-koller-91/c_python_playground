import sys
import unittest
import numpy as np

sys.path.append('..')

from modules import utils as ut     # noqa: E402
from modules import algs    # noqa: E402
from cymodules import algs as calgs     # noqa: E402


class TestOmp(unittest.TestCase):
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


class TestPowerIteration(unittest.TestCase):
    def test_power_iteration(self):
        np.random.seed(123)
        for dtype in [np.float64, np.complex128]:
            for m in [2, 4, 8, 16, 32]:
                a = ut.randn((m, m), dtype)
                a = a + a.conj().T
                v_init = ut.randn((m,), dtype)

                max_iter = 1000
                if m > 8:
                    max_iter = 100000

                mu_power, v_power = algs.power_iteration(a, v_init, eps=1e-15, max_iterations=max_iter)
                mu_power_jit, v_power_jit = algs.power_iteration_jit(a, v_init, eps=1e-15, max_iterations=max_iter)
                mu_power_c, v_power_c = calgs.power_iteration(a, v_init, eps=1e-15, max_iterations=max_iter)

                mu, v = np.linalg.eigh(a)
                idx = np.argmax(np.abs(mu))

                np.testing.assert_allclose(mu_power, mu[idx], atol=1e-6)
                np.testing.assert_allclose(mu_power_jit, mu[idx], atol=1e-6)
                np.testing.assert_allclose(mu_power_c, mu[idx], atol=1e-6)

                np.testing.assert_allclose(np.abs(np.inner(v_power.conj(), v[:, idx])), 1.0)
                np.testing.assert_allclose(np.abs(np.inner(v_power_jit.conj(), v[:, idx])), 1.0)
                np.testing.assert_allclose(np.abs(np.inner(v_power_c.conj(), v[:, idx])), 1.0)
