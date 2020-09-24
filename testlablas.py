import unittest
from cymodules import lablas as lb
import numpy as np


np.random.seed(123)


def rand(shape, dtype):
    x = np.random.randn(*shape).astype(dtype)
    if dtype in [np.complex64, np.complex128]:
        x += 1j * np.random.randn(*shape).astype(dtype)
    return x


class TestGELS(unittest.TestCase):
    def test(self):
        for dtype in [np.float32, np.float64, np.complex64, np.complex128]:
            for (m, n) in [(10, 3), (50, 10), (100, 3), (200, 5)]:
                a = rand((m, n), dtype)
                b = rand((m,), dtype)
                x_np = np.linalg.lstsq(a, b, rcond=None)[0].astype(dtype)
                x_lb = lb.xgels(a, b)
                np.testing.assert_allclose(x_np, x_lb, rtol=1e-6, atol=1e-6)
