import sys
import timeit
import numpy as np
import pandas as pd

sys.path.append('..')

from modules import algs    # noqa: E402
from modules import utils as ut     # noqa: E402
from cymodules import algs as calgs     # noqa: E402


#
# omp
#
d = {
    'dtype': list(),
    'm': list(),
    's': list(),
    'algs.omp': list(),
    'algs.omp_jit': list(),
    'calgs.omp': list(),
    'algs.omp/algs.omp_jit': list(),
    'algs.omp/calgs.omp': list(),
    'algs.omp_jit/calgs.omp': list(),
}

for dtype in [np.float32, np.complex128]:
    for m in [16, 32, 64, 96]:
        for s in [1, 5, 10, 15]:
            print('dtype:', dtype, '| m:', m, '| s:', s)
            a = ut.randn((m, 128), dtype=dtype)
            b = ut.randn((m,), dtype=dtype)

            # call every algorithm once, e.g., to run jit
            x1 = algs.omp(a, b, s)[0]
            x2 = algs.omp_jit(a, b, s)[0]
            x3 = calgs.omp(a, b, s)[0]
            if not np.allclose(x1, x2, atol=1e-6) or not np.allclose(x1, x3, atol=1e-6):
                print('Algorithms did not compute same results.')

            d['algs.omp'].append(np.mean(ut.exec_time_auto(algs.omp, 7, 0.2, a, b, s)))
            d['algs.omp_jit'].append(np.mean(ut.exec_time_auto(algs.omp_jit, 7, 0.2, a, b, s)))
            d['calgs.omp'].append(np.mean(ut.exec_time_auto(calgs.omp, 7, 0.2, a, b, s)))
            d['algs.omp/algs.omp_jit'].append(d['algs.omp'][-1]/d['algs.omp_jit'][-1])
            d['algs.omp/calgs.omp'].append(d['algs.omp'][-1]/d['calgs.omp'][-1])
            d['algs.omp_jit/calgs.omp'].append(d['algs.omp_jit'][-1]/d['calgs.omp'][-1])

            if dtype == np.float32:
                d['dtype'].append('s')
            elif dtype == np.complex128:
                d['dtype'].append('z')
            d['m'].append(m)
            d['s'].append(s)

for dtype in [np.float32, np.complex128]:
    m = 1000
    s = 200
    print('dtype:', dtype, '| m:', m, '| s:', s)
    a = ut.randn((m, 2000), dtype=dtype)
    b = ut.randn((m,), dtype=dtype)

    x1 = algs.omp(a, b, s)[0]
    x2 = algs.omp_jit(a, b, s)[0]
    x3 = calgs.omp(a, b, s)[0]
    if not np.allclose(x1, x2, atol=1e-6) or not np.allclose(x1, x3, atol=1e-6):
        print('Algorithms did not compute same results.')

    d['algs.omp'].append(np.mean(ut.exec_time_auto(algs.omp, 7, 0.2, a, b, s)))
    d['algs.omp_jit'].append(np.mean(ut.exec_time_auto(algs.omp_jit, 7, 0.2, a, b, s)))
    d['calgs.omp'].append(np.mean(ut.exec_time_auto(calgs.omp, 7, 0.2, a, b, s)))
    d['algs.omp/algs.omp_jit'].append(d['algs.omp'][-1]/d['algs.omp_jit'][-1])
    d['algs.omp_jit/calgs.omp'].append(d['algs.omp_jit'][-1]/d['calgs.omp'][-1])

    if dtype == np.float32:
        d['dtype'].append('s')
    elif dtype == np.complex128:
        d['dtype'].append('z')
    d['m'].append(m)
    d['s'].append(s)

df = pd.DataFrame(d)
print(df)
print(df.describe())

#
# power iteration
#
d = {
    'dtype': list(),
    'n': list(),
    'algs.power_iteration': list(),
    'algs.power_iteration_jit': list(),
    'calgs.power_iteration': list(),
    'algs./algs.jit': list(),
    'algs./calgs.': list(),
    'algs.jit/calgs.': list(),
}

for dtype in [np.float64, np.complex128]:
    for n in [2**x for x in range(1, 9)]:
        print('dtype:', dtype, '| n:', n)
        a = ut.randn((n, n), dtype=dtype)
        v_init = ut.randn((n,), dtype=dtype)

        mu_power1, v_power1 = algs.power_iteration(a, v_init, eps=1e-15, max_iterations=500)
        mu_power2, v_power2 = algs.power_iteration_jit(a, v_init, eps=1e-15, max_iterations=500)
        mu_power3, v_power3 = calgs.power_iteration(a, v_init, eps=1e-15, max_iterations=500)

        if not (np.allclose(mu_power1, mu_power2, atol=1e-6) or
            np.allclose(mu_power1, mu_power3, atol=1e-6)):
            print('Algorithms did not compute same results.')

        if not (np.allclose(np.abs(np.inner(v_power1.conj(), v_power2)), 1.0) or
            np.allclose(np.abs(np.inner(v_power1.conj(), v_power3)), 1.0)):
            print('Algorithms did not compute same results.')

        d['algs.power_iteration'].append(
            np.mean(ut.exec_time_auto(algs.power_iteration, 7, 0.2, a, v_init, 1e-15, 500)))
        d['algs.power_iteration_jit'].append(
            np.mean(ut.exec_time_auto(algs.power_iteration_jit, 7, 0.2, a, v_init, 1e-15, 500)))
        d['calgs.power_iteration'].append(
            np.mean(ut.exec_time_auto(calgs.power_iteration, 7, 0.2, a, v_init, 1e-15, 500)))
        d['algs./algs.jit'].append(
            d['algs.power_iteration'][-1]/d['algs.power_iteration_jit'][-1])
        d['algs./calgs.'].append(
            d['algs.power_iteration'][-1]/d['calgs.power_iteration'][-1])
        d['algs.jit/calgs.'].append(
            d['algs.power_iteration_jit'][-1]/d['calgs.power_iteration'][-1])

        if dtype == np.float64:
            d['dtype'].append('d')
        elif dtype == np.complex128:
            d['dtype'].append('z')
        d['n'].append(n)

df = pd.DataFrame(d)
print(df)
print(df.describe())
