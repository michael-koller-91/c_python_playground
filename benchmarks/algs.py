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
