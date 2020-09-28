import timeit
import numpy as np


def exec_time(func, repeat, number, *args, **kwargs):

    def wrapper(func, *args, **kwargs):
        def wrapped():
            return func(*args, **kwargs)
        return wrapped

    wrapped = wrapper(func, *args, **kwargs)

    t = np.array(timeit.repeat(wrapped, repeat=repeat, number=number))
    return np.sum(t) / number / repeat


def exec_time_auto(func, repeat=7, maxtime=0.2, *args, **kwargs):
    """
    This is basically an equivalent of the %timeit magic. The function func is
    called repeatedly with the args and kwargs provided and the execution time
    is measured. The function is called at least 'repeat' times.
    """
    def wrapper(func, *args, **kwargs):
        def wrapped():
            return func(*args, **kwargs)
        return wrapped

    wrapped = wrapper(func, *args, **kwargs)

    # How many times can the function be called within 'maxtime' seconds?
    for i in range(10):
        number = 10 ** i
        x = timeit.timeit(wrapped, number=number)
        if x >= maxtime:
            break

    # the actual timing
    secs = np.array(timeit.repeat(wrapped, repeat=repeat, number=number)) / number

    # print results
    nsecs = secs * 1e9
    s_mean = np.mean(nsecs)
    s_std = np.std(nsecs)
    secs_mean, unit_mean = nsecs_to_unit(s_mean)
    secs_std, unit_std = nsecs_to_unit(s_std)
    print(
        '{:5.1f} {} ± {:6.2f} {} per loop (mean ± std. dev. of {} runs, {} loops each) ({})'
        .format(secs_mean, unit_mean, secs_std, unit_std, repeat, number, func)
    )
    return secs


def nsecs_to_unit(nsecs):
    """
    Convert a given number nsecs of nanoseconds to a representation with a
    corresponding unit. For example, if nsecs = 12345678 nanoseconds, then
    this function returns (12.345678, 'ms').
    """
    if nsecs < 1000:
        unit = 'ns'
    else:
        nsecs /= 1000
        if nsecs < 1000:
            unit = 'µs'
        else:
            nsecs /= 1000
            if nsecs < 1000:
                unit = 'ms'
            else:
                nsecs /= 1000
                unit = 's'
    return nsecs, unit


def randn(shape, dtype):
    """
    Standard normal random numbers of a given data type.
    """
    x = np.random.randn(*shape).astype(dtype)
    if dtype in [np.complex64, np.complex128]:
        x += 1j * np.random.randn(*shape).astype(dtype)
    return x
