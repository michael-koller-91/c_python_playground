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
