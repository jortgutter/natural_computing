import numpy as np


def f(x1, x2):
    fun = lambda x: -x * np.sin(np.sqrt(np.abs(x)))
    return fun(x1) + fun(x2)


x1 = float(input("x1: "))
x2 = float(input("x2: "))


print(f"f({x1}, {x2}) = {f(x1, x2)}")


