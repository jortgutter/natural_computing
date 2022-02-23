import numpy as np


def f(x):
    fun = lambda x: -x * np.sin(np.sqrt(np.abs(x)))
    return np.sum(fun(x))

def print_fitness(x):
    print(f"f({x[0]}, {x[1]}) = {f(x)}")


x = np.array([float(x) for x in input("x: ").split(", ")])

print_fitness(x)

print()
pb = np.array([float(x) for x in input("x*: ").split(", ")])
print_fitness(pb)

print()
top = np.array([float(x) for x in input("global optimum: ").split(", ")])
print_fitness(top)

print()
v     = np.array([float(x) for x in input("v: ").split(", ")])
omega = float(input("ω: "))
alpha = float(input("α: "))
r     = float(input("r: "))

print()
v_new = omega * v + alpha * r * (pb - x) + alpha * r * (top - x)
print(f"v_new = {v_new}")
