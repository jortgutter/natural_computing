import numpy as np


def f(x):
    fun = lambda x: -x * np.sin(np.sqrt(np.abs(x)))
    return np.sum(fun(x))

def print_fitness(x):
    print(f"f({x[0]}, {x[1]}) = {f(x)}")


#x = np.array([float(x) for x in input("x: ").split(", ")])
xs = []
s = " "
while True:
    s = input("x: ")
    if not s:
        break
    try:
        xs.append([float(x) for x in s.split(", ")])
    except:
        print("Invalid input!")

xs = np.array(xs)

top = None
top_f = -np.inf

for x in xs:
    fit = f(x)
    print_fitness(x)

    if top_f < fit:
        top = x
        top_f = fit


print(f"\nGlobal optimum: {top}")
print_fitness(top)


print()
v      = np.array([float(x) for x in input("v: ").split(", ")])
omegas = np.array([float(x) for x in input("ω: ").split(", ")])
alpha  = float(input("α: "))
r      = float(input("r: "))

print()
for omega in omegas:
    v_new = omega * v + alpha * r * (pb - x) + alpha * r * (top - x)
    print(f"ω = {omega}: v_new = {v_new}")
