import matplotlib.pyplot as plt
import numpy as np
import json


def f(x):
    fun = lambda x: -x * np.sin(np.sqrt(np.abs(x)))
    return np.sum(fun(x))


def print_fitness(x):
    print(f"f({x[0]}, {x[1]}) = {f(x)}")


def main():
    params = json.load(open("ex1_init.json", "r"))

    xs = np.array(params["xs"])
    pbs = xs.copy()

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

    v = np.array(params["v"])

    for omega in params["omegas"]:
        xs_cpy = xs.copy()
        pbs_cpy = pbs.copy()
        top_cpy = top.copy()
        top_f_cpy = top_f
        v_cpy = v.copy()

        print(f"\n\n==== ω = {omega} ====")
        for i in range(params["n_it"]):
            print(f"\nIteration {i}:")

            print(f"Global optimum: {top_cpy} (f = {top_f_cpy})")

            v_cpy = omega * v_cpy + params["alpha"] * params["r"] * (pbs_cpy - xs_cpy)\
                                  + params["alpha"] * params["r"] * (top_cpy - xs)
            print(f"vs:\n{v_cpy}")

            xs_cpy = xs_cpy + v_cpy
            print(f"xs:\n{xs_cpy}")

            for idx, x in enumerate(xs_cpy):
                fit = f(x)
                if fit > f(pbs_cpy[idx]):
                    pbs_cpy[idx] = x.copy()
                    if fit > top_f_cpy:
                        top_cpy = x.copy()
                        top_f_cpy = fit


if __name__ == "__main__":
    main()
