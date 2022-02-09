import matplotlib.pyplot as plt
from node import *


def load_data(filename):
    with open(filename, "r") as file:
        xs = [float(term) for term in file.readline().split(", ")]
        ys = [float(term) for term in file.readline().split(", ")]
    return xs, ys


def get_predictions(tree: Node, terminal_value: TerminalValue, x_vals: List[float]) -> List[float]:
    results = []

    for x in x_vals:
        terminal_value.v = x
        results.append(tree.evaluate())

    return results


def protected_div(a, b):
    return 1 if b == 0 else np.divide(a, b)


def protected_log(a):
    return 0 if a <= 0 else np.log(a)


functions = [
    (np.add, 2),
    (np.subtract, 2),
    (np.multiply, 2),
    (protected_div, 2),
    (protected_log, 1),
    (np.exp, 1),
    (np.sin, 1),
    (np.cos, 1)
]

xs, ys = load_data("data_ex8.txt")

term_val = TerminalValue(0)
root = FunctionNode(None, functions=functions, terminal_value=term_val, max_depth=6)
print(root)

res = get_predictions(root, term_val, xs)

plt.plot(xs, ys, label="True", c="green")
plt.plot(xs, res, label="Estimated", c="black")
plt.legend()
plt.show()
