import matplotlib.pyplot as plt
import numpy as np


class Function:
    def __init__(self, parent=None, function=(lambda x: x, 1), children=[]):
        self.p = parent
        self.f, self.arity = function
        self.children = children

        self.depth = 0 if not self.p else self.p.get_depth() + 1

    def get_depth(self):
        return self.depth

    def evaluate(self):
        return self.f(*[c.evaluate() for c in self.children])

    def __repr__(self):
        newline = "\n"
        tab = "\t"
        return f"{tab*self.depth}- Function[{self.depth}]: {self.f}\n{tab*self.depth}{(newline + tab*self.depth).join([str(c) for c in self.children])}"


class Terminal:
    def __init__(self, value=0):
        self.value = value

    def evaluate(self):
        return self.value

    def __repr__(self):
        return f"\t- Terminal: {self.value}"


functions = [
    (np.add, 2),
    (np.subtract, 2),
    (np.multiply, 2),
    (np.divide, 2),
    (np.log, 1),
    (np.exp, 1),
    (np.sin, 1),
    (np.cos, 1)
]


def rand_func(parent, terminal=Terminal(0)):
    f_idx = np.random.choice(range(len(functions)))
    f = functions[f_idx]
    return Function(parent=parent, function=f, children=[terminal]*f[1])

terminal = Terminal(5)
terminal.depth = 2

root = rand_func(None, terminal)

root.children = [rand_func(root, terminal) for _ in range(root.arity)]

print(root)
print(f"\n{root.evaluate()}")