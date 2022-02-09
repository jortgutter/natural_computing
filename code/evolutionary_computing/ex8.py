import matplotlib.pyplot as plt
import numpy as np


class Function:
    def __init__(self, parent=None, function=lambda x: x, children=[]):
        self.p = parent
        self.f = function
        self.children = children

        self.depth = 0 if not self.p else p.get_depth() + 1

    def evaluate(self):
        return self.f(*[c.evaluate() for c in self.children])

    def __repr__(self):
        return f"Function[{self.depth}], {self.f}"

class Terminal:
    def __init__(self, value=0):
        self.value = value

    def evaluate(self):
        return self.value

test2 = Function(function = sum, children=[Terminal(10), Terminal(5)])
test1 = Function(function = np.log, children=[test2])

print(test2.evaluate())
