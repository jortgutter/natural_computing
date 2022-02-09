import matplotlib.pyplot as plt
from node import *

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

root = FunctionNode(None, functions=functions, max_depth=2)

print(root)
print(f"\n{root.evaluate()}")
