import matplotlib.pyplot as plt
from tree import Tree
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


def crossover(tree1: Tree, tree2: Tree):
    # Get a list of the nodes in the tree
    refs1 = tree1.root.get_reference_list()
    refs2 = tree2.root.get_reference_list()

    # Select random node from each tree
    node1_idx = np.random.choice(range(len(refs1)))
    node2_idx = np.random.choice(range(len(refs2)))

    # Retrieve Node objects
    node1 = refs1[node1_idx]
    node2 = refs2[node2_idx]

    print(f"Crossover:\n{node1}\nand\n{node2}")

    # Retrieve Node parents
    parent1 = node1.p
    parent2 = node2.p

    # Update children
    if parent1:
        for i, c1 in enumerate(parent1.children):
            if node1 == c1:
                parent1.children[i] = node2
                break
    else:
        tree1.root = node2

    node1.update_parent(parent2)

    if parent2:
        for i, c2 in enumerate(parent2.children):
            if node2 == c2:
                parent2.children[i] = node1
                break
    else:
        tree2.root = node1

    node2.update_parent(parent1)


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
# root = FunctionNode(None, functions=functions, terminal_value=term_val, max_depth=3)
# print(root)
# print(f"\nNumber of nodes: {root.get_node_count()}")

# res = get_predictions(root, term_val, xs)

# plt.plot(xs, ys, label="True", c="green")
# plt.plot(xs, res, label="Estimated", c="black")
# plt.legend()
# plt.show()

tree1 = Tree(FunctionNode(None, functions=functions, terminal_value=term_val, max_depth=2))
tree2 = Tree(FunctionNode(None, functions=functions, terminal_value=term_val, max_depth=2))

print(tree1)
print()
print(tree2)
print()

crossover(tree1, tree2)
print("\nAfter crossover:")

print(tree1)
print()
print(tree2)
