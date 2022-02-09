import matplotlib.pyplot as plt
from tree import Tree
from tqdm import tqdm
from node import *
import copy


def load_data(filename):
    with open(filename, "r") as file:
        xs = [float(term) for term in file.readline().split(", ")]
        ys = [float(term) for term in file.readline().split(", ")]
    return xs, ys


def get_predictions(tree: Tree, terminal_value: TerminalValue, x_vals: List[float]) -> List[float]:
    results = []

    for x in x_vals:
        terminal_value.v = x
        results.append(tree.root.evaluate())

    return results


def plot_predictions(tree: Tree, xs: List[float], ys: List[float]):
    res = get_predictions(tree, term_val, xs)

    plt.plot(xs, ys, label="True", c="green")
    plt.plot(xs, res, label="Estimated", c="black")
    plt.legend()
    plt.show()


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

POP_SIZE = 1000
N_GEN = 50
P_CROSSOVER = 0.7


def calculate_fitness(tree: Tree, term_val: TerminalValue, xs: List[float], ys: List[float]):
    preds = get_predictions(tree, term_val, xs)

    errors = [abs(y - pred) for y, pred in zip(ys, preds)]

    return sum(errors)


population = [Tree(FunctionNode(None, functions=functions, terminal_value=term_val, max_depth=2)) for _ in range(POP_SIZE)]

optimal_fitness = [np.inf]
for tree in population:
    fitness = calculate_fitness(tree, term_val, xs, ys)
    if fitness < optimal_fitness[0]:
        optimal_fitness[0] = fitness
    tree.update_fitness(fitness)

for gen in tqdm(range(N_GEN)):
    new_pop = []

    for i in range(0, POP_SIZE, 2):
        t1, t2 = np.random.choice(population, size=2, replace=False)

        if np.random.uniform(0, 1) < 0.7:  # Crossover
            o1 = copy.deepcopy(t1)
            o2 = copy.deepcopy(t2)

            crossover(o1, o2)
            o1.update_fitness(calculate_fitness(o1, term_val, xs, ys))
            o2.update_fitness(calculate_fitness(o2, term_val, xs, ys))

            new_pop.extend(sorted([t1, t2, o1, o2], key=lambda x: x.fitness)[:2])

        else:
            new_pop.extend([t1, t2])

    population = new_pop

    for tree in population:
        if tree.fitness < optimal_fitness[-1]:
            optimal_fitness[-1] = tree.fitness

    optimal_fitness.append(optimal_fitness[-1])

plt.plot(range(N_GEN+1), optimal_fitness)
plt.show()