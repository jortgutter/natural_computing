from dataclasses import dataclass
from node import *


@dataclass
class Tree:
    """
    Simple wrapper to easily keep track of a tree, even if the root node changes
    """
    root: Node
    fitness: float = -1

    def update_root(self, new_root):
        self.root = new_root

    def update_fitness(self, fitness):
        self.fitness = fitness

    def __repr__(self):
        return self.root.__repr__()
