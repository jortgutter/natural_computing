from dataclasses import dataclass
from node import *

@dataclass
class Tree:
    root: Node

    def update_root(self, new_root):
        self.root = new_root

    def __repr__(self):
        return self.root.__repr__()
