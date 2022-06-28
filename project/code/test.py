import numpy as np
from itertools import combinations

sample_list = np.array(range(10))
print(sample_list)

pos = np.array(list(combinations(sample_list, 8)))
rand = np.random.permutation(pos)


# print(pos_combinations[0].next())

for x in rand:
    ensemble_member.accept_classes(x)