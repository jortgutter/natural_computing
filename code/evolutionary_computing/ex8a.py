from gplearn.genetic import  SymbolicRegressor
from gplearn.functions import make_function
from sklearn.utils.random import check_random_state
from matplotlib import pyplot as plt
import  numpy as np

def exp(x1):
    return np.exp(x1)
exp = make_function(function=exp,name='exp', arity=1)


def load_data(filename):
    with open(filename, "r") as file:
        xs = [[float(term)] for term in file.readline().split(", ")]
        ys = [float(term) for term in file.readline().split(", ")]
    return xs, ys


xs, ys = load_data("data_ex8.txt")

print(ys)
est_gp = SymbolicRegressor(population_size=1000, verbose=1, init_method='full', p_crossover=0.7,p_subtree_mutation=0, p_hoist_mutation=0, p_point_mutation=0, generations=50,  function_set=('add', 'sub', 'mul', 'div', 'log', 'sin', 'cos', 'tan', exp ))

fit = est_gp.fit(xs, ys)

#Print the S-expression for the Symbolic regressor
print(fit)
prediction = est_gp.predict(xs)

plt.plot(prediction)
plt.plot(ys)
plt.show()