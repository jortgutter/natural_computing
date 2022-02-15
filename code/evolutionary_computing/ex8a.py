import gplearn.fitness
from gplearn.genetic import  SymbolicRegressor
from gplearn.functions import make_function
from matplotlib import pyplot as plt
import  numpy as np

# we had to define the exp function, as gplearn does not have one:
def exp(x1):
    return np.exp(x1)
exp = make_function(function=exp,name='exp', arity=1)

# loads data from the assignment:
def load_data(filename):
    with open(filename, "r") as file:
        xs = [[float(term)] for term in file.readline().split(", ")]
        ys = [float(term) for term in file.readline().split(", ")]
    return xs, ys

# manually defined fitness function:
def fitness_func(ys, y_preds, sample_weight):
    return -sum([abs(y - y_pred) for y, y_pred in zip(ys, y_preds)])

fit_func = gplearn.fitness.make_fitness(fitness_func, greater_is_better=True, wrap=False)

# load data:
xs, ys = load_data("data_ex8.txt")

print(fit_func)
print(ys)

# create regressor object from gplearn
estimator = SymbolicRegressor(
    population_size=1000,
    verbose=1,
    init_method='full',
    p_crossover=0.7,
    p_subtree_mutation=0,
    p_hoist_mutation=0,
    p_point_mutation=0,
    generations=50,
    metric = fit_func,
    function_set=('add', 'sub', 'mul', 'div', 'log', 'sin', 'cos', exp )
)


# perform evolutionary search
fit = estimator.fit(xs, ys)

# print S-expression of best solution:
print(fit)

# retrieve prediction:
prediction = estimator.predict(xs)


# plot results:
plt.plot(xs, prediction, label='prediction')
plt.plot(xs, ys, label='true values')
plt.title('Prediction vs true values after fitting')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

details = estimator.run_details_
print(details.keys())

plt.plot(details['best_fitness'])
plt.xlabel('generation')
plt.ylabel('fitness')
plt.title('Fitness of the best individual per generation')
plt.show()

plt.plot(details['best_length'], label='best')
plt.plot(details['average_length'], label='avg')
plt.xlabel('generation')
plt.ylabel('tree size')
plt.legend()
plt.title('Best and average tree sizes at each generation')
plt.show()

