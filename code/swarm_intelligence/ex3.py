import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


class Particle:
    def __init__(self, x_init, k, dim):
        self.x = x_init
        self.v = np.zeros((len(self.x), dim))
        self.best = self.x


class PSOCluster:
    def __init__(self, k, n_particles):
        self.k = k
        self.n_particles = n_particles

    def cluster(self, data, n_iterations, omega, alpha1, alpha2):
        self.omega = omega
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        # generate initial particles
        data_dim = len(data[0])
        self.particles = []
        for i in range(self.n_particles):
            init_indices = np.random.choice(range(len(data)), self.k, replace=False)
            self.particles.append(Particle(x_init=data[init_indices,:] ,k=self.k, dim=data_dim))

        # take first particle position as initial global best
        self.best = self.particles[0].best

        # iteratively update particles:
        for i in range(n_iterations):
            for particle in self.particles:
                # update and evaluate particle:
                self.update(particle=particle)

                # update global best:
                if self.score(particle.best, data) < self.score(self.best, data):
                    self.best = particle.best

        # return best cluster centers:
        return self.best



    def update(self, particle):
        pass

    def score(self, x, data):
        pass


class KMeans:
    def __init__(self, k ):
        self.k = k


class ArtificialData:
    def generate_data(self, n):
        data = {}
        data['X'] = np.array([[np.random.uniform(-1, 1) for j in range(2)] for i in range(n)])
        data['y'] = np.array([self.classify(x) for x in data['X']])
        return data

    def classify(self, x):
        return 1 if x[0]>=0.7 or x[0] <= 0.3 and x[1] >= -0.2 - x[0] else 0



def plot_2D_data(X, y):
    for i in range(max(y)+1):
        class_members = X[y==i]
        plt.scatter(class_members[:,0], class_members[:,1], label=f'class {i}')
        plt.legend()
        plt.xlabel('z0')
        plt.ylabel('z1')
    plt.show()

# Load iris data:
iris = datasets.load_iris()
iris_data = {'X':iris.data, 'y': iris.target}

# generate artificial data:
artificial_data = ArtificialData().generate_data(n=400)

# parameters for PSO clustering (taken from van der Merwe and Engelbrechts paper):
omega = 0.72
alpha1 = alpha2 = 1.49
n_its = 100
n_trials = 10

iris_PSO_results = []
artificial_PSO_results = []

# perform clustering:
for i in range(n_trials):
    # PSO clustering on artificial data:
    artificial_cluster = PSOCluster(k=2, n_particles=10)
    artificial_PSO_results.append(artificial_cluster.cluster(
        data=artificial_data['X'],
        n_iterations=n_its,
        omega=omega,
        alpha1=alpha1,
        alpha2=alpha2
    ))

    # PSO clustering on iris dataset:
    iris_cluster = PSOCluster(k=3, n_particles=10)
    iris_PSO_results.append(iris_cluster.cluster(
        data=iris_data['X'],
        n_iterations=n_its,
        omega=omega,
        alpha1=alpha1,
        alpha2=alpha2
    ))




plot_2D_data(iris_data['X'][:,:2], iris_data['y'])
plot_2D_data(artificial_data['X'][:,:2], artificial_data['y'])

