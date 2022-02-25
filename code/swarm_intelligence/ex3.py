import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import cluster
from tqdm import tqdm


class Particle:
    def __init__(self, x_init: np.ndarray):
        self.x = x_init
        self.v = np.zeros(self.x.shape)
        self.fitness = np.inf
        self.best = self.x.copy()
        self.best_fitness = np.inf

    def __repr__(self):
        return f"Particle({self.x})"


class PSOCluster:
    def __init__(self, k: int, n_particles: int):
        self.k = k
        self.n_particles = n_particles

    def cluster(self, data: np.ndarray, n_iterations: int, omega: float, alpha1: float, alpha2: float):
        self.omega = omega
        self.alpha1 = alpha1
        self.alpha2 = alpha2

        # generate initial particles
        self.particles = []
        for i in range(self.n_particles):
            init_indices = np.random.choice(range(len(data)), self.k, replace=False)
            self.particles.append(Particle(x_init=data[init_indices, :]))

        # take first particle position as initial global best
        self.best = self.particles[0].best
        self.best_fitness = np.inf

        errors = np.zeros(n_iterations)

        # iteratively update particles:
        for it in tqdm(range(n_iterations)):
            for particle in self.particles:
                error = self.score(particle.x, data)
                particle.fitness = error
                # update best and global best:
                if error < particle.best_fitness:
                    particle.best = particle.x.copy()
                    particle.best_fitness = error
                    if error < self.best_fitness:
                        self.best = particle.x.copy()
                        self.best_fitness = error

            errors[it] = self.best_fitness

            # update particles:
            for particle in self.particles:
                self.update(p=particle)

        # return best cluster centers and their fitness:
        return np.array([self.best, self.best_fitness]), errors

    def update(self, p: Particle):
        # Update velocity
        r = np.random.random(2)
        p.v = self.omega * p.v + self.alpha1 * r[0] * (p.best - p.x) + self.alpha2 * r[1] * (self.best - p.x)
        # Update position
        p.x = p.x + p.v

    def score(self, x: np.ndarray, data: np.ndarray):
        # assign data to nearest cluster centroids:
        clusters = [[] for i in range(self.k)]
        for d in data:
            # calculate distance between data point and all cluster centroids of particle:
            distances = np.array([np.linalg.norm(d - x[i]) for i in range(self.k)])
            # retrieve distance index of the smallest distance:
            c_i = np.argmin(distances)
            # append distance to cluster
            clusters[c_i].append(distances[c_i])

        # calculate quantization error:
        error = np.sum(
            [np.sum(clusters[i]) / len(clusters[i]) if len(clusters[i]) > 0 else 0 for i in range(self.k)]) / self.k
        return error


class ArtificialData:
    def generate_data(self, n):
        data = dict()
        data['X'] = np.array([[np.random.uniform(-1, 1) for j in range(2)] for i in range(n)])
        data['y'] = np.array([self.classify(x) for x in data['X']])
        return data

    @staticmethod
    def classify(x):
        return 1 if x[0] >= 0.7 or x[0] <= 0.3 and x[1] >= -0.2 - x[0] else 0


def plot_2D_data(X, y):
    for i in range(max(y) + 1):
        class_members = X[y == i]
        plt.scatter(class_members[:, 0], class_members[:, 1], label=f'class {i}')
        plt.legend()
        plt.xlabel('z0')
        plt.ylabel('z1')
    plt.show()


# Load iris data:
iris = datasets.load_iris()
iris_data = {'X': iris.data, 'y': iris.target}

# generate artificial data:
artificial_data = ArtificialData().generate_data(n=400)

# parameters for PSO clustering (taken from van der Merwe and Engelbrechts paper):
omega = 0.72
alpha1 = alpha2 = 1.49
n_its = 100
n_trials = 1

iris_PSO_results = np.array([])
iris_PSO_scores = np.array([])
artificial_PSO_results = np.array([])
artificial_PSO_scores = np.array([])

# perform clustering:
for i in range(n_trials):
    print(f'starting trial {i}:')

    # KMeans clustering on artificial data:
    artificial_kmeans = cluster.KMeans(n_clusters=2, init='random', n_init=10, max_iter=100).fit(
        artificial_data['X'])
    print(artificial_kmeans.cluster_centers_)
    iris_kmeans = cluster.KMeans(n_clusters=3, init='random', n_init=10, max_iter=100).fit(
        iris_data['X'])
    print(iris_kmeans.cluster_centers_)

    # PSO clustering on artificial data:
    print(f'PSO artificial')
    artificial_cluster = PSOCluster(k=2, n_particles=10)
    best, errors = artificial_cluster.cluster(
        data=artificial_data['X'],
        n_iterations=n_its,
        omega=omega,
        alpha1=alpha1,
        alpha2=alpha2
    )
    np.append(artificial_PSO_results, best)

    # PSO clustering on iris dataset:
    print(f'PSO iris')
    iris_cluster = PSOCluster(k=3, n_particles=10)
    best, errors = iris_cluster.cluster(
        data=iris_data['X'],
        n_iterations=n_its,
        omega=omega,
        alpha1=alpha1,
        alpha2=alpha2
    )
    np.append(iris_PSO_results, best)


plot_2D_data(iris_data['X'][:, :2], iris_data['y'])
plot_2D_data(artificial_data['X'][:, :2], artificial_data['y'])
