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
        return self.best, self.best_fitness

    def update(self, p: Particle):
        # Update velocity
        r = np.random.random(2)
        p.v = self.omega * p.v + self.alpha1 * r[0] * (p.best - p.x) + self.alpha2 * r[1] * (self.best - p.x)
        # Update position
        p.x = p.x + p.v

    def score(self, x: np.ndarray, data: np.ndarray):
        return quantization_error(self.k, x, data)


def quantization_error(k, x, data):
    # assign data to nearest cluster centroids:
    clusters, labels = get_clusters(k, x, data)
    # calculate quantization error:
    error = np.sum(
        [np.sum(clusters[i]) / len(clusters[i]) if len(clusters[i]) > 0 else 0 for i in range(k)]) / k
    return error


def get_clusters(k, x, data):
    # assign data to nearest cluster centroids:
    cluster_distances = [[] for i in range(k)]
    labels = []
    for d in data:
        # calculate distance between data point and all cluster centroids of particle:
        distances = np.array([np.linalg.norm(d - x[i]) for i in range(k)])
        # retrieve distance index of the smallest distance:
        c_i = np.argmin(distances)
        # append label to labels:
        labels.append(c_i)
        # append distance to cluster
        cluster_distances[c_i].append(distances[c_i])
    return cluster_distances, np.array(labels)


class ArtificialData:
    def generate_data(self, n):
        data = dict()
        data['X'] = np.array([[np.random.uniform(-1, 1) for j in range(2)] for i in range(n)])
        data['y'] = np.array([self.classify(x) for x in data['X']])
        return data

    @staticmethod
    def classify(x):
        return 1 if x[0] >= 0.7 or x[0] <= 0.3 and x[1] >= -0.2 - x[0] else 0


def plot_2D_data(X: np.ndarray, y: np.ndarray, y_hat: np.ndarray, k: int, centroids:np.ndarray, data_name: str, method_name: str):
    colors = ['blue', 'red', 'orange']
    for i in range(k):
        plt.scatter(X[y == i][:, 0], X[y == i][:, 1], marker='o', s=80, c=colors[i], label=f'True class {i}')
    for i in range(k):
        plt.scatter(X[y_hat == i][:, 0], X[y_hat == i][:, 1], marker='.', c=colors[i], label=f'Predicted class {i}')
    plt.scatter(centroids[:,0], centroids[:,1],c='black', marker='D', label='predicted centroids')
    plt.legend()
    plt.xlabel('feature 0')
    plt.ylabel('feature 1')
    plt.title(f'True vs Predicted labels of the {data_name} data\nClustered through {method_name}')
    plt.show()


# Load iris data:
iris = datasets.load_iris()
iris_data = {'X': iris.data, 'y': iris.target}
k_iris = 3

# generate artificial data:
artificial_data = ArtificialData().generate_data(n=400)
k_artificial = 2

# parameters for PSO clustering (taken from van der Merwe and Engelbrechts paper):
omega = 0.72
alpha1 = alpha2 = 1.49
n_its = 100
n_trials = 1

all_scores = {
    'iris_PSO': [],
    'iris_KMeans': [],
    'artificial_PSO': [],
    'artificial_KMeans': []
}

all_centroids = {
    'iris_PSO': [],
    'iris_KMeans': [],
    'artificial_PSO': [],
    'artificial_KMeans': []
}

# perform clustering:
for i in range(n_trials):
    print(f'starting trial {i}:')

    # KMeans clustering on artificial data:
    artificial_kmeans = cluster.KMeans(n_clusters=k_artificial, init='random', n_init=10, max_iter=100).fit(
        artificial_data['X'])
    centroids = artificial_kmeans.cluster_centers_
    all_centroids['artificial_KMeans'].append(centroids)
    all_scores['artificial_KMeans'].append(quantization_error(k=k_artificial, x=centroids, data=artificial_data['X']))

    # KMeans clustering on iris data:
    iris_kmeans = cluster.KMeans(n_clusters=3, init='random', n_init=10, max_iter=100).fit(
        iris_data['X'])
    centroids = iris_kmeans.cluster_centers_
    all_centroids['iris_KMeans'].append(centroids)
    all_scores['iris_KMeans'].append(quantization_error(k=k_iris, x=centroids, data=iris_data['X']))

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
    all_centroids['artificial_PSO'].append(best)
    all_scores['artificial_PSO'].append(quantization_error(k=k_artificial, x=best, data=artificial_data['X']))

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
    all_centroids['iris_PSO'].append(best)
    all_scores['iris_PSO'].append(quantization_error(k=k_iris, x=best, data=iris_data['X']))

mean_quantization_KMeans_artificial = np.mean(all_scores['artificial_KMeans'])
mean_quantization_PSO_artificial = np.mean(all_scores['artificial_PSO'])
mean_quantization_KMeans_iris = np.mean(all_scores['iris_KMeans'])
mean_quantization_PSO_iris = np.mean(all_scores['iris_PSO'])

print(f'Mean Quantization errors over {n_trials} runs:\n')
print(f'\t\t|\tArtif.\t|\tIris')
print('--------+-----------+----------')
print(f'KMeans\t|\t{mean_quantization_KMeans_artificial:.3f}\t|\t{mean_quantization_KMeans_iris:.3f}')
print(f'PSO\t\t|\t{mean_quantization_PSO_artificial:.3f}\t|\t{mean_quantization_PSO_iris:.3f}')

# plot_2D_data(iris_data['X'][:, :2], iris_data['y'])
# plot_2D_data(artificial_data['X'][:, :2], artificial_data['y'])

# Plot KMeans iris data:
centroids = all_centroids['iris_KMeans'][0]
_, y_hat = get_clusters(k=k_iris, x=centroids, data=iris_data['X'])
plot_2D_data(X=iris_data['X'], y=iris_data['y'], y_hat=y_hat, k=k_iris, centroids=centroids, data_name='Iris',
             method_name='KMeans')

# Plot KMeans artificial data:
centroids = all_centroids['artificial_KMeans'][0]
_, y_hat = get_clusters(k=k_artificial, x=centroids, data=artificial_data['X'])
plot_2D_data(X=artificial_data['X'], y=artificial_data['y'], y_hat=y_hat, k=k_artificial, centroids=centroids,
             data_name='Artificial', method_name='KMeans')

# Plot PSO iris data:
centroids = all_centroids['iris_PSO'][0]
_, y_hat = get_clusters(k=k_iris, x=centroids, data=iris_data['X'])
plot_2D_data(X=iris_data['X'], y=iris_data['y'], y_hat=y_hat, k=k_iris, data_name='Iris', centroids=centroids,
             method_name='PSO')

# Plot KMeans artificial data:
centroids = all_centroids['artificial_PSO'][0]
_, y_hat = get_clusters(k=k_artificial, x=centroids, data=artificial_data['X'])
plot_2D_data(X=artificial_data['X'], y=artificial_data['y'], y_hat=y_hat, k=k_artificial, centroids=centroids,
             data_name='Artificial', method_name='PSO')

#
