import numpy as np
import matplotlib.pyplot as plt


class PSOCluster:
    def __init__(self):
        pass


class KMeans:
    def __init__(self, k):
        self.k = k


class ArtificialData:
    data = {}

    def generate_data(self, n):
        self.data['X'] = np.array([[np.random.uniform(-1, 1) for j in range(2)] for i in range(n)])
        self.data['y'] = np.array([self.classify(x) for x in self.data['X']])
        return self.data

    def classify(self, x):
        return 1 if x[0]>=0.7 or x[0] <= 0.3 and x[1] >= -0.2 - x[0] else 0

data = ArtificialData().generate_data(n=400)

def plot_data(X, y, k):

ones = data['X'][data['y']==1]
zeros = data['X'][data['y']==0]
plt.scatter(ones[:,0], ones[:,1])
plt.scatter(zeros[:,0], zeros[:,1])
plt.show()
