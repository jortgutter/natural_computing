import math

# load the data:
locations = [[float(c) for c in x.split()] for x in open('file-tsp.txt', 'r').read().split('\n')]
print(f'Loaded {len(locations)} locations successfully!')

# calculate the distance between two locations
def distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


