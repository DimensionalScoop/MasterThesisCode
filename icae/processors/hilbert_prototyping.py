# +
import numpy as np

#from numba import jit
from hilbertcurve.hilbertcurve import HilbertCurve
from sklearn.preprocessing import MinMaxScaler
# -

import matplotlib.pyplot as plt

dims = 3
samples = 10000
shape = [samples, dims]
aspect_ratios = [1,3] # ,5,100]
test_data = np.random.rand(*shape)
for i, ratio in enumerate(aspect_ratios):
    test_data[:,i] *= ratio

mean = list(range(dims))
cov = np.eye(dims)
test_data = np.random.multivariate_normal(mean, cov, size=samples)


def rotate_2d(coordinates, angle, axis_0=0, axis_1=1):
    coordinates = coordinates.copy()
    r = np.sqrt(coordinates[:,axis_0]**2 + coordinates[:,axis_1]**2)
    theta = np.arctan2(coordinates[:,axis_1], coordinates[:,axis_0])
    coordinates[:,axis_0] = np.cos(theta+angle) * r
    coordinates[:,axis_1] = np.sin(theta+angle) * r
    return coordinates


# +
def hilbert_distance(coordinates, iterations=5):
    count_entries, dims = coordinates.shape
    distances = np.zeros([count_entries])
    cells = 2 ** (iterations * dims)
    transform = HilbertCurve(iterations, dims)
    mm = MinMaxScaler((0, transform.max_x))
    normed_coords = np.rint(mm.fit_transform(coordinates)).astype('int')

    for i in range(len(coordinates)):
        distances[i] = transform.distance_from_coordinates(normed_coords[i])
    return distances

coordinates = test_data.copy()
distances_a = hilbert_distance(coordinates, 5)
distances_b = hilbert_distance(rotate(coordinates), 5)
distances = distances_a#distances_a - distances_b#np.sum(distances_a, distances_b)
# -

distances = []
for a in range(0,90,12):
    angle = a/180*np.pi
    distances.append(hilbert_distance(rotate(coordinates, angle), 15))

distances

for i,dist in enumerate(distances):
    #plt.subplot(len(distances),1,i+1)
    if dims==2:
        plt.scatter(coordinates[:,0],coordinates[:,1],c=dist/dist.max(),marker='.')
    elif dims==3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(coordinates[:,0],coordinates[:,1],coordinates[:,2],c=dist/dist.max(),marker='.')
    plt.show()

if dims==2:
    plt.scatter(coordinates[:,0],coordinates[:,1],c=distances_a/distances_a.max(),marker='.')
if dims==3:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coordinates[:,0],coordinates[:,1],coordinates[:,2],c=distances_a/distances_a.max(),marker='.')
plt.show()
if dims==2:
    plt.scatter(coordinates[:,0],coordinates[:,1],c=distances_b/distances_b.max(),marker='.')
if dims==3:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coordinates[:,0],coordinates[:,1],coordinates[:,2],c=distances_b/distances_b.max(),marker='.')
plt.show()

if dims==2:
    plt.scatter(coordinates[:,0],coordinates[:,1],c=distances/distances.max(),marker='.')
if dims==3:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coordinates[:,0],coordinates[:,1],coordinates[:,2],c=distances/distances.max(),marker='.')
plt.show()

for i,dist in enumerate(distances):
    plt.hist(dist, bins=int(np.sqrt(samples)));
    plt.show()




