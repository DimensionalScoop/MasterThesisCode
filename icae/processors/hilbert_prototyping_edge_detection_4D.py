"""Study to confirm that adding more rotations of the hilbert curve improves
neighborhood relations"""
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
from sklearn.preprocessing import MinMaxScaler
import scipy.signal as signal

import matplotlib.pyplot as plt


def grid(resolution, dims):
    axis = [np.linspace(0,1,resolution) for i in range(dims)]
    axis_matrices = np.meshgrid(*axis)
    coords = np.array([am.ravel() for am in axis_matrices]).T
    return coords


dims = 4
resolution = 20
test_data = grid(resolution, dims)
samples = len(test_data)
shape = [samples, dims]
plt.scatter(test_data[:,0],test_data[:,1],marker='.');

print("Using %.1e MB of RAM"%(test_data.nbytes/1e6))


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
def hd(r,a0=0,a1=1):
    return hilbert_distance(rotate_2d(coordinates,r*np.pi/4,a0,a1), 12)
distances = [hd(0), hd(1), hd(1,1,2), hd(1,0,3)]


# +
def get_distance_matrix(distances):
    distance_matrix = np.ones([resolution-1,resolution-1])
    for coord, dist in zip(coordinates, distances):
        index_x = int(coord[0]*(resolution-2))
        index_y = int(coord[1]*(resolution-2))
        distance_matrix[index_x,index_y] = dist
    return distance_matrix

distance_matrices = [get_distance_matrix(d) for d in distances]
plt.imshow(distance_matrices[0])
plt.show()
plt.imshow(distance_matrices[1])


# +
def plot(d_index):
    if dims==2:
        plt.scatter(coordinates[:,0],coordinates[:,1],c=distances[d_index]/distances[d_index].max(),marker='.')
    if dims>=3:
        if dims>3: print("Warning: Showing 3D prjection of %iD space"%dims)
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(coordinates[:,0],coordinates[:,1],coordinates[:,2],c=distances[d_index]/distances[d_index].max(),marker='.')
    plt.show()

for i in range(len(distance_matrices)):
    plot(i)
# -

from pprint import pprint


# +
def get_edge_kernels(dimensions):
    """>>> get_edge_kernels(2)
[array([[ 1., -1.],
       [ 0.,  0.]]),
 array([[ 1.,  0.],
       [-1.,  0.]]),
 array([[ 1.,  0.],
       [ 0., -1.]])]
    """
    # generate all possible permutations of a kernel with exactly one element -1 and all others 0
    count_permutations = 2*dimensions
    kernels = []
    for p in range(count_permutations):
        kernel = np.zeros(2*dimensions)
        kernel[p] = -1
        kernels.append(kernel.reshape([dimensions,2]))
    
    # set 'top left' element to 1
    for k in kernels:
        k.reshape(-1)[0]=1 
    
    # identity kernel doesn't detect edges
    kernels.pop(0)
    
    return kernels

pprint(get_edge_kernels(2))
# -

import doctest
doctest.testmod()

[np.array([[ 1., -1.]])] == [np.array([[ 1., -1.]])]


# convolve
def get_edge_strenght(distance_matrix):
    dims = len(distance_matrix.shape)
    kernels = get_edge_kernels(dims)
    
    edges = [signal.convolve(distance_matrix,k,mode='valid') for k in kernels]
    edges = [np.abs(e) for e in edges]
    return edges


# +
edges = get_edge_strenght(distance_matrices[0])
edges = [e.flatten() for e in edges]
total_edges = np.concatenate(edges)

plt.ylabel('frequency')
plt.xlabel('edge strenght (a.u.)')
plt.title('One Hilbert parametrization in %dD'%dims)
xlim = [1,max(total_edges)]
_,bins,_ = plt.hist(total_edges,bins=20)
plt.yscale('log')
plt.xlim(*xlim);
# -

for k in range(1,5):
    # Dimensionality: [rotation, kernel type, x, y, z, ...]
    edges = np.array([get_edge_strenght(dm) for dm in distance_matrices[0:k]])
    min_edges = []
    for kernel_type in range(edges.shape[1]):
        flattened_arrays = [m.flatten() for m in edges[:,kernel_type]]
        mins = flattened_arrays[0]
        for i in range(edges.shape[0]-1):
            mins = np.minimum(mins,flattened_arrays[i+1])
        min_edges.append(mins)

    total_edges = np.concatenate(min_edges)

    plt.ylabel('frequency')
    plt.xlabel('edge strenght (a.u.)')
    plt.title('One Hilbert parametrization in %dD'%dims)
    plt.hist(total_edges,alpha=1,bins=bins,histtype='step', label="%d parametrizations"%k)
    plt.yscale('log')
    plt.xlim(*xlim)
    plt.legend()




