import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
from sklearn.preprocessing import MinMaxScaler
import scipy.signal as signal

import matplotlib.pyplot as plt


def grid(resolution):
    x=np.linspace(0,1,resolution)
    y=np.linspace(0,1,resolution)
    xx,yy=np.meshgrid(x,y)
    coords=np.array((xx.ravel(), yy.ravel())).T
    return coords


dims = 2
resolution = 40
test_data = grid(40)
samples = len(test_data)
shape = [samples, dims]
plt.scatter(test_data[:,0],test_data[:,1],marker='.');


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
distances_a = hilbert_distance(coordinates, 12)
distances_b = hilbert_distance(rotate_2d(coordinates,np.pi/4), 12)


# +
def get_distance_matrix(distances):
    distance_matrix = np.ones([resolution-1,resolution-1])
    for coord, dist in zip(coordinates, distances):
        index_x = int(coord[0]*(resolution-2))
        index_y = int(coord[1]*(resolution-2))
        distance_matrix[index_x,index_y] = dist
    return distance_matrix

distance_matrix_a = get_distance_matrix(distances_a)
distance_matrix_b = get_distance_matrix(distances_b)
plt.imshow(distance_matrix_a)
plt.show()
plt.imshow(distance_matrix_b)
# -

if dims==2:
    plt.scatter(coordinates[:,0],coordinates[:,1],c=distances_b/distances_b.max(),marker='.')
if dims==3:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coordinates[:,0],coordinates[:,1],coordinates[:,2],c=distances/distances.max(),marker='.')
plt.show()



plt.hist(distances_a, bins=int(np.sqrt(samples)));


# convolve
def get_edge_strenght(distance_matrix):
    kernel = [[1,0],[-1,0]]
    edge_matrix_y = signal.convolve2d(distance_matrix,kernel,mode='valid')
    kernel = [[1,-1],[0,0]]
    edge_matrix_x = signal.convolve2d(distance_matrix,kernel,mode='valid')
    return np.abs(edge_matrix_x), np.abs(edge_matrix_y)


edges_x, edges_y = get_edge_strenght(distance_matrix_b)
total_edges = np.concatenate([edges_x.flatten(),edges_y.flatten()])
plt.hist(total_edges);
plt.yscale('log');
plt.xlim(1,1e7)

edges_x_a, edges_y_a = get_edge_strenght(distance_matrix_a)
edges_x_b, edges_y_b = get_edge_strenght(distance_matrix_b)
edges_x = np.minimum(edges_x_a, edges_x_b)
edges_y = np.minimum(edges_y_a, edges_y_b)
total_edges = np.concatenate([edges_x.flatten(),edges_y.flatten()])
plt.hist(total_edges);
plt.yscale('log');
plt.xlim(1,1e7)

plt.imshow(edges_x)

plt.imshow(edges_y)

for i,dist in enumerate(distances):
    plt.hist(dist, bins=int(np.sqrt(samples)));
    plt.show()




