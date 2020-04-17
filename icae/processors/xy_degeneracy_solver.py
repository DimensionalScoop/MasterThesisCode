# +
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, distance_matrix


# %load_ext autoreload
# %autoreload 2
# %matplotlib inline

# + {}
def convex_hull_lenght(points):
    """calculates the longest line that fits inside the convex hull of `points`.
       returns the two points defining this line"""

    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    distances = distance_matrix(hull_points, hull_points)

    flattened_index = np.argmax(distances)
    index = np.array(np.unravel_index(flattened_index, distances.shape))
    max_distance_points = hull_points[index]
    return max_distance_points


def rotate_pc_to_x(points, inplace=False):
    """rotates every 2D vector (x,y) in `points` so that x.max()-x.min() becomes maximal"""
    if not inplace:
        points = points.copy()  # don't mess with the input

    (x1, y1), (x2, y2) = convex_hull_lenght(points)
    rotate_by = np.arctan2(y1 - y2, x1 - x2)

    # transform to polar
    r = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    theta = np.arctan2(points[:, 1], points[:, 0])
    # rotate so that the 'shape' created by the `points` points into the x-direction
    theta -= rotate_by
    # transform back to cartesian
    points[:, 0] = r * np.cos(theta)
    points[:, 1] = r * np.sin(theta)

    if not inplace:
        return points


def argsort_by_pc(points):
    """determines how far along the principle component 
       each point is and argsort them accordingly"""

    if len(np.unique(points, axis=0)) > 2:  # with fewer points there is no 'best shape'
        points = rotate_pc_to_x(points)

    y_extend = points[:, 1].max() - points[:, 1].min()
    priorities = points[:, 0] * y_extend + points[:, 1]

    # priorities are abitrary floats. Make them ascending integers for readability
    uniques = np.unique(priorities)
    priorities = np.digitize(priorities, uniques)

    return priorities


# -
if __name__ == "__main__":
    print("Testing degeneracy module")

    def generate_points(rot, size=100):
        points = np.random.multivariate_normal(
            [100, 100], [[1, rot], [rot, 1]], size=size
        )
        return points

    def plot(points):
        plt.scatter(points[:, 0], points[:, 1], marker=".")

    # Testing the conv method (it dosn't work that well because I'm dumb)
    for i in [-1, -0.9, -0.8, -0.3, 0, 0.3, 0.8, 0.9, 1]:
        rot = i
        points = generate_points(rot=rot, size=1000)

        rs = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        ths = np.arctan2(points[:, 1], points[:, 0])

        ths += np.pi / 8
        points[:, 0] = rs * np.cos(ths)
        points[:, 1] = rs * np.sin(ths)

        cov = np.corrcoef(points.T)
        angle = np.arctan2(
            -cov[1, 0] + cov[1, 1], -cov[0, 0] + cov[0, 1]
        )  # np.arccos(cov[0,1])# - np.pi/4
        x = np.cos(angle) * np.arange(-3, 3)
        y = np.sin(angle) * np.arange(-3, 3)
        plt.plot(x + points[:, 0].mean(), y + points[:, 1].mean(), "r-")
        plot(points)
        plt.show()

    # Testing the convex hull method
    for i in range(20):
        rot = (np.random.rand() - 0.5) * 2
        points = generate_points(rot)
        center_x = points[:, 0].sum() / len(points)
        center_y = points[:, 1].sum() / len(points)
        max_distance_points = convex_hull_lenght(points)
        plt.plot(max_distance_points[:, 0], max_distance_points[:, 1], "gv")
        plot(points)
        plt.plot(center_x, center_y, "r*")
        plt.show()
        # plt.plot(hull_points[:,0],hull_points[:,1],'r_')
    # convex_hull_plot_2d(hull)

    for i in range(20):
        rot = (np.random.rand() - 0.5) * 2
        points = generate_points(rot)
        center_x = points[:, 0].sum() / len(points)
        center_y = points[:, 1].sum() / len(points)
        max_distance_points = convex_hull_lenght(points)

        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.set_aspect("equal")
        plt.plot(max_distance_points[:, 0], max_distance_points[:, 1], "gv")
        plot(points)
        plt.plot(center_x, center_y, "r*")

        ax = fig.add_subplot(122)
        ax.set_aspect("equal")
        rotate_pc_to_x(points, inplace=True)
        plot(points)

        plt.show()

    # +
    for i in range(10):
        rot = (np.random.rand() - 0.5) * 2
        points = generate_points(rot)
        center_x = points[:, 0].sum() / len(points)
        center_y = points[:, 1].sum() / len(points)
        max_distance_points = convex_hull_lenght(points)

        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.set_aspect("equal")
        plt.plot(max_distance_points[:, 0], max_distance_points[:, 1], "gv")
        plot(points)
        plt.plot(center_x, center_y, "r*")

        ax = fig.add_subplot(122)
        ax.set_aspect("equal")

        index_ordering = argsort_by_pc(points)

        plt.scatter(
            points[:, 0][index_ordering],
            points[:, 1][index_ordering],
            c=np.arange(len(points)),
        )
        plt.show()


# -

