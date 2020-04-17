"""Transforms spacetime coordinates to 1D Hilbert distances"""

import numpy as np
import torch
import pandas as pd

from hilbertcurve.hilbertcurve import HilbertCurve
from sklearn.preprocessing import MinMaxScaler


def hilbert_distance(coordinates, iterations=10):
    count_entries, dims = coordinates.shape
    distances = np.zeros([count_entries])
    cells = 2 ** (iterations * dims)
    transform = HilbertCurve(iterations, dims)
    mm = MinMaxScaler((0, transform.max_x))
    normed_coords = np.rint(mm.fit_transform(coordinates)).astype("int")

    for i in range(len(coordinates)):
        distances[i] = transform.distance_from_coordinates(normed_coords[i])
    return distances


def rotate_2d(coordinates, angle, axis_0=0, axis_1=1):
    """Rotate all coordinates from `axis_0` to `axis_1` by `angle`"""
    coordinates = coordinates.copy()
    r = np.sqrt(coordinates[:, axis_0] ** 2 + coordinates[:, axis_1] ** 2)
    theta = np.arctan2(coordinates[:, axis_1], coordinates[:, axis_0])
    coordinates[:, axis_0] = np.cos(theta + angle) * r
    coordinates[:, axis_1] = np.sin(theta + angle) * r
    return coordinates


class TorchMinMaxScaler:
    def __init__(self, feature_range=[0, 1]):
        self.minimum = feature_range[0]
        self.maximum = feature_range[1]

    def __call__(self, x):
        return (x - self.minimum) / (self.maximum - self.minimum)


class Hilbertize:
    """Transforms a batch of coordinates from Euclidean space to pseudo Hilbert distances space."""
    # angle that maximizes separation between two Hilbert distance measures
    ANGLE = np.deg2rad(45)

    def __init__(
        self, rotation_axis=((0, 1), (1, 2), (3, 0)), dimension_columns = ['x','y','z','t'], iterations=10
    ):
        # TODO: document rotation_axis
        self.rotation_axis = rotation_axis
        self.dimension_columns = dimension_columns
        self.dims = len(dimension_columns)
        self.dims_output =  1+ len(self.rotation_axis)
        self.cells = 2 ** (iterations * self.dims)
        self.transform = HilbertCurve(iterations, self.dims)
        self.scale_factor = self.transform.max_x
        self.edge_resolution = self.transform.max_x
        
    def _get_output_column_names(self):
        return [f"hilbert_{i}" for i in range(self.dims_output)]

    def __str__(self):
        return (
            f"Hilbertize from {self.dims}D, "
            f"resolution by edge: {self.edge_resolution+1}, "
            f"resolution by volume: {self.cells:.1e} cells."
        )

    def __call__(self, x:pd.DataFrame):
        coordinates = x[self.dimension_columns].values
        non_coordinates = x.drop(columns=self.dimension_columns)
        
        waveforms, dimensions = coordinates.shape
        assert dimensions == self.dims
        assert np.all(coordinates >= 0)
        assert np.all(coordinates <= 1)

        coordinates -= 0.5  # rotate around center of mass
        rotations = [coordinates] + [
            rotate_2d(coordinates, self.ANGLE, *ax) for ax in self.rotation_axis
        ]
        rotations = np.stack(rotations)

        # XXX: this also manipulates all axis that are not part of the rotations
        # this leads to a smaller parameter space, but I'm not sure if that's a bad thing
        rotation_in_01 = rotations / np.sqrt(2) + 0.5
        assert rotation_in_01.max() <= 1
        assert rotation_in_01.min() >= 0

        scaled_rot = rotation_in_01 * self.scale_factor
        scaled_rot = np.rint(scaled_rot).astype("int")

        distances = np.zeros([len(coordinates),self.dims_output])
        for r, coordinates in enumerate(scaled_rot):
            for c, coordinate in enumerate(coordinates):
                distances[c][r] = self.transform.distance_from_coordinates(coordinate)

        distances /= self.cells # scale distances to [0,1]
        
        transformed_x = non_coordinates
        for dim, key in enumerate(self._get_output_column_names()):
            transformed_x[key] = distances[:,dim]
        return transformed_x    

