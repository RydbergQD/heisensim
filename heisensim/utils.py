import numpy as np


def zeros_from_coords(coords_names, coords):
    return np.zeros((len(coords[c_name]) for c_name in coords_names))


def coords_tuple(coords_names, coords):
    return coords_names, zeros_from_coords(coords_names, coords)
