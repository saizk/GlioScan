import numpy as np


def compute_bounding_cube(segmentation):
    # Get the coordinates of all non-zero (glioma) voxels
    indices = np.where(segmentation > 0)
    # Extract min and max coordinates for each dimension
    bounding_cube = [(min(indices[dim]), max(indices[dim]) + 1) for dim in range(segmentation.ndim)]
    return tuple(bounding_cube)


def scale_bounding_cube(bounding_cube, n_pixels=None, factor=None):
    if n_pixels is not None:
        scaled_cube = tuple(
            (max(0, coord[0] + n_pixels), coord[1] + n_pixels) for coord in bounding_cube
        )
    elif factor is not None:
        scaled_cube = tuple(
            (
                max(0, int(coord[0] - (coord[1] - coord[0]) * (factor - 1) / 2)),
                int(coord[1] + (coord[1] - coord[0]) * (factor - 1) / 2)
            ) for coord in bounding_cube
        )
    else:
        raise ValueError("Either n_pixels or factor should be provided!")

    return scaled_cube


def interval_length(t):
    return tuple(b - a for a, b in t)


