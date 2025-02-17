import numpy as np
import mcubes
from typing import Tuple


def get_index_location(
    point: np.ndarray, resolution: np.ndarray, bbox: Tuple[np.ndarray]
):
    normalized_point = (point - bbox[0]) / (bbox[1] - bbox[0])
    index_location = np.floor((normalized_point * resolution) - 1).astype(int)
    return index_location


def get_points(vertices, edges, radii):

    points = []
    points_radii = []

    for e, r in zip(edges, radii):
        v0, v1 = vertices[e[0]], vertices[e[1]]
        length = np.linalg.norm(v0 - v1)
        for pt in np.linspace(v0, v1, max(2, int(np.floor(length / 0.67 / r)))):
            points.append(pt)
            points_radii.append(r)

    return np.array(points), np.array(points_radii)


def get_bbox(vertices, buffer):
    return np.min(vertices, axis=0) - buffer, np.max(vertices, axis=0) + buffer


def get_resolution(min_feature_size, bbox_min, bbox_max):
    return np.ceil((bbox_max - bbox_min) / min_feature_size).astype(int)


def index_to_point(idx, resolution, extent):
    return idx / resolution * (extent[1] - extent[0]) + extent[0]


def get_sphere_distance_function(
    point, point_index, min_feature_size, radius, resolution, bbox
):

    bbox_arr = np.array([bbox[0], bbox[1]]).T
    idx_extent = np.ceil(radius / min_feature_size).astype(int) + 1

    x = np.arange(
        max(0, point_index[0] - idx_extent - 1),
        min(point_index[0] + idx_extent + 1, resolution[0] - 1),
    )
    y = np.arange(
        max(0, point_index[1] - idx_extent - 1),
        min(point_index[1] + idx_extent + 1, resolution[1] - 1),
    )
    z = np.arange(
        max(0, point_index[2] - idx_extent - 1),
        min(point_index[2] + idx_extent + 1, resolution[2] - 1),
    )
    xx, yy, zz = np.meshgrid(x, y, z)

    sphere_distance = (
        np.sqrt(
            (index_to_point(xx, resolution[0], bbox_arr[0]) - point[0]) ** 2
            + (index_to_point(yy, resolution[1], bbox_arr[1]) - point[1]) ** 2
            + (index_to_point(zz, resolution[2], bbox_arr[2]) - point[2]) ** 2
        )
        - radius
    )
    return xx, yy, zz, sphere_distance


def mesh_network(vertices, edges, radii, min_feature_size):

    points, points_radii = get_points(vertices, edges, radii)
    buffered_bbox = get_bbox(vertices, 3 * np.max(radii))
    resolution = get_resolution(min_feature_size, *buffered_bbox)
    volume = np.ones(resolution)

    for point, radius in zip(points, points_radii):
        idx = get_index_location(point, resolution, buffered_bbox)
        x_indices, y_indices, z_indices, dist = get_sphere_distance_function(
            point, idx, min_feature_size, radius, resolution, buffered_bbox
        )

        volume[x_indices.flatten(), y_indices.flatten(), z_indices.flatten()] = np.clip(
            dist.flatten(),
            a_max=volume[x_indices.flatten(), y_indices.flatten(), z_indices.flatten()],
            a_min=None,
        )

    vertices, edges = mcubes.marching_cubes(volume, 0)
    vertices = (
        vertices / resolution * (buffered_bbox[1] - buffered_bbox[0]) + buffered_bbox[0]
    )
    return vertices, edges
