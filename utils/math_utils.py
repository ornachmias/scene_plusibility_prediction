import random

import numpy as np
from math import pi


def clamp_bbox(bbox, max_size):
    formatted_bbox = []
    for b in bbox:
        formatted_bbox.append(max(0, min(b, max_size)))

    return formatted_bbox


def bbox_valid(bbox):
    if bbox is None:
        return False

    return 0 not in [bbox[2] - bbox[0], bbox[3] - bbox[1]]


def deg_to_rad(deg):
    return pi * deg / 180


def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def random_points_in_sphere(center, radius, n_points):
    min_x = center[0] - radius
    max_x = center[0] + radius

    min_y = center[1] - radius
    max_y = center[1] + radius

    min_z = center[2] - radius
    max_z = center[2] + (radius / 2)

    x_values = [random.uniform(min_x, max_x) for _ in range(n_points)]
    y_values = [random.uniform(min_y, max_y) for _ in range(n_points)]
    z_values = [random.uniform(min_z, max_z) for _ in range(n_points)]

    return [(x_values[i], y_values[i], z_values[i]) for i in range(n_points)]


def random_points_above(initial_location, radius, n_locations, limit_z=(None, None)):
    selected_points = []
    while len(selected_points) < n_locations:
        x_location = initial_location[0] + random.uniform(-1 * radius / 2, radius / 2)
        y_location = initial_location[1] + random.uniform(-1 * radius / 2, radius / 2)
        min_z, max_z = initial_location[2], initial_location[2] + radius
        config_min_z, config_max_z = None, None
        if limit_z:
            if limit_z[0] is not None:
                config_min_z = limit_z[0]
            if limit_z[1] is not None:
                config_max_z = limit_z[1]

        if config_min_z and min_z < config_min_z:
            min_z = config_min_z

        if config_max_z and max_z > config_max_z:
            max_z = config_max_z

        z_location = random.uniform(min_z, max_z + radius / 2)
        selected_points.append((x_location, y_location, z_location))

    return selected_points


def get_random_from_normal(a, b, n):
    min_limit = np.min([a, b])
    max_limit = np.max([a, b])

    values = np.random.normal(loc=((a + b) / 2), scale=np.abs(a - b), size=n)
    values = values[(values >= min_limit) & (values <= max_limit)]

    while len(values) < n:
        tmp_values = np.random.normal(loc=((a + b) / 2), scale=np.abs(a - b), size=n)
        tmp_values = tmp_values[(tmp_values >= min_limit) & (tmp_values <= max_limit)]
        values = np.concatenate((values, tmp_values))

    return values
