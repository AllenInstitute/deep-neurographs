import heapq
from copy import deepcopy

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.linalg import svd

from deep_neurographs import utils


# Directional Vectors
def get_directional(neurograph, i, proposal_tangent, window=5):
    # Compute principle axes
    directionals = []
    d = neurograph.optimize_depth
    for branch in neurograph.get_branches(i):
        if branch.shape[0] >= window + d:
            xyz = deepcopy(branch[d:, :])
        elif branch.shape[0] <= d:
            xyz = deepcopy(branch)
        else:
            xyz = deepcopy(branch[d : window + d, :])
        # print(xyz)
        directionals.append(compute_tangent(xyz))

    # Determine best
    max_dot_prod = 0
    arg_max = -1
    for k in range(len(directionals)):
        dot_prod = abs(np.dot(proposal_tangent, directionals[k]))
        if dot_prod >= max_dot_prod:
            max_dot_prod = dot_prod
            arg_max = k

    return directionals[arg_max]


def compute_svd(xyz):
    xyz = xyz - np.mean(xyz, axis=0)
    return svd(xyz)


def compute_tangent(xyz):
    if xyz.shape[0] == 2:
        tangent = (xyz[1] - xyz[0]) / dist(xyz[1], xyz[0])
    else:
        xyz = smooth_branch(xyz, s=10)
        U, S, VT = compute_svd(xyz)
        tangent = VT[0]
    return tangent / np.linalg.norm(tangent)


def compute_normal(xyz):
    U, S, VT = compute_svd(xyz)
    normal = VT[-1]
    return normal / np.linalg.norm(normal)


def get_midpoint(xyz1, xyz2):
    return np.mean([xyz1, xyz2], axis=0)


# Smoothing
def smooth_branch(xyz, s=None):
    if xyz.shape[0] > 5:
        t = np.linspace(0, 1, xyz.shape[0])
        spline_x, spline_y, spline_z = fit_spline(xyz, s=s)
        xyz = np.column_stack((spline_x(t), spline_y(t), spline_z(t)))
    return xyz


def fit_spline(xyz, s=None):
    s = xyz.shape[0] / 5 if not s else xyz.shape[0] / s
    t = np.linspace(0, 1, xyz.shape[0])
    spline_x = UnivariateSpline(t, xyz[:, 0], s=s, k=1)
    spline_y = UnivariateSpline(t, xyz[:, 1], s=s, k=1)
    spline_z = UnivariateSpline(t, xyz[:, 2], s=s, k=1)
    return spline_x, spline_y, spline_z


def sample_path(path, num_points):
    t = np.linspace(0, 1, num_points)
    spline_x, spline_y, spline_z = fit_spline(path, s=10)
    path = np.column_stack((spline_x(t), spline_y(t), spline_z(t)))
    return path.astype(int)


# Image feature extraction
def get_profile(img, xyz_arr, window_size=[5, 5, 5]):
    return [np.max(utils.get_chunk(img, xyz, window_size)) for xyz in xyz_arr]


def fill_path(img, path, val=-1):
    for xyz in path:
        x, y, z = tuple(np.floor(xyz).astype(int))
        # img[x - 1: x + 2, y - 1: y + 2, z - 1: z + 2] = val
        img[x, y, z] = val
    return img


# Miscellaneous
def shortest_path(img, start, end):
    def is_valid_move(x, y, z):
        return (
            0 <= x < shape[0]
            and 0 <= y < shape[1]
            and 0 <= z < shape[2]
            and not visited[x, y, z]
        )

    def get_nbs(x, y, z):
        moves = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ]
        return [
            (x + dx, y + dy, z + dz)
            for dx, dy, dz in moves
            if is_valid_move(x + dx, y + dy, z + dz)
        ]

    img = img - np.min(img) + 1
    start = tuple(start)
    end = tuple(end)

    shape = img.shape
    visited = np.zeros(shape, dtype=bool)
    distances = np.inf * np.ones(shape)
    distances[start] = 0
    previous_nodes = {}

    heap = [(0, start)]
    while heap:
        current_distance, cur_node = heapq.heappop(heap)

        if cur_node == end:
            path = []
            while cur_node != start:
                path.append(cur_node)
                cur_node = previous_nodes[cur_node]
            path.append(start)
            return path[::-1]

        visited[cur_node] = True

        for nb in get_nbs(*cur_node):
            if not visited[nb]:
                new_distance = distances[cur_node] + 1 / img[nb]
                if new_distance < distances[nb]:
                    distances[nb] = new_distance
                    previous_nodes[nb] = cur_node
                    heapq.heappush(heap, (new_distance, nb))
    return None


def transform_path(path, img_origin, patch_centroid, patch_dims):
    img_origin = np.array(img_origin)
    transformed_path = np.zeros((len(path), 3))
    for i, xyz in enumerate(path):
        hat_xyz = utils.patch_to_img(xyz, patch_centroid, patch_dims)
        transformed_path[i, :] = utils.to_world(hat_xyz, shift=-img_origin)
    return smooth_branch(transformed_path, s=10)


def get_optimal_patch(xyz_1, xyz_2, buffer=8):
    return [int(abs(xyz_1[i] - xyz_2[i])) + buffer for i in range(3)]


def compare_edges(xyx_i, xyz_j, xyz_k):
    dist_ij = dist(xyx_i, xyz_j)
    dist_ik = dist(xyx_i, xyz_k)
    return dist_ij < dist_ik


def dist(x, y, metric="l2"):
    """
    Computes distance between "x" and "y".

    Parameters
    ----------

    Returns
    -------
    float

    """
    if metric == "l1":
        return np.linalg.norm(np.subtract(x, y), ord=1)
    else:
        return np.linalg.norm(np.subtract(x, y), ord=2)


def make_line(xyz_1, xyz_2, num_steps):
    xyz_1 = np.array(xyz_1)
    xyz_2 = np.array(xyz_2)
    t_steps = np.linspace(0, 1, num_steps)
    return np.array([(1 - t) * xyz_1 + t * xyz_2 for t in t_steps], dtype=int)


def normalize(x, norm="l2"):
    zero_vec = np.zeros((3))
    return x / abs(dist(zero_vec, x, metric=norm))
