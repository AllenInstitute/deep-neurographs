"""
Created on Sat Nov 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

from copy import deepcopy

import numpy as np
import tensorstore as ts
from scipy.interpolate import UnivariateSpline
from scipy.linalg import svd
from scipy.spatial import distance

from deep_neurographs import utils


# Directional Vectors
def get_directional(
    neurograph, i, proposal_tangent, window=5, n_svd_points=10
):
    directionals = []
    d = n_svd_points
    for branch in neurograph.get_branches(i):
        if branch.shape[0] >= window + d:
            xyz = deepcopy(branch[d:, :])
        elif branch.shape[0] <= d:
            xyz = deepcopy(branch)
        else:
            xyz = deepcopy(branch[d: window + d, :])
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
    """
    Compute singular value decomposition (svd) of an NxD array where N is the
    number of points and D is the dimension of the space.

    Parameters
    ----------
    xyz : numpy.ndarray
        Array containing data points.

    Returns
    -------
    numpy.ndarry
        Unitary matrix having left singular vectors as columns. Of shape
        (N, N) or (N, min(N, D)), depending on full_matrices.
    numpy.ndarray
        Singular values, sorted in non-increasing order. Of shape (K,), with
        K = min(N, D).
    numpy.ndarray
        Unitary matrix having right singular vectors as rows. Of shape (D, D)
        or (K, D) depending on full_matrices.

    """
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


def get_midpoint(xyz_1, xyz_2):
    """
    Computes the midpoint between "xyz_1" and "xyz_2".

    Parameters
    ----------
    xyz_1 : numpy.ndarray
        n-dimensional coordinate.
    xyz_2 : numpy.ndarray
        n-dimensional coordinate.
    """
    return np.mean([xyz_1, xyz_2], axis=0)


# Smoothing
def smooth_branch(xyz, s=None):
    """
    Smooths a Nx3 array of points by fitting a cubic spline. The points are
    assumed to be continuous and the curve that they form does not have any
    branching points.

    Parameters
    ----------
    xyz : numpy.ndarray
        Array of xyz coordinates to be smoothed.
    s : float
        A parameter that controls the smoothness of the spline, where
        "s" in [0, N]. Note that the larger "s", the smoother the spline.

    Returns
    -------
    xyz : numpy.ndarray
        Smoothed points.

    """
    if xyz.shape[0] > 8:
        t = np.linspace(0, 1, xyz.shape[0])
        spline_x, spline_y, spline_z = fit_spline(xyz, s=s)
        xyz = np.column_stack((spline_x(t), spline_y(t), spline_z(t)))
    return xyz.astype(np.float32)


def fit_spline(xyz, s=None):
    """
    Fits a cubic spline to an array containing xyz coordinates.

    Parameters
    ----------
    xyz : numpy.ndarray
        Array of xyz coordinates to be smoothed.
    s : float, optional
        A parameter that controls the smoothness of the spline.

    Returns
    -------
    spline_x : UnivariateSpline
        Spline fit to x-coordinates of "xyz".
    spline_y : UnivariateSpline
        Spline fit to the y-coordinates of "xyz".
    spline_z : UnivariateSpline
        Spline fit to the z-coordinates of "xyz".

    """
    s = xyz.shape[0] / 5 if not s else xyz.shape[0] / s
    t = np.linspace(0, 1, xyz.shape[0])
    spline_x = UnivariateSpline(t, xyz[:, 0], s=s, k=3)
    spline_y = UnivariateSpline(t, xyz[:, 1], s=s, k=3)
    spline_z = UnivariateSpline(t, xyz[:, 2], s=s, k=3)
    return spline_x, spline_y, spline_z


def sample_path(path, n_points):
    if len(path) > 5:
        t = np.linspace(0, 1, n_points)
        spline_x, spline_y, spline_z = fit_spline(path, s=10)
        path = np.column_stack((spline_x(t), spline_y(t), spline_z(t)))
    else:
        path = make_line(path[0], path[-1], 10)
    return path.astype(int)


# Image feature extraction
def get_profile(img, xyz_arr, process_id=None, window=[5, 5, 5]):
    profile = []
    for xyz in xyz_arr:
        if type(img) == ts.TensorStore:
            profile.append(np.max(utils.read_tensorstore(img, xyz, window)))
        else:
            profile.append(np.max(utils.get_chunk(img, xyz, window)))

    if process_id:
        return process_id, profile
    else:
        return profile


def fill_path(img, path, val=-1):
    for xyz in path:
        x, y, z = tuple(np.floor(xyz).astype(int))
        img[x - 1: x + 2, y - 1: y + 2, z - 1: z + 2] = val
    return img


# Proposal optimization
def optimize_alignment(neurograph, img, edge, depth=15):
    """
    Optimizes alignment of edge proposal between two branches by finding
    straight path with the brightest averaged image profile.

    Parameters
    ----------
    neurograph : NeuroGraph
        Predicted neuron reconstruction to be corrected.
    img : numpy.ndarray
        Image chunk that the reconstruction is contained in.
    edge : frozenset
        Edge proposal to be aligned.
    depth : int, optional
        Maximum depth checked during alignment optimization. The default value
        is 15.

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        xyz coordinates of aligned edge proposal.

    """
    if neurograph.is_simple(edge):
        return optimize_simple_alignment(neurograph, img, edge, depth=depth)
    else:
        return optimize_complex_alignment(neurograph, img, edge, depth=depth)


def optimize_simple_alignment(neurograph, img, edge, depth=15):
    """
    Optimizes alignment of edge proposal for simple edges.

    Parameters
    ----------
    neurograph : NeuroGraph
        Predicted neuron reconstruction to be corrected.
    img : numpy.ndarray
        Image chunk that the reconstruction is contained in.
    edge : frozenset
        Edge proposal to be aligned.
    depth : int, optional
        Maximum depth checked during alignment optimization. The default value
        is 15.

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        xyz coordinates of aligned edge proposal.

    """
    i, j = tuple(edge)
    branch_i = neurograph.get_branches(i)[0]
    branch_j = neurograph.get_branches(j)[0]
    d_i, d_j, _ = align(neurograph, img, branch_i, branch_j, depth)
    return branch_i[d_i], branch_j[d_j]


def optimize_complex_alignment(neurograph, img, edge, depth=15):
    """
    Optimizes alignment of edge proposal for complex edges.

    Parameters
    ----------
    neurograph : NeuroGraph
        Predicted neuron reconstruction to be corrected.
    img : numpy.ndarray
        Image chunk that the reconstruction is contained in.
    edge : frozenset
        Edge proposal to be aligned.
    depth : int, optional
        Maximum depth checked during alignment optimization. The default value
        is 15.

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        xyz coordinates of aligned edge proposal.

    """
    i, j = tuple(edge)
    branch = neurograph.get_branches(i if neurograph.is_leaf(i) else j)[0]
    branches = neurograph.get_branches(j if neurograph.is_leaf(i) else i)
    d1, e1, val_1 = align(neurograph, img, branch, branches[0], depth)
    d2, e2, val_2 = align(neurograph, img, branch, branches[1], depth)
    pair_1 = (branch[d1], branches[0][e1])
    pair_2 = (branch[d2], branches[1][e2])
    return pair_1 if val_1 > val_2 else pair_2


def align(neurograph, img, branch_1, branch_2, depth):
    """
    Finds straight line path between end points of "branch_1" and "branch_2"
    that best captures the image signal. This path is determined by checking
    the average image intensity of the line drawn from "branch_1[d_1]" and
    "branch_2[d_2]" with d_1, d_2 in [0, depth].

    Parameters
    ----------
    neurograph : NeuroGraph
        Predicted neuron reconstruction to be corrected.
    img : numpy.ndarray
        Image chunk that the reconstruction is contained in.
    branch_1 : np.ndarray
        Branch corresponding to some predicted neuron. This branch must be
        oriented so that the end points being considered are the coordinates
        in rows 0 through "depth".
    branch_2 : np.ndarray
        Branch corresponding to some predicted neuron. This branch must be
        oriented so that the end points being considered are the coordinates
        in rows 0 through "depth".
    depth : int
        Maximum depth of branch that is optimized over.

    Returns
    -------
    best_xyz_1 : np.ndarray
        Optimal xyz coordinate from "branch_1".
    best_xyz_2 : np.ndarray
        Optimal xyz coordinate from "branch_2".
    best_score : float
        Average brightness of voxels sampled along line between "best_xyz_1"
        and "best_xyz_2".

    """
    best_d1 = None
    best_d2 = None
    best_score = 0
    for d1 in range(min(depth, len(branch_1) - 1)):
        xyz_1 = neurograph.to_img(branch_1[d1], shift=True)
        for d2 in range(min(depth, len(branch_2) - 1)):
            xyz_2 = neurograph.to_img(branch_2[d2], shift=True)
            line = make_line(xyz_1, xyz_2, 10)
            score = np.mean(get_profile(img, line, window=[3, 3, 3]))
            if score > best_score:
                best_score = score
                best_d1 = d1
                best_d2 = d2
    return best_d1, best_d2, best_score


# Miscellaneous
def dist(v_1, v_2, metric="l2"):
    """
    Computes distance between "v_1" and "v_2".

    Parameters
    ----------
    v_1 : np.ndarray
        Vector.
    v_2 : np.ndarray
        Vector.

    Returns
    -------
    float
        Distance between "v_1" and "v_2".

    """
    if metric == "l1":
        return np.sum(v_1 - v_2)
    else:
        return distance.euclidean(v_1, v_2)


def check_dists(xyz_1, xyz_2, xyz_3, radius):
    """
    Checks whether distance between "xyz_1", "xyz_3" and "xyz_2", "xyz_3" is
    sufficiently small. Routine is used during edge proposal generation to
    determine whether to create new vertex at "xyz_2" or draw proposal between
    "xyz_1" and existing node at "xyz_3".

    Parameters
    ----------
    xyz_1 : np.ndarray
        xyz coordinate of leaf node (i.e. source of edge proposal).
    xyz_2 : np.ndarray
        xyz coordinate queried from kdtree (i.e. dest of edge proposal).
    xyz_3 : np.ndarray
        xyz coordinate of existing node in graph that is near "xyz_2".
    radius : float
        Maximum Euclidean length of edge proposal.

    Parameters
    ----------
    bool
        Indication of whether to draw edge proposal between "xyz_1" and
        "xyz_3".

    """
    d_1 = dist(xyz_1, xyz_3) < radius
    d_2 = dist(xyz_2, xyz_3) < 5
    return True if d_1 and d_2 else False


def make_line(xyz_1, xyz_2, n_steps):
    xyz_1 = np.array(xyz_1)
    xyz_2 = np.array(xyz_2)
    t_steps = np.linspace(0, 1, n_steps)
    return np.array([(1 - t) * xyz_1 + t * xyz_2 for t in t_steps], dtype=int)


def normalize(vec, norm="l2"):
    return vec / abs(dist(np.zeros((3)), vec, metric=norm))
