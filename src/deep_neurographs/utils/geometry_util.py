"""
Created on Sat Nov 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.linalg import svd
from scipy.spatial import distance

from deep_neurographs.utils import img_util


# --- Directionals ---
def get_directional(branches, origin, depth):
    """
    Computes the directional vector of a branch or bifurcation in a neurograph
    relative to a specified origin.

    Parameters
    ----------
    neurograph : Neurograph
        The neurograph object containing the branches.
    origin : numpy.ndarray
        The origin point xyz relative to which the directional vector is
        computed.
    depth : numpy.ndarry
        The size of the window in microns around the branch or bifurcation to
        consider for computing the directional vector.

    Returns
    -------
    numpy.ndarray
        The directional vector of the branch or bifurcation relative to the
        specified origin.

    """
    branches = [shift_path(b, origin) for b in branches]
    if len(branches) == 1:
        return tangent(truncate_path(branches[0], depth))
    else:
        branch_1 = truncate_path(branches[0], depth)
        branch_2 = truncate_path(branches[1], depth)
        return tangent(np.concatenate((branch_1, branch_2)))


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


def tangent(xyz_arr):
    """
    Computes the tangent vector at a given point or along a curve defined by
    an array of points.

    Parameters
    ----------
    xyz_arr : numpy.ndarray
        Array containing either two xyz coordinates or an arbitrary number of
        defining a curve.

    Returns
    -------
    numpy.ndarray
        Tangent vector at the specified point or along the curve.

    """
    if len(xyz_arr) == 2:
        d = max(dist(xyz_arr[1], xyz_arr[0]), 0.1)
        tangent_vec = (xyz_arr[1] - xyz_arr[0]) / d
    else:
        _, _, VT = compute_svd(xyz_arr)
        tangent_vec = VT[0]
        if np.dot(tangent_vec, tangent([xyz_arr[0], xyz_arr[-1]])) < 0:
            tangent_vec *= -1
    return tangent_vec / np.linalg.norm(tangent_vec)


def midpoint(xyz_1, xyz_2):
    """
    Computes the midpoint between "xyz_1" and "xyz_2".

    Parameters
    ----------
    xyz_1 : numpy.ndarray
        n-dimensional coordinate.
    xyz_2 : numpy.ndarray
        n-dimensional coordinate.

    Returns
    -------
    numpy.ndarray
        Midpoint of "xyz_1" and "xyz_2".

    """
    return np.mean([xyz_1, xyz_2], axis=0)


# --- Path utils ---
def sample_path(xyz_path, n_points):
    """
    Uniformly samples points from a curve represented as an array.

    Parameters
    ----------
    xyz_arr : np.ndarray
        xyz coordinates that form a continuous path.
    n_points : int
        Number of points to be sampled.

    Returns
    -------
    numpy.ndarray
        Resampled points along curve.

    """
    k = 1 if len(xyz_path) <= 3 else 3
    t = np.linspace(0, 1, n_points)
    spline_x, spline_y, spline_z = fit_spline(xyz_path, k=k, s=0)
    xyz_path = np.column_stack((spline_x(t), spline_y(t), spline_z(t)))
    return xyz_path.astype(int)


def truncate_path(xyz_path, depth):
    """
    Extracts a sub-path of a specified depth from a given input path.

    Parameters
    ----------
    xyz_path : array-like
        xyz coordinates that form a continuous path.
    depth : int
        Path length in microns to extract from input path.

    Returns
    -------
    numpy.ndarray
        Sub-path of a specified depth from a given input path.

    """
    length = 0
    for i in range(1, len(xyz_path)):
        length += dist(xyz_path[i - 1], xyz_path[i])
        if length > depth:
            return np.array(xyz_path[0:i])
    return np.array(xyz_path)


def shift_path(xyz_path, offset):
    """
    Shifts "voxels" by subtracting the min coordinate in "bbox".

    Parameters
    ----------
    voxels : ArrayLike
        Voxel coordinates to be shifted.
    offset : ArrayLike
        ...

    Returns
    -------
    numpy.ndarray
        Voxels shifted by min coordinate in "bbox".

    """
    offset = np.array(offset)
    return [tuple(xyz - offset) for xyz in map(np.array, xyz_path)]


def fill_path(img, path, val=-1):
    """
    Fills a given path in a 3D image array with a specified value.

    Parameters
    ----------
    img : numpy.ndarray
        The 3D image array to fill the path in.
    path : iterable
        A list or iterable containing 3D coordinates (x, y, z) representing
        the path.
    val : int, optional
        The value to fill the path with. Default is -1.

    Returns
    -------
    numpy.ndarray
        The modified image array with the path filled with the specified value.

    """
    for xyz in path:
        x, y, z = tuple(np.floor(xyz).astype(int))
        img[x - 2: x + 3, y - 2: y + 3, z - 2: z + 3] = val
    return img


def path_length(path):
    """
    Computes the path length of "path".

    Parameters
    ----------
    path : list
        xyz coordinates that form a path.

    Returns
    -------
    float
        Path length of "path".

    """
    return np.sum([dist(path[i], path[i - 1]) for i in range(1, len(path))])


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
    numpy.ndarray
        Smoothed points.

    """
    if xyz.shape[0] > 8:
        t = np.linspace(0, 1, xyz.shape[0])
        spline_x, spline_y, spline_z = fit_spline(xyz, s=s)
        xyz = np.column_stack((spline_x(t), spline_y(t), spline_z(t)))
    return xyz.astype(np.float32)


def fit_spline(xyz, k=2, s=None):
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
    UnivariateSpline
        Spline fit to x-coordinates of "xyz".
    UnivariateSpline
        Spline fit to the y-coordinates of "xyz".
    UnivariateSpline
        Spline fit to the z-coordinates of "xyz".

    """
    s = xyz.shape[0] / 10 if not s else xyz.shape[0] / s
    t = np.linspace(0, 1, xyz.shape[0])
    spline_x = UnivariateSpline(t, xyz[:, 0], k=k, s=s)
    spline_y = UnivariateSpline(t, xyz[:, 1], k=k, s=s)
    spline_z = UnivariateSpline(t, xyz[:, 2], k=k, s=s)
    return spline_x, spline_y, spline_z


# --- kd-tree utils ---
def query_ball(kdtree, xyz, radius):
    """
    Queries a KD-tree for points within a given radius from a target point.

    Parameters
    ----------
    kdtree : scipy.spatial.cKDTree
        The KD-tree data structure containing the points to query.
    xyz : numpy.ndarray
        The target 3D coordinate (x, y, z) around which to search for points.
    radius : float
        The radius within which to search for points.

    Returns
    -------
    numpy.ndarray
        An array containing the points within the specified radius from the
        target coordinate.

    """
    idxs = kdtree.query_ball_point(xyz, radius)
    return kdtree.data[idxs]


def kdtree_query(kdtree, xyz):
    """
    Gets the xyz coordinates of the nearest neighbor of "xyz" from "kdtree".

    Parameters
    ----------
    xyz : tuple
        xyz coordinate to be queried.

    Returns
    -------
    tuple
        xyz coordinate of the nearest neighbor of "xyz".

    """
    _, idx = kdtree.query(xyz)
    return tuple(kdtree.data[idx])


# --- Proposal optimization ---
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
    branch_i = neurograph.get_branches(i, ignore_reducibles=True)[0]
    branch_j = neurograph.get_branches(j, ignore_reducibles=True)[0]
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
    branch = neurograph.branches(i if neurograph.is_leaf(i) else j)[0]
    branches = neurograph.branches(j if neurograph.is_leaf(i) else i)
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
    np.ndarray
        Optimal xyz coordinate from "branch_1".
    np.ndarray
        Optimal xyz coordinate from "branch_2".
    float
        Average brightness of voxels sampled along line between "best_xyz_1"
        and "best_xyz_2".

    """
    best_d1 = None
    best_d2 = None
    best_score = 0
    for d1 in range(min(depth, len(branch_1) - 1)):
        xyz_1 = neurograph.to_voxels(branch_1[d1], shift=True)
        for d2 in range(min(depth, len(branch_2) - 1)):
            xyz_2 = neurograph.to_voxels(branch_2[d2], shift=True)
            line = make_line(xyz_1, xyz_2, 10)
            score = np.mean(img_util.get_profile(img, line))
            if score > best_score:
                best_score = score
                best_d1 = d1
                best_d2 = d2
    return best_d1, best_d2, best_score


# --- Miscellaneous ---
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


def make_line(xyz_1, xyz_2, n_steps):
    """
    Generates a series of points representing a straight line between two 3D
    coordinates.

    Parameters
    ----------
    xyz_1 : tuple or array-like
        The starting 3D coordinate (x, y, z) of the line.
    xyz_2 : tuple or array-like
        The ending 3D coordinate (x, y, z) of the line.
    n_steps : int
        The number of steps to interpolate between the two coordinates.

    Returns
    -------
    numpy.ndarray
        An array of shape (n_steps, 3) containing the interpolated 3D
        coordinates representing the straight line between xyz_1 and xyz_2.

    """
    xyz_1 = np.array(xyz_1)
    xyz_2 = np.array(xyz_2)
    t_steps = np.linspace(0, 1, n_steps)
    return np.array([(1 - t) * xyz_1 + t * xyz_2 for t in t_steps], dtype=int)


def normalize(vector, norm="l2"):
    """
    Normalizes a vector to have unit length with respect to a specified norm.

    Parameters
    ----------
    vector : numpy.ndarray
        The input vector to be normalized.
    norm : str, optional
        The norm to use for normalization. Default is "l2".

    Returns
    -------
    numpy.ndarray
        The normalized vector with unit length with respect to the specified
        norm.

    """
    return vector / abs(dist(np.zeros((3)), vector, metric=norm))


def nearest_neighbor(xyz_arr, xyz):
    """
    Finds the nearest neighbor in a list of 3D coordinates to a given target
    coordinate.

    Parameters
    ----------
    xyz_arr : numpy.ndarray
        Array of 3D coordinates to search for the nearest neighbor.
    xyz : numpy.ndarray
        The target 3D coordinate xyz to find the nearest neighbor to.

    Returns
    -------
    tuple[int, float]
        A tuple containing the index of the nearest neighbor in "xyz_arr" and
        the distance between the target coordinate `xyz` and its nearest
        neighbor.

    """
    best_dist = np.inf
    best_xyz = None
    for i, xyz_i in enumerate(xyz_arr):
        if dist(xyz, xyz_i) < best_dist:
            best_dist = dist(xyz, xyz_i)
            best_xyz = xyz_i
    return best_xyz
