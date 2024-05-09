"""
Created on Sat Nov 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

"""

import numpy as np
import tensorstore as ts
from scipy.interpolate import UnivariateSpline
from scipy.linalg import svd
from scipy.spatial import distance

from deep_neurographs import utils


# Directional Vectors
def get_directional(neurograph, i, origin, window_size):
    """
    Computes the directional vector of a branch or bifurcation in a neurograph
    relative to a specified origin.

    Parameters
    ----------
    neurograph : Neurograph
        The neurograph object containing the branches.
    i : int
        The index of the branch or bifurcation in the neurograph.
    origin : numpy.ndarray
        The origin point xyz relative to which the directional vector is
        computed.
    window_size : numpy.ndarry
        The size of the window around the branch or bifurcation to consider
        for computing the directional vector.

    Returns
    -------
    numpy.ndarray
        The directional vector of the branch or bifurcation relative to the
        specified origin.

    """
    branches = neurograph.get_branches(i, ignore_reducibles=True)
    branches = shift_branches(branches, origin)
    if len(branches) == 1:
        return compute_tangent(get_subarray(branches[0], window_size))
    elif len(branches) == 2:
        branch_1 = get_subarray(branches[0], window_size)
        branch_2 = get_subarray(branches[1], window_size)
        branch = np.concatenate((branch_1, branch_2))
        return compute_tangent(branch)
    else:
        return np.array([0, 0, 0])


def get_subarray(arr, window_size):
    """
    Extracts a sub-array of a specified window size from a given input array.

    Parameters
    ----------
    branch : numpy.ndarray
        Array from which the sub-branch will be extracted.
    window_size : int
        Size of the window to extract from "arr".

    Returns
    -------
    numpy.ndarray
        A sub-array of the specified window size. If the input array is
        smaller than the window size, the entire branch array is returned.

    """
    if arr.shape[0] < window_size:
        return arr
    else:
        return arr[0:window_size, :]


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
    """
    Computes the tangent vector at a given point or along a curve defined by
    an array of points.

    Parameters
    ----------
    xyz : numpy.ndarray
        Array containing either two xyz coordinates or an arbitrary number of
        defining a curve.

    Returns
    -------
    numpy.ndarray
        Tangent vector at the specified point or along the curve.

    """
    if xyz.shape[0] == 2:
        tangent = (xyz[1] - xyz[0]) / dist(xyz[1], xyz[0])
    else:
        U, S, VT = compute_svd(xyz)
        tangent = VT[0]
    return tangent / np.linalg.norm(tangent)


def compute_normal(xyz):
    """
    Computes the normal vector of a plane defined by an array of xyz
    coordinates using Singular Value Decomposition (SVD).

    Parameters
    ----------
    xyz : numpy.ndarray
        An array of xyz coordinates that normal vector is to be computed of.

    Returns
    -------
    numpy.ndarray
        The normal vector of the array "xyz".

    """
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
    s = xyz.shape[0] / 10 if not s else xyz.shape[0] / s
    t = np.linspace(0, 1, xyz.shape[0])
    spline_x = UnivariateSpline(t, xyz[:, 0], s=s, k=3)
    spline_y = UnivariateSpline(t, xyz[:, 1], s=s, k=3)
    spline_z = UnivariateSpline(t, xyz[:, 2], s=s, k=3)
    return spline_x, spline_y, spline_z


def sample_curve(xyz, n_pts):
    t = np.linspace(0, 1, n_pts)
    spline_x, spline_y, spline_z = fit_spline(xyz, s=0)
    xyz = np.column_stack((spline_x(t), spline_y(t), spline_z(t)))
    return xyz.astype(int)


# Image feature extraction
def get_profile(img, xyz_arr, process_id=None, window=[5, 5, 5]):
    """
    Computes the maximum intensity profile along a list of 3D coordinates
    in a given image.

    Parameters
    ----------
    img : numpy.ndarray
        The image volume or TensorStore object from which to extract intensity
        profiles.
    xyz_arr : numpy.ndarray
        Array of 3D coordinates xyz representing points in the image volume.
    process_id : int or None, optional
        An optional identifier for the process. Default is None.
    window : numpy.ndarray, optional
        The size of the window around each coordinate for profile extraction.
        Default is [5, 5, 5].

    Returns
    -------
    list, tuple
        If "process_id" is provided, returns a tuple containing the process_id
        and the intensity profile list. If "process_id" is not provided,
        returns only the intensity profile list.

    """
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
        img[x - 1 : x + 2, y - 1 : y + 2, z - 1 : z + 2] = val
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
    branch_i = neurograph.get_branches(i, ignore_reducibles=True)[0]
    branch_j = neurograph.get_branches(j, ignore_reducibles=True,)[0]
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
    min_dist = np.inf
    idx = None
    for i, xyz_i in enumerate(xyz_arr):
        d = dist(xyz, xyz_i)
        if d < min_dist:
            min_dist = d
            idx = i
    return idx, min_dist


def shift_branches(branches, shift):
    """
    Shifts the coordinates of branches in a list of arrays by a specified
    shift vector.

    Parameters
    ----------
    branches : list
        A list containing arrays of 3D coordinates representing branches.
    shift : numpy.ndarray
        The shift vector (dx, dy, dz) by which to shift the coordinates.

    Returns
    -------
    list
        A list containing arrays of shifted 3D coordinates representing the
        branches.

    """
    for i, branch in enumerate(branches):
        branches[i] = branch - shift
    return branches


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
    idxs = kdtree.query_ball_point(xyz, radius, return_sorted=True)
    return kdtree.data[idxs]
