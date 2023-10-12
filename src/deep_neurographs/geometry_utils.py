import numpy as np
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.linalg import svd
from deep_neurographs import utils


# Context Tangent Vectors
def compute_context_vec(
    neurograph,
    i,
    mutable_tangent,
    window_size=5,
    return_pts=False,
    vec_type="tangent",
):
    # Compute context vecs
    branches = get_branches(neurograph, i)
    context_vec_list = []
    xyz_list = []
    ref_xyz = neurograph.nodes[i]["xyz"]
    for branch in branches:
        context_vec, xyz = _compute_context_vec(
            branch, ref_xyz, window_size, vec_type
        )
        context_vec_list.append(context_vec)
        xyz_list.append(xyz)

    # Determine best
    max_dot_prod = 0
    arg_max = -1
    for k in range(len(context_vec_list)):
        dot_prod = abs(np.dot(mutable_tangent, context_vec_list[k]))
        if dot_prod >= max_dot_prod:
            max_dot_prod = dot_prod
            arg_max = k

    # Compute normal
    if return_pts:
        return context_vec_list, branches, xyz_list, arg_max
    else:
        return context_vec_list[arg_max]


def _compute_context_vec(all_xyz, ref_xyz, window_size, vec_type):
    from_start = orient_pts(all_xyz, ref_xyz)
    xyz = get_pts(all_xyz, window_size, from_start)
    if vec_type == "normal":
        vec = compute_normal(xyz)
    else:
        vec = compute_tangent(xyz)
    return vec, np.mean(xyz, axis=0).reshape(1, 3)


def get_branches(neurograph, i):
    nbs = []
    for j in list(neurograph.neighbors(i)):
        if frozenset((i, j)) in neurograph.immutable_edges:
            nbs.append(j)
    return [neurograph.edges[i, j]["xyz"] for j in nbs]


def orient_pts(xyz, ref_xyz):
    return True if all(xyz[0] == ref_xyz) else False


def get_pts(xyz, window_size, from_start):
    if len(xyz) > window_size and from_start:
        return xyz[0:window_size]
    elif len(xyz) > window_size and not from_start:
        return xyz[-window_size:]
    else:
        return xyz


def compute_svd(xyz):
    xyz = xyz - np.mean(xyz, axis=0)
    return svd(xyz)


def compute_tangent(xyz):
    if xyz.shape[0] == 2:
        tangent = (xyz[1] - xyz[0]) / dist(xyz[1], xyz[0])
    else:
        xyz = smooth_branch(xyz)
        U, S, VT = compute_svd(xyz)
        tangent = VT[0]
    return tangent / np.linalg.norm(tangent)


# Smoothing
def smooth_branch(xyz):
    t = np.arange(len(xyz[:, 0]) + 12)
    s = len(t) / 10
    cs_x = UnivariateSpline(t, extend_boundary(xyz[:, 0]), s=s, k=3)
    cs_y = UnivariateSpline(t, extend_boundary(xyz[:, 1]), s=s, k=3)
    cs_z = UnivariateSpline(t, extend_boundary(xyz[:, 2]), s=s, k=3)
    smoothed_x = trim_boundary(cs_x(t))
    smoothed_y = trim_boundary(cs_y(t))
    smoothed_z = trim_boundary(cs_z(t))
    smoothed = np.column_stack((smoothed_x, smoothed_y, smoothed_z))
    return smoothed


def extend_boundary(x, num_boundary_points=6):
    extended_x = np.concatenate(
        (
            np.linspace(x[0], x[1], num_boundary_points, endpoint=False),
            x,
            np.linspace(x[-2], x[-1], num_boundary_points, endpoint=False),
        )
    )
    return extended_x


def trim_boundary(x, num_boundary_points=6):
    return x[num_boundary_points:-num_boundary_points]


# Miscellaneous
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