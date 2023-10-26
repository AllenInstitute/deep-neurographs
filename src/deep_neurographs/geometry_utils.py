import numpy as np
from scipy.interpolate import UnivariateSpline
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


def compute_normal(xyz):
    U, S, VT = compute_svd(xyz)
    normal = VT[-1]
    return normal / np.linalg.norm(normal)


# Smoothing
def smooth_branch(xyz):
    if xyz.shape[0] > 5:
        spl_x, spl_y, spl_z = fit_spline(xyz)
        t = np.arange(xyz.shape[0])
        xyz = np.column_stack((spl_x(t), spl_y(t), spl_z(t)))
    return xyz


def fit_spline(xyz):
    s = xyz.shape[0] / 10
    t = np.arange(xyz.shape[0])
    cs_x = UnivariateSpline(t, xyz[:, 0], s=s, k=3)
    cs_y = UnivariateSpline(t, xyz[:, 1], s=s, k=3)
    cs_z = UnivariateSpline(t, xyz[:, 2], s=s, k=3)
    return cs_x, cs_y, cs_z


def smooth_end(branch_xyz, radii, ref_xyz, num_pts=8):
    smooth_bool = branch_xyz.shape[0] > 10
    if all(branch_xyz[0] == ref_xyz) and smooth_bool:
        return branch_xyz[num_pts:-1, :], radii[num_pts:-1], 0
    elif all(branch_xyz[-1] == ref_xyz) and smooth_bool:
        branch_xyz = branch_xyz[:-num_pts]
        radii = radii[:-num_pts]
        return branch_xyz, radii, -1
    else:
        return branch_xyz, radii, None


# Image feature extraction
def get_profile(
    img, xyz_arr, anisotropy=[1.0, 1.0, 1.0], window_size=[4, 4, 4]
):
    xyz_arr = get_coords(xyz_arr, anisotropy=anisotropy)
    profile = []
    for xyz in xyz_arr:
        img_chunk = utils.read_img_chunk(img, xyz, window_size)
        profile.append(np.max(img_chunk))
    return np.array(profile)


def get_coords(xyz_arr, anisotropy=[1.0, 1.0, 1.0]):
    for i in range(3):
        xyz_arr[:, i] = xyz_arr[:, i] / anisotropy[i]
    return xyz_arr.astype(int)


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


def make_line(xyz_1, xyz_2, num_steps):
    t_steps = np.linspace(0, 1, num_steps)
    return np.array([(1 - t) * xyz_1 + t * xyz_2 for t in t_steps])
