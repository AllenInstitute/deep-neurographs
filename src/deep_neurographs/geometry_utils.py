import numpy as np
from deep_neurographs import utils
from scipy.linalg import svd


# Context Tangent Vectors
def compute_context_vec(neurograph, i, mutable_tangent, window_size=5, return_pts=False, vec_type="tangent"):
    # Compute context vecs
    branches = get_branches(neurograph, i)
    context_vec_list = []
    xyz_list = []
    ref_xyz = neurograph.nodes[i]["xyz"]
    for branch in branches:
        context_vec, xyz = _compute_context_vec(branch, ref_xyz, window_size, vec_type)
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
        tangent = (xyz[1] - xyz[0]) / utils.dist(xyz[1], xyz[0])
    else:
        U, S, VT = compute_svd(xyz)
        tangent = VT[0]
    return tangent / np.linalg.norm(tangent)


