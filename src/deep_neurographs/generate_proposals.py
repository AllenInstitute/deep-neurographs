"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Module used to generate edge proposals.

"""

import numpy as np

from deep_neurographs import geometry

ENDPOINT_DIST = 15
LONG_RANGE_FACTOR = 2
TRIM_SEARCH_DIST = 10


def run(
    neurograph,
    radius,
    complex_bool=True,
    long_range_bool=False,
    trim_endpoints_bool=False
):
    """
    Generates proposals emanating from "leaf".

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that proposals will be generated for.
    radius : float
        Maximum Euclidean distance between endpoints of proposal.
    complex_bool : bool, optional
        Indication of whether to generate complex proposals. The default is
        True.
    long_range_bool : bool, optional
        Indication of whether to generate simple proposals within distance of
        2 * radius of leaf. The default is False.
    trim_endpoints_bool : bool, optional
        Indication of whether to endpoints of branches with exactly one
        proposal. The default is False.

    Returns
    -------
    None

    """
    # Initializations
    connections = dict()
    n_endpoints_trimmed = 0
    neurograph.init_kdtree()
    neurograph.init_kdtree(node_type="leaf")
    if complex_bool or trim_endpoints_bool:
        kdtree = neurograph.kdtree
    else:
        kdtree = neurograph.leaf_kdtree

    # Main
    limit = neurograph.proposals_per_leaf
    for leaf in neurograph.leafs:
        # Generate candidates
        leaf_swc_id = neurograph.nodes[leaf]["swc_id"]
        candidates = get_candidates(neurograph, kdtree, leaf, radius, limit)
        if len(candidates) == 0 and long_range_bool:
            candidates = get_candidates(
                neurograph, kdtree, leaf, radius * LONG_RANGE_FACTOR, -1
            )

        # Parse candidates
        for xyz in candidates:
            # Check whether candidate is valid
            i = get_conection(neurograph, leaf, xyz)
            if neurograph.is_invalid_proposal(leaf, i, complex_bool):
                continue

            # Check whether connection between components already exists
            pair_id = frozenset((leaf_swc_id, neurograph.nodes[i]["swc_id"]))
            if pair_id in connections.keys():
                cur_proposal = connections[pair_id]
                cur_dist = neurograph.proposal_length(cur_proposal)
                if neurograph.dist(leaf, i) < cur_dist:
                    neurograph.remove_proposal(cur_proposal)
                    del connections[pair_id]
                else:
                    continue

            # Add proposal
            neurograph.add_proposal(leaf, i)
            connections[pair_id] = frozenset({leaf, i})

    if trim_endpoints_bool:
        run_trimming(neurograph)


def get_candidates(neurograph, kdtree, leaf, radius, max_candidates):
    """
    Generates proposals for node "leaf" in "neurograph" by finding candidate
    xyz points on distinct connected components nearby.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph built from swc files.
    kdtree : ...
        ...
    leaf : int
        Leaf node that proposals are to be generated from.
    radius : float
        Maximum Euclidean distance between endpoints of proposal.
    max_candidates : bool
        Maximum number of proposals that can be generated from a leaf node.

    Returns
    -------
    list[tuple]
        xyz coordinates of candidates.

    """
    # Generate candidates
    candidates = dict()
    leaf_xyz = neurograph.nodes[leaf]["xyz"]
    for xyz in geometry.query_ball(kdtree, leaf_xyz, radius):
        swc_id = neurograph.xyz_to_swc(xyz)
        if swc_id != neurograph.nodes[leaf]["swc_id"]:
            d = geometry.dist(leaf_xyz, xyz)
            if swc_id not in candidates.keys():
                candidates[swc_id] = {"dist": d, "xyz": tuple(xyz)}
            elif geometry.dist(leaf_xyz, xyz) < candidates[swc_id]["dist"]:
                d = geometry.dist(leaf_xyz, xyz)
                candidates[swc_id] = {"dist": d, "xyz": tuple(xyz)}

    # Check whether to filter
    if max_candidates < 0:
        return [] if len(candidates) > 1 else list_candidates_xyz(candidates)
    else:
        return get_best(candidates, max_candidates)


def get_best(candidates, max_candidates):
    """
    Gets the best candidates to become edge proposals given a dictionary
    containing candidates.

    Parameters
    ----------
    candidates : dict
        Dictionary that maps swc ids to a dictionary containing a distance and
        xyz coordinate.
    max_candidates : int
        Maximum number of proposals that can be generated from a leaf node.

    Returns
    -------
    list[tuple]
        xyz coordinates of candidates.

    """
    while len(candidates) > max_candidates:
        worst_key = None
        for key in candidates.keys():
            if worst_key is None:
                worst_key = key
            elif candidates[key]["dist"] > candidates[worst_key]["dist"]:
                worst_key = key
        del candidates[worst_key]
    return list_candidates_xyz(candidates)


def get_conection(neurograph, leaf, xyz):
    """
    Gets the node that proposal with leaf will connect to.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph containing "leaf".
    leaf : int
        Leaf node.
    xyz : numpy.ndarray
        xyz coordinate.

    Returns
    -------
    int
        Node id.

    """
    edge = neurograph.xyz_to_edge[xyz]
    node, d = get_closer_endpoint(neurograph, edge, xyz)
    if d > ENDPOINT_DIST:
        attrs = neurograph.get_edge_data(*edge)
        idx = np.where(np.all(attrs["xyz"] == xyz, axis=1))[0][0]
        node = neurograph.split_edge(edge, attrs, idx)
    return node


def get_closer_endpoint(neurograph, edge, xyz):
    """
    Gets the node from "edge" that is closer to "xyz".

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph containing "edge".
    edge : tuple
        Edge to be checked.
    xyz : numpy.ndarray
        xyz coordinate.

    Returns
    -------
    tuple
        Node id and its distance from "xyz".

    """
    i, j = tuple(edge)
    d_i = geometry.dist(neurograph.nodes[i]["xyz"], xyz)
    d_j = geometry.dist(neurograph.nodes[j]["xyz"], xyz)
    return (i, d_i) if d_i < d_j else (j, d_j)

            
# --- Trim Endpoints ---
def run_trimming(neurograph):
    # Check whether to trim
    if len(candidates) == 1 and neurograph.is_leaf(node):
        # Check whether node is isolated
        if len(neurograph.nodes[node]["proposals"]) != 1:
            pass
        else:
            candidates = get_candidates(neurograph, node, 1.5 * radius)
            if len(candidates) != 1:
                pass

        # Trim
        trim_bool = trim_endpoints(neurograph, leaf, node)
        if trim_bool:
            n_endpoints_trimmed += 1
            leafs.discard(node)
    print("# Endpoints Trimmed:", n_endpoints_trimmed)


# --- Trim Endpoints ---
def run_trimming(neurograph):
    # Check whether to trim
    if len(candidates) == 1 and neurograph.is_leaf(node):
        # Check whether node is isolated
        if len(neurograph.nodes[node]["proposals"]) != 1:
            pass
        else:
            candidates = get_candidates(neurograph, node, 1.5 * radius)
            if len(candidates) != 1:
                pass

        # Trim
        trim_bool = trim_endpoints(neurograph, leaf, node)
        if trim_bool:
            n_endpoints_trimmed += 1
            leafs.discard(node)
    print("# Endpoints Trimmed:", n_endpoints_trimmed)


def trim_endpoints(neurograph, i, j):
    # Initializations
    branch_i = neurograph.get_branch(i)
    branch_j = neurograph.get_branch(j)

    # Check both orderings
    idx_i, idx_j = trim_endpoints_ordered(branch_i, branch_j)
    idx_jj, idx_ii = trim_endpoints_ordered(branch_j, branch_i)
    d1 = geometry.dist(branch_i[idx_i], branch_j[idx_j])
    d2 = geometry.dist(branch_i[idx_ii], branch_j[idx_jj])

    # Update branches (if applicable)
    if min(d1, d2) + 2 < geometry.dist(branch_i[0], branch_j[0]):
        tangent_i = compute_tangent(branch_i, idx_i)
        tangent_j = compute_tangent(branch_j, idx_j)
        if np.dot(tangent_i, tangent_j) < -0.5:
            print("Location:", neurograph.nodes[i]["xyz"])
            print(geometry.dist(branch_i[0], branch_j[0]), "-->", min(d1, d2))
            print("")
            if d1 < d2:
                trim_to_idx(neurograph, i, idx_i)
                trim_to_idx(neurograph, j, idx_j)
            else:
                trim_to_idx(neurograph, i, idx_ii)
                trim_to_idx(neurograph, j, idx_jj)
            return True
    return False


def trim_endpoints_ordered(branch_1, branch_2):
    idx_1 = trim_endpoint(branch_1, branch_2)
    idx_2 = trim_endpoint(branch_2, branch_1[idx_1::])
    return idx_1, idx_2


def trim_endpoint(branch_1, branch_2):
    idx = 0
    path_length = 0
    best_dist = geometry.dist(branch_1[0], branch_2[0])
    best_idx = None
    trim_search_dist = TRIM_SEARCH_DIST
    while idx + 1 < len(branch_1):
        idx += 1
        path_length += geometry.dist(branch_1[idx - 1], branch_1[idx])
        if geometry.dist(branch_1[idx], branch_2[0]) < best_dist:
            best_idx = idx
            best_dist = geometry.dist(branch_1[idx], branch_2[0])

        # Determine whether to continue trimming
        if path_length > trim_search_dist:
            if best_idx is None:
                break
            elif idx - best_idx > 10:
                break
            else:
                trim_search_dist += 10
    return 0 if best_idx is None else best_idx


def trim_to_idx(neurograph, i, idx):
    """
    Trims the branch emanating from "i".

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph containing node "i"
    i : int
        Leaf node.
    idx : int
        Index used to truncate the branch emanating from "i".

    Returns
    -------
    None

    """
    if idx > 2:
        # Update node
        branch_xyz = neurograph.get_branch(i, key="xyz")
        branch_radii = neurograph.get_branch(i, key="radius")
        neurograph.nodes[i]["xyz"] = branch_xyz[idx]
        neurograph.nodes[i]["radius"] = branch_radii[idx]

        # Update edge
        j = neurograph.leaf_neighbor(i)
        neurograph.edges[i, j]["xyz"] = branch_xyz[idx::]
        neurograph.edges[i, j]["radius"] = branch_radii[idx::]
        for k in range(idx):
            del neurograph.xyz_to_edge[tuple(branch_xyz[k])]


# --- utils ---
def list_candidates_xyz(candidates):
    """
    Lists the xyz coordinates of candidates.

    Parameters
    ----------
    candidates : dict
        Dictionary that maps swc ids to a dictionary containing a distance and
        xyz coordinate.

    Returns
    -------
    list
        xyz coordinates of candidates.

    """
    return [candidates[key]["xyz"] for key in candidates.keys()]


def compute_tangent(branch, idx):
    """
    Computes tangent vector of "branch" after indexing from "idx".

    Parameters
    ----------
    branch : numpy.ndarray
        xyz coordinates that form a path.
    idx : int
        Index of a row in "branch".

    Returns
    -------
    numpy.ndarray
        Tangent vector of "branch".

    """
    end = min(idx + 10, len(branch))
    return geometry.compute_tangent(branch[idx:end])
