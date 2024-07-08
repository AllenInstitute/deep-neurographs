"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Module used to generate edge proposals.

"""

import numpy as np

from deep_neurographs import geometry

TRIM_SEARCH_DIST = 10
ENDPOINT_DIST = 10


def run(
    neurograph, search_radius, complex_bool=True, long_range_proposals=False
):
    """
    Generates proposals emanating from "leaf".

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that proposals will be generated for.
    search_radius : float
        Maximum Euclidean distance between endpoints of proposal.
    complex_bool : bool, optional
        Indication of whether to generate complex proposals. The default is
        True.
    long_range_proposals : bool
        Indication of whether to generate simple proposals within distance of
        2 * search_radius of leaf.

    Returns
    -------
    None

    """
    connections = dict()
    leafs = set(neurograph.leafs)
    n_endpoints_trimmed = 0
    while len(leafs) > 0:
        # Get candidates
        leaf = leafs.pop()
        leaf_swc_id = neurograph.nodes[leaf]["swc_id"]
        candidates = get_candidates(neurograph, leaf, search_radius)
        if len(candidates) == 0 and long_range_proposals:
            candidates = get_candidates(neurograph, leaf, 2 * search_radius)
            candidates = parse_long_range(neurograph, candidates, leaf)

        # Parse candidates
        for xyz in candidates:
            # Get connection
            neurograph, node = get_conection(neurograph, leaf, xyz)
            if not complex_bool and neurograph.degree[node] > 1:
                continue

            # Check for somas
            swc_id = neurograph.nodes[node]["swc_id"]
            soma_bool_1 = swc_id in neurograph.soma_ids.keys()
            soma_bool_2 = leaf_swc_id in neurograph.soma_ids.keys()
            if soma_bool_1 and soma_bool_2:
                continue

            # Check whether already connection exists
            pair_id = frozenset((leaf_swc_id, swc_id))
            if pair_id in connections.keys():
                proposal = connections[pair_id]
                dist_1 = neurograph.dist(leaf, node)
                dist_2 = neurograph.proposal_length(proposal)
                if dist_1 < dist_2:
                    i, j = tuple(proposal)
                    neurograph.nodes[i]["proposals"].remove(j)
                    neurograph.nodes[j]["proposals"].remove(i)
                    del neurograph.proposals[proposal]
                    del connections[pair_id]
                else:
                    continue

            # Add proposal
            neurograph.add_proposal(leaf, node)
            connections[pair_id] = frozenset({leaf, node})

        # Check whether to trim
        if len(candidates) == 1 and neurograph.is_leaf(node):
            # Check whether node is isolated
            if len(neurograph.nodes[node]["proposals"]) != 1:
                continue
            else:
                candidates = get_candidates(neurograph, node, 1.5 * search_radius)
                if len(candidates) != 1:
                    continue

            # Trim
            trim_bool = trim_endpoints(neurograph, leaf, node)
            if trim_bool:
                n_endpoints_trimmed += 1
                leafs.discard(node)
    print("# Endpoints Trimmed:", n_endpoints_trimmed)


def parse_long_range(neurograph, candidates, leaf):
    hit_swc_ids = set()
    filtered_candidates = []
    for xyz in candidates:
        neurograph, i = get_conection(neurograph, leaf, xyz)
        if neurograph.degree[i] > 1:
            continue
        else:
            filtered_candidates.append(xyz)
            hit_swc_ids.add(neurograph.nodes[i]["swc_id"])
    return filtered_candidates if len(hit_swc_ids) == 1 else []


def get_candidates(neurograph, leaf, search_radius):
    """
    Generates proposals for node "leaf" in "neurograph" by finding candidate
    xyz points on distinct connected components nearby.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph built from swc files.
    leaf : int
        Leaf node that proposals are to be generated from.
    search_radius : float
        Maximum Euclidean distance between endpoints of proposal.

    Returns
    -------
    list
        Proposals generated from "leaf".

    """
    candidates = dict()
    dists = dict()
    leaf_xyz = neurograph.nodes[leaf]["xyz"]
    for xyz in neurograph.query_kdtree(leaf_xyz, search_radius):
        try:
            swc_id = neurograph.xyz_to_swc(xyz)
            if swc_id != neurograph.nodes[leaf]["swc_id"]:
                if swc_id not in candidates.keys():
                    candidates[swc_id] = tuple(xyz)
                    dists[swc_id] = geometry.dist(leaf_xyz, xyz)
                elif geometry.dist(leaf_xyz, xyz) < dists[swc_id]:
                    candidates[swc_id] = tuple(xyz)
                    dists[swc_id] = geometry.dist(leaf_xyz, xyz)
        except:
            pass
    return get_best_candidates(neurograph, candidates, dists)


def get_best_candidates(neurograph, candidates, dists):
    if len(candidates) > neurograph.proposals_per_leaf:
        worst = None
        for key, d in dists.items():
            if worst is None:
                worst = key
            elif dists[key] > dists[worst]:
                worst = key
        del candidates[worst]
        del dists[worst]
        return get_best_candidates(neurograph, candidates, dists)
    else:
        return list(candidates.values())


def get_conection(neurograph, leaf, xyz):
    edge = neurograph.xyz_to_edge[xyz]
    node, d = get_closer_endpoint(neurograph, edge, xyz)
    if d > ENDPOINT_DIST:
        attrs = neurograph.get_edge_data(*edge)
        idx = np.where(np.all(attrs["xyz"] == xyz, axis=1))[0][0]
        node = neurograph.split_edge(edge, attrs, idx)
    return neurograph, node


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
