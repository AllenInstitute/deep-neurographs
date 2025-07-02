"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org


Code that generates edge proposals for a given fragments graph.

"""

from collections import deque

import numpy as np
from tqdm import tqdm

from deep_neurographs.utils import geometry_util as geometry

DOT_THRESHOLD = -0.3
SEARCH_SCALING_FACTOR = 1.5
TRIM_SEARCH_DIST = 15


def run(
    fragments_graph, search_radius, complex_bool=False, long_range_bool=True,
):
    """
    Generates proposals for fragments graph.

    Parameters
    ----------
    fragments_graph : FragmentsGraph
        Graph that proposals will be generated for.
    search_radius : float
        Maximum Euclidean distance between endpoints of proposal.
    complex_bool : bool, optional
        Indication of whether to generate complex proposals, meaning proposals
        between leaf and non-leaf nodes. The default is False.
    long_range_bool : bool, optional
        Indication of whether to generate simple proposals within distance of
        "LONG_RANGE_FACTOR" * radius of leaf from leaf without any proposals.
        The default is True.

    Returns
    -------
    None
    """
    # Initializations
    connections = dict()
    kdtree = init_kdtree(fragments_graph, complex_bool)
    if fragments_graph.verbose:
        iterable = tqdm(fragments_graph.get_leafs(), desc="Proposals")
    else:
        iterable = fragments_graph.get_leafs()

    # Main
    for leaf in iterable:
        # Generate potential proposals
        candidates = get_candidates(
            fragments_graph,
            leaf,
            kdtree,
            search_radius,
            fragments_graph.proposals_per_leaf,
            complex_bool,
        )

        # Generate long range proposals (if applicable)
        if len(candidates) == 0 and long_range_bool:
            candidates = get_candidates(
                fragments_graph,
                leaf,
                kdtree,
                search_radius * SEARCH_SCALING_FACTOR,
                -1,
                True,
            )

        # Determine which potential proposals to keep
        for i in candidates:
            leaf_swc_id = fragments_graph.nodes[leaf]["swc_id"]
            node_swc_id = fragments_graph.nodes[i]["swc_id"]
            pair_id = frozenset((leaf_swc_id, node_swc_id))
            if pair_id in connections.keys():
                cur_proposal = connections[pair_id]
                cur_dist = fragments_graph.proposal_length(cur_proposal)
                if fragments_graph.dist(leaf, i) < cur_dist:
                    fragments_graph.remove_proposal(cur_proposal)
                    del connections[pair_id]
                else:
                    continue

            # Add proposal
            fragments_graph.add_proposal(leaf, i)
            connections[pair_id] = frozenset({leaf, i})


def init_kdtree(fragments_graph, complex_bool):
    """
    Initializes a KD-Tree used to generate proposals.

    Parameters
    ----------
    graph : FragmentsGraph
        Graph that proposals will be generated for.
    complex_bool : bool
        Indication of whether to generate complex proposals.

    Returns
    -------
    scipy.spatial.cKDTree
        kdtree built from all xyz coordinates across edges in graph if
        complex_bool is True; otherwise, only built from leaf nodes.

    """
    if complex_bool:
        return fragments_graph.get_kdtree()
    else:
        return fragments_graph.get_kdtree(node_type="leaf")


def get_candidates(
    graph, leaf, kdtree, radius, max_proposals, complex_bool
):
    # Generate candidates
    candidates = list()
    for xyz in search_kdtree(graph, leaf, kdtree, radius, max_proposals):
        i = get_connecting_node(graph, leaf, xyz, radius, complex_bool)
        if i is not None:
            if graph.is_valid_proposal(leaf, i, complex_bool):
                candidates.append(i)

    # Process results
    if max_proposals < 0 and len(candidates) == 1:
        is_leaf = graph.degree[candidates[0]] == 1
        return candidates if is_leaf else []
    else:
        return list() if max_proposals < 0 else candidates


def search_kdtree(fragments_graph, leaf, kdtree, radius, max_proposals):
    """
    Generates proposals extending from node "leaf" by finding candidate xyz
    points on distinct connected components nearby.

    Parameters
    ----------
    fragments_graph : FragmentsGraph
        Graph that proposals will be generated for.
    kdtree : scipy.spatial.cKDTree
        ...
    leaf : int
        Leaf node that proposals are to be generated from.
    radius : float
        Maximum Euclidean distance between endpoints of proposal.
    max_proposals : int
        Maximum number of proposals that can be generated from a leaf node.

    Returns
    -------
    List[Tuple[float]]
        xyz coordinates of candidates.
    """
    # Generate candidates
    candidates = dict()
    leaf_xyz = fragments_graph.node_xyz[leaf]
    for xyz in geometry.query_ball(kdtree, leaf_xyz, radius):
        swc_id = fragments_graph.xyz_to_id(xyz)
        if swc_id != fragments_graph.nodes[leaf]["swc_id"]:
            d = geometry.dist(leaf_xyz, xyz)
            if swc_id not in candidates.keys():
                candidates[swc_id] = {"dist": d, "xyz": tuple(xyz)}
            elif geometry.dist(leaf_xyz, xyz) < candidates[swc_id]["dist"]:
                d = geometry.dist(leaf_xyz, xyz)
                candidates[swc_id] = {"dist": d, "xyz": tuple(xyz)}

    # Check whether to filter
    if max_proposals < 0:
        return [] if len(candidates) > 1 else list_candidates_xyz(candidates)
    else:
        return get_best(candidates, max_proposals)


def get_best(candidates, max_proposals):
    """
    Gets the best candidates to become edge proposals given a dictionary
    containing candidates.

    Parameters
    ----------
    candidates : dict
        Dictionary that maps swc ids to a dictionary containing a distance and
        xyz coordinate.
    max_proposals : int
        Maximum number of proposals that can be generated from a leaf node.

    Returns
    -------
    List[Tuple[float]]
        xyz coordinates of candidates.
    """
    while len(candidates) > max_proposals:
        worst_key = None
        for key in candidates.keys():
            if worst_key is None:
                worst_key = key
            elif candidates[key]["dist"] > candidates[worst_key]["dist"]:
                worst_key = key
        del candidates[worst_key]
    return list_candidates_xyz(candidates)


def get_connecting_node(fragments_graph, leaf, xyz, radius, complex_bool):
    """
    Gets node that proposal emanating from "leaf" will connect to.

    Parameters
    ----------
    fragments_graph : FragmentsGraph
        Graph containing "leaf".
    leaf : int
        Leaf node.
    xyz : numpy.ndarray
        xyz coordinate.

    Returns
    -------
    int
        Node id that proposal will connect to.
    """
    # Check if edge exists
    try:
        edge = fragments_graph.xyz_to_edge[xyz]
    except:
        return None

    # Find connecting node
    node = get_closer_endpoint(fragments_graph, edge, xyz)
    if fragments_graph.dist(leaf, node) < radius:
        return node
    elif complex_bool:
        attrs = fragments_graph.get_edge_data(*edge)
        idx = np.where(np.all(attrs["xyz"] == xyz, axis=1))[0][0]
        if type(idx) is int:
            return fragments_graph.split_edge(edge, attrs, idx)
    return None


def get_closer_endpoint(fragments_graph, edge, xyz):
    """
    Gets node from "edge" that is closer to "xyz".

    Parameters
    ----------
    fragments_graph : FragmentsGraph
        Graph containing "edge".
    edge : tuple
        Edge to be checked.
    xyz : numpy.ndarray
        xyz coordinate.

    Returns
    -------
    int
        Node closer to "xyz".
    """
    i, j = tuple(edge)
    d_i = geometry.dist(fragments_graph.node_xyz[i], xyz)
    d_j = geometry.dist(fragments_graph.node_xyz[j], xyz)
    return i if d_i < d_j else j


# --- Trim Endpoints ---
def run_endpoint_trimming(fragments_graph, search_radius):
    # Initializations
    augmented_search_radius = search_radius * SEARCH_SCALING_FACTOR
    long_range, in_range = deque(), deque()
    for p in fragments_graph.proposals:
        if fragments_graph.proposal_length(p) < augmented_search_radius:
            in_range.append(p)
        else:
            long_range.append(p)

    # Trim endpoints by proposal type
    trim_proposal_endpoints(fragments_graph, in_range, search_radius)
    trim_proposal_endpoints(
        fragments_graph, long_range, augmented_search_radius
    )


def trim_proposal_endpoints(fragments_graph, proposals, max_length):
    while proposals:
        p = proposals.pop()
        is_simple = fragments_graph.is_simple(p)
        is_single = fragments_graph.is_single_proposal(p)
        if is_simple and is_single:
            trim_endpoints_at_proposal(fragments_graph, p, max_length)
        elif fragments_graph.proposal_length(p) > max_length:
            fragments_graph.remove_proposal(p)


def trim_endpoints_at_proposal(fragments_graph, proposal, max_length):
    # Find closest points between proposal branches
    i, j = tuple(proposal)
    pts_i = fragments_graph.edge_attr(i, key="xyz", ignore=True)[0]
    pts_j = fragments_graph.edge_attr(j, key="xyz", ignore=True)[0]
    dist_ij, (idx_i, idx_j) = find_closest_pair(pts_i, pts_j)

    # Update branches (if applicable)
    if dist_ij > max_length:
        fragments_graph.remove_proposal(frozenset((i, j)))
    elif dist_ij + 2 < geometry.dist(pts_i[0], pts_j[0]):
        if compute_dot(pts_i, pts_j, idx_i, idx_j) < DOT_THRESHOLD:
            trim_to_idx(fragments_graph, i, idx_i)
            trim_to_idx(fragments_graph, j, idx_j)

def find_closest_pair(pts1, pts2):
    best_dist, best_idxs = np.inf, (0, 0)
    i, length1 = -1, 0
    while length1 < TRIM_SEARCH_DIST and i < len(pts1) - 1:
        i += 1
        length1 += geometry.dist(pts1[i], pts1[i - 1]) if i > 0 else 0

        # Search other branch
        j, length2 = -1, 0
        while length2 < TRIM_SEARCH_DIST and j < len(pts2) - 1:
            j += 1
            length2 += geometry.dist(pts2[j], pts2[j - 1]) if j > 0 else 0

            # Check distance between points
            dist = geometry.dist(pts1[i], pts2[j])
            if dist < best_dist:
                best_dist = dist
                best_idxs = (i, j)
    return best_dist, best_idxs


def trim_to_idx(fragments_graph, i, idx):
    """
    Trims the end of a branch specified by the leaf node "i".

    Parameters
    ----------
    fragments_graph : FragmentsGraph
        Graph containing node "i"
    i : int
        Leaf node ID.
    idx : int
        Branch is trimmed to the index "idx".

    Returns
    -------
    None
    """
    # Update node
    edge_xyz = fragments_graph.edge_attr(i, key="xyz", ignore=True)[0]
    edge_radii = fragments_graph.edge_attr(i, key="radius", ignore=True)[0]
    fragments_graph.node_xyz[i] = edge_xyz[idx]
    fragments_graph.node_radius[i] = edge_radii[idx]

    # Update edge
    nb = list(fragments_graph.neighbors(i))[0]
    fragments_graph.edges[i, nb]["xyz"] = edge_xyz[idx:]
    fragments_graph.edges[i, nb]["radius"] = edge_radii[idx:]
    for k in range(idx):
        try:
            del fragments_graph.xyz_to_edge[tuple(edge_xyz[k])]
        except KeyError:
            pass


# --- Helpers ---
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
    List[Tuple[int]]
        xyz coordinates of candidates.
    """
    return [candidates[key]["xyz"] for key in candidates.keys()]


def compute_dot(branch1, branch2, idx1, idx2):
    """
    Computes dot product between principal components of "branch1" and
    "branch_2".

    Parameters
    ----------
    branch1 : numpy.ndarray
        xyz coordinates of some branch from a graph.
    branch_2 : numpy.ndarray
        xyz coordinates of some branch from a graph.
    idx1 : int
        Index that "branch1" would be trimmed to (i.e. xyz coordinates from 0
        to "idx1" would be deleted from "branch1").
    idx2 : int
        Index that "branch_2" would be trimmed to (i.e. xyz coordinates from 0
        to "idx2" would be deleted from "branch_2").

    Returns
    -------
    float
        Dot product between principal components of "branch1" and "branch_2".
    """
    # Initializations
    midpoint = geometry.midpoint(branch1[idx1], branch2[idx2])
    b1 = branch1 - midpoint 
    b2 = branch2 - midpoint

    # Main
    dot5 = np.dot(tangent(b1, idx1, 5), tangent(b2, idx2, 5))
    dot10 = np.dot(tangent(b1, idx1, 10), tangent(b2, idx2, 10))
    dot20 = np.dot(tangent(b1, idx1, 20), tangent(b2, idx2, 20))
    return min(dot5, min(dot10, dot20))


def tangent(branch, idx, depth):
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
    end = min(idx + depth, len(branch))
    return geometry.tangent(branch[idx:end])
