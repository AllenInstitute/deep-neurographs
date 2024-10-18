"""
Created on Sat July 15 9:00:00 2023

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Module used to generate edge proposals.

"""

from copy import deepcopy

import numpy as np
from tqdm import tqdm

from deep_neurographs import geometry

DOT_THRESHOLD = -0.3
RADIUS_SCALING_FACTOR = 1.5
TRIM_SEARCH_DIST = 15


def run(
    graph,
    radius,
    complex_bool=False,
    long_range_bool=True,
    progress_bar=True,
    trim_endpoints_bool=True,
):
    """
    Generates proposals emanating from "leaf".

    Parameters
    ----------
    graph : FragmentsGraph
        Graph that proposals will be generated for.
    radius : float
        Maximum Euclidean distance between endpoints of proposal.
    complex_bool : bool, optional
        Indication of whether to generate complex proposals, meaning proposals
        between leaf and non-leaf nodes. The default is False.
    long_range_bool : bool, optional
        Indication of whether to generate simple proposals within distance of
        "LONG_RANGE_FACTOR" * radius of leaf from leaf without any proposals.
        The default is False.
    progress_bar : bool, optional
        Indication of whether to print out a progress bar while generating
        proposals. The default is True.
    trim_endpoints_bool : bool, optional
        Indication of whether to endpoints of branches with exactly one
        proposal. The default is True.

    Returns
    -------
    None

    """
    # Initializations
    connections = dict()
    kdtree = init_kdtree(graph, complex_bool)
    radius *= RADIUS_SCALING_FACTOR if trim_endpoints_bool else 1.0
    if progress_bar:
        iterable = tqdm(graph.get_leafs(), desc="Proposals")
    else:
        iterable = graph.get_leafs()

    # Main
    for leaf in iterable:
        # Generate potential proposals
        candidates = get_candidates(
            graph,
            leaf,
            kdtree,
            radius,
            graph.proposals_per_leaf,
            complex_bool,
        )

        # Generate long range proposals (if applicable)
        if len(candidates) == 0 and long_range_bool:
            candidates = get_candidates(
                graph,
                leaf,
                kdtree,
                radius * RADIUS_SCALING_FACTOR,
                -1,
                True,
            )

        # Determine which potential proposals to keep
        for i in candidates:
            leaf_swc_id = graph.nodes[leaf]["swc_id"]
            pair_id = frozenset((leaf_swc_id, graph.nodes[i]["swc_id"]))
            if pair_id in connections.keys():
                cur_proposal = connections[pair_id]
                cur_dist = graph.proposal_length(cur_proposal)
                if graph.dist(leaf, i) < cur_dist:
                    graph.remove_proposal(cur_proposal)
                    del connections[pair_id]
                else:
                    continue

            # Add proposal
            graph.add_proposal(leaf, i)
            connections[pair_id] = frozenset({leaf, i})

    # Trim endpoints (if applicable)
    n_trimmed = 0
    if trim_endpoints_bool:
        radius /= RADIUS_SCALING_FACTOR
        long_range, in_range = separate_proposals(graph, radius)
        graph, n_trimmed_1 = run_trimming(graph, long_range, radius)
        graph, n_trimmed_2 = run_trimming(graph, in_range, radius)
        n_trimmed = n_trimmed_1 + n_trimmed_2
    return n_trimmed


def init_kdtree(graph, complex_bool):
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
        kdtree.

    """
    if complex_bool:
        return graph.get_kdtree()
    else:
        return graph.get_kdtree(node_type="leaf")


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

    # Process the results
    if max_proposals < 0 and len(candidates) == 1:
        return candidates if graph.is_leaf(candidates[0]) else []
    else:
        return list() if max_proposals < 0 else candidates


def search_kdtree(graph, leaf, kdtree, radius, max_proposals):
    """
    Generates proposals for node "leaf" in "graph" by finding candidate
    xyz points on distinct connected components nearby.

    Parameters
    ----------
    graph : FragmentsGraph
        Graph built from swc files.
    kdtree : ...
        ...
    leaf : int
        Leaf node that proposals are to be generated from.
    radius : float
        Maximum Euclidean distance between endpoints of proposal.
    max_proposals : int
        Maximum number of proposals that can be generated from a leaf node.

    Returns
    -------
    list[tuple]
        xyz coordinates of candidates.

    """
    # Generate candidates
    candidates = dict()
    leaf_xyz = graph.nodes[leaf]["xyz"]
    for xyz in geometry.query_ball(kdtree, leaf_xyz, radius):
        swc_id = graph.xyz_to_swc(xyz)
        if swc_id != graph.nodes[leaf]["swc_id"]:
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
    list[tuple]
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


def get_connecting_node(graph, leaf, xyz, radius, complex_bool):
    """
    Gets the node that proposal with leaf will connect to.

    Parameters
    ----------
    graph : FragmentsGraph
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
    edge = graph.xyz_to_edge[xyz]
    node = get_closer_endpoint(graph, edge, xyz)
    if graph.dist(leaf, node) < radius:
        return node
    elif complex_bool:
        attrs = graph.get_edge_data(*edge)
        idx = np.where(np.all(attrs["xyz"] == xyz, axis=1))[0][0]
        if type(idx) is int:
            return graph.split_edge(edge, attrs, idx)
    return None


def get_closer_endpoint(graph, edge, xyz):
    """
    Gets the node from "edge" that is closer to "xyz".

    Parameters
    ----------
    graph : FragmentsGraph
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
    d_i = geometry.dist(graph.nodes[i]["xyz"], xyz)
    d_j = geometry.dist(graph.nodes[j]["xyz"], xyz)
    return i if d_i < d_j else j


def separate_proposals(graph, radius):
    long_range_proposals = list()
    proposals = list()
    for proposal in graph.proposals:
        i, j = tuple(proposal)
        if graph.dist(i, j) > radius:
            long_range_proposals.append(proposal)
        else:
            proposals.append(proposal)
    return long_range_proposals, proposals


# --- Trim Endpoints ---
def run_trimming(graph, proposals, radius):
    n_endpoints_trimmed = 0
    long_radius = radius * RADIUS_SCALING_FACTOR
    for proposal in deepcopy(proposals):
        i, j = tuple(proposal)
        is_simple = graph.is_simple(proposal)
        is_single = graph.is_single_proposal(proposal)
        trim_bool = False
        if is_simple and is_single:
            graph, trim_bool = trim_endpoints(
                graph, i, j, long_radius
            )
        elif graph.dist(i, j) > radius:
            graph.remove_proposal(proposal)
        n_endpoints_trimmed += 1 if trim_bool else 0
    return graph, n_endpoints_trimmed


def trim_endpoints(graph, i, j, radius):
    # Initializations
    branch_i = graph.branch(i)
    branch_j = graph.branch(j)

    # Check both orderings
    idx_i, idx_j = trim_endpoints_ordered(branch_i, branch_j)
    idx_jj, idx_ii = trim_endpoints_ordered(branch_j, branch_i)
    d1 = geometry.dist(branch_i[idx_i], branch_j[idx_j])
    d2 = geometry.dist(branch_i[idx_ii], branch_j[idx_jj])
    if d2 < d1:
        idx_i = idx_ii
        idx_j = idx_jj

    # Update branches (if applicable)
    if min(d1, d2) > radius:
        graph.remove_proposal(frozenset((i, j)))
        return graph, False
    elif min(d1, d2) + 2 < geometry.dist(branch_i[0], branch_j[0]):
        if compute_dot(branch_i, branch_j, idx_i, idx_j) < DOT_THRESHOLD:
            graph = trim_to_idx(graph, i, idx_i)
            graph = trim_to_idx(graph, j, idx_j)
            return graph, True
    return graph, False


def trim_endpoints_ordered(branch_1, branch_2):
    idx_1 = trim_endpoint(branch_1, branch_2)
    idx_2 = trim_endpoint(branch_2, branch_1[idx_1::])
    return idx_1, idx_2


def trim_endpoint(branch_1, branch_2):
    idx = 0
    path_length = 0
    best_dist = geometry.dist(branch_1[0], branch_2[0])
    best_idx = None
    best_upd = False
    while idx + 1 < len(branch_1):
        idx += 1
        path_length += geometry.dist(branch_1[idx - 1], branch_1[idx])
        if geometry.dist(branch_1[idx], branch_2[0]) < best_dist:
            best_idx = idx
            best_dist = geometry.dist(branch_1[idx], branch_2[0])
            best_upd = True

        # Determine whether to continue trimming
        if path_length > TRIM_SEARCH_DIST:
            if best_idx is None:
                break
            elif not best_upd:
                break
            else:
                best_upd = False
                path_length = 0
    return 0 if best_idx is None else best_idx


def trim_to_idx(graph, i, idx):
    """
    Trims the branch emanating from "i".

    Parameters
    ----------
    graph : FragmentsGraph
        Graph containing node "i"
    i : int
        Leaf node.
    idx : int
        Index used to truncate the branch emanating from "i".

    Returns
    -------
    None

    """
    # Update node
    branch_xyz = graph.branch(i, key="xyz")
    branch_radii = graph.branch(i, key="radius")
    graph.nodes[i]["xyz"] = branch_xyz[idx]
    graph.nodes[i]["radius"] = branch_radii[idx]

    # Update edge
    j = graph.leaf_neighbor(i)
    graph.edges[i, j]["xyz"] = branch_xyz[idx::]
    graph.edges[i, j]["radius"] = branch_radii[idx::]
    for k in range(idx):
        try:
            del graph.xyz_to_edge[tuple(branch_xyz[k])]
        except KeyError:
            pass
    return graph


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


def compute_dot(branch_1, branch_2, idx_1, idx_2):
    """
    Computes dot product between principal components of "branch_1" and
    "branch_2".

    Parameters
    ----------
    branch_1 : numpy.ndarray
        xyz coordinates of some branch from a graph.
    branch_2 : numpy.ndarray
        xyz coordinates of some branch from a graph.
    idx_1 : int
        Index that "branch_1" would be trimmed to (i.e. xyz coordinates from 0
        to "idx_1" would be deleted from "branch_1").
    idx_2 : int
        Index that "branch_2" would be trimmed to (i.e. xyz coordinates from 0
        to "idx_2" would be deleted from "branch_2").

    Returns
    -------
    float
        Dot product between principal components of "branch_1" and "branch_2".

    """
    # Initializations
    b1 = branch_1 - geometry.midpoint(branch_1[idx_1], branch_2[idx_2])
    b2 = branch_2 - geometry.midpoint(branch_1[idx_1], branch_2[idx_2])

    # Main
    dot_5 = np.dot(tangent(b1, idx_1, 5), tangent(b2, idx_2, 5))
    dot_10 = np.dot(tangent(b1, idx_1, 10), tangent(b2, idx_2, 10))
    dot_20 = np.dot(tangent(b1, idx_1, 20), tangent(b2, idx_2, 20))
    return min(dot_5, min(dot_10, dot_20))


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
