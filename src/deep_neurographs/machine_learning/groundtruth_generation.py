"""
Created on Sat March 2 16:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Generates ground truth for edge proposals (i.e. determines whether a proposal
should be accepted or rejected).

"""

import networkx as nx
import numpy as np

from deep_neurographs import geometry
from deep_neurographs import graph_utils as gutils
from deep_neurographs import utils
from deep_neurographs.geometry import dist as get_dist

ALIGNED_THRESHOLD = 3.5
MIN_INTERSECTION = 10


def init_targets(pred_graph, target_graph, strict=True):
    """
    Initializes ground truth for edge proposals.

    Parameters
    ----------
    target_graph : NeuroGraph
        Graph built from ground truth swc files.
    pred_graph : NeuroGraph
        Graph build from predicted swc files.
    strict : bool, optional
        Indication if whether target edges should be determined by using
        stricter criteria that checks if proposals are reasonably well
        aligned. The default is True.

    Returns
    -------
    target_edges : set
        Edge proposals that machine learning model learns to accept.

    """
    # Initializations
    valid_proposals = get_valid_proposals(target_graph, pred_graph)
    lengths = [pred_graph.proposal_length(e) for e in valid_proposals]

    # Add best simple edges
    target_edges = set()
    graph = pred_graph.copy_graph()
    for i in np.argsort(lengths):
        edge = valid_proposals[i]
        created_cycle, _ = gutils.creates_cycle(graph, tuple(edge))
        if not created_cycle:
            graph.add_edges_from([edge])
            target_edges.add(edge)
    return target_edges


def get_valid_proposals(target_graph, pred_graph):
    # Initializations
    valid_proposals = list()
    kdtree = target_graph.get_kdtree()
    invalid_ids, node_to_target = unaligned_components(
        target_graph, pred_graph, kdtree
    )

    # Check whether aligned to same/adjacent target edges (i.e. valid)
    for edge in pred_graph.proposals:
        # Filter invalid and proposals btw different components
        i, j = tuple(edge)
        invalid_i = pred_graph.nodes[i]["swc_id"] in invalid_ids
        invalid_j = pred_graph.nodes[j]["swc_id"] in invalid_ids
        if invalid_i or invalid_j:
            continue
        elif node_to_target[i] != node_to_target[j]:
            continue

        # Check whether proposal is valid
        target_id = node_to_target[i]
        if is_valid(target_graph, pred_graph, kdtree, target_id, edge):
            valid_proposals.append(edge)
    return valid_proposals


def unaligned_components(target_graph, pred_graph, kdtree):
    """
    Detects connected components in "pred_graph" that are unaligned to a
    connected component in "target_graph".

    Parameters
    ----------
    target_graph : NeuroGraph
        Graph built from ground truth swc files.
    pred_graph : NeuroGraph
        Graph build from predicted swc files.

    Returns
    -------
    invalid_ids : set
        IDs in ""pred_graph" that correspond to connected components that
        are unaligned to a connected component in "target_graph".
    node_to_target : dict
        Mapping between nodes and target ids.

    """
    invalid_ids = set()
    node_to_target = dict()
    for component in nx.connected_components(pred_graph):
        aligned, target_id = is_component_aligned(
            target_graph, pred_graph, component, kdtree
        )
        if not aligned:
            i = utils.sample_singleton(component)
            invalid_ids.add(pred_graph.nodes[i]["swc_id"])
        else:
            node_to_target = upd_dict(node_to_target, component, target_id)
    return invalid_ids, node_to_target


def is_component_aligned(target_graph, pred_graph, component, kdtree):
    """
    Determines whether the connected component defined by "node_subset" is
    close to a component in "target_graph". This routine iterates over
    "node_subset" and projects each node onto "target_graph", then
    computes the projection distance. If (on average) each node in
    "node_subset" is less 3.5 microns from a component in the ground truth,
    then "node_subset" is aligned.

    Parameters
    ----------
    target_graph : NeuroGraph
        Graph built from ground truth swc files.
    pred_graph : NeuroGraph
        Graph build from predicted swc files.
    component : ...
        ...

    Returns
    -------
    bool
        Indication of whether "component" is aligned to a connected
        component in "target_graph".

    """
    # Compute distances
    dists = dict()
    for edge in pred_graph.subgraph(component).edges:
        for xyz in pred_graph.edges[edge]["xyz"]:
            hat_xyz = geometry.kdtree_query(kdtree, xyz)
            hat_swc_id = target_graph.xyz_to_swc(hat_xyz)
            d = get_dist(hat_xyz, xyz)
            dists = utils.append_dict_value(dists, hat_swc_id, d)

    # Deterine whether aligned
    hat_swc_id = utils.find_best(dists)
    dists = np.array(dists[hat_swc_id])
    intersects = True if len(dists) > MIN_INTERSECTION else False
    aligned_score = np.mean(dists[dists < np.percentile(dists, 85)])
    if (aligned_score < ALIGNED_THRESHOLD and hat_swc_id) and intersects:
        return True, hat_swc_id
    else:
        return False, None


def is_valid(target_graph, pred_graph, kdtree, target_id, edge):
    """
    Determines whether the proposal connects two branches that correspond to
    either the same or adjacent branches on the ground truth. If either
    condition holds, then the proposal is said to be consistent.

    Parameters
    ----------
    target_graph : NeuroGraph
        Graph built from ground truth swc files.
    pred_graph : NeuroGraph
        Graph build from predicted swc files.
    target_id : str
        swc id of target that the proposal "edge" corresponds to.
    edge : frozenset
        Edge proposal to be checked.

    Returns
    -------
    bool
        Indication of whether proposal is consistent.

    """
    # Find closest edges from target_graph
    i, j = tuple(edge)
    hat_edge_i = proj_branch(target_graph, pred_graph, kdtree, target_id, i)
    hat_edge_j = proj_branch(target_graph, pred_graph, kdtree, target_id, j)

    # Check if edges are identical or None
    if not hat_edge_i or not hat_edge_j:
        return False
    elif hat_edge_i == hat_edge_j:
        return True

    # Check if edges are adjacent
    if is_adjacent(target_graph, hat_edge_i, hat_edge_j):
        hat_branch_i = target_graph.edges[hat_edge_i]["xyz"]
        hat_branch_j = target_graph.edges[hat_edge_j]["xyz"]
        xyz_i = pred_graph.nodes[i]["xyz"]
        xyz_j = pred_graph.nodes[j]["xyz"]
        if is_adjacent_aligned(hat_branch_i, hat_branch_j, xyz_i, xyz_j):
            return True
    else:
        return False


def proj_branch(target_graph, pred_graph, kdtree, target_id, i):
    # Compute projections
    hits = dict()
    for branch in pred_graph.get_branches(i):
        for xyz in branch:
            hat_xyz = geometry.kdtree_query(kdtree, xyz)
            swc_id = target_graph.xyz_to_swc(hat_xyz)
            if swc_id == target_id:
                hat_edge = target_graph.xyz_to_edge[hat_xyz]
                hits = utils.append_dict_value(hits, hat_edge, hat_xyz)

    # Determine closest edge
    min_dist = np.inf
    best_edge = None
    xyz_i = pred_graph.nodes[i]["xyz"]
    if len(hits.keys()) > 1:
        swc_id = pred_graph.nodes[i]["swc_id"]
        for edge in hits.keys():
            nb, d = geometry.nearest_neighbor(hits[edge], xyz_i)
            if d < min_dist:
                min_dist = d
                best_edge = edge
    elif len(hits.keys()) == 1:
        best_edge = list(hits.keys())[0]
    return best_edge


def is_adjacent(neurograph, edge_i, edge_j):
    """
    Determines whether "edge_i" and "edge_j" are adjacent, meaning there
    exists a nodes in "edge_i" and "edge_j" which are neighbors.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that "edge_i" and "edge_j" are contained in.
    edge_i : tuple
        Edge to be checked.
    edge_j : tuple
        Edge to be checked.

    Returns
    -------
    bool
        Indication of whether "edge_i" and "edge_j" are adjacent.

    """
    for i in edge_i:
        for j in edge_j:
            if i == j:
                return True
    return False


def is_adjacent_aligned(hat_branch_i, hat_branch_j, xyz_i, xyz_j):
    hat_branch_i, hat_branch_j = orient_branch(hat_branch_i, hat_branch_j)
    hat_i, _ = geometry.nearest_neighbor(hat_branch_i, xyz_i)
    hat_j, _ = geometry.nearest_neighbor(hat_branch_j, xyz_j)
    hat_path_dist = hat_i + hat_j
    path_dist = geometry.dist(xyz_i, xyz_j)
    return True if 2 * path_dist / (path_dist + hat_path_dist) > 0.5 else False


# -- utils --
def upd_dict_cnts(my_dict, key):
    if key in my_dict.keys():
        my_dict[key] += 1
    else:
        my_dict[key] = 1
    return my_dict


def orient_branch(branch_i, branch_j):
    """
    Flips branches so that "all(branch_i[0] == branch_j[0])" is True.

    Parameters
    ----------
    branch_i : numpy.ndarray
        Array containing xyz coordinates corresponding to some edge in a
        Neurograph.
    branch_j : numpy.ndarray
        Array containing xyz coordinates corresponding to some edge in a
        Neurograph.

    Returns
    -------
    branch_i : numpy.ndarray
        xyz coordinates corresponding to some edge in a Neurograph.
    branch_j : numpy.ndarray
        xyz coordinates corresponding to some edge in a Neurograph.

    """
    # Orient branches
    if all(branch_i[-1] == branch_j[0]):
        branch_i = np.flip(branch_i, axis=0)
    elif all(branch_i[0] == branch_j[-1]):
        branch_j = np.flip(branch_j, axis=0)
    elif all(branch_i[-1] == branch_j[-1]):
        branch_i = np.flip(branch_i, axis=0)
        branch_j = np.flip(branch_j, axis=0)
    return branch_i, branch_j


def upd_dict(node_to_target_id, nodes, target_id):
    for node in nodes:
        node_to_target_id[node] = target_id
    return node_to_target_id
