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


def init_targets(target_neurograph, pred_neurograph):
    # Initializations
    target_edges = set()
    valid_proposals = get_valid_proposals(target_neurograph, pred_neurograph)

    # Add best simple edges
    dists = [pred_neurograph.proposal_length(p) for p in valid_proposals]
    groundtruth_graph = pred_neurograph.copy_graph()
    for idx in np.argsort(dists):
        edge = valid_proposals[idx]
        if not gutils.creates_cycle(groundtruth_graph, tuple(edge)):
            groundtruth_graph.add_edges_from([edge])
            target_edges.add(edge)
    return target_edges


def get_valid_proposals(target_neurograph, pred_neurograph):
    # Detect components unaligned to ground truth
    invalid_proposals = set()
    node_to_target = dict()
    for component in nx.connected_components(pred_neurograph):
        aligned, target_id = is_component_aligned(
            target_neurograph, pred_neurograph, component
        )
        if not aligned:
            i = utils.sample_singleton(component)
            invalid_proposals.add(pred_neurograph.nodes[i]["swc_id"])
        else:
            node_to_target = upd_dict(node_to_target, component, target_id)

    # Check whether aligned to same/adjacent target edges (i.e. valid)
    valid_proposals = list()
    for edge in pred_neurograph.proposals:
        # Filter invalid and proposals btw different components
        i, j = tuple(edge)
        invalid_i = pred_neurograph.nodes[i]["swc_id"] in invalid_proposals
        invalid_j = pred_neurograph.nodes[j]["swc_id"] in invalid_proposals
        if invalid_i or invalid_j:
            continue
        elif node_to_target[i] != node_to_target[j]:
            continue

        # Check whether proposal is valid
        target_id = node_to_target[i]
        if is_valid(target_neurograph, pred_neurograph, target_id, edge):
            valid_proposals.append(edge)
    return valid_proposals


def is_component_aligned(target_neurograph, pred_neurograph, component):
    """
    Determines whether the connected component defined by "node_subset" is
    close to a component in "target_neurograph". This routine iterates over
    "node_subset" and projects each node onto "target_neurograph", then
    computes the projection distance. If (on average) each node in
    "node_subset" is less 3.5 microns from a component in the ground truth,
    then "node_subset" is aligned.

    Parameters
    ----------
    target_neurograph : NeuroGraph
        Graph built from ground truth swc files.
    pred_neurograph : NeuroGraph
        Graph build from predicted swc files.
    component : ...
        ...

    Returns
    -------
    bool
        Indication of whether "component" is aligned to a connected
        component in "target_neurograph".

    """
    # Compute distances
    dists = dict()
    for edge in pred_neurograph.subgraph(component).edges:
        for xyz in pred_neurograph.edges[edge]["xyz"]:
            hat_xyz = target_neurograph.get_projection(tuple(xyz))
            hat_swc_id = target_neurograph.xyz_to_swc(hat_xyz)
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


def is_valid(target_neurograph, pred_neurograph, target_id, edge):
    """
    Determines whether a proposal is valid, meaning it must be consistent and
    aligned.

    Parameters
    ----------
    target_neurograph : NeuroGraph
        Graph built from ground truth swc files.
    pred_neurograph : NeuroGraph
        Graph build from predicted swc files.
    target_id : str
        swc id of target that the proposal "edge" corresponds to.
    edge : frozenset
        Edge proposal to be checked.

    Returns
    -------
    bool
        Indication of whether proposal is consistent
    """
    #aligned = is_proposal_aligned(target_neurograph, pred_neurograph, edge)
    consistent = is_consistent(
        target_neurograph, pred_neurograph, target_id, edge
        )
    return True if consistent else False


def is_consistent(target_neurograph, pred_neurograph, target_id, edge):
    """
    Determines whether the proposal connects two branches that correspond to
    either the same or adjacent branches on the ground truth. If either
    condition holds, then the proposal is said to be consistent.

    Parameters
    ----------
    target_neurograph : NeuroGraph
        Graph built from ground truth swc files.
    pred_neurograph : NeuroGraph
        Graph build from predicted swc files.
    target_id : str
        swc id of target that the proposal "edge" corresponds to.
    edge : frozenset
        Edge proposal to be checked.

    Returns
    -------
    bool
        Indication of whether proposal is consistent

    """
    # Find closest edges from target_neurograph
    i, j = tuple(edge)
    hat_edge_i = proj_branch(target_neurograph, pred_neurograph, target_id, i)
    hat_edge_j = proj_branch(target_neurograph, pred_neurograph, target_id, j)

    # Check if edges are identical or None
    if not hat_edge_i or not hat_edge_j:
        return False
    elif hat_edge_i == hat_edge_j:
        return True

    # Check if edges are adjacent
    if is_adjacent(target_neurograph, hat_edge_i, hat_edge_j):
        hat_branch_i = target_neurograph.edges[hat_edge_i]["xyz"]
        hat_branch_j = target_neurograph.edges[hat_edge_j]["xyz"]
        xyz_i = pred_neurograph.nodes[i]["xyz"]
        xyz_j = pred_neurograph.nodes[j]["xyz"]
        if is_adjacent_aligned(hat_branch_i, hat_branch_j, xyz_i, xyz_j):
            return True
    return False


def is_proposal_aligned(target_neurograph, pred_neurograph, edge):
    xyz_0, xyz_1 = pred_neurograph.proposal_xyz(edge)
    proj_dists = []
    for xyz in geometry.make_line(xyz_0, xyz_1, 10):
        hat_xyz = target_neurograph.get_projection(tuple(xyz))
        proj_dists.append(get_dist(hat_xyz, xyz))
    return True if np.mean(proj_dists) < ALIGNED_THRESHOLD else False


def proj_branch(target_neurograph, pred_neurograph, target_id, i):
    # Compute projections
    hits = dict()
    for branch in pred_neurograph.get_branches(i):
        for xyz in branch:
            hat_xyz = target_neurograph.get_projection(xyz)
            swc_id = target_neurograph.xyz_to_swc(hat_xyz)
            if swc_id == target_id:
                hat_edge = target_neurograph.xyz_to_edge[hat_xyz]
                hits = utils.append_dict_value(hits, hat_edge, hat_xyz)

    # Determine closest edge
    min_dist = np.inf
    best_edge = None
    xyz_i = pred_neurograph.nodes[i]["xyz"]
    if len(hits.keys()) > 1:
        swc_id = pred_neurograph.nodes[i]["swc_id"]
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
