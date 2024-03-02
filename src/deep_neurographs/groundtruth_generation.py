"""
Created on Sat March 2 16:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Generates ground truth for edge proposals (i.e. determines whether a proposal
should be accepted or rejected).

"""

import numpy as np

from deep_neurographs import graph_utils as gutils
from deep_neurographs.geometry import dist as get_dist


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

    # Report % positive examples
    print("% accepts:", len(target_edges) / len(pred_neurograph.proposals))
    return target_edges


def get_valid_proposals(target_neurograph, pred_neurograph):
    # Detect components unaligned to ground truth
    invalid = set()
    pred_densegraph = pred_neurograph.init_densegraph()
    for swc_id, graph in pred_densegraph.graphs.items():
        if not is_component_aligned(target_neurograph, graph):
            invalid.add(swc_id)

    # Find valid proposals
    valid_proposals = list()
    for edge in pred_neurograph.proposals:
        # Check whether aligned to a target component
        i, j = tuple(edge)
        invalid_i = pred_neurograph.nodes[i]["swc_id"] in invalid
        invalid_j = pred_neurograph.nodes[j]["swc_id"] in invalid
        if invalid_i or invalid_j:
            continue

        # Check whether aligned to same/adjacent target edges
        branches_i = pred_neurograph.get_branches(i)
        branches_j = pred_neurograph.get_branches(j)
        if is_mutually_aligned(target_neurograph, branches_i, branches_j):
            valid_proposals.append(edge)

    return valid_proposals


def is_component_aligned(target_neurograph, graph):
    """
    Determines whether the connected component defined by "node_subset" is
    close to a component in "target_densegraph". This routine iterates over
    "node_subset" and projects each node onto "target_densegraph", then
    computes the projection distance. If (on average) each node in
    "node_subset" is less 3.5 microns from a component in the ground truth,
    then "node_subset" is aligned.

    Parameters
    ----------
    target_neurograph : NeuroGraph
        Graph that was generated using the ground truth swc files.
    graph : networkx.Graph
        Graph that corresponds to some swc file from the prediction.

    Returns
    -------
    bool
        Indication of whether "node_subset" is aligned to a connected
        component in "self".

    """
    # Compute projection distances
    dists = np.zeros((len(graph.nodes)))
    for idx, i in enumerate(graph.nodes):
        xyz = graph.nodes[i]["xyz"]
        hat_xyz = target_neurograph.get_projection(xyz)
        dists[idx] = get_dist(hat_xyz, xyz)

    # Determine whether aligned
    idxs = dists < np.percentile(dists, 90)
    return False if np.mean(dists[idxs]) > 4 else True


def is_mutually_aligned(target_neurograph, branches_i, branches_j):
    # Find closest edges from target_neurograph
    hat_edge_i = best_edge_alignment(target_neurograph, branches_i)
    hat_edge_j = best_edge_alignment(target_neurograph, branches_j)

    # Check if edges either identical or adjacent
    identical = hat_edge_i == hat_edge_j
    adjacent = is_adjacent(target_neurograph, hat_edge_i, hat_edge_j)
    if identical or adjacent:
        return True
    else:
        return False


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
            if neurograph.is_nb(i, j):
                return True
    return False


def best_edge_alignment(target_neurograph, branches):
    best_edges = dict()
    for branch in branches:
        best_edge = __best_edge_alignment(target_neurograph, branch)
        best_edges = upd_dict_cnts(best_edges, best_edge)
    return find_best(best_edges)


def __best_edge_alignment(target_neurograph, branch):
    close_edges = dict()
    for xyz in branch:
        hat_xyz = target_neurograph.get_projection(xyz)
        hat_edge = target_neurograph.xyz_to_edge[hat_xyz]
        close_edges = upd_dict_cnts(close_edges, hat_edge)
    return find_best(close_edges)


# -- utils --
def upd_dict_cnts(my_dict, key):
    if key in my_dict.keys():
        my_dict[key] += 1
    else:
        my_dict[key] = 1
    return my_dict


def find_best(my_dict):
    best_key = None
    best_vote_cnt = 0
    for key in my_dict.keys():
        vote_cnt = my_dict[key]
        if vote_cnt > best_vote_cnt:
            best_key = key
            best_vote_cnt = vote_cnt
    return best_key
