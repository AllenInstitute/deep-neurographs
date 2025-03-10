"""
Created on Sat March 2 16:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Generates ground truth for proposals by determining whether a proposal should
be accepted or rejected based on comparing fragments to ground truth tracings.

"""

from collections import defaultdict

import networkx as nx
import numpy as np

from deep_neurographs.utils import geometry_util, util

ALIGNED_THRESHOLD = 4.5


def run(pred_graph, target_graph):
    """
    Initializes ground truth for edge proposals.

    Parameters
    ----------
    target_graph : NeuroGraph
        Graph built from ground truth swc files.
    pred_graph : NeuroGraph
        Graph build from predicted swc files.

    Returns
    -------
    set
        Proposals that machine learning model learns to accept.

    """
    # Initializations
    proposals = get_valid_proposals(target_graph, pred_graph)
    lengths = [pred_graph.proposal_length(p) for p in proposals]
    return proposals

    # Add best simple edges -- prevents loops at branching points
    gt_accepts = set()
    graph = pred_graph.copy_graph()
    for idx in np.argsort(lengths):
        i, j = tuple(proposals[idx])
        if not nx.has_path(graph, i, j):
            graph.add_edge(i, j)
            gt_accepts.add(proposals[idx])
    return gt_accepts


def get_valid_proposals(target_graph, pred_graph):
    # Initializations
    target_kdtree = target_graph.get_kdtree()
    aligned_fragment_ids, node_to_target = find_aligned_fragments(
        target_graph, pred_graph, target_kdtree
    )

    # Check whether aligned to same or connected target edges
    valid_proposals = list()
    for p in pred_graph.proposals:
        i, j = tuple(p)
        is_aligned_i = pred_graph.nodes[i]["swc_id"] in aligned_fragment_ids
        is_aligned_j = pred_graph.nodes[j]["swc_id"] in aligned_fragment_ids
        if is_aligned_i and is_aligned_j:
            if node_to_target[i] == node_to_target[j]:
                # Check whether proposal is valid
                if is_valid(target_graph, pred_graph, target_kdtree, p):
                    valid_proposals.append(p)
    return valid_proposals


def find_aligned_fragments(target_graph, pred_graph, kdtree):
    """
    Detects connected components in "pred_graph" that are aligned to some
    connected component in "target_graph".

    Parameters
    ----------
    target_graph : NeuroGraph
        Graph built from ground truth swc files.
    pred_graph : NeuroGraph
        Graph build from predicted swc files.

    Returns
    -------
    valid_ids : set
        IDs in ""pred_graph" that correspond to connected components that
        are aligned to some connected component in "target_graph".
    node_to_target : dict
        Mapping between nodes and target ids.

    """
    valid_ids = set()
    node_to_target = dict()
    for nodes in nx.connected_components(pred_graph):
        aligned, target_id = is_component_aligned(
            target_graph, pred_graph, nodes, kdtree
        )
        if aligned:
            i = util.sample_once(nodes)
            valid_ids.add(pred_graph.nodes[i]["swc_id"])
            node_to_target = upd_dict(node_to_target, nodes, target_id)
    return valid_ids, node_to_target


def is_component_aligned(target_graph, pred_graph, nodes, kdtree):
    """
    Determines whether the connected component formed by "nodes" is close to
    some component in "target_graph". This routine projects each node onto the
    "target_graph", then computes the projection distance. If the nodes are on
    average less 3.5 microns from a single component in "target_graph", then
    the "nodes" are aligned.

    Parameters
    ----------
    target_graph : NeuroGraph
        Graph built from ground truth swc files.
    pred_graph : NeuroGraph
        Graph build from predicted swc files.
    nodes : Set[int]
        Nodes from a connected component in "pred_graph".

    Returns
    -------
    bool
        Indication of whether connected component "nodes" is aligned to a
        connected component in "target_graph".

    """
    # Compute distances
    dists = defaultdict(list)
    point_cnt = 0
    for edge in pred_graph.subgraph(nodes).edges:
        for xyz in pred_graph.edges[edge]["xyz"]:
            hat_xyz = geometry_util.kdtree_query(kdtree, xyz)
            hat_swc_id = target_graph.xyz_to_id(hat_xyz)
            d = geometry_util.dist(hat_xyz, xyz)
            dists[hat_swc_id].append(d)
            point_cnt += 1

    # Deterine whether aligned
    hat_swc_id = util.find_best(dists)
    dists = np.array(dists[hat_swc_id])
    percent_aligned = len(dists) / point_cnt

    intersects = True if percent_aligned > 0.6 else False
    aligned_score = np.mean(dists[dists < np.percentile(dists, 80)])
    if (aligned_score < ALIGNED_THRESHOLD and hat_swc_id) and intersects:
        return True, hat_swc_id
    else:
        return False, None


def is_valid(target_graph, pred_graph, kdtree, proposal):
    """
    Determines whether the proposal connects two branches that correspond to
    either the same or adjacent branches on the ground truth. If either
    condition holds, then the proposal is said to be consistent.

    Parameters
    ----------
    target_graph : FragmentsGraph
        Graph built from ground truth swc files.
    pred_graph : NeuroGraph
        Graph build from predicted swc files.
    proposal : frozenset
        Proposal to be checked.

    Returns
    -------
    bool
        Indication of whether proposal is consistent.

    """
    # Find edges in target_graph closest to edges connected to proposal
    i, j = tuple(proposal)
    hat_edge_i = project_region(target_graph, pred_graph, kdtree, i)
    hat_edge_j = project_region(target_graph, pred_graph, kdtree, j)

    # Check if closest edges are identical
    if hat_edge_i == hat_edge_j:
        return True

    # Check if edges are adjacent
    if is_connected(hat_edge_i, hat_edge_j):
        result = is_connected_aligned(
            target_graph.edges[hat_edge_i]["xyz"],
            target_graph.edges[hat_edge_j]["xyz"],
            pred_graph.nodes[i]["xyz"],
            pred_graph.nodes[j]["xyz"],
        )
        return result
    return False


def project_region(target_graph, pred_graph, kdtree, i):
    """
    Projects the edges (up to a certain depth) connected to node i onto
    target graph.

    Parameters
    ----------
    ...

    """
    hits = defaultdict(list)
    for edge_xyz_list in pred_graph.truncated_edge_attr_xyz(i, 16):
        for xyz in edge_xyz_list:
            hat_xyz = geometry_util.kdtree_query(kdtree, xyz)
            hat_edge = target_graph.xyz_to_edge[hat_xyz]
            hits[hat_edge].append(hat_xyz)
    return util.find_best(hits)


def is_connected(edge_i, edge_j):
    """
    Determines whether "edge_i" and "edge_j" are adjacent, meaning there
    exists a nodes in "edge_i" and "edge_j" which are neighbors.

    Parameters
    ----------
    edge_i : tuple
        Edge to be checked.
    edge_j : tuple
        Edge to be checked.

    Returns
    -------
    bool
        Indication of whether "edge_i" and "edge_j" are connected.

    """
    return True if set(edge_i).intersection(set(edge_j)) else False


def is_connected_aligned(hat_edge_xyz_i, hat_edge_xyz_j, xyz_i, xyz_j):
    hat_xyz_i = geometry_util.nearest_neighbor(hat_edge_xyz_i, xyz_i)
    hat_xyz_j = geometry_util.nearest_neighbor(hat_edge_xyz_j, xyz_j)
    hat_dist = geometry_util.dist(hat_xyz_i, hat_xyz_j)
    dist = geometry_util.dist(xyz_i, xyz_j)
    return True if 2 * dist / (dist + hat_dist) > 0.5 else False


# -- util --
def upd_dict(node_to_target_id, nodes, target_id):
    for node in nodes:
        node_to_target_id[node] = target_id
    return node_to_target_id
