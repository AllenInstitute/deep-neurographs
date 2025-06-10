"""
Created on Sat March 2 16:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Generates ground truth for proposals by determining whether a proposal should
be accepted or rejected based on comparing fragments to ground truth tracings.

    # Add best simple edges -- prevents loops at branching points
    gt_accepts = set()
    graph = pred_graph.copy_graph()
    lengths = [pred_graph.proposal_length(p) for p in proposals]
    for idx in np.argsort(lengths):
        i, j = tuple(proposals[idx])
        if not nx.has_path(graph, i, j):
            graph.add_edge(i, j)
            gt_accepts.add(proposals[idx])
    return gt_accepts

"""

from collections import defaultdict

import networkx as nx
import numpy as np

from deep_neurographs.utils import geometry_util, util

ALIGNED_THRESHOLD = 4.5


def run(pred_graph, gt_graph):
    """
    Initializes ground truth for edge proposals.

    Parameters
    ----------
    gt_graph : NeuroGraph
        Graph built from ground truth swc files.
    pred_graph : NeuroGraph
        Graph build from predicted swc files.

    Returns
    -------
    set
        Proposals that machine learning model learns to accept.

    """
    # Find connected components in prediction aligned to ground truth
    gt_kdtree = gt_graph.get_kdtree()
    pred_to_target = find_alignments(gt_graph, pred_graph, gt_kdtree)

    # Determine proposal groundtruth
    gt_accepts = list()
    for p in pred_graph.proposals:
        # Extract proposal info
        i, j = tuple(p)
        pred_id_i = pred_graph.nodes[i]["swc_id"]
        pred_id_j = pred_graph.nodes[j]["swc_id"]

        # Check whether aligned to ground truth
        is_aligned_i = pred_id_i in pred_to_target
        is_aligned_j = pred_id_j in pred_to_target
        if is_aligned_i and is_aligned_j:
            # Check whether aligned to same ground truth skeleton
            gt_id_i = pred_to_target[pred_id_i]
            gt_id_j = pred_to_target[pred_id_j]
            if gt_id_i == gt_id_j:
                # Check whether proposal is valid
                if is_valid(gt_graph, pred_graph, gt_kdtree, gt_id_i, p):
                    gt_accepts.append(p)
    return gt_accepts


def find_alignments(gt_graph, pred_graph, kdtree):
    """
    Detects connected components in "pred_graph" that are aligned to some
    connected component in "gt_graph".

    Parameters
    ----------
    gt_graph : NeuroGraph
        Graph built from ground truth swc files.
    pred_graph : NeuroGraph
        Graph build from predicted swc files.

    Returns
    -------
    valid_ids : set
        IDs in ""pred_graph" that correspond to connected components that
        are aligned to some connected component in "gt_graph".
    node_to_target : dict
        Mapping between nodes and target ids.

    """
    valid_ids = set()
    pred_to_target = dict()
    for nodes in nx.connected_components(pred_graph):
        aligned, target_id = is_component_aligned(
            gt_graph, pred_graph, nodes, kdtree
        )
        if aligned:
            i = util.sample_once(nodes)
            pred_id = pred_graph.nodes[i]["swc_id"]
            valid_ids.add(pred_id)
            pred_to_target[pred_id] = target_id
    return pred_to_target


def is_component_aligned(gt_graph, pred_graph, nodes, kdtree):
    """
    Determines whether the connected component formed by "nodes" is close to
    some component in "gt_graph". This routine projects each node onto the
    "gt_graph", then computes the projection distance. If the nodes are on
    average less 3.5 microns from a single component in "gt_graph", then
    the "nodes" are aligned.

    Parameters
    ----------
    gt_graph : NeuroGraph
        Graph built from ground truth swc files.
    pred_graph : NeuroGraph
        Graph build from predicted swc files.
    nodes : Set[int]
        Nodes from a connected component in "pred_graph".

    Returns
    -------
    bool
        Indication of whether connected component "nodes" is aligned to a
        connected component in "gt_graph".

    """
    # Compute distances
    dists = defaultdict(list)
    point_cnt = 0
    for edge in pred_graph.subgraph(nodes).edges:
        for xyz in pred_graph.edges[edge]["xyz"]:
            hat_xyz = geometry_util.kdtree_query(kdtree, xyz)
            hat_swc_id = gt_graph.xyz_to_id(hat_xyz)
            d = geometry_util.dist(hat_xyz, xyz)
            dists[hat_swc_id].append(d)
            point_cnt += 1

    # Compute alignment score
    hat_swc_id = util.find_best(dists)
    dists = np.array(dists[hat_swc_id])
    percent_aligned = len(dists) / point_cnt

    intersects = True if percent_aligned > 0.5 else False
    aligned_score = np.mean(dists[dists < np.percentile(dists, 80)])

    # Deterine whether aligned
    if (aligned_score < ALIGNED_THRESHOLD and hat_swc_id) and intersects:
        return True, hat_swc_id
    else:
        return False, None


def is_valid(gt_graph, pred_graph, kdtree, gt_id, proposal):
    """
    Determines whether the proposal connects two branches that correspond to
    either the same or adjacent branches on the ground truth. If either
    condition holds, then the proposal is said to be consistent.

    Parameters
    ----------
    gt_graph : FragmentsGraph
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
    # Find edges in gt_graph closest to edges connected to proposal
    i, j = tuple(proposal)
    hat_edge_i = find_closest_gt_edge(gt_graph, pred_graph, kdtree, gt_id, i)
    hat_edge_j = find_closest_gt_edge(gt_graph, pred_graph, kdtree, gt_id, j)

    # Check if edge was found
    if hat_edge_i is None or hat_edge_j is None:
        return False

    # Check if closest edges are identical
    if hat_edge_i == hat_edge_j:
        return True

    # Check if edges are adjacent
    if is_connected(hat_edge_i, hat_edge_j):
        # Orient ground truth edges
        hat_edge_xyz_i, hat_edge_xyz_j = orient_edges(
            gt_graph.edges[hat_edge_i]["xyz"],
            gt_graph.edges[hat_edge_j]["xyz"]
        )

        # Find index of closest points on ground truth edges
        xyz_i = pred_graph.nodes[i]["xyz"]
        xyz_j = pred_graph.nodes[i]["xyz"]
        idx_i = find_closest_point(hat_edge_xyz_i, xyz_i)
        idx_j = find_closest_point(hat_edge_xyz_j, xyz_j)

        len_1 = length_to_idx(hat_edge_xyz_i, idx_i)
        len_2 = length_to_idx(hat_edge_xyz_j, idx_j)
        gt_dist = len_1 + len_2
        proposal_dist = pred_graph.proposal_length(proposal)
        return abs(proposal_dist - gt_dist) < 40
    return False


def find_closest_gt_edge(gt_graph, pred_graph, kdtree, gt_id, i):
    depth = 16
    while depth <= 64:
        # Search for edge
        hat_edge_i = project_region(
            gt_graph, pred_graph, kdtree, gt_id, i, depth
        )

        # Check result
        if hat_edge_i is None:
            depth += 16
        else:
            break
    return hat_edge_i


def project_region(gt_graph, pred_graph, kdtree, gt_id, i, depth=16):
    """
    Projects the edges (up to a certain depth) connected to node i onto
    target graph.

    Parameters
    ----------
    ...

    """
    hits = defaultdict(list)
    for edge_xyz_list in pred_graph.truncated_edge_attr_xyz(i, 24):
        for xyz in edge_xyz_list:
            hat_xyz = geometry_util.kdtree_query(kdtree, xyz)
            hat_edge = gt_graph.xyz_to_edge[hat_xyz]
            if gt_graph.nodes[hat_edge[0]]["swc_id"] == gt_id:
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


def orient_edges(xyz_edge_i, xyz_edge_j):
    # Compute distances
    dist_1 = geometry_util.dist(xyz_edge_i[0], xyz_edge_j[0])
    dist_2 = geometry_util.dist(xyz_edge_i[0], xyz_edge_j[-1])
    dist_3 = geometry_util.dist(xyz_edge_i[-1], xyz_edge_j[0])
    dist_4 = geometry_util.dist(xyz_edge_i[-1], xyz_edge_j[-1])

    # Orient coordinates to match at 0-th index
    min_dist = np.min([dist_1, dist_2, dist_3, dist_4])
    if dist_2 == min_dist or dist_4 == min_dist:
        xyz_edge_j = np.flip(xyz_edge_j, axis=0)
    if dist_3 == min_dist or dist_4 == min_dist:
        xyz_edge_i = np.flip(xyz_edge_i, axis=0)
    return xyz_edge_i, xyz_edge_j


def find_closest_point(xyz_list, query_xyz):
    best_dist = np.inf
    best_idx = np.inf
    for idx, xyz in enumerate(xyz_list):
        dist = geometry_util.dist(query_xyz, xyz)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def length_to_idx(xyz_list, idx):
    length = 0
    for i in range(0, idx):
        length += geometry_util.dist(xyz_list[i], xyz_list[i + 1])
    return length
