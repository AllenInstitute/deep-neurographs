"""
Created on Fri March 1 16:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org


This code identifies proposals between fragments that align with the same
ground truth skeleton and are structurally consistent.

    Algorithm
    ---------
    1. Find fragments aligned to a single ground truth skeleton and build a
       dictionary that maps these fragment IDs to the corresponding ground
       truth ID.

    2. Iterate over all proposals generated between fragments. A proposal is
       accepted if:
            - Both fragments align to the same ground truth skeleton.
            - Proposal is structurally consistent, meaning the connection
              preserves geometric continuity and branching topology consistent
              with the ground truth structure.

Note: We use the convention that a fragment refers to a connected component in
      "pred_graph".
"""

from collections import defaultdict

import networkx as nx
import numpy as np

from deep_neurographs.utils import geometry_util, util


def run(gt_graph, pred_graph):
    """
    Initializes ground truth for edge proposals.

    Parameters
    ----------
    gt_graph : FragmentsGraph
        Graph built from ground truth SWC files.
    pred_graph : ProposalGraph
        Graph build from predicted SWC files.

    Returns
    -------
    gt_accepts : List[Frozenset[int]]
        Proposals aligned to and structurally consistent with ground truth.
        Note: a model will learn to accept these proposals.
    """
    # Initializations
    gt_kdtree = gt_graph.get_kdtree()
    pred_to_gt = get_pred_to_gt_mapping(gt_graph, pred_graph, gt_kdtree)

    # Main
    gt_accepts = list()
    for proposal in pred_graph.proposals:
        # Extract proposal info
        i, j = tuple(proposal)
        id_i = pred_graph.node_component_id[i]
        id_j = pred_graph.node_component_id[j]

        # Check if aligned to ground truth skeletons
        if id_i not in pred_to_gt or id_j not in pred_to_gt:
            continue

        # Check if aligned to the same ground truth skeletons
        if pred_to_gt[id_i] != pred_to_gt[id_j]:
            continue

        # Check if structurally consistent
        consistent = is_structurally_consistent(
            gt_graph, pred_graph, gt_kdtree, pred_to_gt[id_i], proposal
        )
        if consistent:
            gt_accepts.append(proposal)
    return gt_accepts


def get_pred_to_gt_mapping(gt_graph, pred_graph, kdtree):
    """
    Gets fragments aligned to a single ground truth skeleton and builds a
    dictionary that maps these fragment IDs to the corresponding ground truth
    ID.

    Parameters
    ----------
    gt_graph : FragmentsGraph
        Graph built from ground truth SWC files.
    pred_graph : FragmentsGraph
        Graph build from predicted SWC files.

    Returns
    -------
    pred_to_gt : dict
        Dictionary that maps fragment IDs to the corresponding ground truth
        ID.
    """
    pred_to_gt = dict()
    for nodes in nx.connected_components(pred_graph):
        gt_id = find_aligned_component(gt_graph, pred_graph, nodes, kdtree)
        if gt_id:
            i = util.sample_once(nodes)
            pred_to_gt[pred_graph.node_component_id[i]] = gt_id
    return pred_to_gt


def find_aligned_component(gt_graph, pred_graph, nodes, kdtree):
    """
    Determines whether a fragment is spatially aligned to a single connected
    component in the ground truth graph. The given nodes are projected onto
    "gt_graph", and the average projection distance is computed. If this
    distance is less than 4.5 µm and all projections fall within the same
    connected component of gt_graph, the fragment is considered aligned.

    Parameters
    ----------
    gt_graph : FragmentsGraph
        Graph built from ground truth SWC files.
    pred_graph : FragmentsGraph
        Graph build from predicted SWC files.
    nodes : Set[int]
        Nodes from a connected component in "pred_graph".

    Returns
    -------
    str or None
        Indication of whether connected component "nodes" is aligned to a
        connected component in "gt_graph".
    """
    # Compute distances
    dists = defaultdict(list)
    point_cnt = 0
    for edge in pred_graph.subgraph(nodes).edges:
        for xyz in pred_graph.edges[edge]["xyz"]:
            hat_xyz = geometry_util.kdtree_query(kdtree, xyz)
            hat_id = gt_graph.xyz_to_component_id(hat_xyz)
            d = geometry_util.dist(hat_xyz, xyz)
            dists[hat_id].append(d)
            point_cnt += 1

    # Compute alignment score
    hat_id = util.find_best(dists)
    dists = np.array(dists[hat_id])
    percent_aligned = len(dists) / point_cnt

    intersects = True if percent_aligned > 0.5 else False
    aligned_score = np.mean(dists[dists < np.percentile(dists, 80)])

    # Deterine whether aligned
    if (aligned_score < 4 and hat_id) and intersects:
        return hat_id
    else:
        return None


def is_structurally_consistent(gt_graph, pred_graph, kdtree, gt_id, proposal):
    """
    Determines whether the proposal connects two branches that correspond to
    either the same or adjacent branches on the ground truth. If either
    condition holds, then the proposal is said to be consistent.

    Parameters
    ----------
    gt_graph : FragmentsGraph
        Graph built from ground truth SWC files.
    pred_graph : FragmentsGraph
        Graph build from predicted SWC files.
    proposal : frozenset
        Proposal to be checked.

    Returns
    -------
    bool
        Indication of whether proposal is structurally consistent.
    """
    # Find irreducible edges in gt_graph closest to edges connected to proposal
    i, j = tuple(proposal)
    hat_edge_i = find_closest_gt_edge(gt_graph, pred_graph, kdtree, gt_id, i)
    hat_edge_j = find_closest_gt_edge(gt_graph, pred_graph, kdtree, gt_id, j)
    if hat_edge_i is None or hat_edge_j is None:
        return False

    # Check if closest edges are identical
    if hat_edge_i == hat_edge_j:
        return True

    # Check if edges are adjacent
    if set(hat_edge_i).intersection(set(hat_edge_j)):
        # Orient ground truth edges
        hat_edge_xyz_i, hat_edge_xyz_j = orient_edges(
            gt_graph.edges[hat_edge_i]["xyz"],
            gt_graph.edges[hat_edge_j]["xyz"]
        )

        # Find index of closest points on ground truth edges
        xyz_i = pred_graph.node_xyz[i]
        xyz_j = pred_graph.node_xyz[j]
        idx_i = find_closest_point(hat_edge_xyz_i, xyz_i)
        idx_j = find_closest_point(hat_edge_xyz_j, xyz_j)

        len_1 = length_up_to(hat_edge_xyz_i, idx_i)
        len_2 = length_up_to(hat_edge_xyz_j, idx_j)
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
            if gt_graph.node_component_id[hat_edge[0]] == gt_id:
                hits[hat_edge].append(hat_xyz)
    return util.find_best(hits)


# --- Helpers ---
def find_closest_point(xyz_list, query_xyz):
    best_dist = np.inf
    best_idx = np.inf
    for idx, xyz in enumerate(xyz_list):
        dist = geometry_util.dist(query_xyz, xyz)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def length_up_to(path_pts, idx):
    """
    Computes the cumulative path length from the start of a 3D point path up
    to a given index.

    Parameters
    ----------
    path_pts : numpy.ndarray
        3D points defining a continuous path.
    idx : int
        Index up to which the cumulative length is computed.

    Returns
    -------
    length : float
         Cumulative path length from the start up to point "idx".
    """
    length = 0
    for i in range(0, idx):
        length += geometry_util.dist(path_pts[i], path_pts[i + 1])
    return length


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
