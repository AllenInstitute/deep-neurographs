"""
Created on Sat March 26 17:30:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Deletes merges from predicted swc files in the case when there are ground
truth swc files.

"""

import networkx as nx
import numpy as np
from deep_neurographs.densegraph import DenseGraph
from deep_neurographs import geometry, utils

CLOSE_THRESHOLD = 3.5
DELETION_RADIUS = 5
MERGE_DIST_THRESHOLD = 30
MIN_INTERSECTION = 10


def delete_merges(
    target_swc_paths,
    pred_swc_paths,
    output_dir,
    deletion_radius=DELETION_RADIUS,
):
    """
    Deletes merges from predicted swc files in the case when there are ground
    truth swc files.

    Parameters
    ----------
    target_swc_paths : list[str]
        List of paths to ground truth swc files.
    pred_swc_paths : list[str]
        List of paths to predicted swc files.
    output_dir : str
        Directory that updated graphs will be written to.
    deletion_radius : int, optional
        Each node within "deletion_radius" is deleted. The default is the
        global variable "DELETION_RADIUS".

    Returns
    -------
    None

    """
    target_densegraph = DenseGraph(target_swc_paths)
    pred_densegraph = DenseGraph(pred_swc_paths)
    for swc_id in pred_densegraph.graphs.keys():
        # Detection
        pred_graph = pred_densegraph.graphs[swc_id]
        merged_nodes = detect_merge(target_densegraph, pred_graph)

        # Deletion
        if len(merged_nodes.keys()) > 0:
            visited = set()
            delete_nodes = set()
            for key_1 in merged_nodes.keys():
                for key_2 in merged_nodes.keys():
                    pair = frozenset((key_1, key_2))
                    if key_1 != key_2 and pair not in visited:
                        sites, d = locate_site(
                            pred_graph, merged_nodes[key_1], merged_nodes[key_2]
                        )
                        if d < MERGE_DIST_THRESHOLD:
                            print(sites, d)
                            # delete just like a connector

        pred_densegraph.graphs[swc_id] = pred_graph

    # Save
    pred_densegraph.save(output_dir)


def detect_merge(target_densegraph, pred_graph):
    """
    Determines whether the "pred_graph" contains a merge mistake. This routine
    projects each node in "pred_graph" onto "target_neurograph", then computes
    the projection distance. ...

    Parameters
    ----------
    target_densegraph : DenseGraph
        Graph built from ground truth swc files.
    pred_graph : networkx.Graph
        Graph build from a predicted swc file.

    Returns
    -------
    set
        Set of nodes that are part of a merge mistake.

    """
    # Compute projections
    hits = dict()
    for i in pred_graph.nodes:
        xyz = tuple(pred_graph.nodes[i]["xyz"])
        hat_xyz = target_densegraph.get_projection(xyz)
        hat_swc_id = target_densegraph.xyz_to_swc[hat_xyz]
        if geometry.dist(hat_xyz, xyz) < CLOSE_THRESHOLD:
            hits = utils.append_dict_value(hits, hat_swc_id, i)

    # Remove spurious intersections
    keys = [key for key in hits.keys() if len(hits[key]) < MIN_INTERSECTION]
    return utils.remove_items(hits, keys)


def locate_site(graph, merged_1, merged_2):
    min_dist = np.inf
    node_pair = [None, None]
    for i in merged_1:
        for j in merged_2:
            xyz_i = graph.nodes[i]["xyz"]
            xyz_j = graph.nodes[j]["xyz"]
            if geometry.dist(xyz_i, xyz_j) < min_dist:
                min_dist = geometry.dist(xyz_i, xyz_j)
                node_pair = [i, j]
        return node_pair, min_dist


def delete_merge(graph, root, radius):
    delete_nodes = get_nearby_nodes(graph, root, radius)
    graph.remove_nodes_from(delete_nodes)
    return graph


def get_nearby_nodes(graph, root, radius):
    nearby_nodes = set()
    for _, j in nx.dfs_edges(graph, source=root, depth_limit=radius):
        nearby_nodes.add(j)
    return nearby_nodes
