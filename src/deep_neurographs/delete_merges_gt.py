"""
Created on Sat March 26 17:30:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Deletes merges from predicted swc files in the case when there are ground
truth swc files.

"""

import os

import networkx as nx
import numpy as np

from deep_neurographs import geometry
from deep_neurographs.densegraph import DenseGraph
from deep_neurographs.utils import swc_util, util

CLOSE_DISTANCE_THRESHOLD = 3.5
DELETION_RADIUS = 3
MERGE_DIST_THRESHOLD = 30
MIN_INTERSECTION = 10


def delete_merges(
    target_swc_paths,
    pred_swc_paths,
    output_dir,
    img_patch_origin=None,
    img_patch_shape=None,
    radius=DELETION_RADIUS,
    save_sites=False,
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
        Directory that updated graphs and merge sites are written to.
    img_patch_origin : list[float], optional
        An xyz coordinate in the image which is the upper, left, front corner
        of am image patch that contains the swc files. The default is None.
    img_patch_shape : list[float], optional
        The xyz dimensions of the bounding box which contains the swc files.
        The default is None.
    radius : int, optional
        Each node within "radius" is deleted. The default is the global
        variable "DELETION_RADIUS".
    save_sites : bool, optional
        Indication of whether to save merge sites. The default is False.

    Returns
    -------
    None

    """
    # Initializations
    target_densegraph = DenseGraph(target_swc_paths)
    pred_densegraph = DenseGraph(
        pred_swc_paths,
        img_patch_origin=img_patch_origin,
        img_patch_shape=img_patch_shape,
    )
    if save_sites:
        util.mkdir(os.path.join(output_dir, "merge_sites"))

    # Run merge deletion
    for swc_id in pred_densegraph.graphs.keys():
        # Detection
        graph = pred_densegraph.graphs[swc_id]
        delete_nodes = detect_merges_neuron(
            target_densegraph,
            graph,
            radius,
            output_dir=output_dir,
            save=save_sites,
        )

        # Finish
        if len(delete_nodes) > 0:
            graph.remove_nodes_from(delete_nodes)
            print("Merge Detected:", swc_id)
            print("# Nodes Deleted:", len(delete_nodes))
            print("")
        pred_densegraph.graphs[swc_id] = graph

    # Save
    pred_densegraph.save(output_dir)


def detect_merges_neuron(
    target_densegraph, graph, radius, output_dir=None, save=False
):
    """
    Determines whether the "graph" contains merge mistakes. This routine
    projects each node in "graph" onto "target_neurograph", then computes
    the projection distance. ...

    Parameters
    ----------
    target_densegraph : DenseGraph
        Graph built from ground truth swc files.
    graph : networkx.Graph
        Graph build from a predicted swc file.
    radius : int
        Each node within "radius" is deleted.
    output_dir : str, optional
        Directory that merge sites are saved in swc files. The default is
        None.
    save : bool, optional
        Indication of whether to save merge sites. The default is False.

    Returns
    -------
    delete_nodes : set
        Nodes that are part of a merge mistake.

    """
    delete_nodes = set()
    for component in nx.connected_components(graph):
        hits = detect_intersections(target_densegraph, graph, component)
        sites = detect_merges(
            target_densegraph, graph, hits, radius, output_dir, save
        )
        delete_nodes = delete_nodes.union(sites)
    return delete_nodes


def detect_intersections(target_densegraph, graph, component):
    """
    Projects each node in "component" onto the closest node in
    "target_densegraph". If the projection distance for a given node is less
    than "CLOSE_DISTANCE_THRESHOLD", then this node is said to 'intersect'
    with ground truth neuron corresponding to "hat_swc_id".

    Parameters
    ----------
    target_densegraph : DenseGraph
        Graph built from ground truth swc files.
    graph : networkx.Graph
        Graph build from a predicted swc file.
    component : iterator
        Nodes that comprise a connected component.

    Returns
    -------
    dict
        Dictionary that records intersections between "component" and ground
        truth graphs stored in "target_densegraph". Each item consists of the
        swc_id of a neuron from the ground truth and the nodes from
        "component" that intersect that neuron.

    """
    # Compute projections
    hits = dict()
    for i in component:
        xyz = tuple(graph.nodes[i]["xyz"])
        hat_xyz = target_densegraph.get_projection(xyz)
        hat_swc_id = target_densegraph.xyz_to_swc[hat_xyz]
        if geometry.dist(hat_xyz, xyz) < CLOSE_DISTANCE_THRESHOLD:
            hits = util.append_dict_value(hits, hat_swc_id, i)

    # Remove spurious intersections
    keys = [key for key in hits.keys() if len(hits[key]) < MIN_INTERSECTION]
    return util.remove_items(hits, keys)


def detect_merges(target_densegraph, graph, hits, radius, output_dir, save):
    """
    Detects merge mistakes in "graph" (i.e. whether "graph" is closely aligned
    with two distinct connected components in "target_densegraph".

    Parameters
    ----------
    target_densegraph : DenseGraph
        Graph built from ground truth swc files.
    graph : networkx.Graph
        Graph build from a predicted swc file.
    hits : dict
        Dictionary that stores intersections between "target_densegraph" and
        "graph", where the keys are swc ids from "target_densegraph" and
        values are nodes from "graph".
    radius : int
        Each node within "radius" is deleted.
    output_dir : str, optional
        Directory that merge sites are saved in swc files. The default is
        None.
    save : bool, optional
        Indication of whether to save merge sites.

    Returns
    -------
    merge_sites : set
        Nodes that are part of a merge site.

    """
    merge_sites = set()
    if len(hits.keys()) > 0:
        visited = set()
        for id_1 in hits.keys():
            for id_2 in hits.keys():
                # Determine whether to visit
                pair = frozenset((id_1, id_2))
                if id_1 == id_2 or pair in visited:
                    continue

                # Check for merge site
                min_dist, sites = locate_site(graph, hits[id_1], hits[id_2])
                visited.add(pair)
                if min_dist < MERGE_DIST_THRESHOLD:
                    merge_nbhd = get_merged_nodes(graph, sites, radius)
                    merge_sites = merge_sites.union(merge_nbhd)
                    if save:
                        dir_name = f"{output_dir}/merge_sites/"
                        filename = "merge-" + graph.nodes[sites[0]]["swc_id"]
                        path = util.set_path(dir_name, filename, "swc")
                        xyz = get_point(graph, sites)
                        swc_util.save_point(path, xyz)
    return merge_sites


def locate_site(graph, merged_1, merged_2):
    """
    Locates the approximate site of where a merge between two neurons occurs.

    Parameters
    ----------
    graph : networkx.Graph
        Graph to be searched.
    merged_1 : list
        List of nodes part of merge.
    merged_2 : list
        List of nodes part of merge.

    Returns
    -------
    node_pair : tuple
        Closest nodes from "merged_1" and "merged_2"
    min_dist : float
        Euclidean distance between nodes in "node_pair".

    """
    min_dist = np.inf
    node_pair = (None, None)
    for i in merged_1:
        for j in merged_2:
            xyz_i = graph.nodes[i]["xyz"]
            xyz_j = graph.nodes[j]["xyz"]
            if geometry.dist(xyz_i, xyz_j) < min_dist:
                min_dist = geometry.dist(xyz_i, xyz_j)
                node_pair = [i, j]
    return min_dist, node_pair


def get_merged_nodes(graph, sites, radius):
    """
    Gets nodes that are falsely merged.

    Parameters
    ----------
    graph : networkx.Graph
        Graph that contains a merge at "sites".
    sites : list
        Nodes in "graph" that are part of a merge mistake.
    radius : int
        Radius about node to be searched.

    Returns
    -------
    merged_nodes : set
        Nodes that are falsely merged.

    """
    i, j = tuple(sites)
    merged_nodes = set(nx.shortest_path(graph, source=i, target=j))
    merged_nodes = merged_nodes.union(get_nbhd(graph, i, radius))
    merged_nodes = merged_nodes.union(get_nbhd(graph, j, radius))
    return merged_nodes


def get_nbhd(graph, i, radius):
    """
    Gets all nodes within a path length of "radius" from node "i".

    Parameters
    ----------
    graph : networkx.Graph
        Graph to searched.
    i : node
        Node that is root of neighborhood to be returned.
    radius : int
        Radius about node to be searched.

    Returns
    -------
    set
        Nodes within a path length of "radius" from node "i".

    """
    return set(nx.dfs_tree(graph, source=i, depth_limit=radius))


def get_point(graph, sites):
    """
    Gets midpoint of merge site defined by the pair contained in "sites".

    Parameters
    ----------
    graph : networkx.Graph
        Graph that contains a merge at "sites".
    sites : list
        Nodes in "graph" that are part of a merge mistake.

    Returns
    -------
    numpy.ndarray
        Midpoint between pair of xyz coordinates in "sites".

    """
    xyz_0 = graph.nodes[sites[0]]["xyz"]
    xyz_1 = graph.nodes[sites[1]]["xyz"]
    return geometry.get_midpoint(xyz_0, xyz_1)
