"""
Created on Sat June 25 9:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Module that removes doubled fragments and trims branches that pass by each
other from a FragmentsGraph.

"""

from collections import defaultdict
from tqdm import tqdm

import networkx as nx
import numpy as np

from deep_neurographs.utils import geometry_util, util

QUERY_DIST = 15


# --- Curvy Removal ---
def remove_curvy(graph, max_length, ratio=0.5):
    """
    Removes connected components with 2 nodes from "graph" that are "curvy",
    based on a specified ratio of endpoint distance to edge length and a
    maximum length threshold.

    Parameters
    ----------
    graph : FragmentsGraph
        Graph generated from fragments of a predicted segmentation.
    max_length : float
        The maximum allowable length (in microns) for an edge to be considered
        for removal.
    ratio : float, optional
        Threshold ratio of endpoint distance to edge length. Components with a
        ratio below this value are considered "curvy" and are removed. The
        default is 0.5.

    Returns
    -------
    int
        Number of fragments removed from the graph.

    """
    deleted_ids = set()
    for nodes in get_line_components(graph):
        i, j = tuple(nodes)
        length = graph.edges[i, j]["length"]
        endpoint_dist = graph.dist(i, j)
        if endpoint_dist / length < ratio and length < max_length:
            deleted_ids.add(graph.edges[i, j]["swc_id"])
            graph = delete_fragment(graph, i, j)
    return graph


# --- Doubles Removal ---
def remove_doubles(graph, max_length, node_spacing):
    """
    Removes connected components from "graph" that are likely to be a double,
    which is caused by ghosting in the image.

    Parameters
    ----------
    graph : FragmentsGraph
        Graph to be searched for doubles.
    max_length : int
        Maximum size of connected components to be searched.
    node_spacing : int
        Expected distance (in microns) between nodes in "graph".

    Returns
    -------
    int
        Number of fragments removed from graph.

    """
    # Initializations
    components = get_line_components(graph)
    deleted_ids = set()
    kdtree = graph.get_kdtree()

    # Main
    desc = "Filter Doubled Fragments"
    for idx in tqdm(np.argsort([len(c) for c in components]), desc=desc):
        i, j = tuple(components[idx])
        swc_id = graph.nodes[i]["swc_id"]
        if swc_id not in deleted_ids:
            if graph.edges[i, j]["length"] < max_length:
                # Check doubles criteria
                n_points = len(graph.edges[i, j]["xyz"])
                hits = compute_projections(graph, kdtree, (i, j))
                if check_doubles_criteria(hits, n_points):
                    graph = delete_fragment(graph, i, j)
                    deleted_ids.add(swc_id)
    return graph


def compute_projections(graph, kdtree, edge):
    """
    Given a fragment defined by "edge", this routine iterates of every xyz in
    the fragment and projects it onto the closest fragment. For each detected
    fragment, the fragment id and projection distance are stored in a
    dictionary called "hits".

    Parameters
    ----------
    graph : FragmentsGraph
        Graph that contains "edge".
    kdtree : KDTree
        KD-Tree that contains all xyz coordinates of every fragment in
        "graph".
    edge : tuple
        Pair of leaf nodes that define a fragment.

    Returns
    -------
    dict
        Dictionary that stores all fragments that were detected and the
        projection distances.

    """
    hits = defaultdict(list)
    query_id = graph.edges[edge]["swc_id"]
    for i, xyz in enumerate(graph.edges[edge]["xyz"]):
        # Compute projections
        best_id = None
        best_dist = np.inf
        for hit_xyz in geometry_util.query_ball(kdtree, xyz, QUERY_DIST):
            hit_id = graph.xyz_to_id(hit_xyz)
            if hit_id is not None and hit_id != query_id:
                if geometry_util.dist(hit_xyz, xyz) < best_dist:
                    best_dist = geometry_util.dist(hit_xyz, xyz)
                    best_id = hit_id

        # Store best
        if best_id:
            hits[best_id].append(best_dist)
        elif i == 15 and len(hits) == 0:
            return hits
    return hits


def check_doubles_criteria(hits, n_points):
    """
    Determines whether the connected component corresponding to "root" is a
    double of another connected component.

    Paramters
    ---------
    hits : dict
        ...
    n_points : int
        Number of nodes that comprise the component being checked.

    Returns
    -------
    bool
        Indication of whether component is a double.

    """
    for dists in hits.values():
        if len(dists) > 10:
            percent_hit = len(dists) / n_points
            if percent_hit > 0.5 and np.std(dists) < 2:
                return True
            elif percent_hit > 0.75 and np.std(dists) < 2.5:
                return True
    return False


def delete_fragment(graph, i, j):
    """
    Deletes nodes "i" and "j" from "graph", where these nodes form a connected
    component.

    Parameters
    ----------
    graph : FragmentsGraph
        Graph that contains nodes to be removed.
    i : int
        Node to be removed.
    j : int
        Node to be removed.

    Returns
    -------
    FragmentsGraph
        Graph with nodes removed.

    """
    graph = remove_xyz_entries(graph, i, j)
    graph.swc_ids.remove(graph.nodes[i]["swc_id"])
    graph.remove_nodes_from([i, j])
    return graph


def remove_xyz_entries(graph, i, j):
    """
    Removes dictionary entries from "graph.xyz_to_edge" corresponding to the
    edge (i, j).

    Parameters
    ----------
    graph : FragmentsGraph
        Graph containing an edge between nodes i and j.
    i : int
        Node ID.
    j : int
        Node ID.

    Returns
    -------
    FragmentsGraph
        Graph with updated "graph.xyz_to_edge" attribute.

    """
    # Subroutine
    def remove_entry(xyz):
        try:
            del graph.xyz_to_edge[tuple(xyz)]
        except KeyError:
            pass
        return graph

    # Main
    graph = remove_entry(graph.nodes[i]["xyz"])
    graph = remove_entry(graph.nodes[j]["xyz"])
    for xyz in graph.edges[i, j]["xyz"]:
        graph = remove_entry(xyz)
    return graph

def upd_hits(hits, key, value):
    """
    Updates "hits" by adding ("key", "value") if this item does not exist.
    Otherwise, checks if "value" is less than "hits[key"]".

    Parameters
    ----------
    hits : dict
        Stores swd_ids of fragments within a certain distance a query fragment
        along with the corresponding distances.
    key : str
        swc id of some fragment.
    value : float
        Distance in microns that fragment corresponding to "key" is from the
        query fragment.

    Returns
    -------
    dict
        Updated version of hits.

    """
    if key in hits:
        if value < hits[key]:
            hits[key] = value
    else:
        hits[key] = value
    return hits


# --- utils ---
def get_line_components(graph):
    """
    Identifies and returns all line components in the given graph. A line
    component is defined as a connected component with exactly two nodes.

    Parameters
    ----------
    graph : networkx.Graph
        Input graph in which line components are to be identified.

    Returns
    -------
    List[set]
        List of sets, where each set contains two nodes representing a
        connected component with exactly two nodes.

    """
    return [c for c in nx.connected_components(graph) if len(c) == 2]
