"""
Created on Sat June 25 9:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Module that removes doubled fragments and trims branches that pass by each
other from a FragmentsGraph.

"""

from collections import defaultdict

import networkx as nx
import numpy as np
from tqdm import tqdm

from deep_neurographs import geometry

QUERY_DIST = 15


# --- Curvy Removal ---
def remove_curvy(fragments_graph, max_length, ratio=0.5):
    """
    Removes connected components with 2 nodes from "fragments_graph" that are
    "curvy" fragments, based on a specified ratio of endpoint distance to edge
    length and a maximum length threshold.

    Parameters
    ----------
    fragments_graph : FragmentsGraph
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
    components = get_line_components(fragments_graph)
    for nodes in tqdm(components, desc="Filter Curvy Fragments"):
        i, j = tuple(nodes)
        length = fragments_graph.edges[i, j]["length"]
        endpoint_dist = fragments_graph.dist(i, j)
        if endpoint_dist / length < ratio and length < max_length:
            print(i, j)
            deleted_ids.add(fragments_graph.edges[i, j]["swc_id"])
            delete_fragment(fragments_graph, i, j)
    return len(deleted_ids)


# --- Doubles Removal ---
def remove_doubles(fragments_graph, max_length, node_spacing):
    """
    Removes connected components from "fragments_graph" that are likely to be
    a double -- caused by ghosting in the image.

    Parameters
    ----------
    fragments_graph : FragmentsGraph
        Graph to be searched for doubles.
    max_length : int
        Maximum size of connected components to be searched.
    node_spacing : int
        Expected distance (in microns) between nodes in "fragments_graph".

    Returns
    -------
    int
        Number of fragments removed from graph.

    """
    # Initializations
    components = get_line_components(fragments_graph)
    deleted_ids = set()
    kdtree = fragments_graph.get_kdtree()

    # Main
    desc = "Filter Doubled Fragments"
    for idx in tqdm(np.argsort([len(c) for c in components]), desc=desc):
        i, j = tuple(components[idx])
        swc_id = fragments_graph.nodes[i]["swc_id"]
        if swc_id not in deleted_ids:
            if fragments_graph.edges[i, j]["length"] < max_length:
                # Check doubles criteria
                n_points = len(fragments_graph.edges[i, j]["xyz"])
                hits = compute_projections(fragments_graph, kdtree, (i, j))
                if check_doubles_criteria(hits, n_points):
                    delete_fragment(fragments_graph, i, j)
                    deleted_ids.add(swc_id)
    return len(deleted_ids)


def compute_projections(fragments_graph, kdtree, edge):
    """
    Given a fragment defined by "edge", this routine iterates of every xyz in
    the fragment and projects it onto the closest fragment. For each detected
    fragment, the fragment id and projection distance are stored in a
    dictionary called "hits".

    Parameters
    ----------
    fragments_graph : graph
        Graph that contains "edge".
    kdtree : KDTree
        KD-Tree that contains all xyz coordinates of every fragment in
        "fragments_graph".
    edge : tuple
        Pair of leaf nodes that define a fragment.

    Returns
    -------
    dict
        Dictionary that stores all fragments that were detected and the
        projection distances.

    """
    hits = defaultdict(list)
    query_id = fragments_graph.edges[edge]["swc_id"]
    for i, xyz in enumerate(fragments_graph.edges[edge]["xyz"]):
        # Compute projections
        best_id = None
        best_dist = np.inf
        for hit_xyz in geometry.query_ball(kdtree, xyz, QUERY_DIST):
            hit_id = fragments_graph.xyz_to_swc(hit_xyz)
            if hit_id is not None and hit_id != query_id:
                if geometry.dist(hit_xyz, xyz) < best_dist:
                    best_dist = geometry.dist(hit_xyz, xyz)
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


def delete_fragment(fragments_graph, i, j):
    """
    Deletes nodes "i" and "j" from "fragments_graph", where these nodes form a
    connected component.

    Parameters
    ----------
    fragments_graph : FragmentsGraph
        Graph that contains nodes to be removed.
    i : int
        Node to be removed.
    j : int
        Node to be removed.

    Returns
    -------
    fragments_graph
        Graph with nodes removed.

    """
    fragments_graph = remove_xyz_entries(fragments_graph, i, j)
    fragments_graph.swc_ids.remove(fragments_graph.nodes[i]["swc_id"])
    fragments_graph.remove_nodes_from([i, j])


def remove_xyz_entries(fragments_graph, i, j):
    """
    Removes dictionary entries from "fragments_graph.xyz_to_edge"
    corresponding to the edge {i, j}.

    Parameters
    ----------
    fragments_graph : graph
        Graph to be updated.
    i : int
        Node in graph.
    j : int
        Node in graph.

    Returns
    -------
    graph
        Updated graph.

    """
    for xyz in fragments_graph.edges[i, j]["xyz"]:
        try:
            del fragments_graph.xyz_to_edge[tuple(xyz)]
        except KeyError:
            pass
    return fragments_graph


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
