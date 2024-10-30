"""
Created on Sat June 25 9:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Module that removes doubled fragments and trims branches that pass by each
other from a FragmentsGraph.

"""
from collections import defaultdict

import numpy as np
from networkx import connected_components
from tqdm import tqdm

from deep_neurographs import geometry
from deep_neurographs.utils import util

COLOR = "1.0 0.0 0.0"
QUERY_DIST = 15


# --- Curvy Removal ---
def remove_curvy(graph, max_length, ratio=0.5):
    deleted_ids = set()
    components = [c for c in connected_components(graph) if len(c) == 2]
    for nodes in tqdm(components, desc="Filter Curvy Fragment"):
        if len(nodes) == 2:
            i, j = tuple(nodes)
            length = graph.edges[i, j]["length"]
            endpoint_dist = graph.dist(i, j)
            if endpoint_dist / length < ratio and length < max_length:
                deleted_ids.add(graph.edges[i, j]["swc_id"])
                delete_fragment(graph, i, j)
    return len(deleted_ids)


# --- Doubles Removal ---
def remove_doubles(graph, max_length, node_spacing, output_dir=None):
    """
    Removes connected components from "neurgraph" that are likely to be a
    double.

    Parameters
    ----------
    graph : FragmentsGraph
        Graph to be searched for doubles.
    max_length : int
        Maximum size of connected components to be searched.
    node_spacing : int
        Expected distance in microns between nodes in "graph".
    output_dir : str or None, optional
        Directory that doubles will be written to. The default is None.

    Returns
    -------
    graph
        Graph with doubles removed.

    """
    # Initializations
    components = [c for c in connected_components(graph) if len(c) == 2]
    deleted_ids = set()
    kdtree = graph.get_kdtree()
    if output_dir:
        util.mkdir(output_dir, delete=True)

    # Main
    desc = "Filter Doubled Fragment"
    for idx in tqdm(np.argsort([len(c) for c in components]), desc=desc):
        i, j = tuple(components[idx])
        swc_id = graph.nodes[i]["swc_id"]
        if swc_id not in deleted_ids:
            if graph.edges[i, j]["length"] < max_length:
                # Check doubles criteria
                n_points = len(graph.edges[i, j]["xyz"])
                hits = compute_projections(graph, kdtree, (i, j))
                if check_doubles_criteria(hits, n_points):
                    if output_dir:
                        graph.to_swc(output_dir, [i, j], color=COLOR)
                    delete_fragment(graph, i, j)
                    deleted_ids.add(swc_id)
    return len(deleted_ids)


def compute_projections(graph, kdtree, edge):
    """
    Given a fragment defined by "edge", this routine iterates of every xyz in
    the fragment and projects it onto the closest fragment. For each detected
    fragment, the fragment id and projection distance are stored in a
    dictionary called "hits".

    Parameters
    ----------
    graph : graph
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
        for hit_xyz in geometry.query_ball(kdtree, xyz, QUERY_DIST):
            hit_id = graph.xyz_to_swc(hit_xyz)
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


def delete_fragment(graph, i, j):
    """
    Deletes nodes "i" and "j" from "graph", where these nodes form a connected
    component.

    Parameters
    ----------
    graph : FragmentsGraph
        Graph that contains nodes to be deleted.
    i : int
        Node to be removed.
    j : int
        Node to be removed.

    Returns
    -------
    graph
        Graph with nodes removed.

    """
    graph = remove_xyz_entries(graph, i, j)
    graph.swc_ids.remove(graph.nodes[i]["swc_id"])
    graph.remove_nodes_from([i, j])


def remove_xyz_entries(graph, i, j):
    """
    Removes dictionary entries from "graph.xyz_to_edge" corresponding to
    the edge {i, j}.

    Parameters
    ----------
    graph : graph
        Graph to be updated.
    i : int
        Node in "graph".
    j : int
        Node in "graph".

    Returns
    -------
    graph
        Updated graph.

    """
    for xyz in graph.edges[i, j]["xyz"]:
        del graph.xyz_to_edge[tuple(xyz)]
    return graph


def upd_hits(hits, key, value):
    """
    Updates "hits" by adding ("key", "value") if this item does not exist.
    Otherwise, checks if "value" is less than "hits[key"]".

    Parameters
    ----------
    hits : dict
        Stores swd_ids of fragments that are within a certain distance a query
        fragment along with the corresponding distances.
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
    if key in hits.keys():
        if value < hits[key]:
            hits[key] = value
    else:
        hits[key] = value
    return hits
