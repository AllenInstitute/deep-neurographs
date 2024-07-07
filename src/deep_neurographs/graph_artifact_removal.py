"""
Created on Sat June 25 9:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Module that removes doubled fragments and trims branches that pass by each
other from a NeuroGraph.

"""
import sys
from time import time

import networkx as nx
import numpy as np

from deep_neurographs import utils
from deep_neurographs.geometry import dist

COLOR = "1.0 0.0 0.0"
MAX_DEPTH = 16


# --- Doubles Removal ---
def remove_doubles(neurograph, max_size, node_spacing, output_dir=None):
    """
    Removes connected components from "neurgraph" that are likely to be a
    double.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph to be searched for doubles.
    max_size : int
        Maximum size of connected components to be searched.
    node_spacing : int
        Expected distance in microns between nodes in "neurograph".
    output_dir : str or None, optional
        Directory that doubles will be written to. The default is None.

    Returns
    -------
    NeuroGraph
        Graph with doubles removed.

    """
    # Initializations
    deleted = set()
    neurograph.init_kdtree()
    nodes = list(nx.connected_components(neurograph))

    # Main
    cnt = 1
    t0, t1 = utils.init_timers()
    for i, idx in enumerate(np.argsort([len(c) for c in nodes])):
        # Determine whether to check component
        node = utils.sample_singleton(nodes[idx])
        swc_id = neurograph.nodes[node]["swc_id"]
        if len(nodes[idx]) == 2 and swc_id not in deleted:
            fragment_size = len(neurograph.edges[tuple(nodes[idx])]["xyz"])
            if fragment_size * node_spacing < max_size:
                # Check doubles criteria
                n_points = len(neurograph.edges[tuple(nodes[idx])]["xyz"])
                hits = compute_hits(neurograph, tuple(nodes[idx]), swc_id)
                if check_doubles_criteria(hits, n_points):
                    if output_dir:
                        neurograph.to_swc(output_dir, nodes[idx], color=COLOR)
                    neurograph = delete_nodes(neurograph, nodes[idx], swc_id)
                    deleted.add(swc_id)

        # Update progress bar
        if i >= cnt * len(nodes) * 0.02:
            cnt, t1 = utils.report_progress(
                i + 1, len(nodes), len(nodes) * 0.02, cnt, t0, t1
            )
    print("\n# Doubles detected:", len(deleted))


def compute_hits(neurograph, edge, query_id):
    hits = dict()
    for i, xyz in enumerate(neurograph.edges[edge]["xyz"]):
        # Compute projections
        best_id = None
        best_dist = np.inf
        for hit_xyz in neurograph.query_kdtree(xyz, 15):
            try:
                hit_id = neurograph.xyz_to_swc(hit_xyz)
                if hit_id != query_id:
                    if dist(hit_xyz, xyz) < best_dist:
                        best_dist = dist(hit_xyz, xyz)
                        best_id = hit_id
            except:
                pass

        # Store best
        if best_id is not None:
            hits = utils.append_dict_value(hits, best_id, best_dist)

        # Check whether to stop
        if i == 8:
            if len(hits) == 0:
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
        if len(dists) > 5:
            if len(dists) / n_points > 0.5 and np.std(dists) < 2:
                return True
            elif len(dists) / n_points > 0.75 and np.std(dists) < 2.5:
                return True
    return False


def delete_nodes(neurograph, nodes, swc_id):
    """
    Deletes "nodes" from "neurograph".

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that contains "nodes".
    nodes : list[int]
        Nodes to be removed.
    swc_id : str
        swc id corresponding to nodes which comprise a connected component in
        "neurograph".

    Returns
    -------
    NeuroGraph
        Graph with nodes removed.

    """
    i, j = tuple(nodes)
    neurograph = remove_xyz_entries(neurograph, i, j)
    neurograph.remove_nodes_from([i, j])
    neurograph.leafs.remove(i)
    neurograph.leafs.remove(j)
    neurograph.swc_ids.remove(swc_id)
    return neurograph


def remove_xyz_entries(neurograph, i, j):
    """
    Removes dictionary entries from "neurograph.xyz_to_edge" corresponding to
    the edge {i, j}.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph to be updated.
    i : int
        Node in "neurograph".
    j : int
        Node in "neurograph".

    Returns
    -------
    NeuroGraph
        Updated graph.

    """
    for xyz in neurograph.edges[i, j]["xyz"]:
        del neurograph.xyz_to_edge[tuple(xyz)]
    return neurograph


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
