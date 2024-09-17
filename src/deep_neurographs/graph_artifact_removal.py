"""
Created on Sat June 25 9:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Module that removes doubled fragments and trims branches that pass by each
other from a NeuroGraph.

"""
from networkx import connected_components
from tqdm import tqdm

import numpy as np

from deep_neurographs import geometry
from deep_neurographs.utils import util

COLOR = "1.0 0.0 0.0"
QUERY_DIST = 15


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
    components = [c for c in connected_components(neurograph) if len(c) == 2]
    deleted = set()
    kdtree = neurograph.get_kdtree()
    if output_dir:
        util.mkdir(output_dir, delete=True)

    # Main
    desc = "Doubles Detection"
    for idx in tqdm(np.argsort([len(c) for c in components]), desc=desc):
        i, j = tuple(components[idx])
        swc_id = neurograph.nodes[i]["swc_id"]
        if swc_id not in deleted:
            if len(neurograph.edges[i, j]["xyz"]) * node_spacing < max_size:
                # Check doubles criteria
                n_points = len(neurograph.edges[i, j]["xyz"])
                hits = compute_projections(neurograph, kdtree, (i, j))
                if check_doubles_criteria(hits, n_points):
                    if output_dir:
                        neurograph.to_swc(
                            output_dir, components[idx], color=COLOR
                        )
                    neurograph = delete(neurograph, components[idx], swc_id)
                    deleted.add(swc_id)
    print("# Doubles detected:", len(deleted))


def compute_projections(neurograph, kdtree, edge):
    """
    Given a fragment defined by "edge", this routine iterates of every xyz in
    the fragment and projects it onto the closest fragment. For each detected
    fragment, the fragment id and projection distance are stored in a
    dictionary called "hits".

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph that contains "edge".
    kdtree : KDTree
        KD-Tree that contains all xyz coordinates of every fragment in
        "neurograph".
    edge : tuple
        Pair of leaf nodes that define a fragment.

    Returns
    -------
    dict
        Dictionary that stores all fragments that were detected and the
        projection distances.

    """
    hits = dict()
    query_id = neurograph.edges[edge]["swc_id"]
    for i, xyz in enumerate(neurograph.edges[edge]["xyz"]):
        # Compute projections
        best_id = None
        best_dist = np.inf
        for hit_xyz in geometry.query_ball(kdtree, xyz, QUERY_DIST):
            hit_id = neurograph.xyz_to_swc(hit_xyz)
            if hit_id is not None and hit_id != query_id:
                if geometry.dist(hit_xyz, xyz) < best_dist:
                    best_dist = geometry.dist(hit_xyz, xyz)
                    best_id = hit_id

        # Store best
        if best_id:
            hits = util.append_dict_value(hits, best_id, best_dist)
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


def delete(neurograph, nodes, swc_id):
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
