"""
Created on Sat June 25 9:00:00 2024

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Module that removes doubled fragments and trims branches that pass by each
other from a NeuroGraph.

"""
import networkx as nx
import numpy as np
from time import time
import sys

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
    components = list(nx.connected_components(neurograph))
    deleted = set()
    doubles_cnt = 0
    
    cnt = 1
    t0, t1 = utils.init_timers()
    chunk_size = int(len(components) * 0.02)

    # Main
    neurograph.init_kdtree()
    for i, idx in enumerate(np.argsort([len(c) for c in components])):
        # Determine whether to inspect fragment
        nodes = components[idx]
        swc_id = get_swc_id(neurograph, nodes)
        xyz_arr = inspect_component(neurograph, nodes)
        if len(xyz_arr) * node_spacing < max_size and swc_id not in deleted:
            if is_double(neurograph, xyz_arr, swc_id):
                if output_dir:
                    neurograph.to_swc(output_dir, nodes, color=COLOR)
                neurograph = remove_component(neurograph, nodes, swc_id)
                doubles_cnt += 1
                deleted.add(swc_id)

        # Update progress bar
        if i >= cnt * chunk_size:
            cnt, t1 = utils.report_progress(
                i + 1, len(components), chunk_size, cnt, t0, t1
            )
    print("\n# Doubles detected:", doubles_cnt)


def is_double(neurograph, fragment, swc_id_i):
    """
    Determines whether the connected component corresponding to "root" is a
    double of another connected component.

    Paramters
    ---------
    neurograph : NeuroGraph
        Graph to be searched for doubles.
    fragment : numpy.ndarray
        Array containing xyz coordinates corresponding to some fragment (i.e.
        connected component in neurograph).
    swc_id_i : str
        swc id corresponding to fragment.

    Returns
    -------
    str or None
        Indication of whether connected component is a double. If True, the
        swc_id of the main fragment (i.e. non doubles) is returned. Otherwise,
        the value None is returned to indicate that query fragment is not a
        double.

    """
    # Compute projections
    hits = dict()
    for xyz_i in fragment:
        hits_i = dict()
        for xyz_j in neurograph.query_kdtree(xyz_i, 15):
            try:
                swc_id_j = neurograph.xyz_to_swc(xyz_j)
                if swc_id_i != swc_id_j:
                    d = dist(xyz_i, xyz_j)
                    hits_i = upd_hits(hits_i, swc_id_j, d)
            except:
                pass
        if len(hits_i) > 0:
            best_swc_id = utils.find_best(hits_i, maximize=False)
            best_dist = hits_i[best_swc_id]
            hits = utils.append_dict_value(hits, best_swc_id, best_dist)

    # Check criteria
    for dists in hits.values():
        percent_hit = len(dists) / len(fragment)
        std = np.std(dists)
        if percent_hit > 0.5 and std < 2:
            return True
        elif percent_hit > 0.7 and std < 1:
            return True
    return False


def inspect_component(neurograph, nodes):
    """
    Determines whether to inspect component for doubles.

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph to be searched.
    nodes : iterable
        Nodes that comprise a connected component.

    Returns
    -------
    numpy.ndarray or list
        Array containing xyz coordinates of nodes.

    """
    if len(nodes) == 2:
        i, j = tuple(nodes)
        return neurograph.edges[i, j]["xyz"]
    else:
        return []


def remove_component(neurograph, nodes, swc_id):
    """
    Removes "nodes" from "neurograph".

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
    Otherwise, checks whether "value" is less than "hits[key"]".
    """
    if key in hits.keys():
        if value < hits[key]:
            hits[key] = value
    else:
        hits[key] = value
    return hits


# --- Trim Passings ---
def trim_passings(neurograph):
    # Initializations
    cnt = 1
    t0, t1 = utils.init_timers()
    chunk_size = int(len(neurograph.leafs) * 0.02)

    # Main
    n_endpoints_trimmed = 0
    for leaf in neurograph.leafs:
        trim_bool = inspect_branch(neurograph, leaf)
        if trim_bool:
            n_endpoints_trimmed += 1
    print("# hits:", n_endpoints_trimmed)


def inspect_branch(neurograph, i):
    trim_bool = False
    swc_id = neurograph.nodes[i]["swc_id"]
    for xyz, radius in get_branch(neurograph, i):
        hits = search_along_branch(neurograph, swc_id, xyz, radius)
        if len(hits) > 0:
            trim_bool = True

        hits = keep_passings(hits)
        
    if trim_bool:
        print("")
    return False


def search_along_branch(neurograph, swc_id_leaf, xyz_leaf, radius):
    hits = dict()
    for xyz in neurograph.query_kdtree(xyz_leaf, radius):
        try:
            swc_id = neurograph.xyz_to_swc(xyz)
            if swc_id != swc_id_leaf:
                hits = utils.append_dict_value(hits, swc_id, xyz)
                d = dist(xyz, xyz_leaf)
                print(swc_id_leaf, swc_id, d, xyz)
        except:
            pass
    return hits


def keep_passings(hits):
    rm_keys = list()
    for swc_id, xyz_coords in hits.items():
        if compute_length(xyz_coords) < 5:
            rm_keys.append(swc_id)
    return utils.remove_items(hits, rm_keys)


# --- utils ---
def get_swc_id(neurograph, nodes):
    """
    Gets the swc id corresponding to "nodes".

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph containing "nodes".
    nodes : list[int]
        Nodes to be checked.

    Returns
    -------
    str
        swc id of "nodes".

    """
    i = utils.sample_singleton(nodes)
    return neurograph.nodes[i]["swc_id"]


def get_branch(neurograph, i):
    """
    Gets the xyz coordinates of the branch emanating from "i".

    Parameters
    ----------
    neurograph : NeuroGraph
        Graph containing node "i".
    i : int
        Leaf node in "neurograph".

    Returns
    -------
    numpy.ndarray
        xyz coordinates of branch emanating from "i".

    """
    j = list(neurograph.neighbors(i))[0]
    xyz_coords = neurograph.oriented_edge((i, j), i)
    radii = neurograph.oriented_edge((i, j), i, key="radius")
    n = max(len(xyz_coords), MAX_DEPTH)
    return zip(xyz_coords[0:n], radii[0:n])


def compute_length(path):
    """
    Computes the length of "path".

    Parameters
    ----------
    path : list
        xyz coordinates that form a path.

    Returns
    -------
    float
        Length of "path".

    """
    return np.sum([dist(path[i], path[i - 1]) for i range(1, len(path))])
